"""CLI:  python -m release_agent {import|verify-import|assess} ..."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from .jira_client import JiraClient, adf_to_text
from .importer import import_dataset, verify_import
from .tools import JiraTools
from .agent import Agent, Budgets
from .report import build_report, overall
from . import scope as S


def load_env():
    f = Path(".env")
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def jira_from_env(a) -> JiraClient:
    base = a.jira_url or os.environ.get("JIRA_BASE_URL")
    if not base:
        sys.exit("JIRA_BASE_URL (or --jira-url) is required")
    return JiraClient(base, os.environ.get("JIRA_EMAIL", ""), os.environ.get("JIRA_API_TOKEN", ""))


def cmd_import(a):
    ds = json.loads(Path(a.dataset).read_text())
    jira = jira_from_env(a)
    print(f"Importing {len(ds['records'])} records into {a.project} with label {ds['dataset_label']}")
    st = import_dataset(jira, ds, a.project, a.issue_type, a.link_type)
    print(json.dumps(st, indent=2))
    v = verify_import(jira, ds, a.project)
    print("verify:", json.dumps(v, indent=2))
    sys.exit(0 if v["ok"] and not st["errors"] else 1)


def cmd_verify(a):
    ds = json.loads(Path(a.dataset).read_text())
    v = verify_import(jira_from_env(a), ds, a.project)
    print(json.dumps(v, indent=2))
    sys.exit(0 if v["ok"] else 1)


def make_llm(a):
    if a.provider == "offline":
        from .llm import OfflinePolicy
        return OfflinePolicy()
    from .llm import AnthropicLLM
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is required for --provider anthropic")
    return AnthropicLLM(a.model)


def exec_summary(llm, decisions, ov, agent) -> str:
    facts = json.dumps([{k: d[k] for k in ("requirement_id", "decision", "rationale")} for d in decisions])
    prompt = (f"EXEC_SUMMARY: Overall decision (computed by code, do not change): {ov}. In 4-6 sentences, summarise why, naming the blocking "
              f"requirements (FAIL) and those needing review (INSUFFICIENT_EVIDENCE) by id. Use only these facts: {facts}")
    try:
        r = agent._model("summary", "You write concise, factual release summaries.", [{"role": "user", "content": prompt}], [])
        t = " ".join(b["text"] for b in r.blocks if b["type"] == "text").strip()
        if t and not t.startswith("Offline baseline"):
            return t
    except Exception as e:
        agent.st.errors.append(f"summary generation failed ({type(e).__name__}); deterministic summary used")
    by = lambda k: ", ".join(d["requirement_id"] for d in decisions if d["decision"] == k) or "none"
    return (f"Overall decision {ov}. FAIL: {by('FAIL')}. INSUFFICIENT_EVIDENCE: {by('INSUFFICIENT_EVIDENCE')}. PASS: {by('PASS')}. "
            "Decisions are computed from per-requirement findings; see each section for evidence and outstanding items.")


def cmd_assess(a):
    jira = jira_from_env(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.jsonl"
    trace.write_text("")
    llm = make_llm(a)
    tools = JiraTools(jira, a.project, a.label, a.release)
    ag = Agent(llm, tools, Budgets(a.max_turns, 8, a.max_tool_calls, a.max_tokens, a.max_seconds), str(trace))
    t0 = time.time()
    ids = ag.run()
    summary_meta = {}
    for rid in ids:
        try:
            d = adf_to_text(tools._issue(rid)["fields"]["description"])
            summary_meta[rid] = {"summary": tools.meta[rid]["summary"], "criterion": d.split("Acceptance criterion:")[-1].strip()}
        except Exception:
            summary_meta[rid] = {"summary": tools.meta[rid]["summary"]}
    decisions = [tools.decisions[r] for r in ids if r in tools.decisions]
    ov = overall(decisions)
    summary = exec_summary(llm, decisions, ov, ag)
    elapsed = time.time() - t0
    model = getattr(llm, "model", "?")
    rep = build_report(project=a.project, label=a.label, target=tools.target(), req_meta=summary_meta, decisions=decisions, tools=tools,
                       usage=ag.st.usage, errors=ag.st.errors, elapsed=elapsed, provider=llm.name, model=model, summary_text=summary,
                       jira_base=jira.base)
    (out / "report.md").write_text(rep)
    metrics = {"overall": ov, "elapsed_seconds": round(elapsed, 1), "provider": llm.name, "model": model, **ag.st.usage,
               "jira_requests": jira.request_count, "errors": ag.st.errors,
               "decisions": {d["requirement_id"]: d["decision"] for d in decisions}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Report: {out / 'report.md'}")


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(prog="release_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    def common(sp):
        sp.add_argument("--jira-url"); sp.add_argument("--project", default=os.environ.get("JIRA_PROJECT_KEY"), required=not os.environ.get("JIRA_PROJECT_KEY"))
    i = sub.add_parser("import"); common(i)
    i.add_argument("--dataset", default="inputs/dataset.json"); i.add_argument("--issue-type", default="Task"); i.add_argument("--link-type", default="Relates")
    i.set_defaults(fn=cmd_import)
    v = sub.add_parser("verify-import"); common(v); v.add_argument("--dataset", default="inputs/dataset.json"); v.set_defaults(fn=cmd_verify)
    s = sub.add_parser("assess"); common(s)
    s.add_argument("--label", required=True, help="dataset scope label, e.g. tyn-release-assessment-v2")
    s.add_argument("--release", required=True, help="target release, e.g. R2.4")
    s.add_argument("--provider", choices=["anthropic", "offline"], default="anthropic")
    s.add_argument("--model"); s.add_argument("--out", default="outputs")
    s.add_argument("--max-turns", type=int, default=14); s.add_argument("--max-tool-calls", type=int, default=250)
    s.add_argument("--max-tokens", type=int, default=600000); s.add_argument("--max-seconds", type=int, default=1200)
    s.set_defaults(fn=cmd_assess)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
