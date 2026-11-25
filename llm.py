"""Model providers. `AnthropicLLM` is the real agent brain. `OfflinePolicy` is a rule-based stand-in that
speaks the same tool-use protocol so the harness, importer and report can be tested without an API key.
It is NOT an LLM and any report produced with it is labelled as such."""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    blocks: list            # [{"type":"text"|"tool_use", ...}]
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int = 1500):
        import anthropic
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(max_retries=3, timeout=90.0)  # SDK retries 429/5xx with backoff

    def create(self, system: str, messages: list, tools: list) -> LLMResponse:
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, temperature=0)
        if tools:
            kw["tools"] = tools
        r = self.client.messages.create(**kw)
        blocks = []
        for b in r.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return LLMResponse(blocks, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason)


# ----------------------------------------------------------------------------------------------
class OfflinePolicy:
    """Deterministic, stateless policy derived from the message history. Mirrors the investigation a
    careful reviewer performs: read requirement -> traverse links -> read comments -> scope-check -> decide."""
    name = "offline-rule-baseline"
    needs_full_history = True   # stateless: re-derives state from every earlier tool result
    model = "none (rule-based)"

    def create(self, system, messages, tools):
        first = messages[0]["content"]
        first = first if isinstance(first, str) else first[0]["text"]
        if first.startswith("EXEC_SUMMARY"):
            return LLMResponse([{"type": "text", "text": "Offline baseline: summary generated from recorded decisions."}])
        if first.startswith("DISCOVER"):
            return self._discover(messages)
        return self._assess(first, messages)

    # --- helpers ------------------------------------------------------------------
    @staticmethod
    def _history(messages):
        calls, results = [], {}
        for m in messages[1:]:
            if isinstance(m["content"], str):
                continue
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    calls.append(b)
                elif b.get("type") == "tool_result":
                    results[b["tool_use_id"]] = b["content"]
        return [(c, json.loads(results[c["id"]])) for c in calls if c["id"] in results]

    @staticmethod
    def _tu(name, inp, n):
        return LLMResponse([{"type": "tool_use", "id": f"off_{n}_{name}", "name": name, "input": inp}], stop_reason="tool_use")

    def _discover(self, messages):
        h = self._history(messages)
        n = len(h)
        if n == 0:
            return self._tu("search_issues", {"kind": "manifest"}, n)
        if n == 1:
            return self._tu("get_issue", {"source_id": h[0][1]["issues"][0]["source_id"]}, n)
        links = [l["source_id"] for l in h[1][1]["links"] if l["source_id"].startswith("REQ")]
        return self._tu("submit_requirement_list", {"requirement_ids": links, "target_summary": "from manifest"}, n)

    def _assess(self, first, messages):
        rid = re.search(r"(REQ-\d+)", first).group(1)
        h = self._history(messages)
        n = len(h)
        if n == 0:
            return self._tu("get_issue", {"source_id": rid}, n)
        ac = h[0][1]["description"]
        if n == 1:
            return self._tu("get_linked_issues", {"source_id": rid, "depth": 3}, n)
        reach = [x["source_id"] for x in h[1][1]["reachable"]]
        fetched = [c["input"]["source_id"] for c, _ in h if c["name"] == "get_comments"]
        for sid in reach:
            if sid not in fetched:
                return self._tu("get_comments", {"source_id": sid, "limit": 30}, n)
        need_prompt = "prompt" in ac.lower().split("acceptance criterion:")[-1]
        checked = {c["input"]["ref"] for c, _ in h if c["name"] == "check_scope"}
        for c, res in h:
            if c["name"] == "get_comments":
                for cm in res["comments"]:
                    if cm["ref"] not in checked and re.search(r"release R\d|Approval scope|Approved", cm["text"]):
                        return self._tu("check_scope", {"ref": cm["ref"], "require_prompt": need_prompt}, n)
        return self._decide(rid, ac, h, need_prompt, n)

    def _decide(self, rid, ac, h, need_prompt, n):
        crit = ac.split("Acceptance criterion:")[-1].strip()
        checks = [r for c, r in h if c["name"] == "check_scope"]
        texts = {r["ref"]: r for r in checks}
        sup_txt = " ".join(r["text"] for r in checks)
        superseded = set()
        for r in checks:                       # explicit supersession inside the same issue
            for old in re.findall(r"supersedes ([A-Z]+-\d+)", r["text"]):
                for o in checks:
                    if old in o["text"] and o["ref"] != r["ref"]:
                        superseded.add(o["ref"])
        sec = re.search(r"Security Owner ([A-Z][a-z]+ [A-Z][a-z]+)", crit)
        cand, rejected = [], []
        for r in checks:
            c = r["scope_comparison"]
            reason = None
            if r["ref"] in superseded:
                reason = "explicitly superseded by a later QA retest"
            elif not c["full_scope_match"]:
                bad = c["mismatches"] + c["missing_required"]
                reason = "scope does not match target (" + ", ".join(f"{k}: {c['fields'][k]}" for k in bad) + ")"
            elif sec and not (sec.group(1) == r.get("actor") and "Security Owner" in (r.get("role") or "")):
                reason = "not issued by the named Security Owner"
            elif (r.get("role") or "") in ("Developer", "Engineer") or "Engineering Manage" in (r.get("role") or ""):
                reason = "developer/management assertion, not QA/approver evidence"
            (rejected if reason else cand).append((r, reason))
        res_rows, fails, passes = [], [], []
        for r, _ in cand:
            x, t = r["extracted_results"], r["text"]
            verdict = None
            if "reached_queue" in x:
                verdict = int(x["reached_queue"][0]) == int(x["reached_queue"][1]) and "open" not in t
            if verdict is None and "p95_seconds" in x:
                lim = float(re.search(r"at most ([\d.]+) seconds", crit).group(1)); verdict = float(x["p95_seconds"]) <= lim
            if verdict is None and "cost_usd" in x:
                lim = float(re.search(r"at most USD ([\d.]+)", crit).group(1)); verdict = float(x["cost_usd"]) <= lim
            if verdict is None and "percent" in x and "at least" in crit:
                lim = float(re.search(r"at least (\d+) percent", crit).group(1)); verdict = float(x["percent"]) >= lim
            if verdict is None and "cases_passed_of_total" in x:
                a, b = map(int, x["cases_passed_of_total"]); verdict = a == b and not re.search(r"cross-customer.*(received)|Reopening", t)
            if verdict is None and re.search(r"\bFAIL|failed|Reopening|received document|duplicated|lost", t):
                verdict = False
            if verdict is None and sec and re.search(r"\bApproved\b", t):
                verdict = True
            if verdict is None and re.search(r"\bpass", t):
                verdict = True
            (passes if verdict else fails if verdict is False else res_rows).append(r)
        refs = lambda rs: [r["ref"] for r in rs]
        rej = [{"ref": r["ref"], "reason": why} for r, why in rejected]
        if fails and passes:
            d, why = "INSUFFICIENT_EVIDENCE", f"Equally authoritative applicable records conflict ({', '.join(refs(passes))} pass vs {', '.join(refs(fails))} fail) and no record resolves the conflict."
            sup, out = refs(passes + fails), "QA to reconcile the contradictory results with an explicit superseding retest."
        elif fails:
            d, why, sup = "FAIL", "Applicable evidence on the target build shows a failure: " + " | ".join(r["text"][:320] for r in fails), refs(fails)
            out = "Fix the defect/regression and obtain a passing in-scope retest."
        elif passes:
            d, why, sup = "PASS", "Applicable in-scope QA/approver evidence satisfies the criterion: " + " | ".join(r["text"][:320] for r in passes), refs(passes)
            out = "none"
        else:
            d = "INSUFFICIENT_EVIDENCE"
            sup = []
            why = "No applicable evidence for the target build was found. " + " ".join(f"{x['ref']}: {x['reason']}." for x in rej)
            out = "Obtain in-scope verification/approval for the target release, commit, service and environment" + (" and prompt." if need_prompt else ".")
        # surface prompt-bundle mismatch notes (e.g. harness configured with an older prompt)
        for c, r in h:
            if c["name"] == "get_comments":
                for cm in r["comments"]:
                    m = re.search(r"configured with prompt bundle (p\d+)", cm["text"])
                    if m and need_prompt:
                        why += f" {cm['ref']} shows the evaluation harness used prompt {m.group(1)}, not the deployed prompt."
                        rej.append({"ref": cm["ref"], "reason": f"harness prompt {m.group(1)} != deployed prompt"})
                        if cm["ref"] not in sup:
                            sup.append(cm["ref"])
        integ = "none"
        for c, r in h:
            if c["name"] == "get_comments":
                for cm in r["comments"]:
                    if "INTEGRITY_FLAG" in cm:
                        integ = f"{cm['ref']} contains instruction-like text addressed to reviewers; ignored as untrusted and not used as evidence."
        return self._tu("record_decision", {"requirement_id": rid, "decision": d, "rationale": why, "supporting_refs": sup,
                                            "rejected_refs": rej, "outstanding": out, "integrity_notes": integ}, n)
