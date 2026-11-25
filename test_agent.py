import json, os, sys, threading, importlib
from http.server import ThreadingHTTPServer
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import mock_jira
from release_agent.jira_client import JiraClient
from release_agent.importer import import_dataset, verify_import
from release_agent.tools import JiraTools
from release_agent.agent import Agent, Budgets
from release_agent.llm import OfflinePolicy, LLMResponse
from release_agent.report import overall

DS = json.loads((ROOT / "inputs/dataset.json").read_text())
LABEL, PROJECT = DS["dataset_label"], "TYN"


@pytest.fixture(scope="module")
def jira():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mock_jira.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    j = JiraClient(f"http://127.0.0.1:{srv.server_address[1]}", "x", "y")
    # a foreign issue WITHOUT the dataset label must never be visible to the agent
    j.bulk_create([{"fields": {"summary": "[REQ-99] foreign", "description": {"type": "doc", "version": 1, "content": []}, "labels": ["other"]}}])
    import_dataset(j, DS, PROJECT, log=lambda *_: None)
    yield j
    srv.shutdown()


def test_import_complete_and_idempotent(jira):
    assert verify_import(jira, DS, PROJECT)["ok"]
    again = import_dataset(jira, DS, PROJECT, log=lambda *_: None)
    assert again["issues_created"] == 0 and again["comments_added"] == 0 and again["links_added"] == 0


def test_scenario_actor_and_timestamp_preserved_in_text(jira):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4"); t.build_index()
    c = t.get_comments("REC-6024", include_routine=True)["comments"]
    assert any(x["actor"] == "Priya Nair" and x["recorded_at"] == "2026-11-08T14:00:00+05:30" for x in c)


def test_scope_label_enforced(jira):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4"); t.build_index()
    assert "foreign" not in json.dumps(t.search_issues(text="foreign"))
    assert json.loads(t.call("get_issue", {"source_id": "TYN-1"}))["error"] == "ToolError"


def test_record_decision_guardrails(jira):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4"); t.build_index()
    r = json.loads(t.call("record_decision", {"requirement_id": "REQ-04", "decision": "PASS", "rationale": "x", "supporting_refs": ["REC-6024#c3"], "outstanding": "none"}))
    assert "never retrieved" in r["message"]                       # unseen ref rejected
    t.call("get_comments", {"source_id": "REC-6024"})
    r = json.loads(t.call("record_decision", {"requirement_id": "REQ-04", "decision": "PASS", "rationale": "x", "supporting_refs": ["REC-6024#c3"], "outstanding": "none"}))
    assert "check_scope" in r["message"]                           # PASS needs a verified scope match
    s = json.loads(t.call("check_scope", {"ref": "REC-6024#c3"}))
    assert s["scope_comparison"]["full_scope_match"]
    ok = json.loads(t.call("record_decision", {"requirement_id": "REQ-04", "decision": "PASS", "rationale": "x", "supporting_refs": ["REC-6024#c3"], "outstanding": "none"}))
    assert ok["decision"] == "PASS"


def test_scope_traps(jira):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4"); t.build_index()
    def chk(sid, idx_pred, rp=False):
        for c in t.get_comments(sid, include_routine=True)["comments"]:
            if idx_pred in c["text"]:
                return t.check_scope(c["ref"], rp)["scope_comparison"]
    assert not chk("REC-2299", "REG-779")["full_scope_match"]            # earlier commit
    assert not chk("REC-1028", "PERF-812")["full_scope_match"]           # staging
    assert not chk("REC-6499", "REDACT-905")["full_scope_match"]         # other service
    assert not chk("REC-2828", "Approved", True)["full_scope_match"]     # staging approval
    assert not chk("REC-4323", "EVAL-55 completed", True)["full_scope_match"]  # prompt not stated


def test_instruction_like_text_is_flagged_and_collapsed_noise(jira):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4"); t.build_index()
    flagged = [c for c in t.get_issue("REC-5716")["substantive_comments"] if "INTEGRITY_FLAG" in c]
    assert flagged
    g = t.get_issue("REC-5957")
    assert g["routine_comments_collapsed"]["count"] >= 60 and len(json.dumps(g)) < 6000   # 62-comment ticket stays small


def run_agent(jira, llm, **b):
    t = JiraTools(jira, PROJECT, LABEL, "R2.4")
    a = Agent(llm, t, Budgets(**b), None, log=lambda *_: None); a.run(); return t, a


def test_offline_end_to_end_decisions(jira):
    t, a = run_agent(jira, OfflinePolicy())
    got = {k: v["decision"] for k, v in t.decisions.items()}
    assert got == {"REQ-01": "PASS", "REQ-02": "FAIL", "REQ-03": "INSUFFICIENT_EVIDENCE", "REQ-04": "PASS", "REQ-05": "FAIL",
                   "REQ-06": "INSUFFICIENT_EVIDENCE", "REQ-07": "PASS", "REQ-08": "FAIL", "REQ-09": "INSUFFICIENT_EVIDENCE",
                   "REQ-10": "PASS", "REQ-11": "INSUFFICIENT_EVIDENCE", "REQ-12": "FAIL"}
    assert overall(list(t.decisions.values())) == "BLOCKED" and not a.st.errors


class Babbler:   # model that never calls a tool -> must fail honestly, not crash
    name, model = "babbler", "x"
    def create(self, system, messages, tools):
        return LLMResponse([{"type": "text", "text": "I think everything is fine."}])


def test_model_failure_reported_not_hidden(jira):
    t, a = run_agent(jira, Babbler())
    # discovery failure falls back to Jira label query; every requirement is INSUFFICIENT with an explicit agent_error
    assert len(t.decisions) == 12 and all(d["decision"] == "INSUFFICIENT_EVIDENCE" and d.get("agent_error") for d in t.decisions.values())
    assert any("discovery failed" in e for e in a.st.errors)


def test_budget_enforced(jira):
    t, a = run_agent(jira, OfflinePolicy(), max_total_tool_calls=10)
    assert any("budget" in e for e in a.st.errors) and a.st.usage["tool_calls"] <= 13


def test_overall_rules():
    d = lambda *x: [{"decision": i} for i in x]
    assert overall(d("PASS", "FAIL", "INSUFFICIENT_EVIDENCE")) == "BLOCKED"
    assert overall(d("PASS", "INSUFFICIENT_EVIDENCE")) == "NEEDS_REVIEW"
    assert overall(d("PASS", "PASS")) == "READY"
