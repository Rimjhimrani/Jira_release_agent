"""Repeatable, idempotent importer: dataset.json -> Jira Cloud project.

Preserves: source id (summary prefix + `src-<id>` label + description header), kind/release/recorded
status (labels + description), full original description, every comment verbatim (actor, role and
recorded timestamp stay in the comment text; native Jira author/time belong to the importer), and
links as Jira issue links.  Re-running only creates what is missing.
"""
from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .jira_client import JiraClient, adf, JiraError

REQUIREMENT_LABEL = "rr-requirement"  # the manifest says requirement records carry this label


def _label(s: str) -> str:
    return re.sub(r"\s+", "_", s.strip())


def issue_payload(rec: dict, project: str, dataset_label: str, issue_type: str) -> dict:
    labels = [dataset_label, f"src-{rec['source_id']}", f"kind-{_label(rec['kind'])}", f"rel-{_label(rec['release'])}"]
    if rec["kind"] == "requirement":
        labels.append(REQUIREMENT_LABEL)
    header = (f"Source ID: {rec['source_id']}\nKind: {rec['kind']}\nRelease: {rec['release']}\n"
              f"Recorded status: {rec['recorded_status']}\nLinked source IDs: {', '.join(rec['links']) or '-'}\n---\n")
    return {"fields": {
        "project": {"key": project},
        "issuetype": {"name": issue_type},
        "summary": f"[{rec['source_id']}] {rec['summary']}"[:255],
        "description": adf(header + rec["description"]),
        "labels": labels,
    }}


def import_dataset(jira: JiraClient, dataset: dict, project: str, issue_type: str = "Task",
                   link_type: str = "Relates", workers: int = 4, log=print) -> dict:
    label = dataset["dataset_label"]
    recs = dataset["records"]
    t0 = time.time()
    stats = {"issues_created": 0, "issues_existing": 0, "comments_added": 0, "links_added": 0, "errors": []}

    # 1) existing issues carrying this dataset label -> {source_id: key}
    existing = {}
    for it in jira.search_all(f'project = "{project}" AND labels = "{label}"', ["labels"]):
        for lb in it["fields"].get("labels", []):
            if lb.startswith("src-"):
                existing[lb[4:]] = it["key"]
    keymap = dict(existing)
    stats["issues_existing"] = len(existing)

    # 2) create missing issues in bulk of 50
    todo = [r for r in recs if r["source_id"] not in keymap]
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        res = jira.bulk_create([issue_payload(r, project, label, issue_type) for r in chunk])
        for r, created in zip(chunk, res.get("issues", [])):
            keymap[r["source_id"]] = created["key"]
            stats["issues_created"] += 1
        for e in res.get("errors", []):
            stats["errors"].append({"stage": "create", "error": e})
        log(f"  created {min(i + 50, len(todo))}/{len(todo)} issues")

    # 3) comments (sequential per issue to keep order; parallel across issues)
    def add_comments(rec):
        key = keymap[rec["source_id"]]
        have, start = 0, 0
        while True:
            page = jira.get_comments(key, start, 100)
            have += len(page.get("comments", []))
            if start + 100 >= page.get("total", 0):
                break
            start += 100
        n = 0
        for body in rec["comments"][have:]:
            jira.add_comment(key, body)  # verbatim: header line carries scenario actor/role/timestamp
            n += 1
        return n

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(add_comments, r): r for r in recs if r["source_id"] in keymap}
        done = 0
        for f in as_completed(futs):
            try:
                stats["comments_added"] += f.result()
            except JiraError as e:
                stats["errors"].append({"stage": "comments", "source_id": futs[f]["source_id"], "error": str(e)})
            done += 1
            if done % 50 == 0:
                log(f"  comments processed for {done}/{len(futs)} issues")

    # 4) links (skip those already present)
    for rec in recs:
        if not rec["links"] or rec["source_id"] not in keymap:
            continue
        key = keymap[rec["source_id"]]
        issue = jira.get_issue(key, "issuelinks")
        linked = {(l.get("outwardIssue") or l.get("inwardIssue"))["key"] for l in issue["fields"].get("issuelinks", [])}
        for tgt in rec["links"]:
            tkey = keymap.get(tgt)
            if not tkey:
                stats["errors"].append({"stage": "link", "source_id": rec["source_id"], "error": f"unknown target {tgt}"})
                continue
            if tkey in linked:
                continue
            try:
                jira.link(link_type, key, tkey)
                stats["links_added"] += 1
            except JiraError as e:
                stats["errors"].append({"stage": "link", "source_id": rec["source_id"], "error": str(e)})

    stats["seconds"] = round(time.time() - t0, 1)
    stats["jira_requests"] = jira.request_count
    return stats


def verify_import(jira: JiraClient, dataset: dict, project: str) -> dict:
    """Post-import check: issue count, comment counts and link counts match the dataset."""
    label = dataset["dataset_label"]
    items = jira.search_all(f'project = "{project}" AND labels = "{label}"', ["labels", "comment", "issuelinks"])
    by_src = {}
    for it in items:
        for lb in it["fields"]["labels"]:
            if lb.startswith("src-"):
                by_src[lb[4:]] = it
    problems = []
    for r in dataset["records"]:
        it = by_src.get(r["source_id"])
        if not it:
            problems.append(f"missing {r['source_id']}")
            continue
        n = jira.get_comments(it["key"], 0, 1).get("total", 0)
        if n != len(r["comments"]):
            problems.append(f"{r['source_id']}: comments {n} != {len(r['comments'])}")
        if len(it["fields"].get("issuelinks", [])) < len(r["links"]):
            problems.append(f"{r['source_id']}: links missing")
    return {"expected_issues": len(dataset["records"]), "found_issues": len(by_src), "problems": problems[:50],
            "ok": not problems and len(by_src) == len(dataset["records"])}
