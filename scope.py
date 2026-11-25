"""Deterministic parsing of recorded-event comments: header, scope fields, results, instruction-like text."""
from __future__ import annotations
import re

HEADER = re.compile(r"^Recorded event (?P<ts>\S+) \| actor (?P<actor>.+?) \| role (?P<role>.+?)\n(?P<body>.*)$", re.S)
FIELD_PATTERNS = {
    "release": re.compile(r"\brelease (R\d+(?:\.\d+)*)", re.I),
    "commit": re.compile(r"\bcommit ([0-9a-f]{7,40})\b", re.I),
    "service": re.compile(r"\bservice ([A-Za-z][\w-]*)", re.I),
    "environment": re.compile(r"\benvironment ([A-Za-z][\w-]*)", re.I),
    "prompt": re.compile(r"\bprompt (?:bundle )?(p\d+)\b", re.I),
    "approval_scope": re.compile(r"\bApproval scope: ([A-Za-z][\w-]*)", re.I),
}
EVIDENCE_HINT = re.compile(
    r"release R\d|commit [0-9a-f]{7}|\bpass|\bfail|approv|defect|supersed|waiv|prompt|cases|p95|cost|latency|"
    r"\d+ of \d+|percent|retest|reopen|duplicate|lost|drop", re.I)
INSTRUCTION_LIKE = re.compile(
    r"automated (release )?review|note for (ai|automated|llm)|ignore (all |the |previous )|do not mention|"
    r"report .{0,40} as (pass|fail|ready)|system prompt|disregard", re.I)
RESULT_PATTERNS = {
    "cases_passed_of_total": re.compile(r"(\d+) of (\d+)[^.;]*?\bpass", re.I),
    "cases_failed_of_total": re.compile(r"(\d+) of (\d+)[^.;]*?\bfail", re.I),
    "reached_queue": re.compile(r"(\d+) of (\d+) escalated conversations reached", re.I),
    "percent": re.compile(r"(\d+(?:\.\d+)?) percent", re.I),
    "p95_seconds": re.compile(r"p95 latency (\d+(?:\.\d+)?) seconds", re.I),
    "cost_usd": re.compile(r"cost USD (\d+(?:\.\d+)?)", re.I),
    "supersedes": re.compile(r"supersedes ([A-Z]+-\d+)", re.I),
}


def parse_comment(text: str) -> dict:
    m = HEADER.match(text)
    header = {"timestamp": m["ts"], "actor": m["actor"], "role": m["role"]} if m else {}
    body = m["body"] if m else text
    return {**header, "body": body.strip(), "has_header": bool(m)}


def is_evidence_bearing(body: str) -> bool:
    return bool(EVIDENCE_HINT.search(body))


def instruction_like(body: str) -> bool:
    return bool(INSTRUCTION_LIKE.search(body))


def extract_fields(body: str) -> dict:
    return {k: (p.search(body).group(1) if p.search(body) else None) for k, p in FIELD_PATTERNS.items()}


def extract_results(body: str) -> dict:
    out = {}
    for k, p in RESULT_PATTERNS.items():
        m = p.search(body)
        if m:
            out[k] = list(m.groups()) if len(m.groups()) > 1 else m.group(1)
    return out


def parse_target(manifest_description: str) -> dict | None:
    import json
    m = re.search(r"Target:\s*(\{.*?\})", manifest_description, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def compare_scope(fields: dict, target: dict, require_prompt: bool) -> dict:
    """Compare comment-stated scope with the target. Missing != matching."""
    env_val = fields.get("environment") or fields.get("approval_scope")
    stated = {"release": fields.get("release"), "commit": fields.get("commit"), "service": fields.get("service"),
              "environment": env_val, "prompt": fields.get("prompt")}
    result, mismatches, missing = {}, [], []
    for k, v in stated.items():
        want = target.get(k)
        if v is None:
            result[k] = "not_stated"
            if k != "prompt" or require_prompt:
                missing.append(k)
        elif v.lower() == str(want).lower():
            result[k] = "match"
        else:
            result[k] = f"MISMATCH (stated {v}, target {want})"
            mismatches.append(k)
    return {"fields": result, "mismatches": mismatches, "missing_required": missing,
            "full_scope_match": not mismatches and not missing,
            "note": "approval_scope used as environment" if fields.get("approval_scope") and not fields.get("environment") else ""}
