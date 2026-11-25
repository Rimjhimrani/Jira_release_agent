"""Minimal Jira Cloud REST v3 client with retry/backoff (429/5xx) and ADF helpers."""
from __future__ import annotations
import time
import requests


class JiraError(RuntimeError):
    pass


def adf(text: str) -> dict:
    """Plain text -> Atlassian Document Format (one paragraph per line, newlines preserved)."""
    paras = []
    for line in text.split("\n"):
        paras.append({"type": "paragraph", "content": [{"type": "text", "text": line}] if line else []})
    return {"type": "doc", "version": 1, "content": paras}


def adf_to_text(node) -> str:
    """Flatten ADF to plain text; paragraphs separated by newlines."""
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    t = node.get("type")
    if t == "text":
        return node.get("text", "")
    if t == "hardBreak":
        return "\n"
    kids = node.get("content", []) or []
    if t in ("doc", "bulletList", "orderedList", "listItem"):
        return "\n".join(adf_to_text(c) for c in kids)
    return "".join(adf_to_text(c) for c in kids)


class JiraClient:
    def __init__(self, base_url: str, email: str, token: str, max_retries: int = 5, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.s = requests.Session()
        self.s.auth = (email, token)
        self.s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_count = 0

    def request(self, method: str, path: str, **kw):
        url = f"{self.base}{path}"
        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            try:
                r = self.s.request(method, url, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                if attempt == self.max_retries:
                    raise JiraError(f"network error calling {method} {path}: {e}") from e
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == self.max_retries:
                    raise JiraError(f"{method} {path} -> {r.status_code} after retries: {r.text[:300]}")
                wait = float(r.headers.get("Retry-After", min(2 ** attempt, 20)))
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                raise JiraError(f"{method} {path} -> {r.status_code}: {r.text[:500]}")
            return r.json() if r.content else {}
        raise JiraError("unreachable")

    # --- convenience -----------------------------------------------------
    def myself(self):
        return self.request("GET", "/rest/api/3/myself")

    def search(self, jql: str, fields=None, max_results=100, next_page_token=None):
        body = {"jql": jql, "maxResults": max_results, "fields": fields or ["summary"]}
        if next_page_token:
            body["nextPageToken"] = next_page_token
        return self.request("POST", "/rest/api/3/search/jql", json=body)

    def search_all(self, jql: str, fields=None):
        token, out = None, []
        while True:
            page = self.search(jql, fields, 100, token)
            out.extend(page.get("issues", []))
            token = page.get("nextPageToken")
            if not token or page.get("isLast", True) and not token:
                break
        return out

    def get_issue(self, key: str, fields: str = "summary,description,labels,issuelinks,status,comment"):
        return self.request("GET", f"/rest/api/3/issue/{key}", params={"fields": fields})

    def get_comments(self, key: str, start_at=0, max_results=100):
        return self.request("GET", f"/rest/api/3/issue/{key}/comment",
                            params={"startAt": start_at, "maxResults": max_results, "orderBy": "created"})

    def bulk_create(self, issues: list[dict]):
        return self.request("POST", "/rest/api/3/issue/bulk", json={"issueUpdates": issues})

    def add_comment(self, key: str, text: str):
        return self.request("POST", f"/rest/api/3/issue/{key}/comment", json={"body": adf(text)})

    def link(self, link_type: str, inward_key: str, outward_key: str):
        return self.request("POST", "/rest/api/3/issueLink", json={
            "type": {"name": link_type}, "inwardIssue": {"key": inward_key}, "outwardIssue": {"key": outward_key}})
