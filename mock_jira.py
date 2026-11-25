"""Tiny Jira Cloud look-alike (stdlib only) for offline testing of the importer and agent.
Implements just the endpoints the project uses. Native author = 'Importer Bot', native time = now.
Env: MOCK_FAIL_EVERY=N makes every Nth request return 429 (to exercise retry logic)."""
import json, os, re, sys, threading, itertools
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
ISSUES, ORDER = {}, []
COUNTER = itertools.count(1)
REQ = itertools.count(1)
LINK_ID = itertools.count(1)
COMMENT_ID = itertools.count(10000)
FAIL_EVERY = int(os.environ.get("MOCK_FAIL_EVERY", "0"))


def now():
    return datetime.now(timezone.utc).isoformat()


def adf_text(n):
    if isinstance(n, dict):
        if n.get("type") == "text":
            return n.get("text", "")
        return "\n".join(adf_text(c) for c in n.get("content", [])) if n.get("type") == "doc" else "".join(adf_text(c) for c in n.get("content", []))
    return ""


def match(issue, jql):
    f = issue["fields"]
    for clause in re.split(r"\s+AND\s+", jql.strip(), flags=re.I):
        m = re.match(r'(\w+)\s*(=|~)\s*"?(.*?)"?$', clause.strip())
        if not m:
            continue
        k, op, v = m.groups()
        if k == "project" and issue["key"].split("-")[0] != v:
            return False
        if k == "labels" and v not in f["labels"]:
            return False
        if k == "text" and op == "~":
            hay = (f["summary"] + " " + adf_text(f["description"]) + " " + " ".join(adf_text(c["body"]) for c in issue["comments"])).lower()
            if v.lower() not in hay:
                return False
    return True


def view(issue, fields):
    f = issue["fields"]; out = {"key": issue["key"], "id": issue["id"], "fields": {}}
    want = set((fields or "summary").split(",")) if isinstance(fields, str) else set(fields or ["summary"])
    for k in ("summary", "description", "labels"):
        if k in want:
            out["fields"][k] = f[k]
    if "status" in want:
        out["fields"]["status"] = {"name": "To Do"}   # native status is importer-controlled; recorded status lives in text
    if "issuelinks" in want:
        out["fields"]["issuelinks"] = issue["links"]
    if "comment" in want:
        out["fields"]["comment"] = {"total": len(issue["comments"])}
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0)); return json.loads(self.rfile.read(n) or b"{}")

    def _route(self, method):
        n = next(REQ)
        if FAIL_EVERY and n % FAIL_EVERY == 0:
            self.send_response(429); self.send_header("Retry-After", "0"); self.send_header("Content-Length", "0"); self.end_headers(); return
        path, _, qs = self.path.partition("?")
        q = dict(x.split("=", 1) for x in qs.split("&") if "=" in x)
        with LOCK:
            if path == "/rest/api/3/myself":
                return self._send(200, {"displayName": "Importer Bot"})
            if path == "/rest/api/3/search/jql" and method == "POST":
                b = self._body(); items = [i for i in (ISSUES[k] for k in ORDER) if match(i, b["jql"])]
                off = int(b.get("nextPageToken") or 0); mx = b.get("maxResults", 50); page = items[off:off + mx]
                res = {"issues": [view(i, b.get("fields")) for i in page], "isLast": off + mx >= len(items)}
                if not res["isLast"]: res["nextPageToken"] = str(off + mx)
                return self._send(200, res)
            if path == "/rest/api/3/issue/bulk" and method == "POST":
                out = []
                for u in self._body()["issueUpdates"]:
                    num = next(COUNTER); key = f"TYN-{num}"; f = u["fields"]
                    ISSUES[key] = {"key": key, "id": str(num), "fields": {"summary": f["summary"], "description": f["description"], "labels": f["labels"]}, "comments": [], "links": []}
                    ORDER.append(key); out.append({"key": key, "id": str(num)})
                return self._send(201, {"issues": out, "errors": []})
            if path == "/rest/api/3/issueLink" and method == "POST":
                b = self._body(); lid = str(next(LINK_ID))
                a, c = ISSUES[b["inwardIssue"]["key"]], ISSUES[b["outwardIssue"]["key"]]
                lt = b["type"]["name"]
                a["links"].append({"id": lid, "type": {"name": lt}, "outwardIssue": {"key": c["key"]}})
                c["links"].append({"id": lid, "type": {"name": lt}, "inwardIssue": {"key": a["key"]}})
                return self._send(201, {})
            m = re.match(r"^/rest/api/3/issue/([A-Z]+-\d+)(/comment)?$", path)
            if m:
                iss = ISSUES.get(m.group(1))
                if not iss: return self._send(404, {"errorMessages": ["Issue does not exist"]})
                if m.group(2):
                    if method == "POST":
                        c = {"id": str(next(COMMENT_ID)), "author": {"displayName": "Importer Bot"}, "created": now(), "body": self._body()["body"]}
                        iss["comments"].append(c); return self._send(201, c)
                    s, mx = int(q.get("startAt", 0)), int(q.get("maxResults", 50))
                    return self._send(200, {"startAt": s, "maxResults": mx, "total": len(iss["comments"]), "comments": iss["comments"][s:s + mx]})
                return self._send(200, view(iss, q.get("fields", "summary").replace("%2C", ",")))
        self._send(404, {"errorMessages": [f"no route {method} {path}"]})

    def do_GET(self): self._route("GET")
    def do_POST(self): self._route("POST")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    print(f"mock jira on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
