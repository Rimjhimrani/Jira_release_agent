"""Fill the results table in README.md from outputs/metrics.json:  python tools/fill_readme.py"""
import json, re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
m = json.loads((root / "outputs/metrics.json").read_text())
d = m["decisions"]
cnt = lambda k: ", ".join(r for r, v in sorted(d.items()) if v == k) or "none"
tbl = f"""| Item | Value |
|---|---|
| Overall decision | **{m['overall']}** |
| Provider / model | {m['provider']} / {m['model']} |
| Wall-clock | {m['elapsed_seconds']} s |
| Model calls / tool calls | {m['model_calls']} / {m['tool_calls']} |
| Tokens in / out | {m['input_tokens']:,} / {m['output_tokens']:,} |
| Jira API requests (assessment) | {m['jira_requests']} |
| PASS | {cnt('PASS')} |
| FAIL | {cnt('FAIL')} |
| INSUFFICIENT_EVIDENCE | {cnt('INSUFFICIENT_EVIDENCE')} |
| Failures during run | {len(m['errors'])} |"""
p = root / "README.md"; s = p.read_text()
s = re.sub(r"<!-- RESULTS_START -->.*<!-- RESULTS_END -->", f"<!-- RESULTS_START -->\n{tbl}\n<!-- RESULTS_END -->", s, flags=re.S)
p.write_text(s); print(tbl)
