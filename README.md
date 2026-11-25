# Jira Release Evidence Agent

A tool-using LLM agent that decides whether the mandatory requirements of a release are satisfied by **evidence recorded in Jira**,
and writes one evidence-backed release-readiness report (overall `READY` / `NEEDS_REVIEW` / `BLOCKED`).

Python 3.11+, pinned deps (`requirements.txt`): `anthropic` (model provider), `requests`, `pytest`.

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill in; .env is git-ignored. No credentials are in this repo.
export PYTHONPATH=src
```
Jira Cloud: create a dedicated project (e.g. key `TYN`), create an API token at id.atlassian.com, put site URL / email / token in `.env`.
The project needs an issue type named `Task` (override with `--issue-type`) and the default `Relates` link type (`--link-type`).

## 2. Import (repeatable, idempotent)
```bash
python -m release_agent import --project TYN --dataset inputs/dataset.json
python -m release_agent verify-import --project TYN
```
Every issue gets: summary `[SOURCE-ID] title`, label `tyn-release-assessment-v2` (dataset scope), `src-<id>`, `kind-*`, `rel-*`
(+ `rr-requirement` for requirements), a description header with source id / recorded status / linked ids followed by the original text,
every comment verbatim (the `Recorded event <ts> | actor | role` header stays in the text; Jira's native author/time are the importer's),
and links as Jira issue links. Re-running creates only what is missing; `verify-import` compares issue/comment/link counts to the dataset.
Rate limits (429/5xx) are retried with backoff.

## 3. Assess
```bash
python -m release_agent assess --project TYN --label tyn-release-assessment-v2 --release R2.4 --out outputs
```
Outputs in `outputs/`: `report.md`, `trace.jsonl` (every model call and tool call: args, timing, tokens, result preview), `metrics.json`.
Budgets (flags): `--max-turns` (per requirement, 14), `--max-tool-calls` (250), `--max-tokens` (600k), `--max-seconds` (1200).

`--provider offline` runs a deterministic rule-based stand-in (no API key) over the same tools; used for tests and for the sample in `outputs/`.

## 4. Test without Jira or an API key
```bash
python tools/mock_jira.py 8089 &        # Jira-compatible stub (supports the endpoints used; MOCK_FAIL_EVERY=37 injects 429s)
python -m release_agent import --jira-url http://127.0.0.1:8089 --project TYN
python -m release_agent assess --jira-url http://127.0.0.1:8089 --project TYN --label tyn-release-assessment-v2 --release R2.4 --provider offline
python -m pytest -q                     # 10 tests: import fidelity/idempotency, scope guard, guardrails, traps, failure + budget handling
```

## 5. Results of the submitted run
<!-- RESULTS_START -->
_Not filled yet. After `assess`, run `python tools/fill_readme.py` to fill this table from `outputs/metrics.json`._
<!-- RESULTS_END -->

`outputs/report.md`, `outputs/trace.jsonl` and `outputs/metrics.json` come from that run (LLM agent against my Jira Cloud project).
If the first line of the report mentions the "offline rule-based baseline", it was generated without an LLM (test mode only).

See `DESIGN.md` for design choices, limits and AI-assistance disclosure.
