# Design notes

## Architecture
`import` → Jira Cloud (source of truth) → `assess`: **discovery** (model reads manifest, submits requirement ids) → **one fresh conversation per
requirement** (model chooses tools, then `record_decision`) → **code** computes the overall decision and writes the report + trace.

Tools (all Jira-backed, all forced into `project = X AND labels = <dataset label>`): `search_issues`, `get_issue`, `get_comments`,
`get_linked_issues` (depth ≤3), `check_scope`, `compare_numbers`, `record_decision`. The dataset JSON is never read at assessment time.

## Choices that map to the brief
- **Model selects actions via tools; no full-dataset prompt.** Typical requirement uses 3-8 calls; tool outputs are capped (7k chars) and paged.
- **Context management.** Per-requirement fresh context; boilerplate/duplicate comments are collapsed (e.g. the 62-comment streaming ticket becomes one line, re-expandable); older bulky tool results are elided from the history.
- **Evidence applicability is enforced in code, not only in the prompt.** `check_scope` parses release/commit/service/environment(/prompt) from the comment and compares with the target read from the Jira manifest; absent ≠ match. `record_decision` rejects PASS/FAIL unless a cited ref passed `check_scope`, and rejects refs the agent never retrieved (no invented citations).
- **Overall decision is computed by code** from the three-state rules, never by the model.
- **Untrusted text.** Ticket/comment text is data. Instruction-like text aimed at reviewers is flagged by the tool layer (`INTEGRITY_FLAG`), the system prompt forbids obeying it, and the report has an integrity-note field.
- **Failure handling.** Model/tool/Jira errors are retried (SDK + client backoff); if a requirement still can't be assessed it is recorded as `INSUFFICIENT_EVIDENCE` with an explicit `agent_error` and listed under run failures — never silently PASS/FAIL. Hard budgets on turns, tool calls, tokens and wall-clock.
- **Discovery is cross-checked** against an independent Jira label query so a model omission can't drop a requirement.

## Known limits
- Not yet run against a live Jira Cloud site or a live model in the authoring environment; Jira interactions were tested against a stub that implements only the endpoints used. Jira text search (`text ~`) is eventually consistent right after import.
- Native Jira *status* is not driven to the recorded status (workflow transitions differ per project); recorded status is preserved in labels/description text and intentionally treated as non-evidence.
- Scope parsing is regex-based on the "release …; commit …; service …; environment …; prompt …" convention; unusual phrasing yields `not_stated` (conservative → INSUFFICIENT_EVIDENCE) rather than a wrong match.
- Judgement calls (conflicting equal-authority records, supersession) are made by the model; code only guards applicability and citation integrity. Non-determinism is reduced with temperature 0, not eliminated.
- The offline policy is dataset-shaped rules for testing, not a general reviewer.

## AI assistance
I used Claude (an agentic coding assistant) to scaffold the importer, the Jira tool layer, the agent loop, the Jira stub used for offline tests, and the test suite. I specified the requirements from the brief, reviewed the generated code, ran the tests, and checked the report decisions against the manifest rules by hand. The assessed model is Claude via the Anthropic API; it only sees Jira data through the tools described above.
