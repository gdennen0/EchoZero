# Agent Task Template

Status: reference
Last reviewed: 2026-04-30


Use this template when assigning work to a disposable worker or starting a
bounded session in a separate worktree.
For prompt phrasing and EchoZero-specific tricks, pair this with
`docs/OPENCLAW-CODEX-PROMPTING.md`.

## Prompt header

- Role:
- Goal:
- Why now:
- User-visible outcome:
- Parent task anchor:
- Lead-dev next step on return:

## Scope and ownership

- Owned paths:
- Forbidden paths:
- Allowed tests or proof lanes:
- Lane/worktree:

## Context package

- Canonical docs to read first:
- Relevant source files to inspect first:
- Relevant tests to inspect first:
- Canonical surface to use:
- Non-canonical surfaces to avoid:

## Locked rules

- Truth-model constraints:
- Sync/timeline/UI constraints:
- Cleanup boundaries:
- Stop and report if:

## Proof lane

- Primary proof command:
- Secondary proof command:
- Perf/hardware/manual proof required:

## Execution contract

- Why this assignment is delegated:
- Stay inside owned paths:
- Do not widen scope into opportunistic cleanup:
- Do not revert unrelated work:
- Report residual risk explicitly:

## Acceptance criteria

- Behavior that must be true:
- Regression that must stay false:
- Deliverables expected:

## Reporting contract

- Files changed:
- Commands run:
- Pass/fail result:
- Parent task anchor restated:
- Strongest failure signal if broken:
- Residual risks/blockers:
- Spawn/session proof:
- Closeout state:

## Optional role add-ons

### `research`

- Exact question to answer:
- References required:
- Uncertainty to call out:

### `impl`

- Concrete change to make:
- Allowed abstractions:
- Explicit anti-patterns to avoid:

### `verify`

- Read-only vs fix-forward:
- Evidence required:

### `review`

- Findings-first requirement:
- Severity/file reference requirement:
