# Agent Task Template

Status: reference
Last reviewed: 2026-05-09

Use this template for one native OpenClaw/Codex disposable worker.
It is intentionally small: the parent owns orchestration, OpenClaw owns spawn
mechanics, and the worker owns one bounded result.

## Spawn defaults

- Runtime: native OpenClaw subagent
- Mode: run
- Cleanup: delete
- Context: isolated unless transcript context is required
- Light context: true
- Agent id: explicit role agent
- Cwd: repo or assigned worktree

## Prompt header

- Role: `research` | `impl` | `verify` | `review`
- Goal:
- Why now:
- User-visible/operator-visible outcome:
- Parent task anchor:
- Lead-dev next step on return:

## Scope and ownership

- Active lane: `EZ app` | `MA3 harness` | `Foundry` | `planning/review`
- Excluded lanes:
- Owned paths:
- Forbidden paths:
- Allowed tests or proof lanes:
- Worktree/cwd:

## Context package

- Canonical docs to read first:
- Relevant source files to inspect first:
- Relevant tests to inspect first:
- Canonical surface to use:
- Non-canonical surfaces to avoid:

## Execution contract

- Stay inside owned paths.
- Do not widen scope into opportunistic cleanup.
- Do not revert unrelated work.
- Do not create a sidecar agent/orchestration framework.
- Stop and report if scope crosses owned paths, conflicts with existing edits,
  or needs operator judgment.
- Report residual risk explicitly.

## Proof lane

- Primary proof command:
- Secondary proof command:
- Perf/hardware/manual proof required:

## Acceptance criteria

- Behavior that must be true:
- Regression that must stay false:
- Deliverables expected:

## Required final payload

Return exactly these fields:

- `status`: `success` or `blocked`
- `changed_files`:
- `tests_run`:
- `summary`:
- `blocker`:
- `residual_risk`:

## Role add-ons

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
