# EchoZero Worker Roles

This repo uses a `lead-dev` plus disposable worker model.

`lead-dev` is the primary engineering interface for EchoZero. It owns planning,
decomposition, integration, and final acceptance. Workers are temporary and
exist to finish bounded assignments without creating architecture drift.

## Core Rule

Use workers only when they reduce risk or save time without creating overlap.

Do not spawn parallel workers into the same file cluster unless the integration
plan is explicit first.

## `lead-dev`

Purpose:

- keep work aligned with first principles
- choose the correct proof lane
- assign bounded file ownership
- integrate worker output
- reject work that violates truth-model or sync boundaries

Must enforce:

- app-first acceptance
- main-only sync semantics
- no speculative refactor during scoped delivery
- proof at the application boundary, not just helper/demo level

## `impl`

Purpose:

- implement a bounded change inside a clearly assigned area

Expected output:

- focused patch
- targeted verification notes
- explicit residual risks or blockers

Default ownership examples:

- timeline app contract: `echozero/application/timeline/**`, `tests/application/**`
- sync lane: `echozero/application/sync/**`, `echozero/infrastructure/sync/**`, sync tests
- Foundry lane: `echozero/foundry/**`, `tests/foundry/**`
- UI lane: `echozero/ui/**`, app-level UI tests

Rules:

- stay inside the assigned slice
- do not widen scope into opportunistic cleanup
- preserve canonical surfaces and locked product rules

## `review`

Purpose:

- find bugs, regressions, missing tests, and broken assumptions

Expected output:

- findings first
- severity plus evidence
- mention remaining risk when no findings are present

Focus areas for EchoZero:

- main/take truth leakage
- stale-state regression
- MA3 sync boundary violations
- widget-only logic bypassing app contracts
- FEEL drift and magic-number UI behavior

## `verify`

Purpose:

- run the narrowest proof lane that can confirm or reject the change

Expected output:

- commands run
- pass/fail result
- strongest failure signal if broken

Typical lanes:

- contract/unit pytest slices
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane appflow-sync`
- `python -m echozero.testing.run --lane appflow-osc`
- `python -m echozero.testing.run --lane appflow-protocol`
- `python -m echozero.testing.run --lane gui-lane-b`

Rules:

- verification reports evidence, not redesign opinions
- choose the smallest lane that proves the claim before expanding

## `research`

Purpose:

- answer bounded questions from code, docs, logs, or test surfaces

Expected output:

- direct answer
- code or doc references
- remaining uncertainty if applicable

Good uses:

- tracing timeline ownership before edits
- mapping sync consequences of a change
- locating the correct test lane before implementation

## Parallelism Rules

- Parallelize by lane or file cluster, not by vague specialization.
- Respect `docs/DEV-LANES.md` when the work touches Foundry/EZ parallel lanes.
- Shared-zone changes need explicit integration review before merge.
- Keep final ownership with `lead-dev`, never with a worker.
