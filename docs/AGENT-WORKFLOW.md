# EchoZero Agent Workflow

This document defines the default operating model for agent-driven work in
EchoZero.
It exists so orchestration, delegation, proof, and cleanup stay consistent
across sessions.
It connects `AGENTS.md`, `docs/WORKER-ROLES.md`, `docs/DEV-LANES.md`, and
`docs/TESTING.md` into one practical workflow.

## Purpose

EchoZero uses an orchestrator-first workflow.

Default stance:

- `lead-dev` owns broad picture, decomposition, architecture, integration, and final acceptance.
- Disposable agents own bounded exploration, implementation, verification, and review tasks.
- Parallelism is allowed only when ownership and proof lanes are explicit first.

This is the default mode for non-trivial work in this repo.

## Operating Model

### `lead-dev` responsibilities

`lead-dev` is the conductor, not the bulk line-by-line worker.

It must:

- read the canonical docs and preserve first principles
- decompose work into bounded assignments
- choose the correct proof lane before edits begin
- assign explicit file/lane ownership
- integrate outputs from disposable agents
- reject work that violates truth-model, sync, FEEL, or app-boundary rules
- close agents and sessions that are no longer needed

### Disposable agent responsibilities

Disposable agents do one bounded job well, then exit.

Typical roles:

- `research`: trace code/docs/tests for one narrow question
- `impl`: implement a bounded change in an assigned file cluster
- `verify`: run the narrowest proof lane and report evidence
- `review`: audit for bugs, regressions, drift, and missing tests

Do not use disposable agents as long-lived project owners.

## When To Spawn

Spawn agents by default for non-trivial work that benefits from bounded parallelism.

Good reasons to spawn:

- one task can be split into disjoint file clusters
- a narrow codebase question can be explored independently
- verification can run in parallel with implementation
- a focused review can happen after implementation without blocking orchestration

Do not spawn when:

- the task is tiny
- the very next blocking step is faster to do directly
- ownership boundaries are unclear
- multiple agents would collide in the same file cluster

## Assignment Contract

Every delegated task must specify:

- goal
- owned paths
- forbidden paths
- proof lane
- expected output
- residual-risk reporting requirement

Use [docs/agent-task-template.md](/Users/march/Documents/GitHub/EchoZero/docs/agent-task-template.md:1)
for the assignment format.

## Ownership Rules

Follow [docs/WORKER-ROLES.md](/Users/march/Documents/GitHub/EchoZero/docs/WORKER-ROLES.md:1)
and [docs/DEV-LANES.md](/Users/march/Documents/GitHub/EchoZero/docs/DEV-LANES.md:1).

Default lane ownership:

- EZ lane: `echozero/application/**`, `echozero/ui/**`, `tests/application/**`, `tests/ui/**`
- Foundry lane: `echozero/foundry/**`, `tests/foundry/**`
- Shared zone: `echozero/inference_eval/**`, `tests/inference_eval/**`, `tests/processors/test_pytorch_audio_classify_preflight.py`

Rules:

- Parallelize by file cluster or lane, not by vague specialization.
- Do not assign overlapping write ownership.
- Shared-zone changes require explicit integration review.
- `lead-dev` keeps final ownership even when all implementation is delegated.

## Proof Rules

No agent task is done without evidence.

Use the smallest lane that proves the claim first, then expand only if needed.

Typical proof surfaces:

- targeted pytest slices
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane appflow-sync`
- `python -m echozero.testing.run --lane appflow-osc`
- `python -m echozero.testing.run --lane appflow-protocol`
- `python -m echozero.testing.run --lane gui-lane-b`
- perf guardrails for hot timeline paths
- packaging/smoke flows for release-affecting changes

Every delegated result must report:

- commands run
- files or feature area covered
- pass/fail result
- strongest failure signal if broken
- residual risks or untested surfaces

## Proof Of Spawn

If `lead-dev` claims agents were spawned or dispatched, it must provide explicit proof.

Visible spawn feedback is the default behavior during active development.

Before or as an agent spawn is initiated, `lead-dev` should send a short notice stating:

- that an agent is being spawned
- the role
- the bounded ownership or task

Use this format after any real delegation:

- `spawned`: agent/session id
- `role`: `research`, `impl`, `verify`, `review`, or session type
- `ownership`: file cluster or bounded scope
- `status`: `active`, `waiting`, `completed`, or `closed`

Do not say an agent was spawned until the spawn call has succeeded.
If no agent was spawned, say `0`.

When an agent returns, `lead-dev` should provide at least a short summary of:

- what the agent concluded or produced
- whether the result was accepted, still in review, or blocked

When an agent is closed, `lead-dev` should report closeout when it is contextually relevant.

The default visible harness during development is:

1. pre-spawn notice
2. spawn proof block
3. return summary
4. closeout proof when relevant

## Status Reporting

When parallel work is active or coordination risk is rising, report status in this compact form:

`active agents / waiting / blocked / open sessions / risk level`

Example:

`2 / 1 / 0 / 3 / medium`

Use short follow-up notes only when there is something actionable or abnormal.

For long-running delegated work, visible status heartbeats are required.

Default timer rules:

- if one or more agents are still active, send a short status heartbeat every `60` seconds
- each heartbeat should name the active agents and the bounded task each one owns
- keep the heartbeat compact unless there is a blocker or a material result

Default heartbeat content:

- compact status line
- active agent id or nickname
- owned task or file cluster
- whether the agent is making progress, waiting on proof, or blocked

Example:

- `2 / 0 / 0 / 2 / medium`
- `James`: `PB-40`, `PB-42` layer-header playback control split
- `Ampere`: `PB-41` inspector/object-info separation

If the work is especially noisy or the channel needs less chatter, `lead-dev` may widen the heartbeat interval, but it should stay explicit.
Default escalated interval ceiling is `120` seconds.

Stuck-agent rule:

- if an agent has not returned, emitted a usable progress update, or shown observable forward motion within `300` seconds, `lead-dev` must explicitly question whether it is stuck
- do not leave a long-running agent silent past the stuck threshold without comment
- after the threshold is crossed, `lead-dev` should do one or more of:
  - wait once with intent and report the result
  - send a redirect or clarification
  - re-scope the task
  - close and replace the agent
  - report that the agent is still legitimately running and why

The purpose of the heartbeat and stuck threshold is operator trust.
Long-running orchestration should stay visible, bounded, and explainable.

## Session Cleanup

Orphaned sessions are a failure mode.

Rules:

- close disposable agents after their output is integrated or rejected
- do not leave completed workers open without a reason
- prefer short-lived agents over indefinite background sessions
- keep recurring audits silent unless there is a finding worth escalation
- periodically check for orphaned subagents and excessive session buildup

If an agent remains open, `lead-dev` should be able to answer:

- why it is still open
- what it owns
- what would allow it to be closed

## Review Standard

EchoZero uses audit-heavy development.

Every meaningful change should be examined through multiple lenses:

- canonical docs
- actual code path
- proof/test surface
- product/operator intent

Review agents should focus especially on:

- main/take truth leakage
- stale-state regressions
- MA3 sync boundary violations
- widget-only logic bypassing application contracts
- FEEL drift and scattered magic numbers

## Preferred Flow

For most non-trivial work, use this sequence:

1. `lead-dev` reads context and decomposes the task.
2. `research` agent traces the relevant code/docs/tests.
3. `impl` agent edits one bounded slice.
4. `verify` agent runs the narrowest proof lane.
5. `review` agent audits findings and residual risk.
6. `lead-dev` integrates, verifies final fit, closes agents, and reports outcome.

Not every task needs every role.
Small tasks may collapse several roles back into `lead-dev`.

## Non-Negotiables

- App-facing work is not accepted without app-path proof.
- Main remains truth; takes remain subordinate.
- MA3 sync remains main-only.
- FEEL owns UI tuning constants.
- Generated/runtime artifacts do not belong in git.
- Do not introduce speculative abstractions or sidecar frameworks when a small local doc or module is enough.
