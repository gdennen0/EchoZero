# EchoZero Agent Workflow

Status: reference
Last reviewed: 2026-05-09

This document is the only repo-level workflow for spawning disposable agent work.
It exists because the old spawning guidance had accumulated too many overlapping
queue, heartbeat, and TaskFlow patterns. That made it easy to lose status or
mistake a completed process for accepted work.

Pair this with [docs/OPENCLAW-CODEX-PROMPTING.md](OPENCLAW-CODEX-PROMPTING.md)
for prompt phrasing and [docs/agent-task-template.md](agent-task-template.md)
for the handoff shape.

## Dispatch v2

Default path for non-trivial EchoZero work:

1. `lead-dev` owns the parent lane and final acceptance.
2. `lead-dev` names one active objective and the excluded lanes.
3. `lead-dev` spawns one bounded native OpenClaw subagent when delegation is
   useful.
4. The subagent runs one role: `research`, `impl`, `verify`, or `review`.
5. The subagent returns one final payload.
6. `lead-dev` verifies the payload, runs or inspects proof as needed, and reports
   accepted/blocked/follow-up status.

Do not build a sidecar agent framework in this repo. The spawning substrate is
OpenClaw; EchoZero owns only the scope, prompt, proof, and acceptance contract.

## Native OpenClaw Spawn Contract

Use native OpenClaw subagents for disposable workers.

Required defaults:

- `runtime`: native subagent
- `mode`: `run`
- `cleanup`: `delete`
- `lightContext`: `true`
- `cwd`: the repo or lane worktree the worker owns
- `agentId`: the specific role agent, never an implicit default
- `context`: isolated by default; use forked context only when the transcript is
  required for correctness

Do not use ACP for EchoZero dispatch unless the operator explicitly asks for ACP
or the task is testing the ACP path.

Do not use TaskFlow for ordinary one-shot implementation, research, verification,
or review. TaskFlow is only for durable queues that must survive resets, waits,
approvals, or multi-step detached orchestration.

After a spawn succeeds, do not poll in a loop. Record the child session id,
report the spawn proof, then wait for the completion event. If the worker is
abnormally quiet or the completion payload is malformed, the parent must say so
and verify directly from repo state before reporting success.

## When To Spawn

Spawn when a bounded worker reduces risk or saves meaningful time.

Good reasons:

- the task can be isolated to one file cluster or lane
- research can happen without edits
- verification can run after a patch without changing scope
- review can independently audit a completed change

Do not spawn when:

- the next step is tiny and faster in the parent
- the spawning path itself is under repair
- file ownership is unclear
- workers would write overlapping paths
- the task requires immediate operator judgment before safe edits

## Parent-Lane Rules

`lead-dev` is the conductor, not a passive relay.

It must always keep:

- one active objective
- one named lane: `EZ app`, `MA3 harness`, `Foundry`, or `planning/review`
- explicitly excluded lanes
- a visible queue state when work is running
- final acceptance authority

Cross-lane discoveries become queued follow-up work unless the operator changes
the active objective.

## Worker Assignment Contract

Every delegated prompt must include:

- role: `research`, `impl`, `verify`, or `review`
- goal and why it matters now
- parent task anchor
- owned paths
- forbidden paths
- canonical docs to read first
- canonical surface to use
- proof lane before edits begin
- stop/report conditions
- final reporting contract

Minimum final payload:

- `status`: `success` or `blocked`
- `changed_files`
- `tests_run`
- `summary`
- `blocker`
- `residual_risk`

A worker completion without that payload is unresolved until `lead-dev` verifies
state directly or reruns the worker with a corrected assignment.

## Spawn Proof

If `lead-dev` says work was delegated, it must show proof after the spawn call
succeeds.

Minimum visible proof:

- worker/session id
- role
- mode
- ownership
- status

Do not claim a worker was spawned before the spawn succeeds.
If no worker was spawned, say so explicitly.

## Queue State

Use a compact queue snapshot when work is active:

- `current task`: short label plus desired end state
- `state`: `queued`, `running`, `blocked`, `review`, or `done`
- `workers`: active session ids and bounded ownership
- `next`: parent-side action after completion

Prefer at most two parallel writer workers. Never run overlapping writers on the
same file cluster. Reader/research/review workers may run in parallel when they
are independent and read-only.

## Completion Gate

Delegated work is not done until `lead-dev` can state:

- the assigned objective
- the owned paths or bounded scope
- the worker final payload or direct parent-side substitute evidence
- the tests/proof that passed or failed
- whether the result is accepted, blocked, or queued for retry/follow-up

Silence is not success. A process exit is not success. A malformed completion
event is not success.

## Stuck Or Malformed Worker Recovery

If a worker is quiet, off-scope, or returns a malformed payload:

1. Mark the worker result unresolved.
2. Inspect the repo diff and relevant tests directly if safe.
3. If the scope is still valid, rerun a corrected worker assignment or continue
   in the parent.
4. If the worker crossed ownership boundaries, stop and ask or split a new
   bounded task.
5. Report the strongest known evidence instead of guessing.

## Session Cleanup

Disposable workers should not become project owners.

Rules:

- use short-lived run-mode workers
- clean up completed sessions unless there is a specific reason to keep them
- do not leave open workers without a named owner/scope
- parent lane keeps the integration decision

## Non-Negotiables

- App-facing work needs app-path proof.
- Main remains truth; takes remain subordinate.
- MA3 sync remains main-only.
- FEEL owns UI tuning constants.
- Generated/runtime artifacts do not belong in git.
- Do not introduce speculative agent frameworks when native OpenClaw dispatch is
  the right substrate.
