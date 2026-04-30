# OpenClaw / Codex Prompting

Status: active
Last reviewed: 2026-04-30



Use this doc when you need to hand a bounded EchoZero task to Codex, OpenClaw,
or any disposable worker session.
It captures the prompt patterns that produce the least drift and the strongest
proof in this repo.

This is not a generic prompt-writing guide.
It is the EchoZero-specific contract for getting good agent output.

## Bottom Line

The best EchoZero prompts do seven things up front:

1. state the exact outcome
2. assign one role and one ownership slice
3. point at the minimum canonical docs and code paths
4. name the canonical surface to use and the non-canonical surface to avoid
5. define the proof lane before edits begin
6. say when to stop and report instead of guessing
7. require residual-risk reporting

If any of those are missing, the odds of drift go up sharply.

## What Works Best In EchoZero

### 1. Outcome-first prompts

Lead with:

- the problem to solve
- why it matters now
- the user-visible or operator-visible outcome

Do not lead with a wall of repo history.
Agents do better when intent is clear before context expands.

### 2. Explicit ownership

Always give:

- owned paths
- forbidden paths
- allowed proof lanes

This matters more than model choice.
Most bad agent output in EchoZero comes from vague scope, not weak coding skill.

### 3. Curated context, not repo dumps

Point the agent at the smallest useful package:

- `STYLE.md`
- `GLOSSARY.md`
- one or two canonical docs
- the exact file cluster being changed
- the narrow test slice or proof lane

Do not tell an agent to "read the repo."
Do tell it which two or three files define the surface.

### 4. Canonical-surface binding

EchoZero has real surface boundaries.
Prompts should name them directly.

Examples:

- app automation work: use `run_echozero.py`, `echozero/ui/qt/automation_bridge.py`,
  `packages/ui_automation/**`, `tests/ui_automation/**`
- do not treat `echozero/testing/gui_dsl.py`, `gui_lane_b.py`, or `demo_app.py`
  as the long-term app-control surface
- timeline truth work: stay in `echozero/application/timeline/**` and app/UI
  tests, not widget-local shortcuts
- sync work: stay in `echozero/application/sync/**`,
  `echozero/infrastructure/sync/**`, and sync proof lanes

This is one of the highest-value prompt clauses in the repo.

### 5. Proof-before-edit prompting

Every real task prompt should declare:

- primary proof command
- secondary proof command
- whether app-path, perf, packaging, or hardware proof is required

Agents produce tighter changes when they know the acceptance lane before they
start editing.

### 6. Negative constraints

EchoZero prompts should usually include some version of:

- do not widen scope into opportunistic cleanup
- do not revert unrelated work in the dirty tree
- do not invent alternate truth models
- do not bypass application contracts with widget-only logic
- do not add a second automation client beside `packages/ui_automation/**`
- do not present simulated proof as human-path proof

Negative constraints prevent "helpful" but harmful expansion.

### 7. Stop conditions

Tell the agent when it must stop and report.

Typical stop conditions:

- the task conflicts with unowned modified files
- the docs disagree and the precedence is unclear
- proof requires unavailable hardware or a missing environment dependency
- the change would cross from app contract into forbidden widget or demo logic

Good prompts reduce silent wrong guesses.

### 8. Reporting contract

Require the result to include:

- files changed
- commands run
- pass/fail result
- strongest failure signal if broken
- residual risks or untested surfaces

For review tasks, require findings first with file/line evidence.

## High-Value Prompt Clauses

These short clauses consistently improve OpenClaw/Codex output in EchoZero.

### Use these almost every time

- `Role:` `research`, `impl`, `verify`, or `review`
- `Owned paths:` exact file cluster
- `Forbidden paths:` where the agent must not edit
- `Canonical docs to read first:` smallest relevant list
- `Canonical surface:` the app/runtime/test boundary that defines truth
- `Primary proof command:` smallest lane that proves the claim
- `Do not revert unrelated work.`
- `Stop and report if scope crosses the owned paths or conflicts with existing edits.`
- `Report residual risk explicitly.`

### Use when the task is UI, automation, or OpenClaw-facing

- `Use the app-owned automation bridge and ui_automation client as the control plane.`
- `Prefer semantic invoke or stable target ids before pointer fallbacks.`
- `Take a snapshot before and after meaningful UI actions when debugging state.`
- `Use screenshots as confirmation, not as the primary locator model.`
- `Do not use demo_app or simulated GUI helpers as the canonical agent path.`

### Use when the task is review-only

- `Review for bugs, regressions, truth-model violations, and missing tests.`
- `Findings first; summary second.`
- `Include severity and file references.`

### Use when the task is long-running or delegated across sessions

- `Parent task anchor:` `<short task label + desired end state>.`
- `Emit visible status heartbeats every 60 seconds while active.`
- `Restate the parent task anchor in spawn proof, heartbeat, and return summary.`
- `If blocked or silent for 300 seconds, treat it as stuck and report.`

## Prompt Shapes By Role

### Research Prompt

Use when you need tracing and references before editing.

```md
Role: research
Goal: trace the canonical implementation and proof surface for <task>.
Why now: <reason>
Parent task anchor: <short label + desired end state>
Lead-dev next step on return: <integrate / verify / decide X>
Owned paths: <docs/files to inspect>
Forbidden paths: no edits
Canonical docs to read first: STYLE.md, GLOSSARY.md, <doc>, <doc>
Deliverable:
- direct answer
- file references
- parent task anchor restated
- recommended proof lane
- remaining uncertainty
```

### Implementation Prompt

Use for the main change.

```md
Role: impl
Goal: implement <specific behavior>.
Why now: <reason>
User-visible outcome: <what becomes true>
Parent task anchor: <short label + desired end state>
Lead-dev next step on return: <integrate / verify / decide X>

Owned paths:
- <path cluster>

Forbidden paths:
- <path cluster>

Canonical docs to read first:
- STYLE.md
- GLOSSARY.md
- <exact repo docs>

Canonical surface:
- <truth boundary or automation boundary>

Proof lane:
- Primary: <command>
- Secondary: <command>
- Manual/perf/hardware: <required or none>

Execution rules:
- stay inside owned paths
- do not widen scope into cleanup
- do not revert unrelated work
- stop and report if conflicting edits or rule conflicts appear
- report residual risk explicitly

Return:
- files changed
- commands run
- pass/fail result
- parent task anchor restated
- residual risks/blockers
```

### Verification Prompt

Use when a worker should only prove or reject a claim.

```md
Role: verify
Goal: prove whether <change/behavior> holds on the canonical path.
Parent task anchor: <short label + desired end state>
Lead-dev next step on return: <integrate / retry proof / decide X>
Owned paths: no edits unless fixing a broken test is explicitly allowed
Proof commands:
- <command>
- <command>
Return:
- commands run
- pass/fail result
- parent task anchor restated
- strongest failure signal
- untested surfaces
```

### Review Prompt

Use when the task is bug-finding, not implementation.

```md
Role: review
Goal: audit <change area> for bugs, regressions, and missing proof.
Parent task anchor: <short label + desired end state>
Lead-dev next step on return: <accept / request fixes / decide X>
Focus:
- truth-model leakage
- stale-state regression
- MA3 sync boundary violations
- widget-only bypasses
- FEEL drift or magic numbers
Return:
- findings first with severity and file references
- parent task anchor restated
- then residual risks or testing gaps
```

## OpenClaw / UI Automation Prompt Pattern

When the task involves driving the EchoZero app, include an explicit control
order.

Use this shape:

```md
Role: impl or verify
Goal: <automation or app-path task>

Canonical surface:
- run_echozero.py
- echozero/ui/qt/automation_bridge.py
- packages/ui_automation/**
- tests/ui_automation/**

Do not use as the canonical agent surface:
- echozero/testing/gui_dsl.py
- echozero/testing/gui_lane_b.py
- echozero/ui/qt/timeline/demo_app.py

Interaction order:
1. attach or launch the real app path
2. call snapshot first
3. prefer invoke or stable target ids
4. use hit targets for timeline interactions
5. use screenshot only for confirmation

Proof:
- targeted ui_automation tests
- appflow or ui-automation lane when the change crosses the app boundary
```

That clause set keeps OpenClaw aligned with the repo's automation policy instead
of drifting into test-only helpers.

## Strong Prompt Patterns

### Tell the agent where to start

Bad:

- "Find the relevant files and fix it."

Better:

- "Start with `echozero/application/timeline/object_action_settings_service.py`,
  `echozero/application/timeline/pipeline_run_service.py`, and
  `tests/test_pipeline_run_service.py`."

### Name the real truth boundary

Bad:

- "Make the UI behave correctly."

Better:

- "Keep workflow semantics in the application layer; do not solve this in
  widget-local code if the behavior belongs in the app contract."

### Pair positive and negative instructions

Bad:

- "Add app automation."

Better:

- "Use the app-owned bridge and `packages/ui_automation/**`; do not add a second
  automation client or treat `gui_dsl` as the long-term control surface."

### Ask for proof, not confidence

Bad:

- "Make sure it works."

Better:

- "Run `python -m echozero.testing.run --lane ui-automation` and report pass/fail
  plus the strongest failure signal if anything breaks."

## Weak Prompt Smells

If a prompt has several of these, rewrite it before dispatch:

- no owned paths
- no forbidden paths
- no proof lane
- broad "clean this up" or "refactor as needed"
- no mention of main/take truth or app-boundary rules on timeline/sync work
- no instruction about simulated versus human-path proof
- no residual-risk reporting requirement

## Recommended Operating Order

For non-trivial work, prompt in this sequence:

1. bounded research prompt
2. bounded implementation prompt
3. bounded verification prompt
4. bounded review prompt

This matches `docs/AGENT-WORKFLOW.md`.

## Copy-Paste Skeleton

Use this when you need one strong default prompt quickly.

```md
Role: <research|impl|verify|review>
Goal: <specific task>
Why now: <reason>
User-visible outcome: <what becomes true>

Owned paths:
- <paths>

Forbidden paths:
- <paths>

Canonical docs to read first:
- STYLE.md
- GLOSSARY.md
- <docs>

Canonical surface:
- <truth boundary / automation boundary>

Proof lane:
- Primary: <command>
- Secondary: <command>
- Manual/perf/hardware: <required or none>

Rules:
- stay inside owned paths
- do not widen scope
- do not revert unrelated work
- stop and report on conflicting edits, missing dependencies, or rule conflicts
- report residual risk explicitly

Return:
- files changed
- commands run
- pass/fail result
- findings or outcome
- residual risks/blockers
```

## Keep It Local

EchoZero does not need a large sidecar prompt framework.
The right implementation is:

- one canonical prompting guide
- one strong task template
- explicit workflow references

Prefer a small local doc over meta-infrastructure.
