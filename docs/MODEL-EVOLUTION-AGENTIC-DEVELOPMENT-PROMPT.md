# Model Evolution Agentic Development Prompt

Status: draft
Last verified: 2026-05-20
Lane: Foundry

## Prompt

You are working in the local EchoZero repository on this machine.

Active objective: implement the first production slice of EchoZero's model evolution architecture, turning user-fixed timeline Events into runtime-shaped training data and candidate runtime model bundles.

Active lane: Foundry.
Excluded lanes: MA3 harness, unrelated EZ app UI work, broad architecture cleanup.

Follow `AGENTS.md` and local repo instructions. Read:

- `docs/STATUS.md`
- `docs/FOUNDRY-TRAINING.md`
- `docs/MODEL-EVOLUTION-ARCHITECTURE.md`
- `docs/TIMELINE-EVENT-80MS-INVESTIGATION-PROMPT.md` if the 80 ms root cause is still unresolved

Do not preload unrelated docs.

## Product Goal

EchoZero should treat model improvement as a core product principle:

> A user fixes Events in the timeline, then EchoZero can improve or create models from those fixes through a simple, reliable path.

The target user experience is calm and minimal:

- Improve current models.
- Create new model.
- Preview candidate behavior.
- Install or discard.
- Roll back if needed.

The implementation should begin in Foundry services and contracts. Do not build a broad UI surface until the service path is real and tested.

## Current Context

Existing work already proved part of the path:

- Shared local review sample folders can be ingested.
- One-vs-rest kick/snare datasets can be built.
- CRNN training can warm-start from an existing model bundle.
- Runtime bundles can be installed with clear identity and class labels.

Known problem:

- The shared review sample pool appears to contain mostly 80 ms clips.
- That likely reflects timeline Event duration, not runtime model input context.
- Training on padded 80 ms clips may mismatch runtime inference, which uses a longer audio window.

Therefore, this architecture must separate:

- Event truth: what the user fixed.
- Runtime-shaped sample: what the model should train on.

## Development Strategy

Work in small, shippable slices. Prefer service-level implementation and focused tests over broad UI construction.

First slice:

1. Add a dedicated Foundry model evolution module or service boundary.
2. Represent fixed Event review truth separately from materialized training audio.
3. Materialize runtime-shaped drum samples from source audio using deterministic window policy.
4. Build one-vs-rest datasets that include all negatives for the selected scope.
5. Resolve lineage from the current installed model or explicit seed model.
6. Produce candidate run specs with explicit profile, identity, class, and lineage metadata.
7. Install only compatible runtime bundles, preserving app selector compatibility.

Do not silently change the existing project-specialized model flow.

## Suggested Module Shape

Prefer a dedicated package if it fits the repo:

```text
echozero/foundry/model_evolution/
  __init__.py
  truth_collector.py
  sample_materializer.py
  lineage.py
  planner.py
  service.py
```

Or use a smaller service-first slice if the repo style suggests it:

```text
echozero/foundry/services/model_evolution_service.py
```

Reuse existing Foundry services for:

- dataset version creation
- one-vs-rest derived datasets
- run spec validation
- CRNN warm-start training
- artifact validation
- runtime bundle install/index

Add new abstraction only when it removes real complexity or protects the Event truth vs training sample boundary.

## Core Requirements

### Review Truth

Collect or represent fixed Events with:

- source project/song identity when available
- source audio path or source audio id
- target class
- Event start/end
- optional corrected onset/anchor if available
- review decision metadata
- source kind

Do not assume Event visual duration is the training clip duration.

### Runtime-Shaped Samples

Materialize samples with a policy compatible with runtime inference.

For binary drum models, use a deterministic window around the Event onset/anchor. Record:

- sample id
- content hash
- source audio identity
- Event start/end
- materialized window start/end
- target class
- source kind
- window policy version

Keep deterministic ordering.

Do not commit generated WAVs, caches, model weights, or local artifacts.

### Negatives

For one-vs-rest training:

- target class examples are positives
- every other eligible class in the selected scope is a negative
- report negative source class counts
- avoid accidental filtering that drops hard negatives

### Lineage

The new model must clearly state what it started from:

- installed current model
- explicit seed bundle
- prior candidate checkpoint

Write lineage into run metadata and installed bundle manifests where practical.

### Profiles

Use explicit training profiles. Do not make all existing paths heavier by default.

Recommended profile names:

- `quick_check`
- `beefy`
- `release_candidate`

Unit tests should not run heavy training. Mock or fake training where existing tests do.

### Promotion

Do not auto-promote a model only because training finished.

A candidate should expose enough metadata for a later promotion gate:

- validation metrics
- threshold policy
- replay/eval scope
- compatibility status
- lineage

If full promotion is too large for the first slice, create the contract and make install explicit.

## UX Direction

Do not build the full UX unless asked, but design the service API so the future UI can be simple.

Future primary surface:

- one primary action: `Improve Models`
- secondary action: `Create New Model`
- current installed model cards
- fixed Event count
- candidate status
- review before install

No raw hyperparameters on the primary surface.

## Tests

Add focused tests for the first slice:

- fixed Event truth is represented without losing Event timing
- materializer creates runtime-shaped windows independent of 80 ms Event duration
- one-vs-rest planning includes all negatives
- lineage resolver chooses the installed current model when requested
- candidate or bundle metadata includes identity, class, profile, and lineage
- existing project-specialized flow still passes its focused tests

Use targeted tests only. Do not run broad full-suite tests unless a change touches broad contracts.

## Verification

Run the smallest meaningful gate. Prefer commands like:

```text
.venv/bin/python -m pytest tests/foundry/<focused_test>.py -q
```

If real training is too expensive, do not run it as a unit gate. Provide the exact manual command.

## Guardrails

- Do not touch MA3.
- Do not redesign unrelated app UI.
- Do not rework broad architecture outside Foundry.
- Do not commit generated artifacts.
- Do not revert unrelated work in the dirty tree.
- Do not break existing project-specialized model behavior.
- Keep generated model files under local artifact/model roots only.

## Desired Final Payload

Return:

```text
status: success | partial | blocked
changed_files:
  - path
verification:
  - command/result
summary:
  - concise bullets
manual_train_command:
  - exact command if applicable
blocker: null or explanation
residual_risk:
  - concise bullets
```

If implementation is split into multiple commits or agents, each agent must report its scope, changed files, tests, summary, blocker, and residual risk.

