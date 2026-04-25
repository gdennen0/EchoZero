# Review Signal Execution Plan

Status: active implementation plan
Last verified: 2026-04-24

Primary source:
- [REVIEW-SIGNAL-FEATURE-SPEC.md](REVIEW-SIGNAL-FEATURE-SPEC.md)

This plan turns the shared review-signal feature into bounded execution waves.

## Strategy

Use the Foundry review slice that already landed as the seed contract, then
expand outward in this order:

1. harden the canonical review signal shape
2. generate queues from canonical EZ project data
3. wire timeline fix mode into the same persisted signal lane
4. add missed-event and boundary-correction capture
5. promote reviewed records into Foundry dataset/training inputs

Why this order:
- phone review already exists and proves the basic review lane
- timeline fix mode must not invent its own review persistence
- downstream training export is unsafe until review provenance is explicit
- project-adaptive models should remain blocked until the shared review signal
  is trustworthy

## Current Baseline

Completed on 2026-04-24:
- Foundry review records now support a nested reusable decision shape with
  `verified`, `rejected`, and `relabeled`
- Foundry review snapshots support deterministic cursor navigation
- phone review now exposes explicit `Prev` and `Next`
- focused proof passes in `tests/foundry/test_review_sessions.py`

Baseline files already in play:
- `echozero/foundry/domain/review.py`
- `echozero/foundry/persistence/review_repository.py`
- `echozero/foundry/services/review_session_service.py`
- `echozero/foundry/review_server.py`
- `echozero/foundry/review_web.py`
- `tests/foundry/test_review_sessions.py`

## Hard Rules

- Main remains truth.
- Timeline writes through the application boundary, never directly from widgets.
- Foundry owns persisted review queues and downstream training consumption.
- Keep parallel work bounded to at most two writers.
- Do not run overlapping writers in the same file cluster without an explicit
  integration plan.
- App-facing timeline changes are not done until proven through the real app
  path.

## Wave 1
Contract Hardening

Goal:
- evolve the seeded Foundry review shape into the canonical shared review
  contract without breaking the existing phone lane

Scope:
- expand decision/provenance fields
- add training-eligibility semantics
- preserve compatibility with current review sessions

Likely files:
- `echozero/foundry/domain/review.py`
- `echozero/foundry/persistence/review_repository.py`
- `echozero/foundry/services/review_session_service.py`
- new helper under `echozero/foundry/services/` if a dedicated review-signal
  service is warranted
- `tests/foundry/test_review_sessions.py`

Done when:
- decision kinds can represent verified, rejected, relabeled,
  boundary-corrected, and missed-event-added semantics
- provenance is explicit enough for later timeline integration
- compatibility with current phone review data is preserved

Proof:
- `./.venv/bin/python -m pytest tests/foundry/test_review_sessions.py -q`
- additional focused Foundry review tests as needed

## Wave 2
Queue Generation From EZ Data

Goal:
- build review queues from canonical project/song/layer/main data instead of
  only imported clip folders or JSON payloads

Scope:
- derive deterministic queue items from real EZ project context
- carry song/layer/event provenance into the review lane
- support project/song/layer-scoped phone review

Likely files:
- `echozero/foundry/services/review_session_service.py`
- `echozero/foundry/review_import.py`
- new queue-builder helper under `echozero/foundry/services/`
- `echozero/application/timeline/app.py` or app-shell integration seams if a
  launch action is added
- focused tests under `tests/foundry/`

Done when:
- a queue can be created for a selected song/layer from real EZ data
- queue ordering is deterministic
- queue items carry enough provenance to map back to the canonical event

Proof:
- focused `tests/foundry/` queue-builder coverage
- launch/creation smoke if a user-facing entrypoint is added

## Wave 3
Timeline Fix-Mode Write-Through

Goal:
- make timeline fix mode a first-class producer of persisted review signal

Scope:
- add or adapt typed timeline intents for explicit review-relevant actions
- persist review signal through the timeline application boundary
- keep widgets as UI only

Likely files:
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/orchestrator_event_edit_mixin.py`
- `echozero/application/timeline/assembler.py` or related assembler helpers
- `echozero/ui/qt/app_shell.py`
- `echozero/ui/qt/timeline/widget_action_contract_mixin.py`
- `echozero/ui/qt/timeline/widget_actions.py`
- focused tests under `tests/application/` and `tests/ui/`

Done when:
- explicit verify, reject, and relabel actions in fix mode write durable review
  signal
- the write path goes through the canonical timeline app contract
- no widget-local alternate truth is introduced

Proof:
- focused timeline application tests
- focused Qt timeline/app-shell tests
- `./.venv/bin/python -m echozero.testing.run --lane appflow`

## Wave 4
Boundary Corrections And Missed Events

Goal:
- capture the two high-value expert corrections that matter most after simple
  verify/reject/relabel

Scope:
- retime/resize should emit `boundary_corrected`
- manual event creation should emit `missed_event_added`
- keep the canonical main event as truth while preserving review provenance

Likely files:
- `echozero/application/timeline/orchestrator_event_edit_mixin.py`
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/intents.py`
- Foundry review contract/service files from Wave 1
- focused tests under `tests/application/` and `tests/ui/`

Done when:
- corrected-boundary reviews preserve both original and corrected context
- manually added events can later become reviewed positive signal
- timeline undo/redo remains coherent

Proof:
- targeted timeline orchestrator tests
- targeted app-shell and undo/redo tests
- `./.venv/bin/python -m echozero.testing.run --lane appflow`

## Wave 5
Phone Queue Productization

Goal:
- move the phone review lane from generic imported sessions to real
  project/song/layer review work

Scope:
- session index reflects project-backed queues
- queue filters align with real EZ review workflows
- phone review stays fast and one-screen

Likely files:
- `echozero/foundry/review_server.py`
- `echozero/foundry/review_web.py`
- `echozero/foundry/services/review_session_service.py`
- app-shell launch or entry files if opening phone review becomes user-facing in
  the desktop app
- `tests/foundry/test_review_sessions.py`

Done when:
- operators can review song/layer queues on the phone without importing manual
  sidecar JSON first
- prev/next and filter semantics still operate deterministically
- queue progress is preserved through persistence

Proof:
- `./.venv/bin/python -m pytest tests/foundry/test_review_sessions.py -q`
- targeted app-shell smoke if a launch surface is added

## Wave 6
Promotion Into Foundry Dataset Signals

Goal:
- convert reviewed records into dataset-ready positives and negatives for
  Foundry

Scope:
- reviewed record export
- relabel pair handling
- corrected-boundary and missed-event materialization
- class-balance and eligibility reporting before training

Likely files:
- `echozero/foundry/services/dataset_service.py`
- `echozero/foundry/services/train_run_service.py`
- `echozero/foundry/services/artifact_service.py`
- `echozero/foundry/contracts/dataset_version.v1.json`
- focused tests under `tests/foundry/`

Done when:
- Foundry can derive training-ready reviewed examples from the shared review
  lane
- untouched events are excluded by rule, not convention
- relabel and missed-event semantics are explicit in exported data

Proof:
- focused Foundry dataset/training tests
- any contract validation tests touched by export changes

## Deferred Work

Not part of this execution plan's done bar:
- project-specific fine-tune scheduling
- project model promotion policy
- runtime project-model selection
- automatic weighting strategies for project-adaptive training

Those stay blocked until Wave 6 is complete and trustworthy.

## Recommended Worker Split

Maximum active writers:
- 2

Recommended split after Wave 1 design is explicit:
- Writer A:
  Foundry review contract, queue building, phone review, dataset promotion
- Writer B:
  timeline application and UI integration for fix-mode write-through
- Verify sidecar:
  targeted proof plus `appflow` replay when timeline surfaces change
- Review sidecar:
  audit for alternate-truth leakage, missing provenance, and test gaps

Do not split two writers across the same timeline file cluster.
Do not let UI work finalize before the application write path is explicit.

## Suggested Immediate Next Slice

If work resumes now, the best next slice is Wave 1 plus the queue-creation seam
for Wave 2.

Why:
- the current Foundry review lane already proves the basic mobile UX
- the next architectural risk is not the web page, it is the shared review
  contract and provenance shape
- once the contract is explicit, timeline integration becomes much safer to
  implement without rework

## Completion Bar

This plan is complete when:
- the shared review signal contract is durable and explicit
- phone review and timeline fix mode both write it
- missed events and corrected boundaries are represented
- Foundry can turn reviewed records into dataset-ready signals
- all relevant proof lanes pass
