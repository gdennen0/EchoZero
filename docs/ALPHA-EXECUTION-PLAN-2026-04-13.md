# Alpha Execution Plan (2026-04-13)

This plan turns the gap matrix into executable tranches for alpha readiness. The tranche order favors the requested real user flow:

`Open EZ -> New Project -> Load Song -> Extract Stems -> Extract Drum Events -> event actions -> MA3 actions`

## Tranche 1: Canonical stem extraction kickoff

- Objective: make `Extract Stems` work through the canonical app-shell runtime for the imported song layer and persist real derived audio layers/takes.
- Files / modules:
  - `EchoZero/ui/qt/app_shell.py`
  - `tests/ui/test_app_shell_runtime_flow.py`
- Acceptance tests:
  - `tests/ui/test_app_shell_runtime_flow.py`
  - `tests/ui/test_timeline_shell.py`
- Demo proof artifact expected:
  - app-shell presentation showing `Imported Song`, `Drums`, `Bass`, `Vocals`, `Other`
  - passing runtime test proving persisted stem layers/takes
- Stop / go criteria:
  - Go when source-song extraction is green and no timeline-shell regressions appear
  - Stop if extraction still relies on fake state mutation or bypasses persistence

Status: started and landed in this slice.

## Tranche 2: Canonical drum-event extraction

- Objective: wire `Extract Drum Events` from the drums stem into persisted event layers through the same runtime/orchestrator path.
- Files / modules:
  - `EchoZero/ui/qt/app_shell.py`
  - `EchoZero/services/orchestrator.py` or a small runtime analysis seam if needed
  - relevant processor/template wiring for onset or classifier execution
  - `tests/ui/test_app_shell_runtime_flow.py`
  - `tests/ui/test_timeline_shell.py`
- Acceptance tests:
  - source-song import -> stem extraction -> drum-event extraction creates event layers and takes
  - clear runtime failure on invalid source-layer selection remains explicit
- Demo proof artifact expected:
  - app-shell screenshot or trace with drums stem plus kick/snare/hat-class event layers
- Stop / go criteria:
  - Go when the source-song flow reaches visible event layers without presentation-state faking
  - Stop if implementation requires UI-only mutation or violates main/take persistence rules

## Tranche 3: Lane B real-input closure for the alpha flow

- Objective: update Lane B so the starter scenario proves the real extraction flow instead of the old explicit failure path.
- Files / modules:
  - `EchoZero/testing/gui_lane_b.py`
  - `EchoZero/testing/gui_dsl.py`
  - `tests/gui/scenarios/e2e_core.json`
  - `tests/testing/test_gui_lane_b.py`
- Acceptance tests:
  - Lane B scenario passes locally and in CI
  - trace JSON captures add-song, extract-stems, and extract-drum-events state transitions
- Demo proof artifact expected:
  - `trace.json`
  - final screenshot from the green starter scenario
- Stop / go criteria:
  - Go when Lane B no longer encodes extraction failure as the expected outcome
  - Stop if the runner starts mutating presentation state directly to fake success

## Tranche 4: Sync proof hardening for alpha

- Objective: close the remaining A6-style proof gap by extending app-level and end-to-end sync coverage around main-only writes and transfer safety.
- Files / modules:
  - `tests/application/test_manual_transfer_push_flow.py`
  - `tests/application/test_manual_transfer_pull_flow.py`
  - `tests/unit/test_ma3_communication_service_protocol.py`
  - `tests/unit/test_ma3_receive_path_integration.py`
  - app/runtime sync seam files if assertions need to move upward
- Acceptance tests:
  - explicit proof that non-main/take data does not cross the MA3 write boundary
  - push/pull divergence, reconnect, and empty-state cases remain green
- Demo proof artifact expected:
  - targeted appflow-sync and appflow-protocol reports
- Stop / go criteria:
  - Go when main-only behavior is demonstrably enforced at the app boundary
  - Stop if proof still depends on lower-level unit assumptions only

## Tranche 5: Alpha packaging and operator validation

- Objective: turn the already-built packaging path into an alpha gate with operator evidence.
- Files / modules:
  - `scripts/build-test-release.ps1`
  - `scripts/smoke-test-release.ps1`
  - `scripts/run-appflow-gates.ps1`
  - appflow summary / delivery docs
- Acceptance tests:
  - packaged smoke lane green
  - milestone manual QA walkthrough completed against the packaged build
- Demo proof artifact expected:
  - release artifact bundle
  - smoke JSON summary
  - walkthrough notes/screens
- Stop / go criteria:
  - Go when packaged app boot and shutdown are repeatable and manual critical-path QA is complete
  - Stop if alpha evidence exists only in dev-run mode

## Tranche 6: Post-alpha defer bucket

- Objective: keep non-critical but important work out of the first alpha gate.
- Files / modules:
  - undo/redo infrastructure
  - wider arbitrary-layer rerun support
  - Foundry/train expansion docs and modules
- Acceptance tests:
  - dedicated undo/redo tests and UI affordance tests
  - arbitrary derived-audio rerun coverage
  - Foundry-specific validation lanes as needed
- Demo proof artifact expected:
  - not required for first alpha signoff
- Stop / go criteria:
  - Stay deferred unless the alpha target explicitly broadens beyond the current app-shell flow

## Current tranche summary

Completed in tranche 1:
- canonical runtime now performs source-song stem extraction
- derived audio layers/takes are persisted through the orchestrator path
- focused runtime and timeline-shell regression tests are green

Remaining before alpha:
- drum-event extraction
- Lane B green path
- sync end-to-end proof closure
- packaged-app manual QA
