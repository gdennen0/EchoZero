# Playback Remediation Task Board

Last updated: 2026-04-18

This board turns the playback/clock audit into executable work.
It is intentionally concrete:
- each task has an ID
- each task has a bounded owner role
- each task has dependencies
- each task has a required proof lane
- each task has a done condition

This board is for near-term stabilization of playback, transport timing, and human-path demo proof.

## Goal

Ship a reliable near-term playback and clock system for EchoZero that:
- separates selection from playback target
- uses real application paths for demos and non-deterministic functional tests
- produces correct audio playback on real hardware
- renders a smooth, trustworthy playhead
- preserves a path to a future bespoke DAW-grade playback engine

## Hard Rules

- Human-path demo rule is mandatory.
- No mock analysis, fake stream, direct audio-callback driving, or widget state injection for demo claims.
- Main app/runtime contracts win over helper/demo convenience.
- Playback target must become independent from selection.
- Transport timing must be owned by the playback/backend side, not by UI polling.

## Execution Order

Work in this order:

1. PB-00 through PB-03
2. PB-10 through PB-14
3. PB-20 decision gate
4. PB-30 through PB-34
5. PB-40 through PB-43
6. PB-50 through PB-54
7. PB-60 through PB-63
8. PB-70 final signoff

## Task Board

### Phase 0: Freeze and Cleanup

#### PB-00: Freeze Invalid Demo Patterns
- Owner: `lead-dev`
- Depends on: none
- Scope:
  - identify all playback demos/tests that use fake streams, callback driving, or presentation injection
  - mark them as simulated proof or remove them from demo claims
- Files:
  - `echozero/testing/**`
  - `tests/testing/**`
  - `tests/ui_automation/**`
- Proof:
  - targeted grep/report in PR notes
  - targeted pytest slice if behavior changes
- Done when:
  - no human-path demo artifact depends on simulated playback

#### PB-01: Remove Selection-as-Playback Fallback
- Owner: `impl`
- Depends on: PB-00
- Scope:
  - remove fallback from selected layer to playback target
  - playback target must be explicit or empty
- Files:
  - `echozero/ui/qt/timeline/runtime_audio.py`
  - related tests
- Proof:
  - targeted runtime-audio tests
  - targeted widget/application tests
- Done when:
  - changing selection alone cannot change playback source

#### PB-02: Quarantine Invalid Recorder Paths
- Owner: `impl`
- Depends on: PB-00
- Scope:
  - quarantine or relabel current simulated playback recorder helpers
  - prevent them from being used for operator demo claims
- Files:
  - `echozero/testing/playback_capture.py`
  - `echozero/testing/app_flow.py`
  - relevant tests/docs
- Proof:
  - targeted pytest slices
  - doc update confirmation
- Done when:
  - recorder helpers clearly indicate simulated vs human-path status

#### PB-03: Human-Path Demo Policy Enforcement
- Owner: `impl`
- Depends on: PB-00
- Scope:
  - add guardrails in harness/demo entrypoints so simulated playback paths cannot be mistaken for human-path demos
- Files:
  - `echozero/testing/**`
  - `docs/TESTING.md`
  - `AGENTS.md`
- Proof:
  - targeted tests
  - doc review
- Done when:
  - invalid demo modes are blocked or explicitly labeled

### Phase 1: State Model Separation

#### PB-10: Introduce Playback Target State
- Owner: `impl`
- Depends on: PB-01
- Scope:
  - add first-class playback target state below presentation
  - keep selection independent
- Files:
  - `echozero/application/timeline/models.py`
  - `echozero/application/presentation/models.py`
  - `echozero/application/timeline/intents.py`
  - `echozero/application/timeline/orchestrator.py`
  - `echozero/application/timeline/assembler.py`
- Proof:
  - targeted application/model tests
  - targeted assembler tests
- Done when:
  - timeline model owns both selection and playback target

#### PB-11: Add Explicit Playback-Target Intent
- Owner: `impl`
- Depends on: PB-10
- Scope:
  - add explicit playback-target intent, e.g. `SetActivePlaybackTarget`
  - deprecate ambiguous selection-coupled route behavior
- Files:
  - `echozero/application/timeline/intents.py`
  - `echozero/application/timeline/orchestrator.py`
  - tests
- Proof:
  - targeted orchestrator tests
- Done when:
  - playback target changes through explicit intent only

#### PB-12: Presentation Flags for Playback Active State
- Owner: `impl`
- Depends on: PB-10
- Scope:
  - add `active_playback_layer_id`
  - add `active_playback_take_id`
  - add `LayerPresentation.is_playback_active`
- Files:
  - `echozero/application/presentation/models.py`
  - `echozero/application/timeline/assembler.py`
  - tests
- Proof:
  - targeted assembler/presentation tests
- Done when:
  - presentation can show selected and playback-active state independently

#### PB-13: App-Shell Refresh Preservation
- Owner: `impl`
- Depends on: PB-10
- Scope:
  - preserve playback target across app-shell refresh/storage reload where valid
  - repair deterministically when target disappears
- Files:
  - `echozero/ui/qt/app_shell.py`
  - tests
- Proof:
  - targeted app-shell runtime tests
- Done when:
  - reload behavior keeps target/selection independent

#### PB-14: Contract Test Pass for Selection vs Playback Target
- Owner: `verify`
- Depends on: PB-10, PB-11, PB-12, PB-13
- Scope:
  - prove selection and playback target can diverge safely
- Proof:
  - targeted pytest slices
  - `python -m echozero.testing.run --lane appflow`
- Done when:
  - all selection-vs-playback-target contract tests pass

### Phase 2: Backend Decision Gate

#### PB-20: Playback Backend Bakeoff
- Owner: `research`
- Depends on: PB-10
- Scope:
  - evaluate current `sounddevice` engine vs temporary more proven backend
  - compare correctness, device behavior, timing exposure, integration cost
- Candidates:
  - current `echozero/audio/*` path
  - Qt Multimedia-backed path (`QAudioSink` and related primitives)
  - any other narrowly justified proven backend
- Deliverable:
  - written decision memo with recommendation
- Proof:
  - code references
  - primary-source references
  - local repro notes on real hardware
- Done when:
  - one backend direction is chosen for near-term stabilization

### Phase 3: Playback Service Boundary

#### PB-30: Define Playback Service Contract
- Owner: `impl`
- Depends on: PB-20
- Scope:
  - create service boundary owning:
    - transport state
    - playback target
    - output format/device session
    - playback timing snapshots
    - mute/solo/audition policy seam
- Files:
  - new/existing playback service modules under `echozero/application/**`
  - app shell integration
- Proof:
  - targeted contract tests
- Done when:
  - UI does not depend on raw backend details

#### PB-31: Move Runtime Audio Mapping Behind Service
- Owner: `impl`
- Depends on: PB-30
- Scope:
  - stop letting widget/runtime layer own core playback semantics
  - runtime audio becomes adapter/infrastructure
- Files:
  - `echozero/ui/qt/timeline/runtime_audio.py`
  - `echozero/ui/qt/app_shell.py`
  - service files
- Proof:
  - targeted playback/runtime tests
- Done when:
  - playback target resolution lives in service layer, not widget convenience code

#### PB-32: Device/Format Negotiation Audit and Fix
- Owner: `impl`
- Depends on: PB-20, PB-30
- Scope:
  - audit sample rate, channel count, dtype, latency, stream format assumptions
  - fix static/noise root cause
- Files:
  - `echozero/audio/engine.py`
  - backend adapter files
  - runtime service tests
- Proof:
  - targeted backend tests
  - real hardware manual validation
- Done when:
  - real playback is audibly correct on target hardware

#### PB-33: Real Playback Smoke Path
- Owner: `verify`
- Depends on: PB-31, PB-32
- Scope:
  - verify load song, play, seek, stop, switch active playback target on real app path
- Proof:
  - `python -m echozero.testing.run --lane appflow`
  - manual smoke note
- Done when:
  - smoke path is green and reproducible

#### PB-34: Review Gate for Service Boundary
- Owner: `review`
- Depends on: PB-30 through PB-33
- Scope:
  - audit for contract leakage, backend coupling, selection fallback regressions
- Proof:
  - review findings
- Done when:
  - no major architecture regression remains unresolved

### Phase 4: UI Redesign

#### PB-40: Add Active Button in Layer Header
- Owner: `impl`
- Depends on: PB-12, PB-30
- Scope:
  - add explicit `Active` button/control in layer title area
  - no selection side effects
- Files:
  - `echozero/ui/qt/timeline/blocks/layer_header.py`
  - `echozero/ui/qt/timeline/widget.py`
  - style files if needed
- Proof:
  - widget tests
  - app-flow lane
- Done when:
  - header click selects, Active button activates playback target

#### PB-41: Object Info Separation
- Owner: `impl`
- Depends on: PB-12, PB-30
- Scope:
  - object info must show selected identity separately from playback target
- Files:
  - `echozero/application/presentation/inspector_contract.py`
  - `echozero/ui/qt/timeline/widget.py`
  - object info panel files
- Proof:
  - inspector contract tests
  - widget tests
- Done when:
  - object info no longer implies selection equals playback target

#### PB-42: Visual Distinction Between Selected and Active
- Owner: `impl`
- Depends on: PB-40, PB-41
- Scope:
  - selected highlight and active-playback indicator must be visually distinct
- Files:
  - timeline header/canvas/style files
- Proof:
  - widget tests
  - human review screenshot
- Done when:
  - a user can identify both states at a glance

#### PB-43: UI Contract Verification
- Owner: `verify`
- Depends on: PB-40, PB-41, PB-42
- Scope:
  - prove UI actions follow contract:
    - row click selects only
    - Active sets playback target only
- Proof:
  - targeted pytest slice
  - `python -m echozero.testing.run --lane gui-lane-b` if still relevant
  - `python -m echozero.testing.run --lane appflow`
- Done when:
  - UI contract proof is green

### Phase 5: Clock and Playhead

#### PB-50: Define Transport Snapshot Model
- Owner: `impl`
- Depends on: PB-30
- Scope:
  - define authoritative playback snapshot:
    - sample position
    - playing state
    - output latency / stream time
    - seek/stop markers
- Files:
  - playback service modules
  - backend adapter files
- Proof:
  - contract tests
- Done when:
  - UI can render from snapshots without treating timer polling as truth

#### PB-51: Implement Presentation Clock and Interpolation
- Owner: `impl`
- Depends on: PB-50
- Scope:
  - UI renders playhead at display cadence using interpolation/extrapolation from transport snapshots
  - drift corrections are gentle, not stepwise
- Files:
  - `echozero/ui/qt/timeline/widget.py`
  - related timing helpers
- Proof:
  - targeted widget/runtime tests
  - perf spot check
- Done when:
  - visible playhead is smooth and monotonic

#### PB-52: Remove Timer-as-Truth Behavior
- Owner: `impl`
- Depends on: PB-51
- Scope:
  - UI timer may trigger repaint cadence, but not define playback truth
- Files:
  - `echozero/ui/qt/timeline/widget.py`
- Proof:
  - targeted tests
- Done when:
  - UI timer is presentation cadence only

#### PB-53: Seek/Stop/Start Edge-Case Audit
- Owner: `verify`
- Depends on: PB-50, PB-51, PB-52
- Scope:
  - verify no backward jumps, stale playhead snaps, or bad restart offsets
- Proof:
  - targeted runtime/playback tests
  - appflow lane
- Done when:
  - edge-case timing regressions are closed

#### PB-54: Perf Guardrail for Playhead Path
- Owner: `verify`
- Depends on: PB-51
- Scope:
  - ensure playhead smoothing does not regress hot-path rendering
- Proof:
  - `pytest tests/benchmarks/benchmark_timeline_phase3.py -q`
- Done when:
  - perf remains acceptable

### Phase 6: Human-Path Demo/Test Rebuild

#### PB-60: Rebuild Playback Demo on Real Human Path
- Owner: `impl`
- Depends on: PB-32, PB-43, PB-53
- Scope:
  - blank project
  - load real song
  - run real stems pipeline
  - wait for completion
  - set Active via real UI
  - begin playback
  - switch Active via real UI only
- Files:
  - human-path demo tooling only
- Proof:
  - generated demo artifact
  - explicit run log
- Done when:
  - demo is reproducible through real app interactions only

#### PB-61: Human-Path Functional Playback Test
- Owner: `impl`
- Depends on: PB-60
- Scope:
  - real playback functional test through automation/live app surface only
- Proof:
  - targeted functional test
- Done when:
  - no fake stream, no callback driving, no presentation injection is used

#### PB-62: Manual Operator Proof
- Owner: `verify`
- Depends on: PB-60, PB-61
- Scope:
  - human validates audio correctness, active-layer switching, and playhead smoothness
- Proof:
  - manual QA note
- Done when:
  - operator can reproduce expected playback behavior

#### PB-63: Demo/Functional Review Gate
- Owner: `review`
- Depends on: PB-60, PB-61, PB-62
- Scope:
  - confirm demo/test path is truly human-path and contract-clean
- Proof:
  - review findings
- Done when:
  - no demo-path bypass remains

### Phase 7: Final Signoff

#### PB-70: Playback Stabilization Signoff
- Owner: `lead-dev`
- Depends on: PB-14, PB-20, PB-34, PB-43, PB-54, PB-63
- Scope:
  - consolidate final evidence
  - confirm near-term playback system is stable enough to build on
- Proof:
  - appflow results
  - targeted slices
  - perf result
  - demo artifact
  - manual QA note
- Done when:
  - playback remediation is accepted as the new baseline

## Decision Log

### D-PB-1
- Decision:
  - selection and playback target must be separate first-class concepts

### D-PB-2
- Decision:
  - backend/engine owns playback time truth

### D-PB-3
- Decision:
  - UI renders a smoothed playhead from authoritative transport snapshots

### D-PB-4
- Decision:
  - demo and non-deterministic functional playback proof must use real human paths only

### D-PB-5
- Decision:
  - current bespoke engine may remain only if it survives the backend bakeoff behind a service boundary

## Suggested Panel Assignment

- Panel A:
  - PB-20 backend bakeoff
  - PB-32 device/format audit
  - PB-50 transport snapshot model

- Panel B:
  - PB-10 through PB-14 state-model separation
  - PB-40 through PB-43 UI separation

- Panel C:
  - PB-51 through PB-54 playhead/presentation clock
  - PB-60 through PB-63 human-path demo/test rebuild

## Minimal First Slice

If only one slice starts now, do this:

1. PB-10
2. PB-11
3. PB-12
4. PB-40
5. PB-41
6. PB-43

Reason:
- it fixes the wrong contract first
- it stops selection from masquerading as playback target
- it creates the correct seam before backend replacement work
