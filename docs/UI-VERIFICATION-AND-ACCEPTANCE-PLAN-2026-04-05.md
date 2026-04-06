# UI Verification and Acceptance Plan - 2026-04-05

## Purpose

Define how timeline UI work is proven in EchoZero before it is accepted.

This plan follows current repo direction:
- prioritize integration and conformance over broad new test expansion
- prove the real UI contract, not just isolated widgets
- keep golden-record comparison as a future acceptance gate, not a prerequisite today

This plan is grounded in:
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/UX-MICRO-TESTS.md`
- `docs/UX-DESIGN-DECISIONS.md`
- `docs/UI-CONTRACT-AUDIT-2026-04-03.md`
- `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`
- `docs/PERF-GUARDRAILS.md`
- current `tests/ui/*`, `tests/application/*`
- `run_timeline_real_data_capture.py`
- `run_timeline_walkthrough.py`
- `echozero/ui/qt/timeline/real_data_fixture.py`

## Acceptance Principle

UI work is accepted when it clears the relevant proof lanes for the change.

The lanes are intentionally narrow:
1. contract/unit
2. real-data smoke
3. walkthrough capture
4. perf guardrails
5. future golden-record comparison

Not every change needs every lane on every iteration, but no timeline-facing change should merge without clearing at least the contract/unit lane and the appropriate smoke/capture lane for the surface it changes.

## What Must Be Proven

At acceptance time, the evidence should answer five questions:

1. Does the code still honor timeline first principles and locked UX decisions?
2. Does the UI still render and behave correctly on realistic analyzed data, not just demo fixtures?
3. Can a human reviewer see the intended behavior in a stable capture artifact?
4. Did the change preserve agreed performance budgets?
5. Is the change moving toward stable visual regression proof without making that a blocker yet?

## Functionality Verification Sweep

The Functionality Verification Sweep is the repeatable procedure for timeline UI work.

### Trigger

Run the sweep when a change touches any of:
- `echozero/ui/qt/timeline/*`
- `echozero/application/timeline/*`
- fixture/capture code that drives the timeline shell
- FEEL constants or layout wiring used by the timeline
- persistence/service code that changes main/take/staleness behavior visible in the UI

### Procedure

1. Identify the contract at risk.
   - Map the change to first-principles rules, UX decisions, FEEL contract, and current audits.
   - Decide which proof lanes are mandatory for this change.

2. Run contract/unit proof first.
   - Confirm the change did not break truth-model, take behavior, FEEL wiring, shell structure, ruler/follow behavior, or other covered contracts.

3. Run real-data smoke for any visible timeline behavior.
   - Use the real-data fixture/capture path to verify the UI still renders against analyzed audio outputs and take lanes.

4. Produce a walkthrough capture when reviewer-visible behavior changed.
   - Use the scripted walkthrough or equivalent recorded run to create reviewer-facing evidence.

5. Run perf guardrails when paint, assembly, scrolling, zoom, layout density, or event-lane behavior changed.
   - Treat threshold regressions as acceptance failures unless explicitly waived.

6. Record outcome and artifacts.
   - Keep pass/fail status explicit.
   - Attach or link produced artifacts in the PR, task note, or handoff.

### Sweep Output

Each sweep should end with a short result block in the PR/task:
- `contract/unit`: pass or fail
- `real-data smoke`: pass or fail
- `walkthrough capture`: pass or fail
- `perf guardrails`: pass or fail/not-required
- `golden-record`: not-run for now, or future-ready note

## Proof Lanes

## 1. Contract / Unit Lane

### Purpose

Catch violations of timeline truth model, FEEL contract, shell structure, and deterministic view logic before doing heavier UI proof.

### Primary Scope

- application timeline contract tests
- UI contract/unit tests
- shell-level deterministic behavior

### Current Evidence Sources

Representative tests already in repo:
- `tests/application/test_timeline_assembler_contract.py`
- `tests/application/test_timeline_orchestrator_take_actions.py`
- `tests/application/test_timeline_assembler_incremental.py`
- `tests/ui/test_timeline_feel_contract.py`
- `tests/ui/test_timeline_shell.py`
- `tests/ui/test_follow_scroll.py`
- `tests/ui/test_ruler_block.py`
- `tests/ui/test_take_row_block.py`
- `tests/ui/test_event_geometry.py`
- `tests/ui/test_event_lane_culling.py`

### Pass Condition

Pass when:
- targeted contract/unit tests exit cleanly
- no failure reintroduces known audit risks such as active-take truth leakage or FEEL drift
- the change either uses existing coverage or adds narrowly scoped tests for the specific contract it changed

### Fail Condition

Fail when:
- any relevant contract/unit test fails
- a change to visible timeline behavior ships without proving the underlying contract it changed
- the change broadens behavior without updating the most directly impacted contract test

### Recommended Output

- console test summary
- optional junit/pytest artifact if CI produces one

### Artifact

Minimum artifact:
- test run log in CI or attached command output in the task/PR

Preferred command shape:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\application tests\ui
```

For focused local verification, narrow to the impacted files first, then expand if needed.

## 2. Real-Data Smoke Lane

### Purpose

Prove the timeline still works on actual analyzed audio data, including stable layers, takes, waveform registration, and stems-first presentation.

### Primary Scope

- real analyzed audio import
- stable layer/take presentation
- event and audio lane population
- real-data screenshot generation

### Current Evidence Sources

- `run_timeline_real_data_capture.py`
- `echozero/ui/qt/timeline/real_data_fixture.py`
- `tests/ui/test_real_data_fixture.py`

### Pass Condition

Pass when:
- the capture script completes without error
- screenshot files are produced for all expected variants
- summary output prints a non-empty layer/take/event picture consistent with the run
- the generated screenshots show the intended UI behavior on real data

Expected variant set today:
- `real_default`
- `real_scrolled`
- `real_zoomed_in`
- `real_zoomed_out`

### Fail Condition

Fail when:
- the script errors
- expected screenshots are missing
- summary counts indicate broken pipeline-to-presentation flow
- the screenshots reveal contract breakage that fixture-only tests would not catch

### Required Output

The script already prints:
- screenshot paths
- `REAL_DATA_SUMMARY`
- `audio=...`
- `working_dir=...`
- `song_version_id=...`
- `layers=...`
- `takes=...`
- `main_events=...`

### Artifact

Required artifacts:
- `artifacts/timeline-real-data/timeline_real_default.png`
- `artifacts/timeline-real-data/timeline_real_scrolled.png`
- `artifacts/timeline-real-data/timeline_real_zoomed_in.png`
- `artifacts/timeline-real-data/timeline_real_zoomed_out.png`
- captured summary text in CI log or task note

Recommended command shape:

```powershell
.\.venv\Scripts\python.exe run_timeline_real_data_capture.py --output-dir artifacts\timeline-real-data
```

## 3. Walkthrough Capture Lane

### Purpose

Create reviewer-visible proof for interaction flow, not just static rendering.

This lane is especially useful when a change affects:
- take lane expansion/collapse
- take selection
- playhead/seek/play/pause flow
- visible shell choreography

### Current Evidence Sources

- `run_timeline_walkthrough.py`
- `echozero/ui/qt/timeline/demo_walkthrough.py`

### Pass Condition

Pass when:
- the walkthrough runs end-to-end without exceptions
- the expected scripted interaction sequence is visible
- capture output is reviewable and shows the behavior change clearly

Current scripted sequence includes:
- toggle take selector open
- select alternate take
- seek
- play
- pause
- collapse take selector

### Fail Condition

Fail when:
- the walkthrough crashes or stalls
- expected interaction states do not appear
- reviewer cannot inspect the changed behavior from the produced artifact

### Required Output

At minimum:
- successful walkthrough run log

Preferred:
- video/gif/screen capture or screenshot sequence attached to the task/PR

### Artifact

Required artifact today:
- reviewer-facing capture from the walkthrough run

Because the wrapper script only launches the walkthrough, artifact capture is currently manual or runner-dependent. Until an automated recorder exists, this remains a manual-proof lane with a required attached capture for behavior-changing UI work.

Recommended command shape:

```powershell
.\.venv\Scripts\python.exe run_timeline_walkthrough.py
```

## 4. Perf Guardrails Lane

### Purpose

Prevent regressions in the two timeline hot paths already called out by repo guardrails:
- cached timeline assembly
- dense event-lane paint

### Current Evidence Sources

- `docs/PERF-GUARDRAILS.md`
- `tests/benchmarks/benchmark_timeline_phase3.py`
- `tests/benchmarks/timeline_phase3_thresholds.json`

### Pass Condition

Pass when:
- benchmark completes
- `pass` is `true`
- no threshold failures are reported
- if `--strict` is used, process exits zero

### Fail Condition

Fail when:
- benchmark exits non-zero under `--strict`
- output JSON contains `pass: false`
- thresholds are exceeded without an explicit, reviewed threshold update

### Required Output

The benchmark emits:
- aggregate JSON to stdout
- threshold values
- measured stats for `assemble_cached`
- measured stats for `event_lane_paint`
- `checks_ms`
- overall `pass`
- `failures`

### Artifact

Required artifact:
- `artifacts/perf/timeline_phase3.json`

Recommended command shape:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:PYTHONPATH='C:\Users\griff\EchoZero'
.\.venv\Scripts\python.exe tests\benchmarks\benchmark_timeline_phase3.py --strict --json-out artifacts\perf\timeline_phase3.json
```

## 5. Future Golden-Record Comparison Lane

### Purpose

Provide future stable acceptance evidence through canonical screenshot or presentation snapshots.

This is intentionally not a prerequisite today.

### Current Status

Future gate only.

The repo already has useful building blocks:
- deterministic fixture presentations
- real-data screenshot capture
- walkthrough flows

What is missing is a stable, reviewed baseline and comparison workflow.

### Near-Term Role

For now, use this lane only as preparation work:
- keep screenshot naming stable
- keep fixture variants deterministic
- avoid adding randomness to capture flows
- preserve output directories that can later become baseline inputs

### Future Pass Condition

Pass will eventually mean:
- generated screenshots/presentation snapshots are compared against approved baselines
- diff is empty or explicitly approved

### Future Artifact

Expected future artifacts:
- approved baseline images or serialized presentation records
- diff images/reports
- acceptance summary showing baseline match

### Rule For Today

Do not block current UI work on golden-record infrastructure.
Do shape new capture outputs so they can become goldens later.

## Lane-to-Change Matrix

### Always Required

- Contract / unit

### Usually Required For Visible Timeline UI Changes

- Real-data smoke
- Walkthrough capture

### Required When Performance Risk Exists

- Perf guardrails

Trigger perf guardrails for changes touching:
- timeline assembly logic
- event-lane paint or culling
- row density/layout affecting visible event count
- zoom/scroll mechanics
- waveform/event rendering strategy

### Future Only

- Golden-record comparison

## Pass/Fail Summary Format

Use this compact format in PRs and task handoff:

```text
UI Verification Sweep
- contract/unit: PASS
- real-data smoke: PASS
- walkthrough capture: PASS
- perf guardrails: PASS
- golden-record: NOT RUN (future gate)

Artifacts
- artifacts/timeline-real-data/timeline_real_default.png
- artifacts/timeline-real-data/timeline_real_scrolled.png
- artifacts/timeline-real-data/timeline_real_zoomed_in.png
- artifacts/timeline-real-data/timeline_real_zoomed_out.png
- artifacts/perf/timeline_phase3.json
- walkthrough capture attachment
```

If a lane is intentionally skipped, state why:
- `not required: no visible UI or perf-path impact`

## Minimal Recommended Cadence

## CI Cadence

On every PR touching timeline UI or timeline application contract:
- run contract/unit lane

On every PR touching visible timeline presentation, fixture wiring, or real-data presentation:
- run real-data smoke lane

On every PR touching timeline hot paths or rendering:
- run perf guardrails lane

Walkthrough capture in CI is optional until automated recording exists.
If CI recording is absent, require manual attachment for behavior-changing UI work.

## Manual Cadence

During active UI iteration:
- run focused contract/unit checks locally before asking for review

Before review on visible behavior changes:
- run real-data smoke
- produce a walkthrough capture

Before merge on perf-sensitive changes:
- run perf guardrails locally or in CI and attach the JSON result

## Acceptance Rule of Thumb

Use the lightest proof that still demonstrates the real risk:
- logic-only contract change: contract/unit may be enough
- visible shell or interaction change: contract/unit plus walkthrough capture
- anything claiming real-data correctness: include real-data smoke
- anything affecting density, paint, assembly, scroll, or zoom: include perf guardrails

## Explicit Non-Goals

This plan does not require:
- broad new UI test expansion for every micro-interaction in `docs/UX-MICRO-TESTS.md`
- golden-record infrastructure before current UI acceptance
- replacing manual reviewer judgment with fixture-only tests

The goal is practical proof:
- contract conformance
- real-data evidence
- reviewer-visible interaction proof
- performance discipline

