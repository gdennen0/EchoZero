# Phase 3: Real-World DAW Proof Pack

_Date: 2026-04-10_

## Purpose

Phase 3 proves workflow fitness against one canonical LD scenario, not just unit or contract correctness.

This runbook is the reproducible signoff pack for:
- reviewer-visible workflow proof
- regression reruns after behavior changes
- documenting where EchoZero intentionally does not mimic a full DAW

Canonical planning authority remains `docs/UNIFIED-IMPLEMENTATION-PLAN.md`.

## Canonical Scenario Pack

### Scenario name

`LD-01: song ingest -> extract all -> review/edit -> divergence decision -> MA3 sync decision`

### Workflow intent

This scenario reflects the real operating path for one song version in an LD programming pass:
1. add or open the target song/version
2. run Extract All on the current source
3. review generated timeline content and inspector state
4. make one deliberate edit pass on cues/takes
5. force one divergence situation that requires an explicit decision
6. complete one MA3 push or pull decision from the main take boundary

### Preconditions

- Use one stable real-data song fixture for reruns. Do not swap audio or MA3 fixture mid-proof.
- Start from a clean EchoZero application state for the selected project/session.
- The song version used for proof must already have:
  - resolvable source media
  - deterministic enough fixture inputs to reproduce screenshots
  - MA3 connectivity path or replay fixture available for the sync decision step
- Record the fixture identifiers used for the run in the evidence folder README or capture notes.

## Required Execution Order

Run the scenario in this exact order. Do not reorder steps if the run is intended to count as Phase 3 evidence.

1. Launch EchoZero and open the project containing the proof song/version.
2. Capture a pre-run screenshot showing the initial timeline/editor state.
3. Add the target song/version if it does not already exist in the project.
4. Enter the song/version and confirm the expected source is attached.
5. Run `Extract All`.
6. Wait for extraction completion and capture a screenshot of the resulting populated timeline.
7. Perform a review pass:
   - inspect generated layers/events
   - click through inspector-visible selections
   - confirm the shell remains coherent under real data
8. Perform one intentional edit pass:
   - adjust at least one cue/event decision
   - if takes are involved, confirm the main-take truth model still drives parent-row truth
9. Create or expose one divergence case that requires user judgment.
10. Capture a screenshot of the divergence state before resolution.
11. Resolve the divergence through the intended EchoZero path.
12. Trigger one MA3 sync decision:
   - push if EchoZero is authoritative for the current state
   - pull/reconcile if the external state is the authority for the test case
13. Capture a screenshot of the post-decision state.
14. Record one continuous video covering at minimum steps 5 through 13.
15. Save all artifacts into the run evidence folder and mark the run pass/fail.

## Required Artifacts

Each counted run must produce all artifacts below.

### Screenshots

- `01-initial-state.png`
- `02-post-extract-all.png`
- `03-divergence-visible.png`
- `04-post-resolution-or-sync.png`

Screenshots must clearly show the relevant timeline/editor state, not just the desktop shell.

### Video

- `phase3-ld-01-walkthrough.mp4`

Video requirements:
- one continuous capture
- covers extraction, review/edit activity, divergence visibility, and sync decision
- no cuts that hide a state transition relevant to pass/fail

### Run metadata

- fixture identifiers used
- date of execution
- operator
- app revision / commit under test
- pass/fail result
- concise notes for any observed gap

## Pass / Fail Criteria

The scenario passes only if every item below is true.

### Pass criteria

- The run followed the required execution order without skipping a mandatory step.
- `Extract All` completed and produced reviewer-visible timeline content.
- The review/edit pass completed without losing shell coherence or inspector truthfulness.
- Any take-related edit preserved the main-take truth model at the parent-row level.
- A divergence state became reviewer-visible before it was resolved.
- The divergence was resolved through an explicit EchoZero path, not by hidden state reset.
- The MA3 decision path operated only from the main-take boundary semantics already established in Phase 1.
- All required screenshots and the continuous video were captured and retained with run metadata.

### Automatic fail conditions

- Missing any required artifact
- Reordered or skipped mandatory workflow steps
- Sync decision taken from non-main truth
- Inspector/timeline state contradicts the underlying workflow state
- Divergence resolved in a way the reviewer cannot observe
- Run depends on ad-hoc manual cleanup not documented in metadata

### Soft-fail / investigate conditions

These do not automatically invalidate the runbook, but the run should not count as clean signoff until explained:
- visual instability that does not change semantic truth
- fixture-specific timing noise that still preserves workflow outcome
- non-blocking operational friction during capture

## Evidence Packaging

Store each run in a dated evidence folder outside tracked source. Minimum structure:

```text
phase3-daw-proof/
  2026-04-10-ld-01/
    01-initial-state.png
    02-post-extract-all.png
    03-divergence-visible.png
    04-post-resolution-or-sync.png
    phase3-ld-01-walkthrough.mp4
    run-notes.md
```

`run-notes.md` should contain the run metadata and a one-paragraph outcome summary.

## Intentional Differences From DAW Precedent

EchoZero uses DAW precedent as a behavior reference, not as an obligation to become a full DAW.

Documented intentional differences for this proof pack:

1. EchoZero is a cue/programming editor around a song source, not a general-purpose multitrack arranger.
2. Main-take truth remains authoritative for sync and parent-row semantics even when alternate takes are visible for review.
3. Video evidence is required as reviewer proof because some workflow confidence is interaction-based, but video is supporting evidence rather than the sole source of truth.
4. Divergence handling is explicit and reviewer-visible; EchoZero should not silently auto-resolve state in places where the operator needs to make a show-control decision.
5. The proof pack values one stable canonical scenario over broad DAW-style feature parity. Reproducibility beats surface-area breadth at this phase.

## Phase 3 Exit Usage

This runbook is considered active when:
- the tracker marks Phase 3 in progress
- new workflow-affecting changes reference this runbook for signoff
- at least one clean rerunnable `LD-01` evidence pack exists for the current behavior baseline

Phase 3 is complete when this runbook is being used as the regression signoff path for real-world workflow proof.
