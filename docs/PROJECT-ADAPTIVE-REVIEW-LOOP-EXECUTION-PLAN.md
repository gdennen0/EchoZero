# Project Adaptive Review Loop Execution Plan

_Last updated: 2026-04-25_

## Goal

Build the next v1 milestone around one canonical operator loop:

1. Import a Setlist into a Project.
2. Run extraction and classification on a small seed batch with the current
   installed runtime bundles.
3. Review detected Events in EZ or phone review.
4. Manually add missed Events that onset detection did not recover and commit
   them as explicit review.
5. Persist every explicit review commit into one canonical review-signal lane.
6. Keep the resulting review datasets easy to find, inspect, and reuse.
7. Let the operator click one EZ action to create a Project-specialized model
   while Foundry runs in the background.
8. Validate, install, and promote the new runtime bundle.
9. Update pending Songs and Layers that have not yet been processed to use the
   promoted bundle.
10. Continue batch by batch until the operator chooses to run the rest of the
    Setlist.

This milestone is intentionally not just “review signal” and not just
“Foundry training.” It is the full adaptive-review operator loop from Project
truth back into better runtime bundles.

## Scope Note

- This is the active execution doc for the broader adaptive-review milestone.
- `docs/PROJECT-ADAPTIVE-REVIEW-LOOP-SPEC.md` is the product contract for this
  milestone.
- `docs/REVIEW-SIGNAL-EXECUTION-PLAN.md` remains the narrower dependency plan
  for the shared review-signal slice.
- `docs/FOUNDRY-TRAINING.md` remains the current repo-truth training and
  artifact reference for Foundry surfaces.

## Locked Constraints

- Main is truth.
- Takes remain subordinate history and candidate lanes.
- Timeline review writes through the application boundary, never directly from
  widgets.
- Only explicit review commits become training signal.
- A manually added missed Event only becomes reusable signal when the operator
  commits it as review.
- Foundry may run in the background, but EZ owns the operator workflow.
- Review datasets must be durable and discoverable, not hidden temp outputs.
- Shared core-dataset growth must be explicit and lineage-backed. Project data
  must not silently bleed into global training inputs.
- Dataset collection, model building, and runtime bundle promotion are separate
  product lanes. Passing a raw manifest path between them is not a sufficient
  boundary.
- Promoting a new model must update pending work, not silently rewrite already
  reviewed main truth.
- Raw manifest paths are not enough for operator-scale model switching.
  Pending classification settings need logical bundle-selection semantics.

## North Star Shape

This loop should settle into five distinct lanes:

1. Review signal lane:
   one canonical durable record of explicit human review from EZ or phone
   review. This lane owns provenance and decision semantics. It does not train
   models.
2. Project dataset lane:
   versioned, discoverable Project-scoped datasets built from review signals.
   This lane exists so the operator can inspect what one Project has taught us
   so far.
3. Shared core-dataset lane:
   one explicit promotion target that accumulates accepted reviewed data across
   Projects. Promotion into this lane must be operator-visible, deduplicated,
   and lineage-backed.
4. Model build lane:
   train a model from a selected dataset version. This lane must work for both
   a Project dataset version and the shared core-dataset version.
5. Runtime bundle lane:
   validate, install, select, and adopt model bundles for pending work. This
   lane must not mutate datasets.

Simple operator rule:

- review creates signal
- signal builds Project dataset versions
- Project dataset versions can train a Project-specialized model
- Project dataset versions can also promote into the shared core dataset
- shared core-dataset versions can train the next general model

## Refactor Direction

The current code already contains most of the pieces, but the boundary lines
are still too soft for this to become the core application loop.

- Keep `ReviewSignalService` as the only durable write boundary for explicit
  review.
- Split `DatasetService` responsibilities. It currently owns folder ingest,
  review-signal materialization, curation/version mutation, binary dataset
  derivation, and integrity validation. Those are separate concerns and should
  not keep growing inside one service.
- Add one explicit dataset-promotion boundary for moving reviewed Project data
  into the shared core dataset with preserved source lineage.
- Keep `ProjectSpecializedModelService` orchestration-only. It should resolve a
  dataset version, launch a build, validate/install the resulting bundle, and
  stop there.
- Keep query and discovery surfaces behind one Foundry query boundary so EZ can
  ask for latest Project dataset, shared core-dataset summary, lineage, and
  promotion history without reaching into JSON state directly.
- Keep collection task-agnostic even if the first model recipe stays drum-first.
  The current specialized-model path can remain kick/snare initially, but data
  collection and dataset promotion must stay general enough to learn from every
  Project we process.

## V1 Operator Contract

The operator should be able to do all of this without leaving the real EZ app
path:

- import a full Setlist
- open a questionable or all-events review batch
- verify, reject, relabel, retime, and add missed Events
- open the current Project review dataset and inspect what has been collected
- click `Create Specialized Model`
- watch background status for train, validate, install, and promote
- click `Use Promoted Model For Pending Songs`
- continue with the next batch

Foundry remains the background training engine and artifact lane, but not the
required primary operator surface for this milestone.

## Current Baseline

Implemented now:

- Setlist import, ordered per-song pipeline execution, and LTC stripping are in
  place.
- Project-backed phone review batches already exist from the real EZ shell.
- Durable review signals already exist in Foundry.
- Project-backed review can already write corrected truth back into main-take
  data when provenance is sufficient.
- Reviewed records already materialize into dataset-ready samples.
- Runtime bundle install and global installed-bundle resolution already exist.

Partially implemented now:

- Timeline fix mode has edit tools for remove/select/promote, but those tools
  still dispatch generic edit mutations instead of explicit review commits.
- Foundry desktop run and artifact surfaces exist, but there is not yet one
  EZ-owned operator flow from reviewed data to promoted bundle.
- Review datasets exist on disk, but there is not yet one clear EZ-first
  surface for browsing, reopening, and promoting them.

Missing now:

- canonical timeline review-commit intents for verify, reject, relabel,
  boundary correction, and missed-event-added
- a shared review-signal write API that does not require a review session item
- an explicit shared core-dataset promotion flow from Project review datasets
- a clean dataset catalog/query surface for both Project and shared datasets
- logical model selection semantics for pending Songs and Layers
- one-button Project-specialized model creation from EZ
- one-click promoted-model adoption for pending work
- a generalized model-recipe boundary beyond the current drum-specific
  specialized flow
- a reviewed-batch progression model for first `N`, next `N`, then rest

Current implementation risk:

- Foundry CLI reachability must be proved on the current branch before more EZ
  wrapping lands. A previous early-return regression existed here, and the
  current worktree already includes active CLI edits, so command reachability
  should remain a proof requirement instead of an assumed truth.

## Execution Order

### Slice 0 — Unblock Train And Promote Plumbing

Goal:
- restore the current Foundry CLI and background-training entrypoints before EZ
  wraps them

Scope:
- remove the early return that makes downstream CLI branches unreachable
- prove `train-folder`, `create-run`, `start-run`, `install-runtime-bundle`,
  `validate-artifact`, and `ui` are reachable again
- keep this slice narrow and regression-focused

Likely files:
- `echozero/foundry/cli.py`
- focused tests under `tests/foundry/`

Done when:
- CLI and EZ wrappers can reach the existing train, validate, and install
  surfaces again
- the current Foundry training doc stops lying about unreachable commands

Proof:
- focused CLI tests for command reachability
- at least one direct smoke for `ui` and `install-runtime-bundle`

### Slice 1 — Canonical Review Commit Boundary

Goal:
- make explicit review commit a first-class service used by both phone review
  and timeline review

Scope:
- extract or add one review-commit API that is not session-item only
- accept explicit decision payloads for verified, rejected, relabeled,
  boundary-corrected, and missed-event-added
- preserve current writeback and dataset-materialization behavior
- keep review provenance explicit and durable

Likely files:
- `echozero/foundry/services/review_signal_service.py`
- `echozero/foundry/services/review_session_service.py`
- `echozero/foundry/domain/review.py`
- `echozero/foundry/services/review_writeback_service.py`
- focused tests under `tests/foundry/`

Done when:
- phone review becomes one producer of the shared write path, not the only one
- timeline review can write the same signal shape without faking a phone queue
- existing project writeback and dataset materialization stay intact

Proof:
- `./.venv/bin/python -m pytest tests/foundry/test_review_signals.py -q`
- additional focused Foundry review-signal tests

### Slice 2 — Timeline Review Commit Actions

Goal:
- let EZ fix mode commit review, not just mutate Events

Scope:
- add typed timeline intents or commands for explicit review commit
- support verify, reject, relabel, boundary correction, and missed-event-added
- preserve undo and redo semantics
- keep widgets as input surfaces only

Likely files:
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/orchestrator_event_edit_mixin.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/widget_controls.py`
- `echozero/ui/qt/timeline/widget_canvas_interaction_mixin.py`
- focused tests under `tests/application/` and `tests/ui/`

Done when:
- deleting a false positive can emit `rejected`
- relabeling can emit `relabeled`
- retiming or resizing can emit `boundary_corrected`
- manual fix-mode promotion of a missed onset can emit `missed_event_added`
- timeline review commits write through the canonical app path

Proof:
- targeted timeline orchestrator tests
- targeted app-shell undo and redo tests
- `./.venv/bin/python -m echozero.testing.run --lane appflow`

### Slice 3 — Missed-Event Review From Manual Add

Goal:
- explicitly cover the operator case where onset detection missed the Event and
  the user adds it manually in EZ

Scope:
- preserve both the canonical created Event and the review provenance that
  explains why it exists
- make manual add from fix mode produce a review commit rather than just a raw
  `CreateEvent`
- capture corrected timing windows for the created Event so downstream dataset
  materialization can use them safely

Likely files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/timeline/orchestrator_event_edit_mixin.py`
- Foundry review-signal and dataset service files from Slices 1 and 2
- focused tests under `tests/application/`, `tests/ui/`, and `tests/foundry/`

Done when:
- a manually added missed Event is visible as main truth
- the same action also produces a durable `missed_event_added` review signal
- the resulting positive sample is materializable for training

Proof:
- targeted end-to-end missed-event tests from fix mode into dataset materialization
- `./.venv/bin/python -m echozero.testing.run --lane appflow`

### Slice 4 — Review Dataset Accessibility

Goal:
- keep extracted review datasets easy to find and use from EZ

Scope:
- add one Project-scoped dataset index or launcher surface from EZ
- expose the latest review dataset version, sample counts, class balance, and
  open-folder/open-manifest actions
- make the current Project review dataset reopenable after restart
- keep Foundry as the source of truth for stored dataset versions

Likely files:
- `echozero/foundry/app.py`
- `echozero/foundry/ui/*` if query surfaces need extension
- `echozero/ui/qt/app_shell.py`
- new EZ-side review-dataset bridge or mixin under `echozero/ui/qt/`
- focused tests under `tests/ui/` and `tests/foundry/`

Done when:
- the operator can find the current Project review dataset from EZ without
  spelunking through the filesystem
- the latest dataset version and output location are visible
- datasets survive restart and remain easy to reopen

Proof:
- targeted EZ shell tests for dataset launch and summary surfaces
- focused Foundry query tests if new dataset-summary APIs are added

### Slice 4B — Shared Core-Dataset Promotion Boundary

Goal:
- make Project-reviewed data explicitly promotable into one reusable shared
  core dataset

Scope:
- add one shared core-dataset lane in Foundry rather than treating
  cross-Project reuse as ad hoc folder copying
- allow promotion from a selected Project review dataset version into that
  shared dataset lane
- preserve lineage back to source Project, dataset version, and review signal
- deduplicate by stable review-signal identity and content hash where possible
- keep promotion explicit and operator-visible; do not auto-merge every
  reviewed sample globally

Likely files:
- extracted dataset-promotion services under `echozero/foundry/services/`
- `echozero/foundry/app.py`
- `echozero/foundry/services/query_service.py`
- EZ-side Project review dataset surfaces under `echozero/ui/qt/`
- focused tests under `tests/foundry/` and `tests/ui/`

Done when:
- the operator can promote reviewed Project data into a shared core dataset in
  one supported flow
- the resulting shared dataset versions are inspectable and lineage-backed
- rerunning promotion does not silently duplicate already promoted samples

Proof:
- focused dataset-promotion and dedupe tests
- focused query tests for shared core-dataset summaries and lineage

### Slice 5 — Logical Model Selection Policy

Goal:
- replace raw model-path dependence with logical bundle-selection semantics

Scope:
- add logical selectors such as current installed bundle or Project-specialized
  bundle
- keep backward compatibility with existing `model_path` and
  `classify_model_path` settings
- ensure pending classification work resolves model location at run time rather
  than baking absolute manifest paths everywhere

Likely files:
- `echozero/application/timeline/object_action_settings_runtime_mixin.py`
- `echozero/application/timeline/object_action_model_picker_options.py`
- `echozero/application/timeline/object_actions/descriptors.py`
- `echozero/models/runtime_bundle_selection.py`
- `echozero/models/runtime_bundle_index.py`
- focused tests under `tests/ui/` and `tests/application/`

Done when:
- pending Songs and Layers can follow a promoted bundle by policy
- existing saved raw paths still resolve for backward compatibility
- changing the active installed bundle does not require hand-editing every Layer

Proof:
- targeted settings and pipeline-resolution tests
- focused compatibility coverage for older saved model-path settings

### Slice 6 — One-Button Specialized Model Creation In EZ

Goal:
- give the operator one EZ action to create a Project-specialized model

Scope:
- add one EZ-owned command such as `Create Specialized Model`
- derive the default dataset scope from a selected Project dataset version, not
  from raw signal rows
- launch Foundry training in the background with a sane default spec
- surface background run state, validation status, and install result in EZ
- keep the first UX intentionally simple; advanced Foundry knobs remain
  available later

Likely files:
- `echozero/ui/qt/app_shell.py`
- new EZ-side adaptive-model action or bridge under `echozero/ui/qt/`
- `echozero/foundry/app.py`
- `echozero/foundry/services/runtime_bundle_install_service.py`
- `echozero/foundry/ui/*` only if shared background-run helpers are reused
- focused tests under `tests/ui/` and `tests/foundry/`

Done when:
- the operator can start a Project-specialized model build from EZ
- the build runs in the background and reports progress back to EZ
- a successful build validates and installs the runtime bundle without requiring
  manual Foundry-only steps

Proof:
- targeted background-run and install tests
- one EZ shell smoke that covers button -> background run -> validation -> install

### Slice 7 — Promote And Adopt For Pending Work

Goal:
- make the newly promoted model immediately useful for the rest of the loop

Scope:
- add one explicit EZ action such as `Use Promoted Model For Pending Songs`
- update pending extraction or classification settings for Songs and Layers that
  have not yet been processed
- support scoped adoption:
  - active Song
  - selected Songs
  - unprocessed remainder of the Setlist
- do not silently rewrite already reviewed main truth

Likely files:
- `echozero/ui/qt/app_shell.py`
- `echozero/application/timeline/object_action_settings_runtime_mixin.py`
- any Project-scoped policy state added in Slice 5
- focused tests under `tests/ui/` and `tests/application/`

Done when:
- the promoted model can become the default for pending work in one step
- previously reviewed Songs stay stable unless the operator explicitly reruns
  them
- the operator does not need to hand-edit model settings on every remaining
  Layer

Proof:
- targeted policy-adoption tests
- one shell-path smoke for “promote -> adopt -> next batch uses new bundle”

### Slice 8 — End-To-End Adaptive Review Proof

Goal:
- prove the real human-path operator loop from Project import through promoted
  bundle reuse

Scope:
- import a real Setlist
- run the first batch
- review detections including at least one missed Event added manually
- materialize review dataset outputs
- create a Project-specialized model from EZ
- validate, install, and adopt it for pending work
- run the next batch with the promoted model

Done when:
- the loop works without hidden manual sidecar steps
- the proof is through the real app path, not helper-only shortcuts
- the operator can explain what was reviewed, what dataset was produced, what
  model was promoted, and which pending Songs now use it

Proof:
- focused app and UI tests
- `./.venv/bin/python -m echozero.testing.run --lane appflow`
- milestone manual walkthrough through the real EZ shell

## What Completes This Milestone

This milestone is complete when all of this is true:

- the operator can manually add missed Events in EZ and that action becomes
  explicit reusable review signal
- review datasets are durable and easy to access from EZ
- reviewed Project data can be explicitly promoted into a shared core dataset
  with preserved lineage
- EZ exposes one simple specialized-model creation action
- the resulting runtime bundle can be validated, installed, and promoted
- pending Songs and Layers can adopt the promoted model without hand-editing
  every Layer
- the full loop is proven through the real app path

## Not In This Done Bar

Not required for this milestone:

- continuous online self-training
- auto-approval from untouched Events
- fully general project-specific scheduling heuristics
- automatic general-model retraining every time new Project data is promoted
- silent global runtime-bundle switching with no operator confirmation
- advanced Foundry hyperparameter editing inside the first EZ one-button UX

## Immediate Next Slice

If implementation starts now, the best opening slice is:

1. prove Foundry CLI reachability on the current branch
2. add the shared explicit review-commit API
3. wire fix-mode missed-event promotion away from raw `CreateEvent` and into
   `missed_event_added`
4. extract the shared core-dataset promotion boundary before expanding the EZ
   model-build UX

Why this first:

- CLI reachability is still a hard dependency for the train or promote lane
- the shared write API removes the biggest architectural mismatch between phone
  review and timeline review
- missed-event capture is now a direct milestone requirement, not a later nice
  to have
- the shared core-dataset boundary is the cleanest place to keep long-term
  dataset growth simple before more UI layers depend on it
