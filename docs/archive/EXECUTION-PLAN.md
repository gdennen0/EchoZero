# Repo Cleanup Execution Plan

Status: historical
Last reviewed: 2026-04-30

This document is archived historical context.
For current implementation truth, use `docs/STATUS.md`.



This plan turns the current cleanup issues `1-12` into executable work.
Use [STATUS.md](../STATUS.md) for current repo truth.
Use [LLM-CLEANUP-BOARD.md](../LLM-CLEANUP-BOARD.md) for campaign posture.
Use [BACKLOG-CLEARANCE-PLAN.md](../BACKLOG-CLEARANCE-PLAN.md) for the current
ordered backlog path after this remediation plan completed.
Use this file for the completed dependency order, task sequencing, and
acceptance record for issues `1-12`.

## Goal

Complete the current cleanup pass so the canonical EchoZero app path is:

- smaller and easier to reason about
- more strongly typed on the main runtime surfaces
- clearer about canonical versus support-only boundaries
- easier to verify through the existing proof lanes

## Hard Rules

- `run_echozero.py` remains the canonical desktop entrypoint.
- Main is truth. Takes remain subordinate history/candidate surfaces.
- Widget code must not invent alternate app truth.
- Support-only surfaces stay explicitly non-canonical.
- Each phase must land with proof through the smallest relevant canonical lane.
- Do not tighten a guardrail before the corresponding files are clean enough to pass it.

## Issue Map

1. `echozero/application/timeline/orchestrator.py`
2. `echozero/ui/qt/timeline/widget.py`
3. `echozero/foundry/ui/main_window.py`
4. `echozero/application/timeline/object_action_settings_service.py`
5. `echozero/ui/qt/app_shell.py`
6. `echozero/ui/qt/timeline/widget_actions.py`
7. `echozero/application/timeline/assembler.py`
8. `echozero/application/presentation/inspector_contract_support.py`
9. remaining large-module boundary headers
10. support-only monoliths like `demo_app.py` and `gui_lane_b.py`
11. oversized test modules
12. explicit export/import boundaries

## Execution Order

1. Phase A
2. Phase B
3. Phase C
4. Phase D
5. Phase E
6. Phase F

## Phase A
Boundary Baseline

### RCP-001
Normalize Large-Module Boundary Headers

Status: complete (2026-04-21)

- Covers: `9`
- Depends on: none
- Files:
  `echozero/audio/engine.py`
  `echozero/foundry/persistence/repositories.py`
  `echozero/foundry/services/baseline_trainer.py`
  `echozero/foundry/services/cnn_trainer.py`
  `echozero/foundry/ui/main_window.py`
  `echozero/infrastructure/sync/ma3_osc.py`
  `echozero/models/provider.py`
  `echozero/persistence/entities.py`
  `echozero/persistence/session.py`
  `echozero/processors/separate_audio.py`
  `echozero/services/orchestrator.py`
  `echozero/testing/demo_suite_scenarios.py`
  `echozero/testing/ma3/simulator.py`
- Steps:
  1. Add the standard top-of-file docstring shape.
  2. Use `Exists to` and `Connects` on canonical surfaces.
  3. Use `Never` wording on support-only surfaces where appropriate.
- Proof:
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  every large boundary-critical module passes the header rule.
- Completed notes:
  boundary headers are now normalized and enforced for the remaining large audio,
  Foundry, persistence, model, sync, processor, and support-only modules in this slice.

### RCP-002
Make Exports and Imports Explicit

Status: complete (2026-04-21)

- Covers: `12`
- Depends on: `RCP-001`
- Files:
  `echozero/application/presentation/inspector_contract*.py`
  `echozero/ui/qt/timeline/runtime_audio.py`
  `echozero/ui/qt/timeline/widget.py`
  `echozero/ui/qt/timeline/widget_actions.py`
  `echozero/ui/qt/app_shell.py`
- Steps:
  1. Add explicit re-exports or direct imports for split modules.
  2. Remove ambiguous broad-module imports where splits already happened.
  3. Resolve `attr-defined` errors by making public surfaces intentional.
- Proof:
  `./.venv/bin/python -m mypy echozero/application/presentation/inspector_contract.py echozero/ui/qt/timeline/widget.py echozero/ui/qt/timeline/widget_actions.py echozero/ui/qt/app_shell.py --follow-imports=silent`
- Done when:
  split modules are imported from stable, explicit boundaries.
- Completed notes:
  explicit public exports now exist for `inspector_contract.py` and
  `ui/qt/timeline/runtime_audio.py`, and the canonical proof lane now passes
  for `inspector_contract.py`, `widget.py`, `widget_actions.py`, and
  `app_shell.py`. The remaining timeline-shell type cleanup moved out of this
  phase and into the deeper structural refactor work in Phase B and beyond.

## Phase B
Canonical Runtime Core

### RCP-101
Finish `app_shell.py` Decomposition

Status: complete (2026-04-21)

- Covers: `5`
- Depends on: `RCP-002`
- Files:
  `echozero/ui/qt/app_shell.py`
  new helper modules under `echozero/ui/qt/`
- Steps:
  1. Extract object-action session helpers.
  2. Extract pipeline-run lifecycle helpers.
  3. Keep `app_shell.py` focused on runtime orchestration only.
  4. Tighten return annotations and remove untyped `presentation()` leaks.
- Proof:
  `./.venv/bin/python -m pytest tests/ui/test_app_shell_runtime_flow.py tests/ui/test_app_shell_undo_redo.py tests/ui/test_app_shell_timeline_state.py tests/ui/test_app_shell_layer_storage.py -q`
  `./.venv/bin/python -m echozero.testing.run --lane appflow`
- Done when:
  `app_shell.py` reads as an orchestration root instead of a mixed conversion/storage file.
- Current notes:
  object-action session handling and pipeline-run refresh handling now live in
  `echozero/ui/qt/app_shell_object_actions.py`, with the canonical app-shell
  proof slice and mypy lane passing after the extraction. Undo/history
  classification and snapshot restore now live in
  `echozero/ui/qt/app_shell_history.py`, and both the focused app-shell proof
  slice and `python -m echozero.testing.run --lane appflow` pass after that
  split. Storage-backed timeline reconciliation now lives in
  `echozero/ui/qt/app_shell_storage_sync.py`, and project reload plus
  song/version lifecycle flows now live in
  `echozero/ui/qt/app_shell_project_lifecycle.py`; the focused proof slice,
  targeted mypy lane, and `python -m echozero.testing.run --lane appflow` all
  pass after those extractions. Runtime audio sync, preview lookup, shell
  shutdown, source-layer reselection, and object-action service wiring now live
  in `echozero/ui/qt/app_shell_runtime_support.py`, with the same proof lanes
  still green after restoring the `resolve_installed_binary_drum_bundles`
  module-level compatibility seam used by focused runtime tests. The remaining
  public facade methods now live in
  `echozero/ui/qt/app_shell_editing_mixin.py`,
  `echozero/ui/qt/app_shell_project_mixin.py`, and
  `echozero/ui/qt/app_shell_object_action_mixin.py`, which brings
  `app_shell.py` down to a small orchestration root under the repo's module-size
  target while keeping the focused proof slice, targeted mypy lane, and
  `python -m echozero.testing.run --lane appflow` green.

### RCP-102
Split `object_action_settings_service.py`

Status: complete (2026-04-21)

- Covers: `4`
- Depends on: `RCP-101`
- Files:
  `echozero/application/timeline/object_action_settings_service.py`
  new helper modules under `echozero/application/timeline/`
- Steps:
  1. Split scoped-config loading and persistence helpers.
  2. Split copy/plan preview behavior.
  3. Split runtime binding and model-path resolution helpers.
  4. Fix `Ok | Err` narrowing and `str | None` template-id flow.
- Proof:
  `./.venv/bin/python -m mypy echozero/application/timeline/object_action_settings_service.py --follow-imports=silent`
  targeted pytest for pipeline config and action settings slices
- Done when:
  the service has clear internal ownership slices and the current mypy failures are gone.
- Current notes:
  scoped config load/store, default hydration, and config persistence now live in
  `echozero/application/timeline/object_action_scoped_config.py`. The first
  `RCP-102` pass also fixed the file's `Ok | Err` narrowing and
  `str | None` template-id debt on the scoped-config path, bringing the
  dedicated mypy lane green for
  `object_action_settings_service.py` plus the new helper module. Focused proof
  also passes for `tests/test_pipeline_config.py` and the app-shell runtime
  settings slice covering saved settings, song-default scope, copy flows, and
  extract-classified-drums settings. Session copy/preview behavior now lives in
  `echozero/application/timeline/object_action_settings_copy_mixin.py`, and
  runtime binding plus model-path helpers now live in
  `echozero/application/timeline/object_action_settings_runtime_mixin.py`.
  `object_action_settings_service.py` now reads as the orchestration root across
  those helper slices, dropped from 1255 lines to 811, and keeps the
  compatibility exports used by focused runtime tests explicit via
  `__all__`. The dedicated mypy lane remains green across the service plus all
  three helper modules, and the focused pytest slice remains green for pipeline
  config plus app-shell settings coverage.

### RCP-103
Clean `assembler.py`

Status: complete (2026-04-21)

- Covers: `7`
- Depends on: `RCP-102`
- Files:
  `echozero/application/timeline/assembler.py`
- Steps:
  1. Add missing generic parameters.
  2. Add missing helper parameter and return annotations.
  3. Remove `Any`-style return leakage from presentation assembly.
- Proof:
  `./.venv/bin/python -m mypy echozero/application/timeline/assembler.py --follow-imports=silent`
  `./.venv/bin/python -m pytest tests/application/test_timeline_assembler_contract.py -q`
- Done when:
  assembler types are self-describing enough to serve as a canonical presentation boundary.
- Current notes:
  `assembler.py` now has explicit ID, event-ref, diff-preview, and batch-plan-row
  annotations across the cached signature, layer/take/event assembly helpers,
  sync-diff adapters, and transfer-plan lookup helpers. That removes the
  remaining generic `tuple`/`list`/`set` leakage and the implicit `Any` return
  paths on the presentation boundary, bringing the dedicated mypy lane green
  without changing assembler behavior. The canonical assembler contract pytest
  slice also remains green after the type cleanup.

## Phase C
Timeline Truth Mutation

### RCP-201
Split `orchestrator.py` by Concern

Status: complete (2026-04-21)

- Covers: `1`
- Depends on: `RCP-103`
- Files:
  `echozero/application/timeline/orchestrator.py`
  new helper modules under `echozero/application/timeline/`
- Steps:
  1. Separate selection/editing flows from transfer-plan flows.
  2. Separate preset/live-sync helpers from core intent handling.
  3. Extract manual-pull/manual-push lookup and conversion helpers.
  4. Fix the current type debt while each slice is isolated.
- Proof:
  `./.venv/bin/python -m pytest tests/application/test_timeline_orchestrator_take_actions.py tests/application/test_manual_transfer_push_flow.py tests/application/test_manual_transfer_pull_flow.py tests/application/test_transfer_plan_batch_apply.py -q`
  `./.venv/bin/python -m mypy echozero/application/timeline/orchestrator.py --follow-imports=silent`
- Done when:
  `TimelineOrchestrator.handle()` is the public entrypoint over smaller, typed internal slices.
- Current notes:
  first `RCP-201` type cleanup landed on the manual-pull preview/apply path and
  its shared helpers, which tightened the manual-pull event/target-layer flow
  and updated `echozero/application/sync/diff_service.py` to accept a covariant
  source-event sequence for pull previews. The next pass extracted the full
  selection and event-editing surface into
  `echozero/application/timeline/orchestrator_selection_mixin.py`, moving layer
  selection, take/event selection, event mutation, event-ref resolution, and
  selected-record grouping out of the main orchestrator file. After that split,
  `echozero/application/timeline/orchestrator.py` dropped from 2574 lines to
  1791, and the dedicated mypy lane now passes cleanly for
  `orchestrator.py` plus the new mixin module. The canonical transfer proof
  slice also remains green. The next pass extracted transfer preset persistence
  and live-sync guardrail resets into
  `echozero/application/timeline/orchestrator_sync_preset_mixin.py`, leaving
  `orchestrator.py` as the orchestration root over both mixins and dropping the
  file again from 1791 lines to 1651 while keeping the dedicated mypy lane and
  canonical transfer proof slice green. The next pass extracted manual
  push/pull provider option loading, raw-option normalization, and track/target
  lookup helpers into
  `echozero/application/timeline/orchestrator_transfer_lookup_mixin.py`, which
  dropped `orchestrator.py` again from 1651 lines to 1485 while keeping the
  same mypy lane and transfer proof slice green. The next pass extracted manual
  pull target resolution, imported-layer creation, and imported take/main-take
  helpers into
  `echozero/application/timeline/orchestrator_manual_pull_import_mixin.py`,
  which brought `orchestrator.py` down again from 1485 lines to 1346 with the
  dedicated mypy lane still green across all orchestrator mixins and the
  canonical transfer proof slice still passing. The final pass extracted
  transfer-plan preview/apply execution, plan row rebuilding, plan counter
  helpers, and transfer-flow reset helpers into
  `echozero/application/timeline/orchestrator_transfer_plan_mixin.py`, which
  brought `orchestrator.py` down again from 1346 lines to 875 while keeping the
  dedicated mypy lane green across all orchestrator mixins and the canonical
  transfer proof slice still passing. `TimelineOrchestrator.handle()` now reads
  as the public entrypoint over smaller, typed concern slices, so this item is
  complete.

## Phase D
Presentation and Qt Timeline Surface

### RCP-301
Finish `inspector_contract_support.py` Split

Status: complete (2026-04-21)

- Covers: `8`
- Depends on: `RCP-201`
- Files:
  `echozero/application/presentation/inspector_contract_support.py`
  new helper modules under `echozero/application/presentation/`
- Steps:
  1. Extract lookup helpers into one module.
  2. Extract preview/source-resolution helpers into one module.
  3. Extract context action builders into one module.
  4. Keep `inspector_contract.py` as the public builder surface.
- Proof:
  `./.venv/bin/python -m pytest tests/application/test_inspector_contract.py tests/ui/test_timeline_style.py -q`
  `./.venv/bin/python -m mypy echozero/application/presentation/inspector_contract.py echozero/application/presentation/inspector_contract_support.py --follow-imports=silent`
- Done when:
  inspector contract assembly, types, and helpers live in separate obvious files.
- Current notes:
  the first `RCP-301` pass extracted layer/take/event lookup helpers into
  `echozero/application/presentation/inspector_contract_lookup.py` and rewired
  both `inspector_contract.py` and `inspector_contract_support.py` to import
  that dedicated lookup surface directly. That dropped
  `inspector_contract_support.py` from 889 lines to 828 while keeping the
  focused inspector mypy lane green for `inspector_contract.py`,
  `inspector_contract_support.py`, and the new lookup module. The canonical
  inspector/style pytest slice also remains green. The next pass extracted
  preview/source-resolution helpers into
  `echozero/application/presentation/inspector_contract_preview.py`, dropping
  `inspector_contract_support.py` again from 828 lines to 719 while keeping the
  same focused mypy lane and inspector/style pytest slice green. The final pass
  extracted context-action builders into
  `echozero/application/presentation/inspector_contract_context_actions.py` and
  rewired `inspector_contract.py` to import the public action/formatting surface
  directly from the new helper modules. That brought
  `inspector_contract_support.py` down again from 719 lines to 153 while
  keeping the focused mypy lane green across `inspector_contract.py`,
  `inspector_contract_support.py`, `inspector_contract_lookup.py`,
  `inspector_contract_preview.py`, and
  `inspector_contract_context_actions.py`. The canonical inspector/style pytest
  slice also remains green, so this item is complete.

### RCP-302
Split `widget_actions.py`

Status: complete (2026-04-21)

- Covers: `6`
- Depends on: `RCP-301`
- Files:
  `echozero/ui/qt/timeline/widget_actions.py`
  new helper modules under `echozero/ui/qt/timeline/`
- Steps:
  1. Separate contract-action routing from dialog orchestration.
  2. Separate transfer helpers from general widget action handling.
  3. Replace broad runtime-shell `object` usage with a small typed protocol.
  4. Fix typed `LayerId`/`TakeId`/`EventId` conversions at entrypoints.
- Proof:
  `./.venv/bin/python -m pytest tests/ui/test_timeline_shell.py -q`
  `./.venv/bin/python -m mypy echozero/ui/qt/timeline/widget_actions.py --follow-imports=silent`
- Done when:
  widget action routing is typed and clearly separated from UI dialogs.
- Current notes:
  the first `RCP-302` pass extracted general contract-action routing, song and
  version actions, live-sync actions, event-preview routing, and the shared
  `LayerId`/`TakeId`/`EventId` coercion helpers into
  `echozero/ui/qt/timeline/widget_action_contract_mixin.py`. That dropped
  `widget_actions.py` from 1220 lines to 780 while keeping the focused widget
  mypy lane green for `widget_actions.py` plus the new mixin, and the canonical
  `tests/ui/test_timeline_shell.py` slice remained green. The next pass
  extracted transfer workspace routing, transfer preset actions, transfer plan
  actions, manual pull timeline popup helpers, and transfer summary/label
  helpers into `echozero/ui/qt/timeline/widget_action_transfer_mixin.py`,
  bringing `widget_actions.py` down again from 780 lines to 199 while keeping
  the same focused mypy lane and timeline-shell proof slice green. The final
  pass replaced the remaining broad runtime-shell `getattr(...)` edge in
  `widget_actions.py` with typed capability protocols for settings-session and
  object-action runtime methods, and kept the focused mypy lane green across
  `widget_actions.py`, `widget_action_contract_mixin.py`, and
  `widget_action_transfer_mixin.py`. The canonical timeline-shell pytest slice
  also remains green, so this item is complete.

### RCP-303
Split `widget.py`

Status: complete (2026-04-21)

- Covers: `2`
- Depends on: `RCP-302`
- Files:
  `echozero/ui/qt/timeline/widget.py`
  new helper modules under `echozero/ui/qt/timeline/`
- Steps:
  1. Separate canvas/rendering from shell/container wiring.
  2. Separate hit-target/context-menu/inspector handling.
  3. Separate transport/runtime-audio/follow-scroll concerns.
  4. Reduce Qt nullable handling and `object` payload coercion at the edges.
- Proof:
  `./.venv/bin/python -m pytest tests/ui/test_timeline_shell.py tests/ui/test_timeline_style.py tests/ui/test_runtime_audio.py -q`
  `./.venv/bin/python -m mypy echozero/ui/qt/timeline/widget.py --follow-imports=silent`
- Done when:
  the timeline widget is no longer a multi-concept monolith.
- Current notes:
  `widget.py` is now a small orchestration root over
  `widget_canvas.py`, `widget_controls.py`, `widget_viewport.py`,
  `widget_runtime_mixin.py`, and `widget_contract_mixin.py`. The split keeps
  the historical import surface explicit by re-exporting the widget-level
  compatibility names used by existing tests, brings `widget.py` down to a
  small shell file, and keeps the focused timeline-shell/runtime-audio pytest
  slice plus the dedicated mypy lane green.

## Phase E
Foundry UI Cleanup

### RCP-401
Split `main_window.py`

Status: complete (2026-04-21)

- Covers: `3`
- Depends on: `RCP-303`
- Files:
  `echozero/foundry/ui/main_window.py`
  new helper widgets/modules under `echozero/foundry/ui/`
- Steps:
  1. Separate dataset UI, run queue UI, metrics/results UI, and export/actions.
  2. Replace `object`-typed rows/results with typed view models.
  3. Centralize nullable Qt header/item/selection handling.
  4. Add the standard boundary header if still missing.
- Proof:
  `./.venv/bin/python -m pytest tests/foundry -q`
  `./.venv/bin/python -m mypy echozero/foundry/ui/main_window.py --follow-imports=silent`
- Done when:
  Foundry UI is split by screen concern and the major mypy failures are removed.
- Current notes:
  `main_window.py` is now a small composition root over
  `main_window_dataset_mixin.py`, `main_window_workspace_mixin.py`,
  `main_window_run_mixin.py`, `main_window_types.py`, and
  `main_window_worker.py`. The split also replaced the broad `object`-typed row
  surfaces with typed view models, centralized nullable Qt item/header helpers,
  restored the `QDesktopServices` monkeypatch seam used by existing UI tests,
  and fixed the Foundry background-run notification/thread teardown path so the
  full `tests/foundry` lane and the focused mypy lane remain green.

## Phase F
Support Surfaces and Test Monoliths

### RCP-501
Split Support-Only Monoliths

Status: complete (2026-04-21)

- Covers: `10`
- Depends on: `RCP-401`
- Files:
  `echozero/ui/qt/timeline/demo_app.py`
  `echozero/testing/gui_lane_b.py`
- Steps:
  1. Separate scenario/state builders from execution logic.
  2. Keep support-only headers explicit.
  3. Verify canonical runtime does not import these surfaces.
- Proof:
  targeted support-lane pytest slices
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  support-only surfaces are smaller and clearly non-canonical.
- Current notes:
  `demo_app.py` now stays as the public support entrypoint over
  `demo_app_runtime.py`, `demo_app_mutations.py`, and
  `demo_app_services.py`, which separates builders from support-only dispatch
  and mutation logic while keeping the historical `build_demo_app` import
  surface stable. `gui_lane_b.py` now stays as the public runner entrypoint
  over `gui_lane_b_support.py`, with the historical monkeypatch seams restored
  at the root module for `_render_for_hit_testing` and `_write_frame_video`.
  The focused support-lane pytest slices and `scripts/check_repo_hygiene.py`
  remain green after the split.

### RCP-502
Split Oversized Test Modules

Status: complete (2026-04-21)

- Covers: `11`
- Depends on: `RCP-501`
- Files:
  `tests/ui/test_timeline_shell.py`
  `tests/ui/test_app_shell_runtime_flow.py`
  `tests/test_persistence.py`
  `tests/test_audio_engine.py`
  `tests/test_session.py`
  `tests/ui/test_runtime_audio.py`
  other `>=800` line test files as needed
- Steps:
  1. Split by behavior area, not by arbitrary chunk size.
  2. Move repeated builders into shared fixtures/helpers.
  3. Update subsystem READMEs so the new test slices remain discoverable.
- Proof:
  targeted pytest on each split area
  `./.venv/bin/python scripts/check_repo_hygiene.py`
- Done when:
  major test monoliths are decomposed into stable behavior-oriented files.
- Current notes:
  the historical high-traffic test paths now stay as thin wrapper entrypoints:
  `tests/ui/test_timeline_shell.py`, `tests/ui/test_app_shell_runtime_flow.py`,
  `tests/ui/test_runtime_audio.py`, `tests/test_persistence.py`,
  `tests/test_audio_engine.py`, and `tests/test_session.py` all import smaller
  behavior-oriented `*_cases.py` modules while their full prior bodies live in
  non-collected `*_support.py` modules. That keeps the old proof commands
  stable, reduces the public wrapper files to single-digit line counts, and
  makes failure narrowing possible by running only the relevant case modules.
  The wrapper-path pytest lane, the split-area pytest lane, and
  `scripts/check_repo_hygiene.py` are all green after the decomposition.

## Acceptance Ladder

Run these at the end of each phase as applicable:

1. `./.venv/bin/python scripts/check_repo_hygiene.py`
2. `./.venv/bin/python scripts/check_architecture_boundaries.py`
3. `./.venv/bin/python scripts/check_canonical_launcher.py`
4. phase-specific targeted `pytest`
5. phase-specific targeted `mypy`
6. `./.venv/bin/python -m echozero.testing.run --lane appflow` for canonical app-path changes

## Completion Standard

This plan is complete when:

- the canonical runtime hotspots are materially smaller and more typed
- split modules have explicit public boundaries
- large support-only and test-only surfaces are clearly separated from product truth
- the current cleanup work can be tracked phase-by-phase in repo docs instead of chat history
