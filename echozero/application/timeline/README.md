# Timeline Application

Status: canonical subsystem map
Last verified: 2026-04-21

This package is the canonical timeline application contract.
It owns timeline truth, intent handling, presentation shaping, and object-action runtime behavior.
Widgets and demo helpers must consume this package, not replace it.

## Start Here

- `models.py`: timeline state, selection, playback target, and core entities
- `intents.py`: typed timeline commands
- `app.py`: application façade used by the app shell
- `orchestrator.py`: truth mutation and intent handling
- `assembler.py`: presentation shaping for the UI
- `object_actions/`: descriptors, sessions, and settings plans for object-owned pipeline actions
- `object_action_settings_service.py`: orchestration root for scoped config, copy/session, and runtime-binding helpers
- `operation_progress_service.py`: background operation-progress lifecycle visible to the app shell

## Canonical Entry Path

`run_echozero.py` -> `echozero/ui/qt/app_shell.py` -> `TimelineApplication` -> `TimelineOrchestrator` / `TimelineAssembler`

If behavior matters to the user, it is not done until it is proven through this path.

## Invariants

- Main is truth.
- Takes are subordinate.
- MA3 sync is main-only.
- Staleness changes only when upstream main changes.
- Engine stays ignorant of UI/editor semantics.
- UI selection and widget behavior must not invent alternate truth.

## Primary Tests

- `tests/application/test_timeline_orchestrator_take_actions.py`
- `tests/application/test_timeline_assembler_contract.py`
- `tests/application/test_manual_transfer_push_flow.py`
- `tests/application/test_manual_transfer_pull_flow.py`
- `tests/application/test_transfer_plan_batch_apply.py`
- `tests/ui/test_app_shell_runtime_flow.py`
- `tests/ui/test_app_shell_undo_redo.py`

## Proof Lanes

- Targeted pytest slices first
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane appflow-sync` for sync-sensitive changes

## Forbidden Shortcuts

- Do not add widget-only truth mutation here or around it.
- Do not let demo fixtures define runtime behavior.
- Do not make take selection behave like alternate live truth.
- Do not push MA3 sync logic into subordinate takes.
