# LLM Cleanup Board

_Updated: 2026-04-18_

This board is the repo-local campaign for making the canonical EchoZero lane
smaller, less ambiguous, and easier for LLMs to load correctly.

## Goal

Make the repo answer these questions quickly and consistently:

- what is the canonical app entrypoint?
- where does playback truth live?
- where does timeline truth live?
- which UI files are canonical vs transitional?
- which proof lanes are app-real vs simulated/dev-only?

## Keep / Split / Delete / Rename

### Keep

- `run_echozero.py`
- `echozero/application/timeline/**`
- `echozero/application/playback/**`
- `echozero/application/presentation/**`
- `echozero/ui/qt/app_shell.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/**`
- `echozero/ui/FEEL.py`
- `docs/AGENT-CONTEXT.md`
- `docs/TESTING.md`

### Split

- `echozero/ui/qt/app_shell.py`
  - split composition/runtime shell from project-native timeline bootstrap
  - split runtime-service adapters from shell behavior
- `echozero/ui/qt/timeline/widget.py`
  - split manual-pull dialog stack
  - split object-info panel
  - split transfer-action helpers and dialog orchestration
- `echozero/application/timeline/orchestrator.py`
  - split manual transfer, event editing, and playback-target concerns
- `echozero/application/presentation/inspector_contract.py`
  - split layer/event/action section builders from shared contract types
- `tests/ui/test_timeline_shell.py`
  - split input/selection, transfer flows, transport/follow-scroll, and inspector coverage

### Delete Or Quarantine

- demo-only timeline helpers that are not required by `run_echozero.py`
- any fixture loader or preview surface that leaks into canonical app imports
- obsolete compatibility aliases once all current callers have migrated
- repo-local runtime outputs and snapshots if they reappear in git

### Rename

- transitional helper modules should say what they are:
  - `*_runtime_bridge.py`
  - `*_project_timeline.py`
  - `*_manual_pull.py`
  - `*_object_panel.py`
- avoid generic names like `helpers.py`, `misc.py`, or `utils.py`

## Ranked Top-20 Files

Ranked by comprehension cost: size, centrality, and chance of conflicting truths.

1. `echozero/ui/qt/timeline/widget.py`
2. `echozero/ui/qt/app_shell.py`
3. `echozero/application/timeline/orchestrator.py`
4. `echozero/application/presentation/inspector_contract.py`
5. `echozero/foundry/ui/main_window.py`
6. `echozero/persistence/session.py`
7. `echozero/application/timeline/assembler.py`
8. `echozero/models/provider.py`
9. `echozero/foundry/services/baseline_trainer.py`
10. `echozero/ui/qt/timeline/demo_app.py`
11. `packages/ui_automation/src/ui_automation/adapters/echozero/provider.py`
12. `echozero/audio/engine.py`
13. `echozero/ui/qt/timeline/runtime_audio.py`
14. `echozero/ui/qt/launcher_surface.py`
15. `echozero/application/playback/runtime.py`
16. `echozero/application/timeline/app.py`
17. `echozero/ui/qt/automation_bridge.py`
18. `run_echozero.py`
19. `echozero/application/presentation/models.py`
20. `echozero/ui/qt/timeline/blocks/layer_header.py`

## Current Campaign

### Phase 1

- extract project-native timeline bootstrap from `app_shell.py`
- extract manual-pull UI stack from `timeline/widget.py`
- extract object-info panel from `timeline/widget.py`

### Phase 2

- split transfer-action handlers out of `TimelineWidget`
- split runtime pipeline actions from widget shell logic
- split app-shell runtime service adapters out of `app_shell.py`

### Phase 3

- shrink `orchestrator.py`
- shrink `inspector_contract.py`
- split `tests/ui/test_timeline_shell.py`

## Working Rule

If a file contains more than one of these at the same time, it should usually be
split:

- composition root
- domain/application truth assembly
- widget painting/layout
- dialog orchestration
- filesystem/runtime IO
- demo/dev-only helpers
