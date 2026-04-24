# LLM Cleanup Board

Status: active cleanup campaign
Last verified: 2026-04-21

_Updated: 2026-04-21_

This board is the repo-local campaign for making the canonical EchoZero lane
smaller, less ambiguous, and easier for LLMs to load correctly.

Use [STATUS.md](STATUS.md) for current implementation truth.
Use this board for cleanup sequencing and progress.
Use [EXECUTION-PLAN.md](EXECUTION-PLAN.md) for the ordered remediation plan across issues `1-12`.
Use [BACKLOG-CLEARANCE-PLAN.md](BACKLOG-CLEARANCE-PLAN.md) for the ordered
post-remediation backlog-clearance pass.

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
  - keep shrinking shell behavior into named helper modules
  - preserve `app_shell.py` as orchestration, not conversion/storage detail
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

### Completed

- [x] add `docs/STATUS.md` as the canonical current-state map
- [x] add subsystem maps for timeline app, presentation, timeline UI, and Foundry
- [x] refresh `docs/index.md` and `AGENTS.md` so the front door points at current truth
- [x] add `Status:` / `Last verified:` markers to major canonical docs
- [x] label support-only surfaces like `gui_lane_b`, `fixture_loader.py`, and `test_harness.py`
- [x] extract `echozero/ui/qt/app_shell_timeline_state.py` from `app_shell.py`
- [x] extract `echozero/ui/qt/app_shell_layer_storage.py` from `app_shell.py`
- [x] split `echozero/application/presentation/inspector_contract.py` into public builders, shared types, and support helpers
- [x] normalize the remaining large-module boundary headers and enforce them in `scripts/check_repo_hygiene.py`
- [x] make explicit import and export boundaries pass on the canonical shell/timeline proof lane
- [x] extract object-action session and pipeline-run helper logic into `echozero/ui/qt/app_shell_object_actions.py`
- [x] extract undo/history classification and snapshot restore logic into `echozero/ui/qt/app_shell_history.py`
- [x] shrink `echozero/application/timeline/orchestrator.py` by separating selection/edit from transfer-plan flows
- [x] continue the remaining `app_shell.py` cleanup by moving storage-sync and project-lifecycle details behind thinner runtime helpers
- [x] shrink `echozero/application/presentation/inspector_contract.py` into smaller contract builders
- [x] split `echozero/ui/qt/timeline/widget.py` and `widget_actions.py` by rendering, input, and dispatch roles
- [x] split `tests/ui/test_timeline_shell.py` by behavior area instead of one giant proof surface
- [x] continue the strict type-clean pass outward from the canonical shell lane into adjacent timeline/UI helpers
- [x] add repo guardrails for cleaned orchestration roots, wrapper entrypoints, and missing boundary headers

### Next Expansion

- [ ] extend size guardrails from cleaned roots into remaining large canonical mixins once those files are split further
- [ ] choose the next decomposition targets among `widget_canvas.py`, `object_info_panel.py`, and `main_window_run_mixin.py`
- [ ] reduce doc sprawl by rolling planning/audit docs into fewer current-state docs
- [ ] continue the type-clean pass from shell helpers into the remaining large timeline/UI hotspots

## Current Wins

- `app_shell.py`, `widget.py`, `widget_actions.py`, `main_window.py`, `demo_app.py`, and the historical wrapper test files are now thin orchestration roots instead of mixed-concern monoliths.
- `app_shell_project_timeline.py` now stays as the public root over focused storage/audio and presentation-overlay helpers, instead of mixing baseline assembly, waveform registration, and selector formatting in one file.
- The canonical docs front door now has one fast current-truth doc plus code-adjacent subsystem maps.
- Canonical versus support-only UI surfaces are clearer in both docs and module headers.
- The hygiene guardrail now enforces boundary headers plus size ceilings for cleaned public roots and wrapper entrypoints, without tightening limits on still-active large mixins prematurely.
- The cleanup execution plan is now reflected in repo docs as completed work instead of open chat-only status.

## Working Rule

If a file contains more than one of these at the same time, it should usually be
split:

- composition root
- domain/application truth assembly
- widget painting/layout
- dialog orchestration
- filesystem/runtime IO
- demo/dev-only helpers
