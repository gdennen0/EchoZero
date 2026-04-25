# EchoZero Status

Status: active canonical current-state reference
Last verified: 2026-04-25

Use this file to answer one question quickly: what is true in the repo now.
This file is not a plan, board, or historical design note.
When older docs disagree with current implementation, start here, then verify in code.

## Repo Posture

- Canonical desktop entrypoint: `run_echozero.py`
- Canonical codebase: `echozero/`
- Main Stage Zero shell surface: `echozero/ui/qt/app_shell.py`
- Timeline app contract: `echozero/application/timeline/*`
- Sync boundary: `echozero/application/sync/*` and `echozero/infrastructure/sync/ma3_adapter.py`
- Foundry lane: `echozero/foundry/*`
- EZ1 code is removed from this branch and should be treated as history only

## Timeline

- Main is truth for playback, export, sync, and freshness comparisons.
- Takes are subordinate history, rerun results, comparison candidates, and merge inputs.
- The canonical timeline app path is:
  `run_echozero.py` -> `echozero/ui/qt/app_shell.py` -> `echozero/application/timeline/*` -> `echozero/ui/qt/timeline/*`
- `echozero/application/timeline/orchestrator.py` owns intent handling and truth mutation.
- `echozero/application/timeline/assembler.py` owns presentation shaping.
- Widgets must not invent alternate truth or bypass the application contract.
- Song import defaults are machine-local app preferences (`Song Import` section), including:
  - import-time pipeline actions
  - import-time LTC channel strip behavior
- Setlist drag/drop accepts files and folders (recursive audio discovery) and resolves multi-file imports in natural order.
- Multi-file import prompts for destination mode (append, before/after target song, or versions on target song when applicable).
- Multi-file import can run import-pipeline actions through the same queued action path as context-menu runs.
- Single-file add-song/add-version imports can use that same queued action path when runtime request/state/switch support is available.
- Batch import pipeline execution is ordered:
  - songs/versions import first
  - queued per-song pipeline actions execute sequentially in import order
- Queued pipeline progress is surfaced in the main timeline pipeline status banner.
- Import pipeline actions can run for each imported file on:
  - drag/drop imports
  - add-song dialog imports
  - add-version dialog imports
- Stereo files with detected single-channel LTC are auto-split:
  - program channel is imported as song audio
  - both program and LTC mono artifacts are retained in `audio/split_channels/`
  - left/right flipped LTC/program layouts are supported
- Setlist ordering supports move up/down, drag reorder, and batch move top/bottom.

## Sync

- MA3 sync is main-only.
- Non-main takes do not sync directly.
- Sync behavior belongs to the application and infrastructure sync boundaries, not widget-local logic.
- Sync changes require app-boundary guardrail proof, not just isolated helper tests.

## MA3 Transfer

- Operator-first MA3 push routing lives in
  `echozero/application/timeline/ma3_push_intents.py`,
  `echozero/application/timeline/orchestrator_ma3_push_mixin.py`, and
  `echozero/ui/qt/timeline/widget_action_ma3_push_mixin.py`.
- Manual pull workspace state and import behavior live in
  `echozero/application/timeline/orchestrator.py`,
  `echozero/application/timeline/orchestrator_manual_pull_import_mixin.py`,
  `echozero/ui/qt/timeline/manual_pull.py`, and
  `echozero/ui/qt/timeline/widget_action_transfer_workspace_mixin.py`.
- Saved MA3 routing truth remains `layer -> ma3_track_coord`.
- Active song versions carry `ma3_timecode_pool_no`.
- New songs receive the next unused project-local MA3 timecode pool by default.
- New versions of an existing song inherit the source version's MA3 timecode pool.
- Pull defaults to the selected layer route when present; otherwise it falls back to the active song version MA3 timecode pool.
- Pull workspace source selection is operator-first:
  - choose the MA3 timecode pool once
  - see every available track group and track for that pool at the same time
  - click a track group or track instead of stepping through dependent dropdowns
- Pull import mode is destination-driven:
  - new EZ layer target -> import into `main`
  - existing EZ layer target -> import into a new take
- Pull imports auto-link newly created or previously unlinked EZ event layers to the source MA3 track coord.
- When multiple MA3 source tracks are selected for pull planning, the target options can expose `+ Create New Layer Per Source Track...`.

## Playback

- Playback is a real app/runtime concern, not a demo-only surface.
- The current app shell wires local runtime audio through the real presentation/app path.
- Demo helpers, simulated GUI lanes, and screenshot-oriented tools are support surfaces only and do not count as human-path demo proof.

## UI Automation

- Canonical automation control plane:
  `echozero/ui/qt/automation_bridge.py` and `packages/ui_automation/**`
- Canonical app automation proof lanes:
  `python -m echozero.testing.run --lane appflow`
  `python -m echozero.testing.run --lane ui-automation`
- `echozero/testing/gui_dsl.py`, `echozero/testing/gui_lane_b.py`, and `echozero/ui/qt/timeline/demo_app.py`
  are support surfaces, not the canonical app-control plane.

## Foundry

- Foundry is real product code, not a throwaway experiment.
- Current entry surfaces:
  `python -m echozero.foundry.cli`
  `python -m echozero.foundry.app`
  `echozero/foundry/ui/main_window.py`
- Foundry contracts live under `echozero/foundry/contracts/`.
- Training/export/validation flow documentation lives in `docs/FOUNDRY-TRAINING.md`.

## Documentation Status

- Strong: architecture, agent workflow, and testing/proof documentation.
- Good: repo front door and current orientation docs.
- Weak: code-adjacent subsystem maps and "implemented now" summaries were previously sparse.
- Current cleanup direction:
  add subsystem READMEs, reduce doc sprawl, and keep current-truth docs separate from plans and audits.

## Immediate Cleanup Focus

- Split `echozero/ui/qt/app_shell.py`
- Split `echozero/application/timeline/orchestrator.py`
- Split `echozero/application/presentation/inspector_contract.py`
- Split `echozero/application/timeline/object_action_settings_service.py`
- Tighten type clarity on canonical app/UI paths
- Keep canonical versus support-only surfaces explicit in docs and file headers

## Best Next Docs

- `AGENTS.md`
- `docs/AGENT-CONTEXT.md`
- `docs/TESTING.md`
- `docs/SONG-IMPORT-BATCH-LTC-WORKFLOW.md`
- `echozero/application/timeline/README.md`
- `echozero/application/presentation/README.md`
- `echozero/ui/qt/timeline/README.md`
- `echozero/foundry/README.md`
