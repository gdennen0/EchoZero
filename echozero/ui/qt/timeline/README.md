# Timeline Qt Surface

Status: canonical subsystem map
Last verified: 2026-04-21

This package contains the Stage Zero timeline Qt surface.
It exists to render and dispatch the timeline experience through the canonical application contract.
The runtime path is real product code, but several support modules here are intentionally non-canonical.

## Canonical Runtime Surface

- `widget.py`: main timeline widget surface
- `widget_actions.py`: Qt-to-application dispatch helpers
- `object_info_panel.py`: inspector-adjacent timeline details and actions
- `runtime_audio.py`: local runtime audio controller for timeline playback/preview
- `style.py` and `blocks/*`: reusable Stage Zero rendering primitives

Canonical app path:
`run_echozero.py` -> `echozero/ui/qt/app_shell.py` -> this package

## Song Import + Setlist Reorder

The timeline Qt surface also owns the operator-facing import and setlist reorder flows:

- folder/file drag-drop routing and multi-file import prompts
- import-mode selection (append, before/after target, target-song versions)
- import pipeline-action prompt and dispatch wiring
- setlist move up/down, drag reorder, and batch move actions

Implementation and persistence/runtime details are documented in
`docs/SONG-IMPORT-BATCH-LTC-WORKFLOW.md`.

## Support-Only Surfaces

These are useful, but they are not canonical runtime truth:

- `demo_app.py`: demo-only fixture app for tests and screenshots
- `demo_app_runtime.py`: support-only demo intent/runtime shell
- `demo_app_mutations.py`: support-only demo presentation mutation helpers
- `demo_app_services.py`: support-only demo service shims
- `fixture_loader.py`: synthetic presentation loader for UI development
- `real_data_fixture.py`: support fixture builder
- `test_harness.py`: deterministic test harness

Do not treat those modules as the long-term app runtime or automation control plane.

## Primary Tests

- `tests/ui/test_timeline_shell.py`
- `tests/ui/timeline_shell_*_cases.py` with shared helpers in `tests/ui/timeline_shell_support.py`
- `tests/ui/test_runtime_audio.py`
- `tests/ui/runtime_audio_*_cases.py` with shared helpers in `tests/ui/runtime_audio_support.py`
- `tests/ui/test_timeline_style.py`
- `tests/ui/test_take_row_block.py`
- `tests/ui/test_timeline_feel_contract.py`
- `tests/ui/test_shared_shell_style.py`
- `tests/ui_automation/test_echozero_backend.py`
- `tests/ui_automation/test_session.py`

## Proof Lanes

- Targeted Qt/UI pytest slices first
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane ui-automation`
- `python -m echozero.testing.run --lane gui-lane-b` for deterministic simulated GUI regression coverage only

## Forbidden Shortcuts

- Do not invent timeline truth in widgets.
- Do not bypass application contracts with convenience UI logic.
- Do not present simulated fixture behavior as real app-path proof.
- Do not build a second automation control plane beside `packages/ui_automation/**` and the Qt automation bridge.
