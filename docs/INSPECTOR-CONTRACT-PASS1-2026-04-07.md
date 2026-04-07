# Inspector Contract Pass 1

Date: 2026-04-07

## Goal

Unify the Stage Zero timeline Object Palette and right-click context menus on one self-describing inspector contract so both surfaces describe the same object and expose the same action set.

## Contract Schema

The shared contract lives in `echozero/application/presentation/inspector_contract.py`.

- `InspectorObjectIdentity`
  - `object_id`
  - `object_type`
  - `label`
- `InspectorFactRow`
  - `label`
  - `value`
- `InspectorSection`
  - `section_id`
  - `label`
  - `rows`
- `InspectorAction`
  - `action_id`
  - `label`
  - `enabled`
  - `kind`
  - `group`
  - `params`
- `InspectorContextSection`
  - `section_id`
  - `label`
  - `actions`
- `TimelineInspectorHitTarget`
  - `kind`
  - `layer_id`
  - `take_id`
  - `event_id`
  - `time_seconds`
- `InspectorContract`
  - `title`
  - `identity`
  - `sections`
  - `context_sections`
  - `empty_state`

## Mapping Rules

- Selection-driven palette rendering uses `build_timeline_inspector_contract(presentation)`.
- Right-click context menus use `build_timeline_inspector_contract(presentation, hit_target=...)`.
- Event contracts resolve main-row events against `layer.main_take_id`, which preserves main-is-truth semantics.
- Take-event contracts resolve against subordinate `TakeLanePresentation` rows and expose take actions from the same contract.
- Empty layers retain the no-takes indication through `main take: none` and `takes: none`.

## Migration Summary

- Added a presentation-layer inspector contract and mapper.
- Refactored `ObjectInfoPanel` to render contract text instead of bespoke event/layer branch logic.
- Added `TimelineCanvas` hit-target resolution plus right-click context menu generation from `InspectorContextSection` actions.
- Routed menu actions back into existing intents for seek, nudge, duplicate, mute, solo, gain, and take actions.
- Added focused tests for:
  - no selection
  - layer selection
  - main event
  - take event
  - no-takes layer

## Test Command

```powershell
py -m pytest tests\application\test_inspector_contract.py tests\ui\test_timeline_shell.py tests\application\test_timeline_orchestrator_take_actions.py tests\application\test_timeline_assembler_contract.py
```

## Test Result

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.0.2, pluggy-1.6.0
collecting ... collected 66 items
...
============================= 66 passed in 1.84s ==============================
```
