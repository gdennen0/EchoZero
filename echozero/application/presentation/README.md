# Presentation Layer

Status: canonical subsystem map
Last verified: 2026-04-21

This package shapes typed UI-facing presentation models from application state.
It exists so the UI consumes a stable contract instead of re-deriving app truth locally.
It must stay read-only with respect to domain and application truth.

## Start Here

- `models.py`: typed presentation dataclasses consumed by Qt and automation surfaces
- `inspector_contract.py`: public inspector contract builders and selection routing
- `inspector_contract_types.py`: stable inspector dataclasses shared across UI and tests
- `inspector_contract_support.py`: preview, lookup, and context-action helpers behind the public contract
- `action_descriptors.py`: small shared descriptor helpers
- `object_action_settings.py`: presentation helpers for action-settings surfaces

## Canonical Role

- Present application-backed state to the UI
- Make inspector actions explicit and typed
- Keep selection/playback context readable without moving truth into widgets

## Invariants

- Presentation is derived from application state.
- Presentation must not mutate truth.
- Inspector actions must remain app-backed, not widget-invented.
- Canonical labels and action ids must stay stable enough for tests and automation to target them.

## Primary Tests

- `tests/application/test_inspector_contract.py`
- `tests/application/test_timeline_assembler_contract.py`
- `tests/ui/test_timeline_shell.py`
- `tests/ui/test_timeline_style.py`

## Forbidden Shortcuts

- Do not let widgets synthesize alternate inspector state.
- Do not collapse app identity, presentation identity, and display labels into one loose layer.
- Do not move runtime mutation into presentation helpers.
