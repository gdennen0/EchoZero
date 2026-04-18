# Automation Surface Policy

_Updated: 2026-04-18_

This is the fast policy doc for choosing the correct automation surface in
EchoZero.

Use this before writing or extending tests, agents, or automation tooling.

## Canonical Rule

EchoZero has one automation model and one preferred app-control surface.

- EchoZero owns the semantic automation contract.
- The live app automation bridge is the canonical control plane.
- `packages/ui_automation/**` is the preferred client boundary.
- OpenClaw, MCP, and ACP wrappers are thin clients over that boundary.

They are not alternate truth sources.

## Use This

### Real app automation

Use for:
- agent-driven app control
- app-path automation tests
- lifecycle flows
- timeline/sync/operator flows meant to reflect real app behavior

Canonical surfaces:
- `run_echozero.py`
- `echozero/ui/qt/launcher_surface.py`
- `echozero/ui/qt/app_shell.py`
- `echozero/ui/qt/automation_bridge.py`
- `packages/ui_automation/**`
- `tests/ui_automation/**`

## Internal Only

### App harness support

`echozero/testing/app_flow.py`

Allowed use:
- canonical app setup and lifecycle support in tests

Not allowed as the long-term agent/control API.

### Simulated GUI scenario coverage

- `echozero/testing/gui_dsl.py`
- `echozero/testing/gui_lane_b.py`

Allowed use:
- deterministic simulated GUI regression coverage

Not allowed as the primary app automation model.

### Demo and fixture builders

- `echozero/ui/qt/timeline/demo_app.py`
- `echozero/testing/demo_suite.py`
- `echozero/testing/demo_suite_scenarios.py`

Allowed use:
- fixtures
- screenshot/demo generation
- explicitly simulated proof

Not allowed on the canonical runtime or agent control path.

### Legacy/ambiguous E2E scaffolding

- `echozero/testing/e2e/**`

Treat as internal scaffolding until explicitly rebuilt on top of the canonical
bridge and client model.

## Hard Prohibitions

- Do not add a second app automation client beside `packages/ui_automation/**`.
- Do not expose runtime-only privileged actions that bypass user-equivalent app
  paths after launch.
- Do not treat direct widget mutation, presenter injection, or private harness
  access as app automation.
- Do not market simulated proof as app acceptance.
- Do not add alternate launcher paths for automation.

## Decision Test

If you are about to add a test or tool, ask:

1. Does it launch or attach to the real app path?
2. Does it use the app-owned automation model?
3. Does it act through app actions, pointer/keyboard input, or narrow OS
   fallback only?

If any answer is no, it is not the canonical automation surface.
