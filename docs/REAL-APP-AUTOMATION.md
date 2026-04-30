## Real App Automation

Status: reference
Last reviewed: 2026-04-30


EchoZero automation should target the canonical app path, not demo-only helper
surfaces.

### Canonical runtime

- `run_echozero.py` is the only primary desktop launcher.
- `echozero/ui/qt/launcher_surface.py` is the shared shell assembly used by:
  - the launcher
  - `echozero/testing/app_flow.py`
  - the live automation bridge backend

This means the app a developer launches and the app the automation harness
drives now share the same widget/controller construction path.

### Current control stack

Use these layers in order:

1. `AppShellRuntime`
2. `LauncherSurface`
3. `AutomationBridgeServer`
4. `EchoZeroAutomationProvider`

Interpretation:

- `AppShellRuntime` is the canonical non-demo runtime.
- `LauncherSurface` wraps that runtime in the actual Stage Zero shell widget and
  project lifecycle actions.
- `AutomationBridgeServer` exposes the live app surface over localhost.
- `EchoZeroAutomationProvider` is the preferred public provider name and now
  points at the live bridge client.
- `HarnessEchoZeroAutomationProvider` is internal test support for in-process
  proof and should not be treated as the default agent/app automation entrypoint.

### What this is for

Use this path when you need to:

- run the real app in development
- drive user-visible flows end to end
- inspect timeline hit targets and current selection
- capture screenshots from the running shell
- verify operator flows without dropping back to demo-only adapters

### What this is not

- `echozero/testing/e2e/adapters.py` still contains a demo-oriented driver and
  should not be treated as the primary app-path harness.
- `HarnessEchoZeroAutomationProvider` is not the default external automation
  surface.

### Proof lanes

Minimum proof for changes in this area:

- `tests/testing/test_app_flow_harness.py`
- `tests/ui_automation/test_echozero_backend.py`
- `tests/ui_automation/test_bridge_server.py`
- `tests/ui/test_run_echozero_launcher.py`
- `python -m echozero.testing.run --lane appflow`
