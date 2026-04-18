## Real App Automation

EchoZero automation should target the canonical app path, not demo-only helper
surfaces.

### Canonical runtime

- `run_echozero.py` is the only primary desktop launcher.
- `echozero/ui/qt/launcher_surface.py` is the shared shell assembly used by:
  - the launcher
  - `echozero/testing/app_flow.py`
  - `packages/ui_automation/.../echozero/provider.py`

This means the app a developer launches and the app the automation harness
drives now share the same widget/controller construction path.

### Current control stack

Use these layers in order:

1. `AppShellRuntime`
2. `LauncherSurface`
3. `AppFlowHarness`
4. `EchoZeroAutomationProvider`

Interpretation:

- `AppShellRuntime` is the canonical non-demo runtime.
- `LauncherSurface` wraps that runtime in the actual Stage Zero shell widget and
  project lifecycle actions.
- `AppFlowHarness` runs that shell in-process for proof lanes.
- `EchoZeroAutomationProvider` exposes semantic snapshot/click/drag/scroll/key
  control over the real shell surface.

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

### Proof lanes

Minimum proof for changes in this area:

- `tests/testing/test_app_flow_harness.py`
- `tests/ui_automation/test_echozero_backend.py`
- `tests/ui/test_run_echozero_launcher.py`
- `python -m echozero.testing.run --lane appflow`
