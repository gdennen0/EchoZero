## Human Task Tests

This is the current app-path test inventory for the goal:

- start the real app
- drive it through user-visible tasks
- prove the main operator flows through automation and app contracts

Run the full current suite with:

```bash
.venv/bin/python -m echozero.testing.run --lane humanflow-all
```

## Included Tests

- [tests/ui/test_run_echozero_launcher.py](/Users/march/Documents/GitHub/EchoZero/tests/ui/test_run_echozero_launcher.py:1)
  Proves the canonical launcher starts the app shell, wires lifecycle actions, and handles shutdown.
- [tests/testing/test_app_shell_profiles.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_app_shell_profiles.py:1)
  Proves the app-shell builder stays on the canonical runtime path.
- [tests/testing/test_app_flow_harness.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_app_flow_harness.py:1)
  Proves the test harness can start the app, drive launcher actions, and manage sync doubles.
- [tests/ui/test_app_shell_runtime_flow.py](/Users/march/Documents/GitHub/EchoZero/tests/ui/test_app_shell_runtime_flow.py:1)
  Proves real app runtime flows such as project lifecycle, song import, versions, stems, drum events, and classification.
- [tests/ui_automation/test_session.py](/Users/march/Documents/GitHub/EchoZero/tests/ui_automation/test_session.py:1)
  Proves the automation session API shape used by higher-level app control.
- [tests/ui_automation/test_echozero_backend.py](/Users/march/Documents/GitHub/EchoZero/tests/ui_automation/test_echozero_backend.py:1)
  Proves semantic app automation over the real shell surface:
  import, stems, drum events, classification, drag, scroll, transport play/pause/stop, pointer movement, and sync enable/disable.
- [tests/ui_automation/test_bridge_server.py](/Users/march/Documents/GitHub/EchoZero/tests/ui_automation/test_bridge_server.py:1)
  Proves the live HTTP automation bridge can expose health and snapshot against a real EchoZero surface.
- [tests/testing/test_gui_dsl.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_gui_dsl.py:1)
  Proves scenario parsing/validation for scripted human-style flows.
- [tests/testing/test_gui_lane_b.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_gui_lane_b.py:1)
  Proves the higher-level operator scenario runner, screenshots, and trace/video outputs.
- [tests/application/test_manual_transfer_pull_flow.py](/Users/march/Documents/GitHub/EchoZero/tests/application/test_manual_transfer_pull_flow.py:1)
  Proves app-contract behavior for pull flows.
- [tests/application/test_manual_transfer_push_flow.py](/Users/march/Documents/GitHub/EchoZero/tests/application/test_manual_transfer_push_flow.py:1)
  Proves app-contract behavior for push flows.
- [tests/application/test_sync_adapters.py](/Users/march/Documents/GitHub/EchoZero/tests/application/test_sync_adapters.py:1)
  Proves sync adapter behavior at the application boundary.
- [tests/application/test_live_sync_guardrail_contracts.py](/Users/march/Documents/GitHub/EchoZero/tests/application/test_live_sync_guardrail_contracts.py:1)
  Proves the sync safety rules remain app-backed.

## Current Coverage Shape

Covered now:

- app startup through the single launcher
- project new/save/save-as through automation
- song import
- stem extraction
- drum-event extraction
- drum classification
- event selection
- event drag
- event nudge/duplicate through scenario runner
- push transfer surface
- pull transfer surface
- sync enable/disable
- screenshots and automation snapshots

Not yet fully covered:

- native OS dialogs as true system-owned UI
- every transport/ruler gesture variant
- every inspector/live-sync branch
- packaged-build UI automation
- MA3 real hardware interaction

Those remaining gaps should be treated as the next expansion list, not as solved.
