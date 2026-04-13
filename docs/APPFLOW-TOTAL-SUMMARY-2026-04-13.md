# Appflow Total Summary (2026-04-13)

## What Was Delivered

- canonical `run_echozero.py` launcher with compatibility delegation from `run_timeline_demo.py`
- app shell runtime lifecycle coverage, launcher flow coverage, timeline shell coverage, and unsaved-close protection
- built-in appflow lanes for `appflow`, `appflow-sync`, `appflow-osc`, `appflow-protocol`, and `appflow-all`
- deterministic MA3 simulator coverage, OSC loopback coverage, protocol fixtures, and receive-path integration coverage
- release packaging and packaged smoke validation scripts aligned to `artifacts/releases/test`
- one-command gate runner: `scripts/run-appflow-gates.ps1`

## How To Run Everything

- Full targeted validation:
  - `C:/Users/griff/EchoZero/.venv/Scripts/python.exe -m pytest tests/testing tests/unit/test_ma3_communication_service_protocol.py tests/unit/test_ma3_receive_path_integration.py tests/unit/test_ma3_event_contract.py tests/ui/test_run_echozero_launcher.py tests/ui/test_app_shell_runtime_flow.py tests/ui/test_timeline_shell.py tests/application/test_manual_transfer_pull_flow.py tests/application/test_manual_transfer_push_flow.py -q`
- One-command local appflow gate:
  - `powershell -File scripts/run-appflow-gates.ps1`
- Full demo suite command:
  - `powershell -File scripts/run-full-demo-suite.ps1`
  - optional audio path:
    - `powershell -File scripts/run-full-demo-suite.ps1 -AudioPath "C:/path/song.wav"`
- Individual release steps:
  - `powershell -File scripts/build-test-release.ps1`
  - `powershell -File scripts/smoke-test-release.ps1`

## What Remains

- manual UX QA walkthrough in the packaged app flow
- real MA3 hardware validation before release signoff
