from __future__ import annotations

import argparse
import sys

import pytest


LANES: dict[str, list[str]] = {
    "appflow": [
        "tests/testing/test_app_flow_harness.py",
        "tests/testing/test_app_shell_profiles.py",
        "tests/testing/test_simulated_ma3_bridge.py",
    ],
    "appflow-sync": [
        "tests/testing/test_app_flow_harness.py",
        "-k",
        "sync",
    ],
    "appflow-osc": [
        "tests/testing/test_ma3_osc_loopback.py",
        "tests/testing/test_app_flow_harness.py",
        "-k",
        "osc",
    ],
    "appflow-protocol": [
        "tests/testing/test_simulated_ma3_bridge.py",
        "tests/application/test_sync_adapters.py",
        "tests/application/test_live_sync_guardrail_contracts.py",
    ],
    "appflow-all": [
        "tests/testing/test_app_flow_harness.py",
        "tests/testing/test_app_shell_profiles.py",
        "tests/testing/test_simulated_ma3_bridge.py",
        "tests/application/test_sync_adapters.py",
        "tests/application/test_live_sync_guardrail_contracts.py",
        "tests/ui/test_run_echozero_launcher.py",
        "tests/ui/test_app_shell_runtime_flow.py",
        "tests/ui/test_timeline_shell.py",
        "tests/application/test_manual_transfer_pull_flow.py",
        "tests/application/test_manual_transfer_push_flow.py",
    ],
    "gui-lane-b": [
        "tests/testing/test_gui_dsl.py",
        "tests/testing/test_gui_lane_b.py",
    ],
    "humanflow-all": [
        "tests/ui/test_run_echozero_launcher.py",
        "tests/testing/test_app_shell_profiles.py",
        "tests/testing/test_app_flow_harness.py",
        "tests/ui/test_app_shell_runtime_flow.py",
        "tests/ui_automation/test_session.py",
        "tests/ui_automation/test_echozero_backend.py",
        "tests/ui_automation/test_bridge_server.py",
        "tests/testing/test_gui_dsl.py",
        "tests/testing/test_gui_lane_b.py",
        "tests/application/test_manual_transfer_pull_flow.py",
        "tests/application/test_manual_transfer_push_flow.py",
        "tests/application/test_sync_adapters.py",
        "tests/application/test_live_sync_guardrail_contracts.py",
    ],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run built-in EchoZero testing lanes.")
    parser.add_argument("--lane", required=True, choices=sorted(LANES))
    parsed = parser.parse_args(argv)

    args = ["-q", *LANES[parsed.lane]]
    print(f"lane={parsed.lane}")
    result = pytest.main(args)
    if result == 0:
        print("result=passed")
    else:
        print(f"result=failed exit_code={result}")
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
