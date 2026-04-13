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
        "tests/unit/test_ma3_communication_service_protocol.py",
        "tests/unit/test_ma3_receive_path_integration.py",
        "tests/unit/test_ma3_event_contract.py",
    ],
    "appflow-all": [
        "tests/testing/test_app_flow_harness.py",
        "tests/testing/test_app_shell_profiles.py",
        "tests/testing/test_simulated_ma3_bridge.py",
        "tests/unit/test_ma3_communication_service_protocol.py",
        "tests/unit/test_ma3_receive_path_integration.py",
        "tests/unit/test_ma3_event_contract.py",
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
