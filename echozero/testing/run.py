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
