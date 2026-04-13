from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from echozero.testing.gui_lane_b import run_scenario_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EchoZero Lane B GUI scenarios.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args(argv)

    output_dir = Path(parsed.output)
    trace = run_scenario_file(scenario_path=parsed.scenario, output_dir=output_dir)
    print(json.dumps({"scenario": parsed.scenario, "output": str(output_dir), "steps": len(trace)}, indent=2))
    return 0 if all(step["status"] == "passed" for step in trace) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
