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
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--real-analysis",
        action="store_true",
        help="Use the canonical runtime analysis stack instead of mock analysis executors.",
    )
    parsed = parser.parse_args(argv)

    output_dir = Path(parsed.output)
    trace = run_scenario_file(
        scenario_path=parsed.scenario,
        output_dir=output_dir,
        record_video=parsed.record_video,
        fps=parsed.fps,
        use_mock_analysis=not parsed.real_analysis,
    )
    print(
        json.dumps(
            {
                "scenario": parsed.scenario,
                "output": str(output_dir),
                "steps": len(trace),
                "record_video": parsed.record_video,
                "fps": parsed.fps,
                "real_analysis": parsed.real_analysis,
            },
            indent=2,
        )
    )
    return 0 if all(step["status"] == "passed" for step in trace) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
