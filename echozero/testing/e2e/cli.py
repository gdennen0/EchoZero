"""CLI entrypoint for running a single E2E scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import FoundryDriverPlaceholder, StageZeroDriver, create_stage_zero_driver
from .reporter import write_run_report
from .runner import Runner
from .scenario import load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one EchoZero E2E scenario.")
    parser.add_argument("scenario", help="Path to a JSON or YAML scenario file.")
    parser.add_argument(
        "--driver",
        choices=("stage-zero", "foundry"),
        default="stage-zero",
        help="Driver adapter to use.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional artifact output directory. Defaults to artifacts/e2e/<scenario-name>.",
    )
    args = parser.parse_args(argv)

    scenario = load_scenario(args.scenario)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path("artifacts") / "e2e" / scenario.name
    driver: StageZeroDriver | FoundryDriverPlaceholder
    if args.driver == "stage-zero":
        driver = create_stage_zero_driver()
    else:
        driver = FoundryDriverPlaceholder()

    try:
        result = Runner(driver).run(scenario, artifacts_dir)
        json_path, markdown_path = write_run_report(result, artifacts_dir)
    finally:
        close = getattr(driver, "close", None)
        if callable(close):
            close()

    print(f"scenario={scenario.name}")
    print(f"passed={result.passed}")
    print(f"json={json_path}")
    print(f"markdown={markdown_path}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
