from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from echozero.testing.demo_suite_scenarios import (
    SCENARIO_ORDER,
    SCENARIO_RUNNERS,
    ScenarioResult,
    collect_reference_artifacts,
    find_latest_smoke_report,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/demo-suite")


def timestamp_run_id(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def create_run_folder(output_root: Path, run_id: str) -> Path:
    run_folder = output_root / run_id
    try:
        run_folder.mkdir(parents=True, exist_ok=True)
        return run_folder.resolve()
    except (FileNotFoundError, PermissionError):
        fallback_root = Path.cwd() / output_root.name
        fallback_run_folder = fallback_root / run_id
        fallback_run_folder.mkdir(parents=True, exist_ok=True)
        return fallback_run_folder.resolve()


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_manifest(*, run_id: str, run_folder: Path, scenario_results: list[ScenarioResult]) -> dict[str, Any]:
    passed = sum(1 for result in scenario_results if result.status == "passed")
    skipped = sum(1 for result in scenario_results if result.status == "skipped")
    missing = sum(1 for result in scenario_results if result.status == "missing")
    artifact_count = sum(len(result.artifacts) for result in scenario_results)
    return {
        "run_timestamp": run_id,
        "run_id": run_id,
        "run_folder": str(run_folder.resolve()),
        "scenario_statuses": [asdict(result) for result in scenario_results],
        "counts": {
            "passed": passed,
            "skipped": skipped,
            "missing": missing,
            "total": len(scenario_results),
            "artifacts": artifact_count,
        },
    }


def _write_summary(run_folder: Path, manifest: dict[str, Any]) -> Path:
    lines = [
        f"run_id: {manifest['run_id']}",
        f"run_folder: {manifest['run_folder']}",
        f"passed: {manifest['counts']['passed']}",
        f"skipped: {manifest['counts']['skipped']}",
        f"missing: {manifest['counts']['missing']}",
        "",
    ]
    for scenario in manifest["scenario_statuses"]:
        lines.append(f"[{scenario['status']}] {scenario['group']}/{scenario['name']}")
        for artifact_name, artifact_path in sorted(scenario["artifacts"].items()):
            lines.append(f"  {artifact_name}: {artifact_path}")
        for note in scenario["notes"]:
            lines.append(f"  note: {note}")
    summary_path = run_folder / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def get_selected_scenarios(requested_scenarios: list[str] | None = None) -> list[str]:
    if not requested_scenarios:
        return list(SCENARIO_ORDER)

    unknown = sorted(set(requested_scenarios) - set(SCENARIO_ORDER))
    if unknown:
        raise ValueError(f"unknown scenarios requested: {', '.join(unknown)}")

    requested = set(requested_scenarios)
    return [name for name in SCENARIO_ORDER if name in requested]


def run_demo_suite(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    record: bool = False,
    fps: int = 8,
    audio_path: Path | None = None,
    scenarios: list[str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    run_id = timestamp_run_id()
    run_folder = create_run_folder(output_root, run_id)
    scenario_results: list[ScenarioResult] = []

    for scenario_name in get_selected_scenarios(scenarios):
        runner = SCENARIO_RUNNERS[scenario_name]
        scenario_results.extend(
            runner(
                run_folder=run_folder,
                record=record,
                fps=fps,
                audio_path=audio_path,
            )
        )

    manifest = build_manifest(run_id=run_id, run_folder=run_folder, scenario_results=scenario_results)
    summary_path = _write_summary(run_folder, manifest)
    manifest["summary_path"] = _relative_path(summary_path, run_folder)
    manifest_path = _write_json(run_folder / "manifest.json", manifest)
    return run_folder, manifest_path, manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoZero full demo suite runner")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--list-scenarios", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list_scenarios:
        for scenario_name in SCENARIO_ORDER:
            print(scenario_name)
        return 0

    try:
        run_folder, manifest_path, manifest = run_demo_suite(
            output_root=args.output_root,
            record=args.record,
            fps=args.fps,
            audio_path=args.audio_path,
            scenarios=args.scenario,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"run_folder={run_folder}")
    print(f"manifest={manifest_path}")
    print(f"passed={manifest['counts']['passed']}")
    print(f"skipped={manifest['counts']['skipped']}")
    print(f"missing={manifest['counts']['missing']}")
    return 0


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "SCENARIO_ORDER",
    "SCENARIO_RUNNERS",
    "ScenarioResult",
    "build_manifest",
    "collect_reference_artifacts",
    "create_run_folder",
    "find_latest_smoke_report",
    "get_selected_scenarios",
    "main",
    "run_demo_suite",
    "timestamp_run_id",
]


if __name__ == "__main__":
    raise SystemExit(main())
