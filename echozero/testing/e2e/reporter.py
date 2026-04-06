"""Structured report emission for E2E runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .runner import RunResult


def write_run_report(run_result: RunResult, artifacts_dir: str | Path) -> tuple[Path, Path]:
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "run-summary.json"
    markdown_path = root / "run-summary.md"
    json_path.write_text(json.dumps(asdict(run_result), indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown_summary(run_result), encoding="utf-8")
    return json_path, markdown_path


def _markdown_summary(run_result: RunResult) -> str:
    lines = [
        f"# E2E Run: {run_result.scenario_name}",
        "",
        f"- Status: {'PASS' if run_result.passed else 'FAIL'}",
        f"- Driver: {run_result.driver_name}",
        f"- Duration Ms: {run_result.duration_ms}",
        f"- Artifacts Dir: {run_result.artifacts_dir}",
        "",
        "| Step | Kind | Status | Duration Ms | Error |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for step in run_result.steps:
        error = (step.error or "").replace("\n", " ").replace("|", "/")
        lines.append(
            f"| {step.name} | {step.kind} | {'PASS' if step.passed else 'FAIL'} | "
            f"{step.duration_ms} | {error} |"
        )
    lines.append("")
    return "\n".join(lines)
