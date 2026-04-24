"""Run, artifact, and eval summary formatters for the Foundry window.
Exists to keep readable summary assembly out of the run action mixin.
Connects Foundry domain reports to the text shown in the run and artifact panels.
"""

from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry import FoundryApp
from echozero.foundry.domain import CompatibilityReport, EvalReport, TrainRun


class _FoundryWindowRunSummaryMixin:
    _root: Path
    _app: FoundryApp

    def _format_run_summary(self, run: TrainRun) -> str:
        checkpoints = sorted(run.checkpoints_dir(self._root).glob("epoch_*.json"))
        exports_dir = run.exports_dir(self._root)
        metrics_path = exports_dir / "metrics.json"
        run_summary_path = exports_dir / "run_summary.json"

        lines = [
            f"Run ID: {run.id}",
            f"Status: {run.status.value}",
            f"Dataset version: {run.dataset_version_id}",
            f"Backend / Device: {run.backend} / {run.device}",
            f"Epochs: {run.spec.get('training', {}).get('epochs')}",
            f"Batch size: {run.spec.get('training', {}).get('batchSize')}",
            f"Learning rate: {run.spec.get('training', {}).get('learningRate')}",
            f"Checkpoints: {len(checkpoints)}",
            f"Exports dir: {exports_dir}",
            f"metrics.json: {'yes' if metrics_path.exists() else 'no'}",
            f"run_summary.json: {'yes' if run_summary_path.exists() else 'no'}",
        ]
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            final_metrics = (metrics.get("finalEval") or {}).get("metrics", {})
            if final_metrics:
                lines.append(
                    "Final eval: "
                    f"macro_f1={final_metrics.get('macro_f1', 'n/a')} "
                    f"accuracy={final_metrics.get('accuracy', 'n/a')}"
                )
        return "\n".join(lines)

    def _format_artifact_summary(self, artifact_id: str | None) -> str:
        if not artifact_id:
            return "No artifact selected."
        artifact = self._app.get_artifact(artifact_id)
        if artifact is None:
            return f"Artifact not found: {artifact_id}"

        validation_note = "Run Validate Selected Artifact to check compatibility."
        diagnostic_lines: list[str] = []
        report: CompatibilityReport | None
        try:
            report = self._app.artifacts.validate_compatibility(artifact.id)
        except Exception:
            report = None
        if report is not None:
            validation_note = (
                f"Validation: ok={report.ok}, errors={len(report.errors)}, "
                f"warnings={len(report.warnings)}"
            )
            diagnostic_lines = self._format_compatibility_issue_lines(report)

        lines = [
            f"Artifact ID: {artifact.id}",
            f"Run ID: {artifact.run_id}",
            f"Manifest: {artifact.path}",
            f"Weights path: {artifact.manifest.get('weightsPath', 'n/a')}",
            f"Classes: {', '.join(artifact.manifest.get('classes', [])) or '(none)'}",
            f"Consumer: {artifact.consumer_hints.get('consumer', 'n/a')}",
            validation_note,
        ]
        if diagnostic_lines:
            lines.append("Diagnostics:")
            lines.extend(diagnostic_lines)
        return "\n".join(lines)

    def _format_compatibility_issue_lines(
        self,
        report: CompatibilityReport,
    ) -> list[str]:
        issues: list[tuple[str, object]] = []
        issues.extend(("error", detail) for detail in list(report.error_details))
        issues.extend(("warning", detail) for detail in list(report.warning_details))

        if not issues:
            issues.extend(("error", message) for message in list(report.errors))
            issues.extend(("warning", message) for message in list(report.warnings))

        lines: list[str] = []
        for default_severity, issue in issues:
            if isinstance(issue, dict):
                severity = str(issue.get("severity") or default_severity).upper()
                code = str(issue.get("code", ""))
                path = str(issue.get("path", ""))
                message = str(issue.get("message", ""))
            else:
                severity = default_severity.upper()
                code = ""
                path = ""
                message = str(issue)

            line = f"- [{severity}]"
            if code:
                line += f" {code}"
            if path:
                line += f" @ {path}"
            if message:
                line += f": {message}"
            lines.append(line)
        return lines

    def _format_eval_summary(self, report: EvalReport) -> str:
        metrics = report.metrics or report.aggregate_metrics or {}
        return "\n".join(
            [
                f"Eval ID: {report.id}",
                f"Run ID: {report.run_id}",
                f"Split: {report.split_name}",
                f"macro_f1: {metrics.get('macro_f1', 'n/a')}",
                f"accuracy: {metrics.get('accuracy', 'n/a')}",
                f"Summary keys: {', '.join(sorted(report.summary.keys())) or '(none)'}",
            ]
        )
