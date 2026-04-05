from __future__ import annotations

from uuid import uuid4

from echozero.foundry.domain import EvalReport
from echozero.foundry.persistence import EvalReportRepository


class EvalService:
    def __init__(self, repository: EvalReportRepository):
        self._repo = repository

    def record_eval(
        self,
        run_id: str,
        *,
        classification_mode: str,
        metrics: dict,
        threshold_policy: dict | None = None,
        confusion: dict | None = None,
        summary: dict | None = None,
    ) -> EvalReport:
        report = EvalReport(
            id=f"evr_{uuid4().hex[:12]}",
            run_id=run_id,
            classification_mode=classification_mode,
            metrics=metrics,
            threshold_policy=threshold_policy,
            confusion=confusion,
            summary=summary or {},
        )
        return self._repo.save(report)

    @staticmethod
    def compute_threshold_policy(metrics: dict, *, target_metric: str = "f1") -> dict:
        candidates = metrics.get("threshold_candidates", [])
        if not candidates:
            return {"metric": target_metric, "threshold": 0.5, "source": "default"}

        best = max(candidates, key=lambda c: float(c.get(target_metric, 0.0)))
        return {
            "metric": target_metric,
            "threshold": float(best.get("threshold", 0.5)),
            "source": "candidates",
        }
