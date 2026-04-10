from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .core import (
    EvalContract,
    EvalCore,
    EvalRequest,
    EvalResult,
    InferenceContract,
    InferenceCore,
    InferenceRequest,
    contract_fingerprint,
)


@dataclass(slots=True)
class FoundrySharedAdapter:
    """Foundry entrypoint for the shared inference/eval core contracts."""

    inference_core: InferenceCore | None = None
    eval_core: EvalCore | None = None

    def inference_contract_from_run_spec(
        self,
        run_spec: Mapping[str, Any],
        *,
        class_map: Sequence[str],
    ) -> InferenceContract:
        data = run_spec.get("data")
        preprocessing = dict(data) if isinstance(data, Mapping) else {}
        model = run_spec.get("model")
        model_signature = None
        if isinstance(model, Mapping):
            model_signature = str(model.get("type") or "") or None

        return InferenceContract(
            preprocessing=preprocessing,
            class_map=tuple(class_map),
            model_signature=model_signature,
        )

    def eval_contract_from_run_spec(self, run_spec: Mapping[str, Any]) -> EvalContract:
        classification_mode = str(run_spec.get("classificationMode", "multiclass"))
        return EvalContract(classification_mode=classification_mode, split_name="test")

    def build_inference_request(
        self,
        run_spec: Mapping[str, Any],
        *,
        class_map: Sequence[str],
        inputs: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any] | None = None,
    ) -> InferenceRequest:
        contract = self.inference_contract_from_run_spec(run_spec, class_map=class_map)
        return InferenceRequest(contract=contract, inputs=tuple(dict(item) for item in inputs), context=dict(context or {}))

    def to_eval_payload(self, request: EvalRequest, result: EvalResult) -> dict[str, Any]:
        """Shape payload to match current Foundry eval service/report expectations."""

        return {
            "run_id": request.run_id,
            "classification_mode": request.contract.classification_mode,
            "dataset_version_id": request.dataset_version_id,
            "split_name": request.contract.split_name,
            "metrics": dict(result.metrics),
            "aggregate_metrics": dict(result.aggregate_metrics),
            "per_class_metrics": dict(result.per_class_metrics),
            "baseline": dict(result.baseline),
            "threshold_policy": dict(result.threshold_policy) if result.threshold_policy is not None else None,
            "confusion": dict(result.confusion) if result.confusion is not None else None,
            "summary": dict(result.summary or request.summary),
        }

    def contract_fingerprint_from_run_spec(self, run_spec: Mapping[str, Any], *, class_map: Sequence[str]) -> str:
        inference_contract = self.inference_contract_from_run_spec(run_spec, class_map=class_map)
        eval_contract = self.eval_contract_from_run_spec(run_spec)
        return contract_fingerprint(inference_contract, eval_contract)


def create_foundry_adapter(
    *,
    inference_core: InferenceCore | None = None,
    eval_core: EvalCore | None = None,
) -> FoundrySharedAdapter:
    return FoundrySharedAdapter(inference_core=inference_core, eval_core=eval_core)
