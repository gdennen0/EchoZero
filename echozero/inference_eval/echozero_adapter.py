from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .core import EvalCore, InferenceContract, InferenceCore, contract_fingerprint


@dataclass(slots=True)
class EchoZeroSharedAdapter:
    """EchoZero app entrypoint for shared inference/eval contracts."""

    inference_core: InferenceCore | None = None
    eval_core: EvalCore | None = None

    def inference_contract_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        class_map: Sequence[str] | None = None,
        model_signature: str | None = None,
    ) -> InferenceContract:
        preprocessing = checkpoint.get("inference_preprocessing")
        if not isinstance(preprocessing, Mapping):
            preprocessing = {}

        resolved_class_map: tuple[str, ...]
        if class_map is not None:
            resolved_class_map = tuple(class_map)
        else:
            classes = checkpoint.get("classes")
            if isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)):
                resolved_class_map = tuple(str(label) for label in classes)
            else:
                resolved_class_map = tuple()

        signature = model_signature or str(checkpoint.get("model_type") or "") or None
        return InferenceContract(
            preprocessing=dict(preprocessing),
            class_map=resolved_class_map,
            model_signature=signature,
        )

    @staticmethod
    def runtime_summary_payload(*, model_id: str, fingerprint: str, prediction_count: int) -> dict[str, Any]:
        return {
            "modelId": model_id,
            "sharedContractFingerprint": fingerprint,
            "predictionCount": int(prediction_count),
        }

    def contract_fingerprint_from_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        class_map: Sequence[str] | None = None,
        model_signature: str | None = None,
    ) -> str:
        inference_contract = self.inference_contract_from_checkpoint(
            checkpoint,
            class_map=class_map,
            model_signature=model_signature,
        )
        # EchoZero runtime-only path uses default eval scaffold until explicit eval lane is wired.
        from .core import EvalContract

        eval_contract = EvalContract()
        return contract_fingerprint(inference_contract, eval_contract)


def create_echozero_adapter(
    *,
    inference_core: InferenceCore | None = None,
    eval_core: EvalCore | None = None,
) -> EchoZeroSharedAdapter:
    return EchoZeroSharedAdapter(inference_core=inference_core, eval_core=eval_core)
