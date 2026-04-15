from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Mapping

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None


class TrainingNumericsError(RuntimeError):
    """Raised when non-finite values are detected during training/eval."""


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> dict[str, object]:
    """Apply reproducibility settings and return the effective configuration."""
    random.seed(seed)
    np.random.seed(seed)

    torch_deterministic_applied = False
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch_deterministic_applied = True
            except Exception:
                torch_deterministic_applied = False
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        else:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = False

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "torchDeterministicApplied": torch_deterministic_applied,
    }


def compute_config_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_finite_array(name: str, values: np.ndarray, *, context: str | None = None) -> None:
    arr = np.asarray(values)
    if arr.size == 0:
        return
    mask = ~np.isfinite(arr)
    if not np.any(mask):
        return
    bad_index = tuple(int(i) for i in np.argwhere(mask)[0].tolist())
    bad_value = arr[bad_index]
    where = f" ({context})" if context else ""
    raise TrainingNumericsError(
        f"non-finite value detected in {name}{where}: index={bad_index}, value={bad_value!r}"
    )


def ensure_finite_tensor(name: str, value: "torch.Tensor", *, context: str | None = None) -> None:
    if torch is None:
        return
    if value.numel() == 0:
        return
    finite = torch.isfinite(value)
    if bool(torch.all(finite).item()):
        return

    bad_index = tuple(int(i) for i in torch.nonzero(~finite, as_tuple=False)[0].tolist())
    bad_value = float(value[bad_index].detach().cpu().item())
    where = f" ({context})" if context else ""
    raise TrainingNumericsError(
        f"non-finite value detected in {name}{where}: index={bad_index}, value={bad_value!r}"
    )
