"""Runtime classifier model catalog helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassifierModelCandidate:
    label: str
    manifest_path: Path
    weights_path: Path | None
    bundle_dir: Path
    display_name: str
    family: str = "binary_drum"
    model_instance: str = ""
    compatibility_status: str = "ready"
    is_compatible: bool = True
    is_current_default: bool = False
    eval_score: float | None = None
    eval_summary: str | None = None
    created_at: str | None = None
    released_at: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeClassifierModelCatalog:
    candidates: tuple[ClassifierModelCandidate, ...]

    def candidates_for_label(self, label: str) -> tuple[ClassifierModelCandidate, ...]:
        normalized = _normalize_label(label)
        return tuple(candidate for candidate in self.candidates if candidate.label == normalized)

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({candidate.label for candidate in self.candidates}))


def build_runtime_classifier_model_catalog(*, models_dir: Path | None = None) -> RuntimeClassifierModelCatalog:
    root = Path(models_dir) if models_dir is not None else Path("models")
    candidates: list[ClassifierModelCandidate] = []
    if root.exists():
        for manifest_path in sorted(root.glob("*/*.manifest.json")):
            candidate = describe_binary_drum_manifest_candidate(manifest_path)
            if candidate is not None:
                candidates.append(candidate)
    return RuntimeClassifierModelCatalog(tuple(candidates))


def describe_binary_drum_manifest_candidate(
    manifest_path: Path,
    *,
    label: str | None = None,
) -> ClassifierModelCandidate | None:
    payload = _read_manifest(manifest_path)
    if payload is None:
        return None
    resolved_label = _manifest_binary_label(payload)
    if resolved_label is None:
        return None
    if label is not None and resolved_label != _normalize_label(label):
        return None
    weights_path = _resolve_weights_path(manifest_path, payload.get("weightsPath"))
    compatible = weights_path is not None and weights_path.exists()
    status = "ready" if compatible else "missing_weights"
    display_name = _display_name(payload, manifest_path, resolved_label)
    eval_score, eval_summary = _eval_summary(payload)
    return ClassifierModelCandidate(
        label=resolved_label,
        manifest_path=manifest_path.resolve(),
        weights_path=weights_path.resolve() if weights_path is not None else None,
        bundle_dir=manifest_path.parent.resolve(),
        display_name=display_name,
        family=str(payload.get("family") or payload.get("modelFamily") or "binary_drum"),
        model_instance=str(payload.get("modelInstance") or payload.get("model_instance") or manifest_path.parent.name),
        compatibility_status=status,
        is_compatible=compatible,
        is_current_default=bool(payload.get("isCurrentDefault") or payload.get("currentDefault")),
        eval_score=eval_score,
        eval_summary=eval_summary,
        created_at=_string_or_none(payload.get("createdAt") or payload.get("created_at")),
        released_at=_string_or_none(payload.get("releasedAt") or payload.get("released_at")),
    )


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_binary_label(manifest: dict[str, Any]) -> str | None:
    classes = manifest.get("classes")
    if not isinstance(classes, list):
        return None
    normalized = tuple(_normalize_label(value) for value in classes)
    if len(normalized) != 2 or "other" not in normalized:
        return None
    for item in normalized:
        if item != "other":
            return item
    return None


def _normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    return label


def _resolve_weights_path(manifest_path: Path, raw: object) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    return path if path.is_absolute() else manifest_path.parent / path


def _display_name(payload: dict[str, Any], manifest_path: Path, label: str) -> str:
    for key in ("displayName", "display_name", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{label.title()} · {manifest_path.parent.name}"


def _eval_summary(payload: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in ("eval", "evaluation", "metrics"):
        value = payload.get(key)
        if isinstance(value, dict):
            score = value.get("f1") or value.get("accuracy") or value.get("score")
            try:
                numeric = float(score) if score is not None else None
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None:
                return numeric, f"Eval {numeric:.2f}"
    return None, None


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ClassifierModelCandidate",
    "RuntimeClassifierModelCatalog",
    "build_runtime_classifier_model_catalog",
    "describe_binary_drum_manifest_candidate",
]
