"""
Classifier model catalog for installed runtime bundles.
Exists so operator-facing model pickers can reason about model family, label compatibility, and readiness.
Connects Foundry-installed manifests to timeline settings without exposing raw filesystem scans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .paths import ensure_installed_models_dir
from .runtime_bundle_index import load_binary_drum_bundle_index

_BINARY_DRUM_FAMILY = "binary_drum"


@dataclass(frozen=True, slots=True)
class ClassifierModelCandidate:
    """One installed classifier model candidate with compatibility/readiness metadata."""

    family: str
    label: str
    model_instance: str
    manifest_path: Path
    weights_path: Path | None
    display_name: str
    created_at: str | None = None
    released_at: str | None = None
    eval_summary: str = ""
    eval_score: float | None = None
    is_current_default: bool = False
    compatibility_status: str = "ready"

    @property
    def is_compatible(self) -> bool:
        """Return whether this candidate can be selected for its output label."""

        return self.compatibility_status == "ready"


@dataclass(frozen=True, slots=True)
class RuntimeClassifierModelCatalog:
    """Searchable catalog of installed classifier model candidates."""

    candidates: tuple[ClassifierModelCandidate, ...]

    def candidates_for_label(
        self,
        label: str,
        *,
        family: str = _BINARY_DRUM_FAMILY,
        compatible_only: bool = True,
    ) -> tuple[ClassifierModelCandidate, ...]:
        """Return candidates compatible with one classifier family and output label."""

        normalized_label = _normalize_label(label)
        matches = tuple(
            candidate
            for candidate in self.candidates
            if candidate.family == family and candidate.label == normalized_label
        )
        if compatible_only:
            matches = tuple(candidate for candidate in matches if candidate.is_compatible)
        return tuple(sorted(matches, key=_candidate_sort_key))

    def labels(self, *, family: str = _BINARY_DRUM_FAMILY) -> tuple[str, ...]:
        """Return output labels that have at least one ready candidate."""

        labels = {
            candidate.label
            for candidate in self.candidates
            if candidate.family == family and candidate.is_compatible
        }
        return tuple(sorted(labels))


def build_runtime_classifier_model_catalog(
    *,
    models_dir: Path | None = None,
) -> RuntimeClassifierModelCatalog:
    """Discover installed runtime classifier manifests and group them as model candidates."""

    root = (models_dir or ensure_installed_models_dir()).resolve()
    current_defaults = _current_default_manifest_paths(root)
    candidates: list[ClassifierModelCandidate] = []
    seen: set[Path] = set()
    for manifest_path in sorted(root.rglob("*.manifest.json")):
        resolved_manifest = manifest_path.resolve()
        if resolved_manifest in seen:
            continue
        seen.add(resolved_manifest)
        candidate = describe_binary_drum_manifest_candidate(
            resolved_manifest,
            current_default_manifest_paths=current_defaults,
        )
        if candidate is not None:
            candidates.append(candidate)
    return RuntimeClassifierModelCatalog(tuple(candidates))


def describe_binary_drum_manifest_candidate(
    manifest_path: Path,
    *,
    label: str | None = None,
    current_default_manifest_paths: dict[str, Path] | None = None,
) -> ClassifierModelCandidate | None:
    """Describe one manifest as a binary-drum classifier candidate when possible."""

    resolved_manifest = manifest_path.expanduser().resolve()
    manifest = _load_json_object(resolved_manifest)
    if manifest is None:
        return None
    resolved_label = _manifest_binary_label(manifest)
    if resolved_label is None:
        return None
    requested_label = _normalize_label(label) if label is not None else None
    if requested_label is not None and resolved_label != requested_label:
        return None
    weights_path = _resolve_weights_path(resolved_manifest, manifest.get("weightsPath"))
    status = "ready" if weights_path is not None and weights_path.exists() else "missing_weights"
    defaults = current_default_manifest_paths or {}
    return ClassifierModelCandidate(
        family=_BINARY_DRUM_FAMILY,
        label=resolved_label,
        model_instance=_model_instance_name(resolved_manifest),
        manifest_path=resolved_manifest,
        weights_path=weights_path.resolve() if weights_path is not None else None,
        display_name=_display_name(resolved_manifest, manifest),
        created_at=_manifest_date(manifest, ("createdAt", "created_at")),
        released_at=_manifest_date(
            manifest,
            ("releasedAt", "released_at", "releaseDate", "release_date"),
        ),
        eval_summary=_eval_summary(resolved_manifest, manifest),
        eval_score=_eval_score(resolved_manifest, manifest),
        is_current_default=defaults.get(resolved_label) == resolved_manifest,
        compatibility_status=status,
    )


def binary_drum_family_name() -> str:
    """Return the stable classifier-family key for one-vs-rest drum models."""

    return _BINARY_DRUM_FAMILY


def _current_default_manifest_paths(root: Path) -> dict[str, Path]:
    defaults: dict[str, Path] = {}
    for raw_label, record in load_binary_drum_bundle_index(root).items():
        bundle_dir = (root / record.bundle_dir).resolve()
        manifest_path = (bundle_dir / record.manifest_file).resolve()
        if manifest_path.exists():
            defaults[_normalize_label(raw_label)] = manifest_path
    return defaults


def _candidate_sort_key(candidate: ClassifierModelCandidate) -> tuple[object, ...]:
    score = candidate.eval_score if candidate.eval_score is not None else -1.0
    date = candidate.released_at or candidate.created_at or ""
    return (
        0 if candidate.is_current_default else 1,
        0 if candidate.is_compatible else 1,
        -score,
        _reverse_iso_date_key(date),
        candidate.display_name.lower(),
        str(candidate.manifest_path),
    )


def _reverse_iso_date_key(value: str) -> str:
    if not value:
        return "9999-99-99"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return "9999-99-99"
    return f"{9999 - parsed.year:04d}-{12 - parsed.month:02d}-{31 - parsed.day:02d}"


def _manifest_binary_label(manifest: dict[str, object]) -> str | None:
    classes = manifest.get("classes")
    if not isinstance(classes, list):
        return None
    normalized = tuple(_normalize_label(value) for value in classes)
    if len(normalized) != 2 or "other" not in normalized:
        return None
    labels = tuple(value for value in normalized if value and value != "other")
    if len(labels) != 1:
        return None
    return labels[0]


def _resolve_weights_path(manifest_path: Path, raw_weights_path: object) -> Path | None:
    if not isinstance(raw_weights_path, str) or not raw_weights_path.strip():
        return None
    weights_path = Path(raw_weights_path).expanduser()
    if weights_path.is_absolute():
        return weights_path
    return manifest_path.parent / weights_path


def _display_name(manifest_path: Path, manifest: dict[str, object]) -> str:
    for key in ("displayName", "display_name", "name", "modelName", "model_name"):
        value = manifest.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    bundle_name = manifest_path.parent.name
    if bundle_name and bundle_name != ".":
        return bundle_name.replace("-", " ").replace("_", " ").title()
    return _model_instance_name(manifest_path).replace("-", " ").replace("_", " ").title()


def _model_instance_name(manifest_path: Path) -> str:
    name = manifest_path.name
    if name.endswith(".manifest.json"):
        return name[: -len(".manifest.json")]
    return manifest_path.stem


def _manifest_date(manifest: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        normalized = _normalize_date(manifest.get(key))
        if normalized is not None:
            return normalized
    training_summary = manifest.get("trainingSummary")
    if isinstance(training_summary, dict):
        for key in keys:
            normalized = _normalize_date(training_summary.get(key))
            if normalized is not None:
                return normalized
    return None


def _normalize_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    if len(text) >= 10:
        try:
            return datetime.fromisoformat(text[:10]).date().isoformat()
        except ValueError:
            return None
    return None


def _eval_summary(manifest_path: Path, manifest: dict[str, object]) -> str:
    metrics = _eval_metrics(manifest_path, manifest)
    if not metrics:
        return ""
    parts: list[str] = []
    for key, label in (("macro_f1", "F1"), ("f1", "F1"), ("accuracy", "Accuracy")):
        value = _float_metric(metrics.get(key))
        if value is not None:
            parts.append(f"{label} {value:.2f}")
        if parts and label == "F1":
            break
    return " · ".join(parts[:2])


def _eval_score(manifest_path: Path, manifest: dict[str, object]) -> float | None:
    metrics = _eval_metrics(manifest_path, manifest)
    for key in ("macro_f1", "f1", "accuracy"):
        value = _float_metric(metrics.get(key))
        if value is not None:
            return value
    return None


def _eval_metrics(manifest_path: Path, manifest: dict[str, object]) -> dict[str, object]:
    eval_summary = manifest.get("evalSummary")
    if isinstance(eval_summary, dict):
        normalized = _normalize_metrics_keys(eval_summary)
        if normalized:
            return normalized
    training_summary = manifest.get("trainingSummary")
    if isinstance(training_summary, dict):
        embedded = training_summary.get("metrics")
        if isinstance(embedded, dict):
            normalized = _normalize_metrics_keys(embedded)
            if normalized:
                return normalized
    metrics_path = manifest_path.parent / "metrics.json"
    metrics_payload = _load_json_object(metrics_path)
    if metrics_payload is None:
        return {}
    final_eval = metrics_payload.get("finalEval")
    if isinstance(final_eval, dict):
        metrics = final_eval.get("metrics")
        if isinstance(metrics, dict):
            return _normalize_metrics_keys(metrics)
    metrics = metrics_payload.get("metrics")
    if isinstance(metrics, dict):
        return _normalize_metrics_keys(metrics)
    return _normalize_metrics_keys(metrics_payload)


def _normalize_metrics_keys(metrics: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metrics.items():
        metric_key = str(key).strip()
        normalized[metric_key] = value
        snake_key = metric_key.replace("MacroF1", "macro_f1").replace("macroF1", "macro_f1")
        snake_key = snake_key.replace("Accuracy", "accuracy").replace("F1", "f1")
        normalized[snake_key] = value
    return normalized


def _float_metric(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_label(value: object) -> str:
    label = str(value or "").strip().lower()
    if label in {"symbol", "cymbol"}:
        return "cymbal"
    return label
