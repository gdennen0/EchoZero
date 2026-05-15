"""Local find-similar mini-model artifacts.
Exists so a selected sound can save matched positives as a lightweight prototype.
Connects timbre fingerprints to app-managed JSON artifacts for later candidate scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.event_comparison_service import (
    TimbreFingerprintSettings,
    build_timbre_fingerprint_preview,
    compare_timbre_fingerprint_similarity,
)
from echozero.application.timeline.models import EventRef
from echozero.models.paths import ensure_installed_models_dir


SCHEMA = "echozero.find-similar-mini-model.v1"
MODEL_KIND = "timbre_prototype"
INDEX_FILENAME = "_index.json"
INDEX_SCHEMA = "echozero.find-similar-mini-model-index.v1"


@dataclass(frozen=True, slots=True)
class AudioEventTrainingSample:
    """One audio event slice available for local find-similar training or scoring."""

    event_ref: EventRef
    label: str
    audio_path: str | None
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True, slots=True)
class TimbreMiniModelResult:
    """Saved local prototype metadata for one find-similar mini-model."""

    artifact_path: Path
    positive_sample_count: int
    centroid: tuple[float, ...]
    anchor_event_ref: EventRef
    model_id: str


@dataclass(frozen=True, slots=True)
class TimbreMiniModelScore:
    """A candidate score from a saved local find-similar mini-model."""

    event_ref: EventRef
    score: float


@dataclass(frozen=True, slots=True)
class TimbreMiniModelRegistryEntry:
    """List/load metadata for one saved mini-model artifact."""

    model_id: str
    label: str
    artifact_path: Path
    created_at: str
    positive_sample_count: int
    anchor_event_ref: EventRef | None = None


def ensure_find_similar_models_dir() -> Path:
    """Create and return the app-managed local find-similar model directory."""

    path = ensure_installed_models_dir() / "find-similar"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_timbre_mini_models(*, models_dir: Path | None = None) -> tuple[TimbreMiniModelRegistryEntry, ...]:
    """Return saved local timbre mini-models, newest first.

    The registry is intentionally rebuildable from artifacts so a lost/corrupt index does not hide
    user-created local models.
    """

    root = _models_root(models_dir)
    entries = _scan_model_entries(root)
    _write_index(root, entries)
    return tuple(sorted(entries, key=lambda entry: entry.created_at, reverse=True))


def load_timbre_mini_model_by_id(
    identifier: str | Path,
    *,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a saved mini-model by registry id or artifact path."""

    return load_timbre_mini_model(resolve_timbre_mini_model(identifier, models_dir=models_dir))


def resolve_timbre_mini_model(identifier: str | Path, *, models_dir: Path | None = None) -> Path:
    """Resolve a registry id, filename, or path to a saved mini-model artifact path."""

    raw = Path(identifier) if isinstance(identifier, Path) else Path(str(identifier))
    if raw.exists():
        return raw
    root = _models_root(models_dir)
    candidate = root / raw.name
    if candidate.exists():
        return candidate
    wanted = str(identifier)
    for entry in list_timbre_mini_models(models_dir=root):
        if entry.model_id == wanted or entry.artifact_path.name == wanted:
            return entry.artifact_path
    raise FileNotFoundError(f"No saved timbre mini-model found for {identifier!r}")


def delete_timbre_mini_model(identifier: str | Path, *, models_dir: Path | None = None) -> bool:
    """Delete a saved mini-model artifact and refresh the registry index."""

    root = _models_root(models_dir)
    try:
        artifact_path = resolve_timbre_mini_model(identifier, models_dir=root)
    except FileNotFoundError:
        return False
    if not _is_within(artifact_path, root):
        raise ValueError("Refusing to delete a mini-model outside the find-similar model directory")
    artifact_path.unlink(missing_ok=True)
    _write_index(root, _scan_model_entries(root))
    return True


def train_timbre_mini_model(
    *,
    anchor_sample: AudioEventTrainingSample,
    positive_samples: Sequence[AudioEventTrainingSample],
    output_dir: Path | None = None,
    settings: TimbreFingerprintSettings | None = None,
    created_at: datetime | None = None,
) -> TimbreMiniModelResult:
    """Train and save a centroid timbre prototype from matched positive event samples."""

    resolved_settings = settings or TimbreFingerprintSettings(sample_count=64, padding_ms=20.0)
    samples = _dedupe_samples((anchor_sample, *tuple(positive_samples)))
    embeddings: list[tuple[AudioEventTrainingSample, tuple[float, ...]]] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    for sample in samples:
        embedding = _embedding_for_sample(sample, settings=resolved_settings, audio_cache=audio_cache)
        if embedding is None:
            continue
        embeddings.append((sample, embedding))
    if not embeddings:
        raise ValueError("No positive audio samples could be embedded for the mini-model")

    centroid = _centroid(tuple(embedding for _sample, embedding in embeddings))
    timestamp = created_at or datetime.now(timezone.utc)
    artifact_root = _models_root(output_dir)
    model_id = _model_id(anchor_sample, timestamp)
    artifact_path = _unique_artifact_path(artifact_root, anchor_sample, timestamp, model_id=model_id)
    payload = _artifact_payload(
        model_id=model_id,
        anchor_sample=anchor_sample,
        embeddings=tuple(embeddings),
        centroid=centroid,
        settings=resolved_settings,
        created_at=timestamp,
    )
    _atomic_write_json(artifact_path, payload)
    _write_index(artifact_root, _scan_model_entries(artifact_root))
    return TimbreMiniModelResult(
        artifact_path=artifact_path,
        positive_sample_count=len(embeddings),
        centroid=centroid,
        anchor_event_ref=anchor_sample.event_ref,
        model_id=model_id,
    )


def score_timbre_mini_model(
    *,
    artifact_path: Path,
    candidate_sample: AudioEventTrainingSample,
) -> TimbreMiniModelScore:
    """Score one candidate event against a saved centroid timbre prototype."""

    return score_timbre_mini_model_candidates(
        artifact_path=artifact_path,
        candidate_samples=(candidate_sample,),
    )[0]


def score_timbre_mini_model_candidates(
    *,
    artifact_path: Path,
    candidate_samples: Sequence[AudioEventTrainingSample],
) -> tuple[TimbreMiniModelScore, ...]:
    """Score candidate events against a saved centroid, reusing one slice cache."""

    payload = load_timbre_mini_model(artifact_path)
    settings = TimbreFingerprintSettings(
        sample_count=int(payload.get("settings", {}).get("sample_count", 64)),
        padding_ms=float(payload.get("settings", {}).get("padding_ms", 20.0)),
    )
    centroid = tuple(float(value) for value in payload["centroid"])
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    scores: list[TimbreMiniModelScore] = []
    for sample in candidate_samples:
        embedding = _embedding_for_sample(sample, settings=settings, audio_cache=audio_cache)
        score = 0.0 if embedding is None else compare_timbre_fingerprint_similarity(centroid, embedding)
        scores.append(TimbreMiniModelScore(event_ref=sample.event_ref, score=score))
    return tuple(scores)


def load_timbre_mini_model(artifact_path: Path) -> dict[str, Any]:
    """Load and validate one saved local find-similar timbre prototype artifact."""

    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported mini-model schema: {payload.get('schema')!r}")
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError(f"Unsupported mini-model kind: {payload.get('model_kind')!r}")
    centroid = payload.get("centroid")
    if not isinstance(centroid, list) or not centroid:
        raise ValueError("Mini-model artifact is missing a centroid")
    if not payload.get("model_id"):
        payload["model_id"] = _legacy_model_id(Path(artifact_path), payload)
    return payload


def _embedding_for_sample(
    sample: AudioEventTrainingSample,
    *,
    settings: TimbreFingerprintSettings,
    audio_cache: dict[str, tuple[np.ndarray, int]],
) -> tuple[float, ...] | None:
    return build_timbre_fingerprint_preview(
        audio_path=sample.audio_path,
        start_seconds=sample.start_seconds,
        end_seconds=max(sample.end_seconds, sample.start_seconds + 0.02),
        settings=settings,
        audio_cache=audio_cache,
    )


def _dedupe_samples(samples: Sequence[AudioEventTrainingSample]) -> tuple[AudioEventTrainingSample, ...]:
    deduped: list[AudioEventTrainingSample] = []
    seen: set[tuple[str, str, str]] = set()
    for sample in samples:
        key = (
            str(sample.event_ref.layer_id),
            str(sample.event_ref.take_id),
            str(sample.event_ref.event_id),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return tuple(deduped)


def _centroid(embeddings: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    max_size = max(len(embedding) for embedding in embeddings)
    rows = []
    for embedding in embeddings:
        row = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if row.size != max_size:
            row = np.interp(
                np.linspace(0.0, 1.0, max_size),
                np.linspace(0.0, 1.0, row.size),
                row,
            ).astype(np.float32)
        rows.append(row)
    centroid = np.mean(np.vstack(rows), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-9:
        centroid = centroid / norm
    return tuple(float(value) for value in centroid)


def _artifact_payload(
    *,
    model_id: str,
    anchor_sample: AudioEventTrainingSample,
    embeddings: Sequence[tuple[AudioEventTrainingSample, tuple[float, ...]]],
    centroid: tuple[float, ...],
    settings: TimbreFingerprintSettings,
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model_id": model_id,
        "model_kind": MODEL_KIND,
        "created_at": created_at.astimezone(timezone.utc).isoformat(),
        "anchor_event_ref": _event_ref_payload(anchor_sample.event_ref),
        "anchor_label": anchor_sample.label,
        "settings": {
            "sample_count": int(settings.sample_count),
            "padding_ms": float(settings.padding_ms),
        },
        "positive_sample_count": len(embeddings),
        "centroid": [float(value) for value in centroid],
        # Privacy note: we intentionally do not persist source audio paths or raw embeddings.
        # Stored sample metadata is only enough to describe/review what local events trained the prototype.
        "positive_samples": [
            {
                "event_ref": _event_ref_payload(sample.event_ref),
                "label": sample.label,
                "start_seconds": float(sample.start_seconds),
                "end_seconds": float(sample.end_seconds),
                "embedding_size": len(embedding),
            }
            for sample, embedding in embeddings
        ],
    }


def _event_ref_payload(event_ref: EventRef) -> dict[str, str]:
    return {
        "layer_id": str(event_ref.layer_id),
        "take_id": str(event_ref.take_id),
        "event_id": str(event_ref.event_id),
    }


def _event_ref_from_payload(payload: object) -> EventRef | None:
    if not isinstance(payload, dict):
        return None
    try:
        return EventRef(
            LayerId(str(payload["layer_id"])),
            TakeId(str(payload["take_id"])),
            EventId(str(payload["event_id"])),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _model_id(sample: AudioEventTrainingSample, timestamp: datetime) -> str:
    anchor_key = f"{sample.event_ref.layer_id}:{sample.event_ref.take_id}:{sample.event_ref.event_id}"
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return hashlib.sha1(f"{stamp}:{anchor_key}:{uuid4().hex}".encode("utf-8")).hexdigest()[:16]


def _legacy_model_id(path: Path, payload: dict[str, Any]) -> str:
    source = f"{path.name}:{payload.get('created_at', '')}:{payload.get('anchor_label', '')}"
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def _unique_artifact_path(
    root: Path,
    sample: AudioEventTrainingSample,
    timestamp: datetime,
    *,
    model_id: str,
) -> Path:
    label = _safe_slug(sample.label or str(sample.event_ref.event_id))
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    base = f"{stamp}-{label}-{model_id[:8]}"
    candidate = root / f"{base}.json"
    while candidate.exists():
        candidate = root / f"{base}-{uuid4().hex[:8]}.json"
    return candidate


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:36] or "sound"


def _models_root(models_dir: Path | None) -> Path:
    root = models_dir if models_dir is not None else ensure_find_similar_models_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _scan_model_entries(root: Path) -> tuple[TimbreMiniModelRegistryEntry, ...]:
    entries: list[TimbreMiniModelRegistryEntry] = []
    for artifact_path in root.glob("*.json"):
        if artifact_path.name == INDEX_FILENAME:
            continue
        try:
            payload = load_timbre_mini_model(artifact_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        entries.append(_entry_from_payload(artifact_path, payload))
    return tuple(entries)


def _entry_from_payload(artifact_path: Path, payload: dict[str, Any]) -> TimbreMiniModelRegistryEntry:
    return TimbreMiniModelRegistryEntry(
        model_id=str(payload.get("model_id") or _legacy_model_id(artifact_path, payload)),
        label=str(payload.get("anchor_label") or artifact_path.stem),
        artifact_path=artifact_path,
        created_at=str(payload.get("created_at") or ""),
        positive_sample_count=int(payload.get("positive_sample_count") or 0),
        anchor_event_ref=_event_ref_from_payload(payload.get("anchor_event_ref")),
    )


def _write_index(root: Path, entries: Sequence[TimbreMiniModelRegistryEntry]) -> None:
    payload = {
        "schema": INDEX_SCHEMA,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": [
            {
                "model_id": entry.model_id,
                "label": entry.label,
                "artifact_path": entry.artifact_path.name,
                "created_at": entry.created_at,
                "positive_sample_count": entry.positive_sample_count,
                "anchor_event_ref": (
                    _event_ref_payload(entry.anchor_event_ref) if entry.anchor_event_ref is not None else None
                ),
            }
            for entry in sorted(entries, key=lambda item: item.created_at, reverse=True)
        ],
    }
    _atomic_write_json(root / INDEX_FILENAME, payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


__all__ = [
    "AudioEventTrainingSample",
    "TimbreMiniModelRegistryEntry",
    "TimbreMiniModelResult",
    "TimbreMiniModelScore",
    "delete_timbre_mini_model",
    "ensure_find_similar_models_dir",
    "list_timbre_mini_models",
    "load_timbre_mini_model",
    "load_timbre_mini_model_by_id",
    "resolve_timbre_mini_model",
    "score_timbre_mini_model",
    "score_timbre_mini_model_candidates",
    "train_timbre_mini_model",
]
