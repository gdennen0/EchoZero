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


@dataclass(frozen=True, slots=True)
class TimbreMiniModelScore:
    """A candidate score from a saved local find-similar mini-model."""

    event_ref: EventRef
    score: float


def ensure_find_similar_models_dir() -> Path:
    """Create and return the app-managed local find-similar model directory."""

    path = ensure_installed_models_dir() / "find-similar"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    artifact_root = output_dir if output_dir is not None else ensure_find_similar_models_dir()
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / _artifact_filename(anchor_sample, timestamp)
    payload = _artifact_payload(
        anchor_sample=anchor_sample,
        embeddings=tuple(embeddings),
        centroid=centroid,
        settings=resolved_settings,
        created_at=timestamp,
    )
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TimbreMiniModelResult(
        artifact_path=artifact_path,
        positive_sample_count=len(embeddings),
        centroid=centroid,
        anchor_event_ref=anchor_sample.event_ref,
    )


def score_timbre_mini_model(
    *,
    artifact_path: Path,
    candidate_sample: AudioEventTrainingSample,
) -> TimbreMiniModelScore:
    """Score one candidate event against a saved centroid timbre prototype."""

    payload = load_timbre_mini_model(artifact_path)
    settings = TimbreFingerprintSettings(
        sample_count=int(payload.get("settings", {}).get("sample_count", 64)),
        padding_ms=float(payload.get("settings", {}).get("padding_ms", 20.0)),
    )
    embedding = _embedding_for_sample(candidate_sample, settings=settings, audio_cache={})
    score = 0.0 if embedding is None else compare_timbre_fingerprint_similarity(
        tuple(float(value) for value in payload["centroid"]),
        embedding,
    )
    return TimbreMiniModelScore(event_ref=candidate_sample.event_ref, score=score)


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
    anchor_sample: AudioEventTrainingSample,
    embeddings: Sequence[tuple[AudioEventTrainingSample, tuple[float, ...]]],
    centroid: tuple[float, ...],
    settings: TimbreFingerprintSettings,
    created_at: datetime,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
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


def _artifact_filename(sample: AudioEventTrainingSample, timestamp: datetime) -> str:
    anchor_key = f"{sample.event_ref.layer_id}:{sample.event_ref.take_id}:{sample.event_ref.event_id}"
    digest = hashlib.sha1(anchor_key.encode("utf-8")).hexdigest()[:10]
    label = _safe_slug(sample.label or str(sample.event_ref.event_id))
    stamp = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{label}-{digest}.json"


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:36] or "sound"


__all__ = [
    "AudioEventTrainingSample",
    "TimbreMiniModelResult",
    "TimbreMiniModelScore",
    "ensure_find_similar_models_dir",
    "load_timbre_mini_model",
    "score_timbre_mini_model",
    "train_timbre_mini_model",
]
