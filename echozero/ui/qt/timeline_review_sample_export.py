"""Timeline-review sample export helpers for the Qt app shell runtime.
Exists because timeline fix-mode corrections should materialize shareable class-folder clips by default.
Connects committed review signals to deterministic local sample exports under the machine default EchoZero dir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from echozero.foundry.domain.review import ReviewDecisionKind, ReviewSignal
from echozero.foundry.review_samples import (
    review_sample_label_dir,
    review_sample_target_label,
    review_sample_training_role,
)
from echozero.foundry.services.review_audio_clip_service import ReviewAudioClipService
from echozero.foundry.services.review_event_state import normalize_review_label

logger = logging.getLogger(__name__)

_REVIEW_EXPORT_ROOT_ENV = "ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT"


def export_timeline_review_sample(
    *,
    signal: ReviewSignal,
    class_label: str,
    source_audio_path: str,
    start_seconds: float,
    end_seconds: float,
    event_id: str,
    decision_kind: ReviewDecisionKind,
    clip_service: ReviewAudioClipService | None = None,
) -> dict[str, object]:
    """Materialize one committed timeline-review sample clip into its class folder."""

    normalized_class = normalize_review_label(class_label)
    export_root = _default_review_export_root()
    training_role = review_sample_training_role(decision_kind)
    class_dir = review_sample_label_dir(
        export_root,
        class_label=normalized_class,
        training_role=training_role,
    )
    source_audio = Path(source_audio_path).expanduser()
    if not source_audio.exists():
        return {
            "status": "skipped",
            "reason": "missing_source_audio",
            "source_audio_path": str(source_audio),
        }

    sample_start, sample_end, window_policy = _review_sample_window(
        class_label=normalized_class,
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
    )
    clip_path = (clip_service or ReviewAudioClipService()).materialize_event_clip(
        source_audio_path=source_audio,
        clip_cache_dir=class_dir,
        clip_stem=_clip_stem(
            signal_id=signal.id,
            event_id=str(event_id),
            decision_kind=decision_kind,
            class_label=normalized_class,
        ),
        start_seconds=sample_start,
        end_seconds=sample_end,
    )
    if clip_path is None:
        return {
            "status": "skipped",
            "reason": "clip_materialization_failed",
            "source_audio_path": str(source_audio),
            "event_start_seconds": float(start_seconds),
            "event_end_seconds": float(end_seconds),
            "start_seconds": sample_start,
            "end_seconds": sample_end,
        }

    source_manifest_path = _portable_manifest_path(source_audio, export_root=export_root)
    clip_manifest_path = _portable_manifest_path(clip_path, export_root=export_root)
    manifest_row = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "signal_id": signal.id,
        "item_id": signal.item_id,
        "event_id": str(event_id),
        "class_label": normalized_class,
        "training_role": training_role.value,
        "target_label": review_sample_target_label(
            class_label=normalized_class,
            training_role=training_role,
        ),
        "decision_kind": decision_kind.value,
        "review_outcome": signal.review_outcome.value,
        "source_audio_path": source_manifest_path,
        "clip_path": clip_manifest_path,
        "event_start_seconds": float(start_seconds),
        "event_end_seconds": float(end_seconds),
        "event_duration_seconds": max(0.0, float(end_seconds) - float(start_seconds)),
        "start_seconds": sample_start,
        "end_seconds": sample_end,
        "sample_duration_seconds": max(0.0, sample_end - sample_start),
        "sample_window_policy": window_policy,
        "export_contract": {
            "schema": "echozero.review_sample_export.v2",
            "layout": "<root>/<training_role>/<class_label>/<clip>.wav",
            "training_role_values": ["positive", "negative"],
            "negative_target_label": "other",
        },
    }
    _append_manifest_line(export_root / "manifest.jsonl", manifest_row)
    return {
        "status": "exported",
        "class_label": normalized_class,
        "clip_path": str(clip_path.resolve()),
        "manifest_path": str((export_root / "manifest.jsonl").resolve()),
        "event_start_seconds": float(start_seconds),
        "event_end_seconds": float(end_seconds),
        "start_seconds": sample_start,
        "end_seconds": sample_end,
        "sample_window_policy": window_policy,
    }


def safe_export_timeline_review_sample(
    *,
    signal: ReviewSignal,
    class_label: str,
    source_audio_path: str,
    start_seconds: float,
    end_seconds: float,
    event_id: str,
    decision_kind: ReviewDecisionKind,
    clip_service: ReviewAudioClipService | None = None,
) -> dict[str, object]:
    """Best-effort wrapper that never interrupts timeline review commit workflows."""

    try:
        return export_timeline_review_sample(
            signal=signal,
            class_label=class_label,
            source_audio_path=source_audio_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            event_id=event_id,
            decision_kind=decision_kind,
            clip_service=clip_service,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for runtime-only failures
        logger.warning("Timeline review sample export failed: %s", exc, exc_info=True)
        return {"status": "skipped", "reason": "export_error", "detail": str(exc)}


def review_sample_export_root(*, ensure_exists: bool = False) -> Path:
    """Return the canonical local folder used for timeline review sample exports."""

    root = _default_review_export_root()
    if ensure_exists:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _append_manifest_line(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True))
        stream.write("\n")


def _default_review_export_root() -> Path:
    explicit = os.getenv(_REVIEW_EXPORT_ROOT_ENV)
    if explicit:
        return Path(explicit).expanduser().resolve()
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return (Path(local_app_data) / "EchoZero" / "data" / "tmp" / "review_samples").resolve()
    return (Path.home() / ".echozero" / "data" / "tmp" / "review_samples").resolve()


def _review_sample_window(
    *,
    class_label: str,
    start_seconds: float,
    end_seconds: float,
) -> tuple[float, float, dict[str, object]]:
    event_start = max(0.0, float(start_seconds))
    event_end = max(event_start, float(end_seconds))
    return (
        event_start,
        event_end,
        {
            "schema": "echozero.review_sample_window.v1",
            "kind": "event_span",
            "anchor": "event_bounds",
            "class_label": normalize_review_label(class_label),
            "duration_seconds": max(0.0, event_end - event_start),
        },
    )


def _portable_manifest_path(path: Path, *, export_root: Path) -> str:
    """Return a manifest-safe path that does not require absolute machine paths."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(export_root).as_posix()
    except ValueError:
        return candidate.name


def _clip_stem(
    *,
    signal_id: str,
    event_id: str,
    decision_kind: ReviewDecisionKind,
    class_label: str,
) -> str:
    payload = f"{signal_id}|{event_id}|{decision_kind.value}|{class_label}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]
    return f"timeline_review_{digest}"
