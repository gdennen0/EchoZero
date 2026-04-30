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
) -> dict[str, object]:
    """Materialize one committed timeline-review sample clip into its class folder."""

    normalized_class = normalize_review_label(class_label)
    export_root = _default_review_export_root()
    class_dir = export_root / _safe_segment(normalized_class)
    source_audio = Path(source_audio_path).expanduser()
    if not source_audio.exists():
        return {
            "status": "skipped",
            "reason": "missing_source_audio",
            "source_audio_path": str(source_audio),
        }

    clip_path = ReviewAudioClipService().materialize_event_clip(
        source_audio_path=source_audio,
        clip_cache_dir=class_dir,
        clip_stem=_clip_stem(
            signal_id=signal.id,
            event_id=str(event_id),
            decision_kind=decision_kind,
            class_label=normalized_class,
        ),
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
    )
    if clip_path is None:
        return {
            "status": "skipped",
            "reason": "clip_materialization_failed",
            "source_audio_path": str(source_audio),
            "start_seconds": float(start_seconds),
            "end_seconds": float(end_seconds),
        }

    manifest_row = {
        "ts_utc": datetime.now(UTC).isoformat(),
        "signal_id": signal.id,
        "item_id": signal.item_id,
        "event_id": str(event_id),
        "class_label": normalized_class,
        "decision_kind": decision_kind.value,
        "review_outcome": signal.review_outcome.value,
        "source_audio_path": str(source_audio.resolve()),
        "clip_path": str(clip_path.resolve()),
        "start_seconds": float(start_seconds),
        "end_seconds": float(end_seconds),
    }
    _append_manifest_line(export_root / "manifest.jsonl", manifest_row)
    return {
        "status": "exported",
        "class_label": normalized_class,
        "clip_path": str(clip_path.resolve()),
        "manifest_path": str((export_root / "manifest.jsonl").resolve()),
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


def _safe_segment(value: str) -> str:
    text = str(value).strip() or "event"
    safe = "".join(character if character.isalnum() else "_" for character in text)
    safe = safe.strip("_")
    return safe or "event"


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
