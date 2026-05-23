"""Tests for timeline review sample export windows.
Exists because training samples must preserve reviewed Event start/end bounds.
Connects timeline review commits to canonical shared review sample exports.
"""

from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.domain.review import (
    ReviewDecisionKind,
    ReviewOutcome,
    ReviewPolarity,
    ReviewSignal,
)
from echozero.ui.qt.timeline_review_sample_export import export_timeline_review_sample


def test_timeline_review_export_uses_event_span_for_drum_samples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "drums.wav"
    source.write_bytes(b"fixture")
    export_root = tmp_path / "review_samples"
    monkeypatch.setenv("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", str(export_root))
    clip_service = _FakeClipService()

    result = export_timeline_review_sample(
        signal=ReviewSignal(
            id="sig_1",
            session_id="session",
            item_id="item",
            audio_path=str(source),
            predicted_label="kick",
            target_class="kick",
            polarity=ReviewPolarity.POSITIVE,
            review_outcome=ReviewOutcome.CORRECT,
        ),
        class_label="kick",
        source_audio_path=str(source),
        start_seconds=12.0,
        end_seconds=12.23,
        event_id="evt_1",
        decision_kind=ReviewDecisionKind.VERIFIED,
        clip_service=clip_service,
    )

    assert result["status"] == "exported"
    assert clip_service.start_seconds == 12.0
    assert clip_service.end_seconds == 12.23
    assert result["event_start_seconds"] == 12.0
    assert result["event_end_seconds"] == 12.23
    assert result["start_seconds"] == 12.0
    assert result["end_seconds"] == 12.23
    row = json.loads((export_root / "manifest.jsonl").read_text(encoding="utf-8"))
    assert round(row["event_duration_seconds"], 2) == 0.23
    assert round(row["sample_duration_seconds"], 2) == 0.23
    assert row["sample_window_policy"]["kind"] == "event_span"
    assert row["training_role"] == "positive"
    assert row["target_label"] == "kick"
    assert row["export_contract"]["schema"] == "echozero.review_sample_export.v2"
    assert row["clip_path"].startswith("positive/kick/")


def test_timeline_review_export_routes_rejected_samples_to_negative_role(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "drums.wav"
    source.write_bytes(b"fixture")
    export_root = tmp_path / "review_samples"
    monkeypatch.setenv("ECHOZERO_REVIEW_SAMPLE_EXPORT_ROOT", str(export_root))

    result = export_timeline_review_sample(
        signal=ReviewSignal(
            id="sig_2",
            session_id="session",
            item_id="item",
            audio_path=str(source),
            predicted_label="kick",
            target_class="kick",
            polarity=ReviewPolarity.NEGATIVE,
            review_outcome=ReviewOutcome.INCORRECT,
        ),
        class_label="kick",
        source_audio_path=str(source),
        start_seconds=12.0,
        end_seconds=12.08,
        event_id="evt_2",
        decision_kind=ReviewDecisionKind.REJECTED,
        clip_service=_FakeClipService(),
    )

    row = json.loads((export_root / "manifest.jsonl").read_text(encoding="utf-8"))
    assert result["status"] == "exported"
    assert row["training_role"] == "negative"
    assert row["target_label"] == "other"
    assert row["clip_path"].startswith("negative/kick/")
    assert (export_root / row["clip_path"]).exists()


class _FakeClipService:
    start_seconds: float | None = None
    end_seconds: float | None = None

    def materialize_event_clip(
        self,
        *,
        source_audio_path: Path,
        clip_cache_dir: Path,
        clip_stem: str,
        start_seconds: float,
        end_seconds: float,
    ) -> Path:
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        clip_cache_dir.mkdir(parents=True, exist_ok=True)
        path = clip_cache_dir / f"{clip_stem}.wav"
        path.write_bytes(b"clip")
        return path
