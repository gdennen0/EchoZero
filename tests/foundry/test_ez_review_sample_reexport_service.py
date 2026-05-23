"""Tests for EZ bundle review sample re-export.
Exists because project truth should be able to rebuild shared training clips cleanly.
Connects .ez review metadata to the canonical review-sample folder contract.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from echozero.foundry.services.ez_review_sample_reexport_service import (
    EzReviewSampleReexportService,
)


def test_reexport_ez_review_samples_uses_bundled_audio_and_roles(tmp_path: Path):
    project_path = tmp_path / "project.ez"
    _write_project_bundle(project_path)

    result = EzReviewSampleReexportService().reexport(
        [project_path],
        output_root=tmp_path / "samples",
    )

    rows = _read_jsonl(result.manifest_path)
    assert result.report["exported_sample_count"] == 2
    assert result.report["counts_by_training_role"] == {"negative": 1, "positive": 1}
    assert sorted(row["target_label"] for row in rows) == ["other", "snare"]
    assert len(list((result.output_root / "positive" / "snare").glob("*.wav"))) == 1
    assert len(list((result.output_root / "negative" / "kick").glob("*.wav"))) == 1
    assert {row["source_audio_member"] for row in rows} == {"audio/generated/drums.wav"}
    assert {round(float(row["event_duration_seconds"]), 2) for row in rows} == {0.5}


def test_reexport_ez_review_samples_can_filter_labels(tmp_path: Path):
    project_path = tmp_path / "project.ez"
    _write_project_bundle(project_path)

    result = EzReviewSampleReexportService().reexport(
        [project_path],
        output_root=tmp_path / "samples",
        labels=("snare",),
    )

    rows = _read_jsonl(result.manifest_path)
    assert [row["class_label"] for row in rows] == ["snare"]
    assert len(list((result.output_root / "positive" / "snare").glob("*.wav"))) == 1
    assert not (result.output_root / "negative" / "kick").exists()


def test_reexport_ez_review_samples_can_include_promoted_events_with_tails(tmp_path: Path):
    project_path = tmp_path / "project.ez"
    _write_project_bundle(project_path)

    result = EzReviewSampleReexportService().reexport(
        [project_path],
        output_root=tmp_path / "samples",
        include_promoted_events=True,
    )

    rows = _read_jsonl(result.manifest_path)
    promoted = next(row for row in rows if row["event_id"] == "promoted_kick_event")
    assert result.report["exported_sample_count"] == 3
    assert result.report["include_promoted_events"] is True
    assert promoted["source_kind"] == "promoted_event"
    assert promoted["training_role"] == "positive"
    assert promoted["target_label"] == "kick"
    assert promoted["sample_window_policy"]["kind"] == "estimated_audio_tail"
    assert promoted["sample_window_policy"]["span_estimate"]["consensus_method"] == "agreement"
    assert set(promoted["sample_window_policy"]["span_estimate"]["method_durations"]) == {
        "relative_rms_decay",
        "relative_peak_decay",
        "cumulative_energy",
    }
    assert round(float(promoted["event_start_seconds"]), 2) == 3.0
    assert float(promoted["event_duration_seconds"]) > 0.08


def _write_project_bundle(path: Path) -> None:
    db_path = path.parent / "project.db"
    audio_path = path.parent / "drums.wav"
    _write_wave(audio_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            create table object_contents (
                id text primary key,
                object_id text not null,
                revision_id text not null,
                content_kind text not null,
                payload_json text not null default '{}',
                source_ref_json text,
                analysis_build_json text,
                created_at text not null
            );
            create table takes (
                id text primary key,
                layer_id text not null,
                label text not null,
                origin text not null,
                is_main integer not null default 0,
                is_archived integer not null default 0,
                source_json text,
                data_json text,
                created_at text not null,
                notes text default ''
            );
            """
        )
        connection.execute(
            """
            insert into object_contents
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "source_content",
                "source_object",
                "rev_source",
                "generated_audio",
                json.dumps({"audio_file": "audio/generated/drums.wav"}),
                None,
                None,
                "2026-05-22T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            insert into object_contents
            values (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "event_content",
                "event_object",
                "rev_event",
                "event_set",
                json.dumps({"take_id": "take_events", "event_count": 3}),
                json.dumps({"content_id": "source_content"}),
                None,
                "2026-05-22T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            insert into takes
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "take_events",
                "layer_events",
                "Take 1",
                "pipeline",
                1,
                0,
                None,
                json.dumps(
                    {
                        "type": "EventSet",
                        "layers": [
                            {
                                "id": "layer_snare",
                                "name": "Snare",
                                "events": [
                                    _event(
                                        "snare_event",
                                        "Snare",
                                        "verified",
                                        "correct",
                                        1.0,
                                        0.5,
                                    ),
                                    _event(
                                        "kick_event",
                                        "Kick",
                                        "rejected",
                                        "incorrect",
                                        2.0,
                                        0.5,
                                    ),
                                    _promoted_event(
                                        "promoted_kick_event",
                                        "Kick",
                                        3.0,
                                        0.08,
                                    ),
                                ],
                            }
                        ],
                    }
                ),
                "2026-05-22T00:00:00+00:00",
                "",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(db_path, "project.db")
        archive.write(audio_path, "audio/generated/drums.wav")


def _event(
    event_id: str,
    label: str,
    decision_kind: str,
    review_outcome: str,
    time_seconds: float,
    duration_seconds: float,
) -> dict[str, object]:
    return {
        "id": event_id,
        "time": time_seconds,
        "duration": duration_seconds,
        "classifications": {"class": label.lower(), "label": label},
        "metadata": {
            "review": {
                "schema": "echozero.event_review.v1",
                "review_outcome": review_outcome,
                "decision_kind": decision_kind,
                "original_label": label,
                "reviewed_at": "2026-05-22T00:00:00+00:00",
            }
        },
    }


def _promoted_event(
    event_id: str,
    label: str,
    time_seconds: float,
    duration_seconds: float,
) -> dict[str, object]:
    return {
        "id": event_id,
        "time": time_seconds,
        "duration": duration_seconds,
        "classifications": {"class": label.lower(), "label": label},
        "metadata": {
            "detection": {
                "schema": "echozero.event_detection.v1",
                "promotion_state": "promoted",
                "threshold_passed": True,
            }
        },
    }


def _write_wave(path: Path, *, sample_rate: int = 22050) -> None:
    times = np.linspace(0.0, 4.0, sample_rate * 4, endpoint=False)
    wave = 0.25 * np.sin(2.0 * np.pi * (180.0 * times + 13.0 * times * times))
    sf.write(path, wave.astype(np.float32), sample_rate)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
