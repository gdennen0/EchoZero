"""Tests for the shared review sample doctor.
Exists because contaminated review exports can silently poison specialized drum training.
Connects Foundry repair reports to clean pools consumed by shared-review model workflows.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from echozero.foundry.services.review_sample_doctor_service import ReviewSampleDoctorService


def test_review_sample_doctor_routes_rejected_samples_to_other(tmp_path: Path):
    source_root = tmp_path / "review_samples"
    _write_wave(source_root / "kick" / "k1.wav", frequency=70.0)
    _write_wave(source_root / "kick" / "k2.wav", frequency=75.0)
    _write_manifest(
        source_root,
        [
            {"clip_path": "kick/k1.wav", "class_label": "kick", "decision_kind": "verified"},
            {"clip_path": "kick/k2.wav", "class_label": "kick", "decision_kind": "rejected"},
        ],
    )

    result = ReviewSampleDoctorService().audit_and_repair(
        source_root,
        output_root=tmp_path / "doctor",
        labels=("kick",),
    )

    clean_rows = _read_jsonl(result.clean_root / "manifest.jsonl")
    assert result.report["clean_sample_count"] == 2
    assert result.report["counts_by_target_label"] == {"kick": 1, "other": 1}
    assert result.report["counts_by_training_role"] == {"negative": 1, "positive": 1}
    assert sorted(row["target_label"] for row in clean_rows) == ["kick", "other"]
    assert sorted(row["training_role"] for row in clean_rows) == ["negative", "positive"]
    assert any(str(row["clip_path"]).startswith("negative/kick/") for row in clean_rows)
    assert any(str(row["clip_path"]).startswith("positive/kick/") for row in clean_rows)
    assert len(list((result.clean_root / "positive" / "kick").glob("*.wav"))) == 1
    assert len(list((result.clean_root / "negative" / "kick").glob("*.wav"))) == 1


def test_review_sample_doctor_quarantines_conflicting_content(tmp_path: Path):
    source_root = tmp_path / "review_samples"
    _write_wave(source_root / "kick" / "k1.wav", frequency=70.0)
    (source_root / "snare").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "kick" / "k1.wav", source_root / "snare" / "s1.wav")
    _write_wave(source_root / "snare" / "s2.wav", frequency=220.0)
    _write_manifest(
        source_root,
        [
            {"clip_path": "kick/k1.wav", "class_label": "kick", "decision_kind": "rejected"},
            {"clip_path": "snare/s1.wav", "class_label": "snare", "decision_kind": "verified"},
            {"clip_path": "snare/s2.wav", "class_label": "snare", "decision_kind": "verified"},
        ],
    )

    result = ReviewSampleDoctorService().audit_and_repair(
        source_root,
        output_root=tmp_path / "doctor",
        labels=("kick", "snare"),
    )

    quarantine_rows = _read_jsonl(result.quarantine_root / "manifest.jsonl")
    assert result.report["conflicting_content_group_count"] == 1
    assert result.report["quarantine_reason_counts"] == {"conflicting_content": 2}
    assert result.report["clean_sample_count"] == 1
    assert [row["source_clip_path"] for row in quarantine_rows] == [
        "kick/k1.wav",
        "snare/s1.wav",
    ]
    assert {row["target_label"] for row in quarantine_rows} == {"other", "snare"}
    assert len(list((result.clean_root / "positive" / "snare").glob("*.wav"))) == 1


def test_review_sample_doctor_can_recover_latest_review_conflict(tmp_path: Path):
    source_root = tmp_path / "review_samples"
    _write_wave(source_root / "snare" / "s1.wav", frequency=220.0)
    (source_root / "kick").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / "snare" / "s1.wav", source_root / "kick" / "k1.wav")
    _write_manifest(
        source_root,
        [
            {
                "clip_path": "snare/s1.wav",
                "class_label": "snare",
                "decision_kind": "verified",
                "reviewed_at": "2026-05-18T10:00:00+00:00",
            },
            {
                "clip_path": "kick/k1.wav",
                "class_label": "kick",
                "decision_kind": "rejected",
                "reviewed_at": "2026-05-18T11:00:00+00:00",
            },
        ],
    )

    result = ReviewSampleDoctorService().audit_and_repair(
        source_root,
        output_root=tmp_path / "doctor",
        labels=("kick", "snare"),
        conflict_policy="latest_review_wins",
    )

    clean_rows = _read_jsonl(result.clean_root / "manifest.jsonl")
    quarantine_rows = _read_jsonl(result.quarantine_root / "manifest.jsonl")
    assert result.report["conflict_policy"] == "latest_review_wins"
    assert result.report["action_counts"] == {"clean": 1, "quarantine": 1}
    assert clean_rows[0]["reason"] == "latest_review_wins"
    assert clean_rows[0]["source_clip_path"] == "kick/k1.wav"
    assert clean_rows[0]["target_label"] == "other"
    assert quarantine_rows[0]["reason"] == "conflict_superseded_by_latest_review"
    assert quarantine_rows[0]["source_clip_path"] == "snare/s1.wav"


def test_review_sample_doctor_dedupes_same_target_content(tmp_path: Path):
    source_root = tmp_path / "review_samples"
    _write_wave(source_root / "kick" / "k1.wav", frequency=70.0)
    shutil.copy2(source_root / "kick" / "k1.wav", source_root / "kick" / "k2.wav")
    _write_manifest(
        source_root,
        [
            {"clip_path": "kick/k1.wav", "class_label": "kick", "decision_kind": "verified"},
            {"clip_path": "kick/k2.wav", "class_label": "kick", "decision_kind": "verified"},
        ],
    )

    result = ReviewSampleDoctorService().audit_and_repair(
        source_root,
        output_root=tmp_path / "doctor",
        labels=("kick",),
    )

    quarantine_rows = _read_jsonl(result.quarantine_root / "manifest.jsonl")
    assert result.report["action_counts"] == {"clean": 1, "dedupe": 1}
    assert result.report["quarantine_reason_counts"] == {"duplicate_content": 1}
    assert quarantine_rows[0]["reason"] == "duplicate_content"
    assert len(list((result.clean_root / "positive" / "kick").glob("*.wav"))) == 1


def _write_wave(path: Path, *, frequency: float, sample_rate: int = 22050) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 0.18
    times = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2.0 * np.pi * frequency * times) * np.exp(-20.0 * times)
    sf.write(path, wave.astype(np.float32), sample_rate)


def _write_manifest(source_root: Path, rows: list[dict[str, str]]) -> None:
    (source_root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
