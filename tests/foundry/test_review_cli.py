from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.cli import main
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.foundry.services.review_signal_service import ReviewSignalService
from echozero.foundry.domain.review import ReviewOutcome
from tests.foundry.audio_fixtures import write_percussion_dataset
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture


def test_cli_import_review_session_from_jsonl(tmp_path: Path, capsys):
    items_path = tmp_path / "review_items.jsonl"
    items_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "audioPath": str(tmp_path / "kick.wav"),
                        "predictedLabel": "kick",
                        "targetClass": "kick",
                        "polarity": "positive",
                        "score": 0.98,
                    }
                ),
                json.dumps(
                    {
                        "audioPath": str(tmp_path / "snare.wav"),
                        "predictedLabel": "kick",
                        "targetClass": "snare",
                        "polarity": "negative",
                        "score": 0.11,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "kick.wav").write_bytes(b"RIFFtest")
    (tmp_path / "snare.wav").write_bytes(b"RIFFtest")

    assert main(["--root", str(tmp_path), "import-review-session", str(items_path), "--name", "Field Review"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["name"] == "Field Review"
    assert payload["items"] == 2
    assert payload["classes"] == ["kick", "snare"]


def test_cli_serve_review_session_delegates_to_review_server(tmp_path: Path, monkeypatch):
    captured: list[tuple[Path, str, str, int]] = []

    def fake_serve_review_session(root: Path, session_id: str, *, host: str, port: int) -> int:
        captured.append((root, session_id, host, port))
        return 0

    monkeypatch.setattr("echozero.foundry.cli.serve_review_session", fake_serve_review_session)

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "serve-review-session",
                "rev_demo123",
                "--host",
                "0.0.0.0",
                "--port",
                "8510",
            ]
        )
        == 0
    )
    assert captured == [(tmp_path, "rev_demo123", "0.0.0.0", 8510)]


def test_cli_import_review_folder_with_target_class(tmp_path: Path, capsys):
    samples = tmp_path / "clips"
    samples.mkdir(parents=True, exist_ok=True)
    source_root = tmp_path / "source"
    write_percussion_dataset(source_root, sample_count=1)
    (samples / "hit_01.wav").write_bytes((source_root / "kick" / "k1.wav").read_bytes())

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "import-review-folder",
                str(samples),
                "--name",
                "Phone Arcade",
                "--target-class",
                "kick",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["name"] == "Phone Arcade"
    assert payload["items"] == 1
    assert payload["classes"] == ["kick"]


def test_cli_create_project_review_session_with_questionable_filter(tmp_path: Path, capsys):
    _ez_path, working_dir, _refs = _build_project_review_fixture(tmp_path)

    assert (
        main(
            [
                "--root",
                str(working_dir),
                "create-project-review-session",
                str(working_dir),
                "--name",
                "Questionable Hits",
                "--questionable-score-threshold",
                "0.8",
                "--item-limit",
                "2",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["name"] == "Questionable Hits"
    assert payload["items"] == 2
    assert payload["classes"] == ["kick", "snare"]


def test_cli_extract_project_review_dataset(tmp_path: Path, capsys):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(working_dir)
    session = service.create_project_session(
        working_dir,
        name="Kick Corrections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.CORRECT,
        corrected_label=None,
        review_note="export eligibility setup",
    )

    assert (
        main(
            [
                "--root",
                str(working_dir),
                "extract-project-review-dataset",
                str(working_dir),
                "--song-id",
                refs["alpha_song_id"],
                "--layer-id",
                "layer_alpha_kick",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "review_extraction"
    assert payload["dataset_id"].startswith("ds_")
    assert payload["version_id"].startswith("dsv_")
    assert payload["sample_count"] >= 1


def test_cli_extract_review_signal(tmp_path: Path, capsys):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    service = ReviewSessionService(working_dir)
    session = service.create_project_session(
        working_dir,
        name="Kick Corrections",
        song_id=refs["alpha_song_id"],
        layer_id="layer_alpha_kick",
    )
    reviewed = service.set_item_review(
        session.id,
        session.items[0].item_id,
        outcome=ReviewOutcome.INCORRECT,
        corrected_label="tom",
        review_note="deferred then explicit materialize",
    )
    signal_id = ReviewSignalService.build_signal_id(reviewed.id, reviewed.items[0].item_id)

    assert (
        main(
            [
                "--root",
                str(working_dir),
                "extract-review-signal",
                reviewed.id,
                signal_id,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "review_extraction"
    assert payload["status"] == "materialized"
    assert str(payload["dataset_id"]).startswith("ds_")
    assert str(payload["version_id"]).startswith("dsv_")
