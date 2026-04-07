from __future__ import annotations

import json
from pathlib import Path

import pytest

from echozero.foundry.cli import main
from echozero.foundry.persistence import StateFormatError, TrainRunRepository


def test_repository_rejects_legacy_state_until_explicit_migration(tmp_path: Path):
    state_dir = tmp_path / "foundry" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    legacy_payload = {
        "run_legacy": {
            "id": "run_legacy",
            "dataset_version_id": "dsv_legacy",
            "status": "queued",
            "spec": {},
            "spec_hash": "abc",
            "backend": "pytorch",
            "device": "cpu",
            "created_at": "2026-04-07T19:00:00+00:00",
            "updated_at": "2026-04-07T19:00:00+00:00",
        }
    }
    (state_dir / "train_runs.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    repo = TrainRunRepository(tmp_path)
    with pytest.raises(StateFormatError, match="migrate-state"):
        repo.get("run_legacy")


def test_cli_migrate_state_wraps_legacy_file(tmp_path: Path, capsys):
    state_dir = tmp_path / "foundry" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "datasets.json").write_text(
        json.dumps(
            {
                "ds_1": {
                    "id": "ds_1",
                    "name": "Legacy",
                    "source_kind": "folder",
                    "source_ref": None,
                    "metadata": {},
                    "created_at": "2026-04-07T19:00:00+00:00",
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["--root", str(tmp_path), "migrate-state"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "datasets.json" in payload["migrated"]

    saved = json.loads((state_dir / "datasets.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "foundry.state.datasets.v1"
    assert saved["version"] == 1
    assert "items" in saved
    assert "ds_1" in saved["items"]
