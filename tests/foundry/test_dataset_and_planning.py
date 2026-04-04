from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import CurationState


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "snare").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x00" * 32)
    (root / "kick" / "k2.wav").write_bytes(b"RIFF" + b"\x00" * 32)
    (root / "snare" / "s1.wav").write_bytes(b"RIFF" + b"\x00" * 32)


def test_dataset_ingest_plan_and_curation(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _write_samples(dataset_dir)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)

    assert version.stats["sample_count"] == 3
    assert sorted(version.class_map) == ["kick", "snare"]

    plan = app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=7, balance_strategy="undersample_min")
    assert plan["split_plan"]["seed"] == 7
    assert plan["balance_plan"]["strategy"] == "undersample_min"

    reject_id = version.samples[0].sample_id
    curated = app.datasets.apply_curation(version.id, {reject_id: CurationState.REJECTED})
    assert curated.stats["sample_count"] == 2
