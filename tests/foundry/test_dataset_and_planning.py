from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import CurationState
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_dataset_ingest_plan_and_curation(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)

    assert version.stats["sample_count"] == 8
    assert sorted(version.class_map) == ["kick", "snare"]
    assert version.taxonomy["namespace"] == "percussion.one_shot"
    assert version.label_policy["allowed_labels"] == ["kick", "snare"]
    assert version.manifest["content_hash_algorithm"] == "sha256"

    plan = app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=7, balance_strategy="undersample_min")
    assert plan["split_plan"]["seed"] == 7
    assert plan["split_plan"]["planner"] == "content_hash_grouped_v1"
    assert plan["split_plan"]["leakage"]["ok"] is True
    assert plan["split_plan"]["assignments"]
    assert plan["balance_plan"]["strategy"] == "undersample_min"
    persisted = app.datasets.get_version(version.id)
    assert persisted is not None
    assert {sample.split_assignment for sample in persisted.samples} <= {"train", "val", "test"}

    reject_id = version.samples[0].sample_id
    curated = app.datasets.apply_curation(version.id, {reject_id: CurationState.REJECTED})
    assert curated.stats["sample_count"] == 7
    assert reject_id not in curated.split_plan["assignments"]


def test_split_planning_keeps_duplicate_content_hashes_in_one_split(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir, sample_count=2)
    duplicate = (dataset_dir / "kick" / "k1.wav").read_bytes()
    (dataset_dir / "snare" / "s1.wav").write_bytes(duplicate)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Duplicates", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)
    plan = app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=11, balance_strategy="none")

    sample_by_id = {sample.sample_id: sample for sample in version.samples}
    duplicate_ids = [sample.sample_id for sample in version.samples if sample.content_hash == version.samples[0].content_hash]

    assert len(duplicate_ids) == 2
    assert len({plan["split_plan"]["assignments"][sample_id] for sample_id in duplicate_ids}) == 1
    assert plan["split_plan"]["leakage"]["duplicate_hashes_across_splits"] == []
    assert {sample_by_id[sample_id].label for sample_id in duplicate_ids} == {"kick", "snare"}
