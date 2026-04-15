from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import CurationState
from echozero.foundry.persistence import DatasetVersionRepository
from echozero.foundry.services.dataset_service import DatasetService
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
    assert plan["split_plan"]["policy"] == "grouped_anti_leakage_v2"
    assert plan["split_plan"]["dataset_manifest_hash"] == version.manifest_hash
    assert plan["split_plan"]["leakage"]["ok"] is True
    assert plan["split_plan"]["assignments"]
    assert plan["split_plan"]["reproducibility"]["assignment_fingerprint"]
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
    assert plan["split_plan"]["leakage"]["duplicate_groups_across_splits"] == []
    assert {sample_by_id[sample_id].label for sample_id in duplicate_ids} == {"kick", "snare"}


def test_split_planning_keeps_explicit_groups_in_one_split_even_without_duplicate_audio(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir, sample_count=2)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Grouped", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)

    repo = DatasetVersionRepository(tmp_path)
    version = repo.get(version.id)
    assert version is not None
    version.samples[0].group_id = "session:alpha"
    version.samples[1].group_id = "session:alpha"
    version.samples[0].source_provenance["group_id"] = "session:alpha"
    version.samples[1].source_provenance["group_id"] = "session:alpha"
    repo.save(version)

    plan = app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=19, balance_strategy="none")
    grouped_ids = [version.samples[0].sample_id, version.samples[1].sample_id]

    assert len({plan["split_plan"]["assignments"][sample_id] for sample_id in grouped_ids}) == 1
    assert plan["split_plan"]["leakage"]["duplicate_groups_across_splits"] == []



def test_split_planning_is_reproducible_for_same_seed(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir, sample_count=3)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Repro", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)

    first = app.split_balance.plan_splits(version, validation_split=0.2, test_split=0.2, seed=17)
    second = app.split_balance.plan_splits(version, validation_split=0.2, test_split=0.2, seed=17)

    assert first["assignments"] == second["assignments"]
    assert first["reproducibility"]["assignment_fingerprint"] == second["reproducibility"]["assignment_fingerprint"]
    assert first["reproducibility"]["group_fingerprint"] == second["reproducibility"]["group_fingerprint"]



def test_balance_plan_warns_on_skew_and_recommends_defaults(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir, sample_count=4)
    (dataset_dir / "snare" / "s2.wav").unlink()
    (dataset_dir / "snare" / "s3.wav").unlink()
    (dataset_dir / "snare" / "s4.wav").unlink()

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Skewed", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)
    balance = app.split_balance.plan_balance(version, strategy="none")

    assert balance["is_skewed"] is True
    assert balance["majority_to_minority_ratio"] == 4.0
    assert balance["warnings"]
    assert balance["recommended_training_overrides"] == {
        "classWeighting": "balanced",
        "rebalanceStrategy": "oversample",
    }



def test_dataset_version_integrity_detects_manifest_drift(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Integrity", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)
    version.manifest_hash = "broken"

    report = DatasetService.validate_version_integrity(version)

    assert report["ok"] is False
    assert any("manifest_hash" in error for error in report["errors"])
