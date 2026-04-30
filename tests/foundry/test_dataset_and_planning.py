from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import (
    CurationState,
    DatasetSample,
    DatasetVersion,
    ReviewDecisionKind,
    ReviewOutcome,
)
from echozero.foundry.persistence import DatasetVersionRepository
from echozero.foundry.services.dataset_service import DatasetService
from tests.foundry.audio_fixtures import (
    write_percussion_dataset,
    write_unreadable_audio_file,
    write_zero_byte_audio_file,
)
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture
from tests.foundry.test_review_sessions import _mark_project_event_review_state


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


def test_dataset_ingest_skips_invalid_audio_and_records_manifest_stats(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_percussion_dataset(dataset_dir, sample_count=1)
    write_zero_byte_audio_file(dataset_dir / "kick" / "bad_zero.wav")
    write_unreadable_audio_file(dataset_dir / "snare" / "bad_text.wav")

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Graceful Ingest", source_kind="folder_import")
    version = app.datasets.ingest_from_folder(dataset.id, dataset_dir)

    assert version.stats["sample_count"] == 2
    assert version.stats["skipped_invalid_count"] == 2
    assert version.stats["skipped_invalid_by_reason"] == {"unreadable": 1, "zero_byte": 1}
    assert sorted(item["relative_path"] for item in version.manifest["skipped_sources"]) == [
        "kick/bad_zero.wav",
        "snare/bad_text.wav",
    ]


def test_dataset_ingest_rejects_folders_with_only_invalid_audio(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    write_zero_byte_audio_file(dataset_dir / "kick" / "bad_zero.wav")
    write_unreadable_audio_file(dataset_dir / "snare" / "bad_text.wav")

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Only Invalid", source_kind="folder_import")

    with pytest.raises(ValueError, match="No valid audio samples found"):
        app.datasets.ingest_from_folder(dataset.id, dataset_dir)


def test_dataset_service_derives_binary_version_from_review_dataset_samples(tmp_path: Path):
    service = DatasetService(tmp_path)
    dataset = service.create_dataset(
        "Review Samples",
        source_kind="project_review_export",
        metadata={"project_ref": "project:fixture"},
    )
    samples = [
        DatasetSample(
            sample_id="sm_kick",
            audio_ref="kick.wav",
            label="kick",
            content_hash="hash-kick",
            source_provenance={"review_item_id": "ri-kick", "review_polarity": "positive"},
            group_id="content:hash-kick",
            curation_state=CurationState.ACCEPTED,
        ),
        DatasetSample(
            sample_id="sm_snare",
            audio_ref="snare.wav",
            label="snare",
            content_hash="hash-snare",
            source_provenance={"review_item_id": "ri-snare", "review_polarity": "positive"},
            group_id="content:hash-snare",
            curation_state=CurationState.ACCEPTED,
        ),
        DatasetSample(
            sample_id="sm_tom",
            audio_ref="tom.wav",
            label="tom",
            content_hash="hash-tom",
            source_provenance={"review_item_id": "ri-tom", "review_polarity": "positive"},
            group_id="content:hash-tom",
            curation_state=CurationState.ACCEPTED,
        ),
    ]
    source_version = DatasetVersion(
        id="dsv_review",
        dataset_id=dataset.id,
        version=1,
        manifest_hash=DatasetService.compute_manifest_hash(samples),
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare", "tom"],
        samples=samples,
        taxonomy={
            "schema": "foundry.taxonomy.v1",
            "namespace": "percussion.one_shot",
            "version": 1,
            "labels": [],
        },
        label_policy={
            "schema": "foundry.label_policy.v1",
            "classification_mode": "multiclass",
            "unit": "one_shot",
            "allowed_labels": ["kick", "snare", "tom"],
            "unknown_label": None,
        },
        manifest={
            "schema": "foundry.review_dataset_manifest.v1",
            "deterministic_order": [sample.sample_id for sample in samples],
            "content_groups": {
                "content:hash-kick": ["sm_kick"],
                "content:hash-snare": ["sm_snare"],
                "content:hash-tom": ["sm_tom"],
            },
        },
        split_plan={
            "assignments": {
                "sm_kick": "train",
                "sm_snare": "val",
                "sm_tom": "test",
            },
            "train_ids": ["sm_kick"],
            "val_ids": ["sm_snare"],
            "test_ids": ["sm_tom"],
        },
        stats={"sample_count": 3},
        created_at=datetime.now(UTC),
    )
    DatasetVersionRepository(tmp_path).save(source_version)

    derived = service.derive_binary_dataset_version(source_version.id, positive_label="kick")

    assert derived.class_map == ["kick", "other"]
    assert derived.label_policy["classification_mode"] == "binary"
    assert derived.label_policy["allowed_labels"] == ["kick", "other"]
    assert {sample.sample_id: sample.label for sample in derived.samples} == {
        "sm_kick": "kick",
        "sm_snare": "other",
        "sm_tom": "other",
    }
    assert derived.split_plan["assignments"] == source_version.split_plan["assignments"]
    assert derived.lineage["source_version_id"] == source_version.id


def test_dataset_service_derives_binary_version_using_review_polarity(tmp_path: Path):
    service = DatasetService(tmp_path)
    dataset = service.create_dataset(
        "Review Samples",
        source_kind="project_review_export",
        metadata={"project_ref": "project:fixture"},
    )
    samples = [
        DatasetSample(
            sample_id="sm_kick_negative",
            audio_ref="kick_negative.wav",
            label="kick",
            content_hash="hash-kick-negative",
            source_provenance={"review_item_id": "ri-kick-negative", "review_polarity": "negative"},
            group_id="content:hash-kick-negative",
            curation_state=CurationState.ACCEPTED,
        ),
        DatasetSample(
            sample_id="sm_kick_positive",
            audio_ref="kick_positive.wav",
            label="kick",
            content_hash="hash-kick-positive",
            source_provenance={"review_item_id": "ri-kick-positive", "review_polarity": "positive"},
            group_id="content:hash-kick-positive",
            curation_state=CurationState.ACCEPTED,
        ),
        DatasetSample(
            sample_id="sm_snare_positive",
            audio_ref="snare_positive.wav",
            label="snare",
            content_hash="hash-snare-positive",
            source_provenance={"review_item_id": "ri-snare-positive", "review_polarity": "positive"},
            group_id="content:hash-snare-positive",
            curation_state=CurationState.ACCEPTED,
        ),
    ]
    source_version = DatasetVersion(
        id="dsv_review_polarity",
        dataset_id=dataset.id,
        version=1,
        manifest_hash=DatasetService.compute_manifest_hash(samples),
        sample_rate=22050,
        audio_standard="mono_wav_pcm16",
        class_map=["kick", "snare"],
        samples=samples,
        taxonomy={"schema": "foundry.taxonomy.v1"},
        label_policy={"schema": "foundry.label_policy.v1", "classification_mode": "multiclass"},
        manifest={"schema": "foundry.project_review_dataset_manifest.v1"},
        created_at=datetime.now(UTC),
    )
    DatasetVersionRepository(tmp_path).save(source_version)

    derived = service.derive_binary_dataset_version(source_version.id, positive_label="kick")

    assert {sample.sample_id: sample.label for sample in derived.samples} == {
        "sm_kick_negative": "other",
        "sm_kick_positive": "kick",
        "sm_snare_positive": "other",
    }


def test_dataset_service_exports_project_review_dataset_from_canonical_project_truth(tmp_path: Path):
    _ez_path, working_dir, refs = _build_project_review_fixture(tmp_path)
    app = FoundryApp(tmp_path)

    _mark_project_event_review_state(
        working_dir,
        layer_id="layer_alpha_kick",
        event_id="evt_alpha_kick_01",
        promotion_state="demoted",
        review_state="corrected",
        review_outcome=ReviewOutcome.INCORRECT,
        decision_kind=ReviewDecisionKind.REJECTED,
        original_label="kick",
        corrected_label=None,
        review_note="operator removed a false kick",
    )
    _mark_project_event_review_state(
        working_dir,
        layer_id="layer_alpha_kick",
        event_id="evt_alpha_kick_02",
        promotion_state="promoted",
        review_state="signed_off",
        review_outcome=ReviewOutcome.CORRECT,
        decision_kind=ReviewDecisionKind.VERIFIED,
        original_label="kick",
        corrected_label=None,
        review_note=None,
    )

    exported = app.extract_project_review_dataset(
        working_dir,
        project_ref=refs["project_ref"],
    )

    assert exported.dataset_id
    assert exported.stats["sample_count"] == 2
    assert exported.stats["review_positive_count"] == 1
    assert exported.stats["review_negative_count"] == 1
    assert exported.stats["reviewed_item_count"] == 2
    assert sorted(sample.label for sample in exported.samples) == ["kick", "kick"]
    assert {sample.source_provenance["review_polarity"] for sample in exported.samples} == {
        "negative",
        "positive",
    }
    dataset = app.datasets.get_dataset(exported.dataset_id)
    assert dataset is not None
    assert dataset.source_kind == "project_review_export"
    assert dataset.metadata["project_ref"] == refs["project_ref"]
