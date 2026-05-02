from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import (
    ChampionModelRecord,
    ContributionPolicy,
    ContributionRecord,
    LibrarySampleState,
    LibrarySnapshot,
    ModelCandidateRecord,
    SampleLibraryRecord,
    TrainingRecipeName,
)
from echozero.foundry.persistence import (
    ChampionModelRepository,
    ContributionRepository,
    LibrarySnapshotRepository,
    ModelCandidateRepository,
    SampleLibraryRepository,
)
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_library_first_repositories_round_trip(tmp_path: Path):
    sample_record = SampleLibraryRecord(
        id="lib_sample",
        audio_ref="/tmp/audio.wav",
        label="kick",
        state=LibrarySampleState.APPROVED,
        source_type="review_dataset",
        source_ref="dsv_1",
        provenance={"dataset_version_id": "dsv_1"},
        contribution_policy=ContributionPolicy.CONTRIBUTABLE,
    )
    snapshot = LibrarySnapshot(
        id="snap_1",
        name="Snapshot One",
        sample_ids=["lib_sample"],
        sample_count=1,
        class_counts={"kick": 1},
        source_summary={"review_dataset": 1},
    )
    candidate = ModelCandidateRecord(
        id="cand_1",
        snapshot_id="snap_1",
        recipe_name=TrainingRecipeName.QUICK.value,
        dataset_id="ds_1",
        dataset_version_id="dsv_1",
        run_id="run_1",
        artifact_id="art_1",
        status="completed",
    )
    champion = ChampionModelRecord(
        scope="local.default",
        candidate_id="cand_1",
        artifact_id="art_1",
    )
    contribution = ContributionRecord(
        id="contrib_1",
        sample_id="lib_sample",
        policy=ContributionPolicy.CONTRIBUTABLE,
        status="ready",
        payload={"sample_id": "lib_sample"},
    )

    SampleLibraryRepository(tmp_path).save(sample_record)
    LibrarySnapshotRepository(tmp_path).save(snapshot)
    ModelCandidateRepository(tmp_path).save(candidate)
    ChampionModelRepository(tmp_path).save(champion)
    ContributionRepository(tmp_path).save(contribution)

    assert SampleLibraryRepository(tmp_path).get("lib_sample") == sample_record
    assert LibrarySnapshotRepository(tmp_path).get("snap_1") == snapshot
    assert ModelCandidateRepository(tmp_path).get("cand_1") == candidate
    assert ChampionModelRepository(tmp_path).get("local.default") == champion
    assert ContributionRepository(tmp_path).get("contrib_1") == contribution


def test_continuous_training_service_runs_from_library_samples(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Library Source")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    records = app.sample_library.record_dataset_version(version)

    assert len(records) == len(version.samples)
    assert app.summarize_sample_library()["approved_count"] == len(version.samples)

    candidate = app.continuous_training.train_challenger(
        name="Local Drums",
        recipe_name=TrainingRecipeName.QUICK,
    )

    assert candidate.status == "completed"
    assert candidate.artifact_id is not None
    assert candidate.eval_report_id is not None
    assert candidate.dataset_id.startswith("ds_")
    assert candidate.dataset_version_id.startswith("dsv_")
    assert app.continuous_training.list_candidates()[0].id == candidate.id

    champion = app.continuous_training.promote_candidate(candidate.id)

    assert champion.candidate_id == candidate.id
    assert champion.artifact_id == candidate.artifact_id
    assert app.champions.get_champion() == champion
