from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import (
    LibrarySampleState,
    SampleLibraryRecord,
)
from echozero.foundry.persistence import SampleLibraryRepository
from tests.foundry.audio_fixtures import write_percussion_dataset


def test_sample_library_repository_round_trip(tmp_path: Path):
    sample_record = SampleLibraryRecord(
        id="lib_sample",
        audio_ref="/tmp/audio.wav",
        label="kick",
        state=LibrarySampleState.APPROVED,
        source_type="review_dataset",
        source_ref="dsv_1",
        provenance={"dataset_version_id": "dsv_1"},
    )

    SampleLibraryRepository(tmp_path).save(sample_record)

    assert SampleLibraryRepository(tmp_path).get("lib_sample") == sample_record


def test_continuous_training_service_kicks_off_run_from_library_samples(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Library Source")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    records = app.sample_library.record_dataset_version(version)

    assert len(records) == len(version.samples)
    assert app.summarize_sample_library()["approved_count"] == len(version.samples)

    run = app.continuous_training.kickoff_run(
        name="Local Drums",
        epochs=1,
    )

    assert run.status.value == "completed"
    assert run.dataset_version_id.startswith("dsv_")
    assert app.list_runs()[0].id == run.id
    assert len(app.list_artifacts_for_run(run.id)) == 1
    assert len(app.list_eval_reports_for_run(run.id)) == 1
