from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import Dataset, DatasetSample, DatasetVersion
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository
from echozero.foundry.services.query_service import FoundryQueryService
from tests.foundry.audio_fixtures import write_percussion_dataset


def _build_run_spec(version_id: str) -> dict:
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version_id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 19},
    }


def test_query_service_reads_foundry_ui_state(tmp_path: Path):
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Drums")
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.25, test_split=0.25, seed=9, balance_strategy="none")

    run = app.create_run(version.id, _build_run_spec(version.id))
    app.start_run(run.id)

    service = FoundryQueryService(
        dataset_repo=app._dataset_repo,
        version_repo=app._dataset_version_repo,
        artifact_repo=app._artifact_repo,
        eval_repo=app._eval_repo,
        run_service=app.runs,
        root=tmp_path,
    )

    datasets = service.list_datasets()
    runs = service.list_runs()
    artifacts = service.list_artifacts_for_run(run.id)
    eval_reports = service.list_eval_reports_for_run(run.id)

    assert [item.id for item in datasets] == [dataset.id]
    assert [item.id for item in runs] == [run.id]
    assert len(artifacts) == 1
    assert service.list_artifacts() == artifacts
    assert service.get_artifact(artifacts[0].id) == artifacts[0]
    assert len(eval_reports) == 1
    assert eval_reports[0].run_id == run.id


def test_query_service_filters_project_review_dataset_versions_and_returns_latest(tmp_path: Path):
    cache_dir = tmp_path / "foundry" / "cache" / "review_projects" / "fixture" / "clips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latest_clip = cache_dir / "kick_latest.wav"
    prior_clip = cache_dir / "kick_prior.wav"
    foreign_clip = cache_dir / "kick_foreign.wav"
    latest_clip.write_bytes(b"latest")
    prior_clip.write_bytes(b"prior")
    foreign_clip.write_bytes(b"foreign")

    dataset_repo = DatasetRepository(tmp_path)
    version_repo = DatasetVersionRepository(tmp_path)
    now = datetime.now(UTC)
    matching_dataset = dataset_repo.save(
        Dataset(
            id="ds_project_review",
            name="Review Samples - Active Project",
            source_kind="project_review_export",
            source_ref=str(tmp_path),
            metadata={
                "schema": "foundry.project_review_dataset.v1",
                "review_dataset_key": "ez_project:project:active",
                "queue_source_kind": "ez_project",
                "project_ref": "project:active",
            },
            created_at=now - timedelta(minutes=10),
        )
    )
    dataset_repo.save(
        Dataset(
            id="ds_foreign_review",
            name="Review Samples - Foreign Project",
            source_kind="project_review_export",
            source_ref=str(tmp_path),
            metadata={
                "schema": "foundry.project_review_dataset.v1",
                "review_dataset_key": "ez_project:project:foreign",
                "queue_source_kind": "ez_project",
                "project_ref": "project:foreign",
            },
            created_at=now - timedelta(minutes=9),
        )
    )
    version_repo.save(
        DatasetVersion(
            id="dsv_prior",
            dataset_id=matching_dataset.id,
            version=1,
            manifest_hash="hash-prior",
            sample_rate=22050,
            audio_standard="mono_wav_pcm16",
            class_map=["kick"],
            samples=[DatasetSample(sample_id="sm_prior", audio_ref=str(prior_clip), label="kick")],
            manifest={"schema": "foundry.review_dataset_manifest.v1"},
            created_at=now - timedelta(minutes=5),
        )
    )
    version_repo.save(
        DatasetVersion(
            id="dsv_latest",
            dataset_id=matching_dataset.id,
            version=2,
            manifest_hash="hash-latest",
            sample_rate=22050,
            audio_standard="mono_wav_pcm16",
            class_map=["kick"],
            samples=[DatasetSample(sample_id="sm_latest", audio_ref=str(latest_clip), label="kick")],
            manifest={"schema": "foundry.review_dataset_manifest.v1"},
            created_at=now - timedelta(minutes=1),
        )
    )
    version_repo.save(
        DatasetVersion(
            id="dsv_foreign",
            dataset_id="ds_foreign_review",
            version=1,
            manifest_hash="hash-foreign",
            sample_rate=22050,
            audio_standard="mono_wav_pcm16",
            class_map=["kick"],
            samples=[DatasetSample(sample_id="sm_foreign", audio_ref=str(foreign_clip), label="kick")],
            manifest={"schema": "foundry.review_dataset_manifest.v1"},
            created_at=now - timedelta(minutes=2),
        )
    )

    app = FoundryApp(tmp_path)
    service = FoundryQueryService(
        dataset_repo=app._dataset_repo,
        version_repo=app._dataset_version_repo,
        artifact_repo=app._artifact_repo,
        eval_repo=app._eval_repo,
        run_service=app.runs,
        root=tmp_path,
    )

    matches = service.list_project_review_dataset_versions(project_ref="project:active")
    latest = service.get_latest_project_review_dataset_version(project_ref="project:active")

    assert [item.version_id for item in matches] == ["dsv_latest", "dsv_prior"]
    assert latest is not None
    assert latest.dataset_id == "ds_project_review"
    assert latest.dataset_name == "Review Samples - Active Project"
    assert latest.version_id == "dsv_latest"
    assert latest.version_number == 2
    assert latest.sample_count == 1
    assert latest.queue_source_kind == "ez_project"
    assert latest.project_ref == "project:active"
    assert latest.dataset_folder_path == cache_dir.resolve()
    assert latest.version_artifact_path == (
        tmp_path / "foundry" / "state" / "dataset_versions.json"
    ).resolve()


def test_foundry_app_query_methods_delegate_to_query_service(tmp_path: Path):
    app = FoundryApp(tmp_path)

    class StubQueryService:
        def list_datasets(self):
            return ["datasets"]

        def list_runs(self):
            return ["runs"]

        def list_artifacts(self):
            return ["artifacts"]

        def list_artifacts_for_run(self, run_id: str):
            return [run_id, "artifacts_for_run"]

        def get_artifact(self, artifact_id: str):
            return {"artifact_id": artifact_id}

        def list_eval_reports_for_run(self, run_id: str):
            return [run_id, "eval_reports"]

        def list_project_review_dataset_versions(
            self,
            *,
            project_ref: str,
            queue_source_kind: str | None = "ez_project",
        ):
            return [project_ref, queue_source_kind, "review_dataset_versions"]

        def get_latest_project_review_dataset_version(
            self,
            *,
            project_ref: str,
            queue_source_kind: str | None = "ez_project",
        ):
            return {"project_ref": project_ref, "queue_source_kind": queue_source_kind}

    app.queries = StubQueryService()

    assert app.list_datasets() == ["datasets"]
    assert app.list_runs() == ["runs"]
    assert app.list_artifacts() == ["artifacts"]
    assert app.list_artifacts_for_run("run-123") == ["run-123", "artifacts_for_run"]
    assert app.get_artifact("artifact-123") == {"artifact_id": "artifact-123"}
    assert app.list_eval_reports_for_run("run-123") == ["run-123", "eval_reports"]
    assert app.list_project_review_dataset_versions(project_ref="project:active") == [
        "project:active",
        "ez_project",
        "review_dataset_versions",
    ]
    assert app.get_latest_project_review_dataset_version(project_ref="project:active") == {
        "project_ref": "project:active",
        "queue_source_kind": "ez_project",
    }
