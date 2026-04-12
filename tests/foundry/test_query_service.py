from __future__ import annotations

from pathlib import Path

from echozero.foundry.app import FoundryApp
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
        artifact_repo=app._artifact_repo,
        eval_repo=app._eval_repo,
        run_service=app.runs,
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

    app.queries = StubQueryService()

    assert app.list_datasets() == ["datasets"]
    assert app.list_runs() == ["runs"]
    assert app.list_artifacts() == ["artifacts"]
    assert app.list_artifacts_for_run("run-123") == ["run-123", "artifacts_for_run"]
    assert app.get_artifact("artifact-123") == {"artifact_id": "artifact-123"}
    assert app.list_eval_reports_for_run("run-123") == ["run-123", "eval_reports"]
