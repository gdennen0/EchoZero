from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import CompatibilityReport
from tests.foundry.audio_fixtures import write_percussion_dataset


def _load_contract_schema() -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "echozero" / "foundry" / "contracts" / "compatibility_report.v1.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _assert_issue_shape(items: list[dict[str, object]], *, expected_severity: str) -> None:
    for item in items:
        assert set(item.keys()) == {"code", "path", "message", "severity"}
        assert isinstance(item["code"], str)
        assert isinstance(item["path"], str)
        assert isinstance(item["message"], str)
        assert item["severity"] == expected_severity


def test_compatibility_report_contract_schema_parity_guard() -> None:
    schema = _load_contract_schema()
    report = CompatibilityReport(
        artifact_id="art_123",
        consumer="PyTorchAudioClassify",
        ok=False,
        errors=["runtime mismatch"],
        warnings=["non-fatal issue"],
        error_details=[
            {
                "code": "runtime_consumer_mismatch",
                "path": "manifest.runtime.consumer",
                "message": "manifest.runtime.consumer must match the validated consumer",
                "severity": "error",
            }
        ],
        warning_details=[
            {
                "code": "optional_field_missing",
                "path": "manifest.foo",
                "message": "manifest.foo missing; using default",
                "severity": "warning",
            }
        ],
    )

    payload = report.to_contract_payload()

    assert set(payload.keys()) == set(schema["properties"].keys())
    assert set(schema["required"]) == set(schema["properties"].keys())

    assert payload["schema"] == "foundry.compatibility_report.v1"
    assert payload["artifactId"] == "art_123"
    assert payload["consumer"] == "PyTorchAudioClassify"
    assert payload["ok"] is False
    assert payload["errors"] == ["runtime mismatch"]
    assert payload["warnings"] == ["non-fatal issue"]
    assert isinstance(payload["checkedAt"], str)

    _assert_issue_shape(payload["error_details"], expected_severity="error")
    _assert_issue_shape(payload["warning_details"], expected_severity="warning")


def test_validate_artifact_contract_payload_contains_structured_diagnostics(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    write_percussion_dataset(samples)

    app = FoundryApp(tmp_path)
    dataset = app.datasets.create_dataset("Contract Drums", source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(version.id, validation_split=0.2, test_split=0.2, seed=53, balance_strategy="none")

    run_spec = {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.01, "seed": 59},
    }

    run = app.create_run(version.id, run_spec)
    run = app.start_run(run.id)
    artifact = app.finalize_artifact(
        run.id,
        {
            "weightsPath": "model.pth",
            "classes": list(version.class_map),
            "classificationMode": "multiclass",
            "runtime": {"consumer": "OtherRuntime", "backend": "pytorch", "device": "cpu"},
            "inferencePreprocessing": run_spec["data"],
        },
    )

    report = app.validate_artifact(artifact.id)
    payload = report.to_contract_payload()

    schema = _load_contract_schema()
    required = set(schema["required"])
    assert required.issubset(payload.keys())
    assert payload["artifactId"] == artifact.id
    assert payload["ok"] is False
    assert payload["errors"] == report.errors
    assert payload["warnings"] == report.warnings

    assert payload["error_details"], "expected at least one structured error detail"
    _assert_issue_shape(payload["error_details"], expected_severity="error")
    _assert_issue_shape(payload["warning_details"], expected_severity="warning")
