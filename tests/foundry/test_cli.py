from __future__ import annotations

import json
from pathlib import Path

from echozero.foundry.cli import main


def _write_samples(root: Path) -> None:
    (root / "kick").mkdir(parents=True, exist_ok=True)
    (root / "kick" / "k1.wav").write_bytes(b"RIFF" + b"\x00" * 32)


def test_cli_dataset_ingest_and_run(tmp_path: Path, capsys):
    samples = tmp_path / "samples"
    _write_samples(samples)

    assert main(["--root", str(tmp_path), "create-dataset", "Drums"]) == 0
    out = capsys.readouterr().out
    dataset_id = json.loads(out)["id"]

    assert main(["--root", str(tmp_path), "ingest-folder", dataset_id, str(samples)]) == 0
    out = capsys.readouterr().out
    version_id = json.loads(out)["version_id"]

    run_spec = json.dumps(
        {
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
            "training": {"epochs": 1, "batchSize": 2, "learningRate": 0.001},
        }
    )
    assert main(["--root", str(tmp_path), "create-run", version_id, run_spec]) == 0
    out = capsys.readouterr().out
    run_id = json.loads(out)["run_id"]

    assert main(["--root", str(tmp_path), "start-run", run_id]) == 0
