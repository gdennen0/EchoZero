from __future__ import annotations

import argparse
import json
from pathlib import Path

from echozero.foundry import FoundryApp
from echozero.foundry.persistence import EvalReportRepository, ModelArtifactRepository
from echozero.foundry.ui import run_foundry_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ez-foundry", description="EchoZero Foundry standalone CLI")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Foundry workspace root")

    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("create-dataset")
    ds.add_argument("name")

    ingest = sub.add_parser("ingest-folder")
    ingest.add_argument("dataset_id")
    ingest.add_argument("folder")

    plan = sub.add_parser("plan-version")
    plan.add_argument("version_id")
    plan.add_argument("--val", type=float, default=0.15)
    plan.add_argument("--test", type=float, default=0.10)
    plan.add_argument("--seed", type=int, default=42)
    plan.add_argument("--balance", default="none")

    train_folder = sub.add_parser("train-folder")
    train_folder.add_argument("name")
    train_folder.add_argument("folder")
    train_folder.add_argument("--val", type=float, default=0.15)
    train_folder.add_argument("--test", type=float, default=0.10)
    train_folder.add_argument("--seed", type=int, default=42)
    train_folder.add_argument("--balance", default="none")
    train_folder.add_argument("--epochs", type=int, default=4)
    train_folder.add_argument("--batch-size", type=int, default=4)
    train_folder.add_argument("--learning-rate", type=float, default=0.01)
    train_folder.add_argument("--sample-rate", type=int, default=22050)
    train_folder.add_argument("--max-length", type=int, default=22050)
    train_folder.add_argument("--n-fft", type=int, default=2048)
    train_folder.add_argument("--hop-length", type=int, default=512)
    train_folder.add_argument("--n-mels", type=int, default=128)
    train_folder.add_argument("--fmax", type=int, default=8000)

    run = sub.add_parser("create-run")
    run.add_argument("dataset_version_id")
    run.add_argument("spec_json")

    start = sub.add_parser("start-run")
    start.add_argument("run_id")

    complete = sub.add_parser("complete-run")
    complete.add_argument("run_id")
    complete.add_argument("--metrics", default="{}")

    art = sub.add_parser("finalize-artifact")
    art.add_argument("run_id")
    art.add_argument("manifest_json")

    val = sub.add_parser("validate-artifact")
    val.add_argument("artifact_id")

    sub.add_parser("ui", help="Launch standalone Foundry UI")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    app = FoundryApp(args.root)

    if args.command == "create-dataset":
        dataset = app.datasets.create_dataset(args.name)
        print(json.dumps({"id": dataset.id, "name": dataset.name}, indent=2))
        return 0

    if args.command == "ingest-folder":
        version = app.datasets.ingest_from_folder(args.dataset_id, args.folder)
        print(json.dumps({"version_id": version.id, "samples": len(version.samples)}, indent=2))
        return 0

    if args.command == "plan-version":
        planned = app.plan_version(
            args.version_id,
            validation_split=args.val,
            test_split=args.test,
            seed=args.seed,
            balance_strategy=args.balance,
        )
        print(json.dumps(planned, indent=2))
        return 0

    if args.command == "train-folder":
        dataset = app.datasets.create_dataset(args.name, source_ref=str(Path(args.folder).resolve()))
        version = app.datasets.ingest_from_folder(dataset.id, args.folder, sample_rate=args.sample_rate)
        app.plan_version(
            version.id,
            validation_split=args.val,
            test_split=args.test,
            seed=args.seed,
            balance_strategy=args.balance,
        )
        run = app.create_run(
            version.id,
            _default_run_spec(
                version.id,
                sample_rate=args.sample_rate,
                max_length=args.max_length,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                fmax=args.fmax,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            ),
        )
        run = app.start_run(run.id)
        eval_reports = EvalReportRepository(args.root).list_for_run(run.id)
        artifacts = ModelArtifactRepository(args.root).list_for_run(run.id)
        payload = {
            "dataset_id": dataset.id,
            "dataset_version_id": version.id,
            "run_id": run.id,
            "status": run.status.value,
            "eval_report_ids": [report.id for report in eval_reports],
            "artifact_ids": [artifact.id for artifact in artifacts],
            "exports_dir": str(run.exports_dir(args.root)),
        }
        print(json.dumps(payload, indent=2))
        return 0 if run.status.value == "completed" else 1

    if args.command == "create-run":
        run = app.create_run(args.dataset_version_id, json.loads(args.spec_json))
        print(json.dumps({"run_id": run.id, "status": run.status.value}, indent=2))
        return 0

    if args.command == "start-run":
        run = app.start_run(args.run_id)
        eval_reports = EvalReportRepository(args.root).list_for_run(run.id)
        artifacts = ModelArtifactRepository(args.root).list_for_run(run.id)
        print(
            json.dumps(
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "eval_report_ids": [report.id for report in eval_reports],
                    "artifact_ids": [artifact.id for artifact in artifacts],
                    "exports_dir": str(run.exports_dir(args.root)),
                },
                indent=2,
            )
        )
        return 0 if run.status.value == "completed" else 1

    if args.command == "complete-run":
        run = app.runs.complete_run(args.run_id, metrics=json.loads(args.metrics))
        print(json.dumps({"run_id": run.id, "status": run.status.value}, indent=2))
        return 0

    if args.command == "finalize-artifact":
        artifact = app.finalize_artifact(args.run_id, json.loads(args.manifest_json))
        print(json.dumps({"artifact_id": artifact.id, "path": str(artifact.path)}, indent=2))
        return 0

    if args.command == "validate-artifact":
        report = app.validate_artifact(args.artifact_id)
        print(json.dumps({"ok": report.ok, "errors": report.errors, "warnings": report.warnings}, indent=2))
        return 0

    if args.command == "ui":
        return run_foundry_ui(args.root)

    parser.error("Unknown command")
    return 2


def _default_run_spec(
    version_id: str,
    *,
    sample_rate: int,
    max_length: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmax: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": version_id,
            "sampleRate": sample_rate,
            "maxLength": max_length,
            "nFft": n_fft,
            "hopLength": hop_length,
            "nMels": n_mels,
            "fmax": fmax,
        },
        "training": {
            "epochs": epochs,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
