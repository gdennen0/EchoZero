from __future__ import annotations

import argparse
import json
from pathlib import Path

from echozero.foundry import FoundryApp
from echozero.foundry.persistence import EvalReportRepository, ModelArtifactRepository, migrate_foundry_state
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
    train_folder.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    train_folder.add_argument("--rebalance", choices=["none", "oversample"], default="none")
    train_folder.add_argument("--augment-train", action="store_true")
    train_folder.add_argument("--augment-noise-std", type=float, default=0.02)
    train_folder.add_argument("--augment-gain-jitter", type=float, default=0.10)
    train_folder.add_argument("--augment-copies", type=int, default=1)
    train_folder.add_argument("--next-level", action="store_true", help="Enable v1.5 imbalance+augmentation defaults")
    train_folder.add_argument(
        "--trainer-profile",
        choices=["baseline_v1", "stronger_v1"],
        default="baseline_v1",
        help="Training profile. Default preserves the current baseline behavior.",
    )
    train_folder.add_argument(
        "--optimizer",
        choices=["sgd_constant", "sgd_optimal"],
        default="sgd_constant",
        help="Optimizer schedule used by the selected trainer profile.",
    )
    train_folder.add_argument("--regularization-alpha", type=float, default=0.0001)
    train_folder.add_argument("--average-weights", action="store_true")
    train_folder.add_argument("--early-stopping-patience", type=int)
    train_folder.add_argument("--min-epochs", type=int)
    train_folder.add_argument("--synthetic-mix-enabled", action="store_true")
    train_folder.add_argument("--synthetic-mix-ratio", type=float, default=0.0)
    train_folder.add_argument("--synthetic-mix-cap", type=int)
    train_folder.add_argument("--gate-macro-f1-floor", type=float)
    train_folder.add_argument("--gate-max-regression-vs-reference", type=float)
    train_folder.add_argument("--gate-max-real-vs-synth-gap", type=float)
    train_folder.add_argument(
        "--gate-per-class-recall-floor",
        action="append",
        default=[],
        metavar="LABEL=VALUE",
        help="Repeatable per-class recall floor gate entry.",
    )
    train_folder.add_argument("--reference-run-id")
    train_folder.add_argument("--reference-artifact-id")

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

    sub.add_parser("migrate-state", help="Explicitly migrate legacy foundry/state JSON to v1 envelopes")

    sub.add_parser("ui", help="Launch standalone Foundry UI")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "migrate-state":
        result = migrate_foundry_state(args.root)
        print(json.dumps({"migrated": result}, indent=2))
        return 0

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
        class_weighting = args.class_weighting
        rebalance = args.rebalance
        augment_train = args.augment_train
        augment_noise_std = args.augment_noise_std
        augment_gain_jitter = args.augment_gain_jitter
        augment_copies = args.augment_copies
        if args.next_level:
            class_weighting = "balanced"
            rebalance = "oversample"
            augment_train = True
            if args.augment_noise_std == 0.02:
                augment_noise_std = 0.03
            if args.augment_gain_jitter == 0.10:
                augment_gain_jitter = 0.15
            if args.augment_copies == 1:
                augment_copies = 2

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
                class_weighting=class_weighting,
                rebalance_strategy=rebalance,
                augment_train=augment_train,
                augment_noise_std=augment_noise_std,
                augment_gain_jitter=augment_gain_jitter,
                augment_copies=augment_copies,
                trainer_profile=args.trainer_profile,
                optimizer=args.optimizer,
                regularization_alpha=args.regularization_alpha,
                average_weights=args.average_weights,
                early_stopping_patience=args.early_stopping_patience,
                min_epochs=args.min_epochs,
                synthetic_mix_enabled=args.synthetic_mix_enabled,
                synthetic_mix_ratio=args.synthetic_mix_ratio,
                synthetic_mix_cap=args.synthetic_mix_cap,
                gate_macro_f1_floor=args.gate_macro_f1_floor,
                gate_max_regression_vs_reference=args.gate_max_regression_vs_reference,
                gate_max_real_vs_synth_gap=args.gate_max_real_vs_synth_gap,
                gate_per_class_recall_floors=_parse_per_class_recall_floors(args.gate_per_class_recall_floor),
                reference_run_id=args.reference_run_id,
                reference_artifact_id=args.reference_artifact_id,
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
        print(json.dumps(report.to_contract_payload(), indent=2))
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
    class_weighting: str = "none",
    rebalance_strategy: str = "none",
    augment_train: bool = False,
    augment_noise_std: float = 0.02,
    augment_gain_jitter: float = 0.10,
    augment_copies: int = 1,
    trainer_profile: str = "baseline_v1",
    optimizer: str = "sgd_constant",
    regularization_alpha: float = 0.0001,
    average_weights: bool = False,
    early_stopping_patience: int | None = None,
    min_epochs: int | None = None,
    synthetic_mix_enabled: bool = False,
    synthetic_mix_ratio: float = 0.0,
    synthetic_mix_cap: int | None = None,
    gate_macro_f1_floor: float | None = None,
    gate_max_regression_vs_reference: float | None = None,
    gate_max_real_vs_synth_gap: float | None = None,
    gate_per_class_recall_floors: dict[str, float] | None = None,
    reference_run_id: str | None = None,
    reference_artifact_id: str | None = None,
) -> dict:
    training = {
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "seed": seed,
        "classWeighting": class_weighting,
        "rebalanceStrategy": rebalance_strategy,
        "augmentTrain": augment_train,
        "augmentNoiseStd": augment_noise_std,
        "augmentGainJitter": augment_gain_jitter,
        "augmentCopies": augment_copies,
        "trainerProfile": trainer_profile,
        "optimizer": optimizer,
        "regularizationAlpha": regularization_alpha,
        "averageWeights": average_weights,
    }
    if early_stopping_patience is not None:
        training["earlyStoppingPatience"] = early_stopping_patience
    if min_epochs is not None:
        training["minEpochs"] = min_epochs
    if synthetic_mix_enabled or synthetic_mix_ratio > 0 or synthetic_mix_cap is not None:
        training["syntheticMix"] = {
            "enabled": synthetic_mix_enabled,
            "ratio": synthetic_mix_ratio,
            "cap": synthetic_mix_cap,
        }

    promotion: dict[str, object] = {}
    gate_policy: dict[str, object] = {}
    if gate_macro_f1_floor is not None:
        gate_policy["macro_f1_floor"] = gate_macro_f1_floor
    if gate_max_regression_vs_reference is not None:
        gate_policy["max_regression_vs_reference"] = gate_max_regression_vs_reference
    if gate_max_real_vs_synth_gap is not None:
        gate_policy["max_real_vs_synth_gap"] = gate_max_real_vs_synth_gap
    if gate_per_class_recall_floors:
        gate_policy["per_class_recall_floors"] = gate_per_class_recall_floors
    if gate_policy:
        promotion["gate_policy"] = gate_policy
    if reference_run_id:
        promotion["reference_run_id"] = reference_run_id
    if reference_artifact_id:
        promotion["reference_artifact_id"] = reference_artifact_id

    payload = {
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
        "training": training,
    }
    if promotion:
        payload["promotion"] = promotion
    return payload


def _parse_per_class_recall_floors(entries: list[str]) -> dict[str, float]:
    floors: dict[str, float] = {}
    for entry in entries:
        label, separator, raw_value = entry.partition("=")
        if not separator or not label.strip() or not raw_value.strip():
            raise ValueError(f"Invalid --gate-per-class-recall-floor value: {entry}")
        floors[label.strip()] = float(raw_value)
    return floors


if __name__ == "__main__":
    raise SystemExit(main())
