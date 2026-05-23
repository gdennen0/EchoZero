from __future__ import annotations

import argparse
import json
from pathlib import Path

from echozero.foundry import FoundryApp
from echozero.foundry.domain import LibrarySampleState
from echozero.foundry.persistence import (
    EvalReportRepository,
    ModelArtifactRepository,
    migrate_foundry_state,
)
from echozero.foundry.review_server import serve_review_session
from echozero.foundry.services.project_specialized_model_service import (
    ProjectSpecializedModelService,
)
from echozero.foundry.services.shared_review_specialized_model_service import (
    SharedReviewSpecializedModelService,
)
from echozero.foundry.ui import run_foundry_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ez-foundry", description="EchoZero Foundry standalone CLI"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Foundry workspace root")

    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("create-dataset")
    ds.add_argument("name")

    ingest = sub.add_parser("ingest-folder")
    ingest.add_argument("dataset_id")
    ingest.add_argument("folder")

    sample_library_record = sub.add_parser("record-sample-library")
    sample_library_record.add_argument("version_id")
    sample_library_record.add_argument(
        "--state",
        choices=[state.value for state in LibrarySampleState],
        default=LibrarySampleState.APPROVED.value,
    )

    sample_library_summary = sub.add_parser("sample-library-summary")

    sample_library_train = sub.add_parser("train-sample-library")
    sample_library_train.add_argument("name")
    sample_library_train.add_argument("--epochs", type=int, default=4)
    sample_library_train.add_argument("--scope", default="local.default")
    sample_library_train.add_argument(
        "--refresh-version-id",
        help="Record this dataset version into the sample library before training.",
    )
    sample_library_train.add_argument(
        "--refresh-state",
        choices=[state.value for state in LibrarySampleState],
        default=LibrarySampleState.APPROVED.value,
        help="Library state to apply when --refresh-version-id is used.",
    )

    review_import = sub.add_parser("import-review-session")
    review_import.add_argument("items_path")
    review_import.add_argument("--name")
    review_import.add_argument("--session-id")

    review_project = sub.add_parser("create-project-review-session")
    review_project.add_argument("project_path")
    review_project.add_argument("--name")
    review_project.add_argument("--session-id")
    review_project.add_argument("--song-id")
    review_project.add_argument("--song-version-id")
    review_project.add_argument("--layer-id")
    review_project.add_argument("--questionable-score-threshold", type=float)
    review_project.add_argument("--item-limit", type=int)

    review_import_folder = sub.add_parser("import-review-folder")
    review_import_folder.add_argument("folder")
    review_import_folder.add_argument("--name")
    review_import_folder.add_argument("--session-id")
    review_import_folder.add_argument("--target-class")

    review_serve = sub.add_parser("serve-review-session")
    review_serve.add_argument("session_id")
    review_serve.add_argument("--host", default="127.0.0.1")
    review_serve.add_argument("--port", type=int, default=8421)

    extract_review_dataset = sub.add_parser("extract-project-review-dataset")
    extract_review_dataset.add_argument("project_path")
    extract_review_dataset.add_argument("--project-ref")
    extract_review_dataset.add_argument("--song-id")
    extract_review_dataset.add_argument("--song-version-id")
    extract_review_dataset.add_argument("--layer-id")
    extract_review_dataset.add_argument("--queue-source-kind", default="ez_project")

    extract_signal = sub.add_parser("extract-review-signal")
    extract_signal.add_argument("session_id")
    extract_signal.add_argument("signal_id")

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
    train_folder.add_argument(
        "--next-level", action="store_true", help="Enable v1.5 imbalance+augmentation defaults"
    )
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

    train_grouped_binary = sub.add_parser("train-grouped-binary-models")
    train_grouped_binary.add_argument("source_version_id")
    train_grouped_binary.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="TARGET=LABEL1,LABEL2",
        help=(
            "Repeatable grouped-binary model spec. Use TARGET alone for TARGET vs other, "
            "or TARGET=LABEL1,LABEL2 to collapse several source labels into one target class."
        ),
    )
    train_grouped_binary.add_argument("--install-runtime", action="store_true")
    train_grouped_binary.add_argument("--models-dir", type=Path)

    train_noah_shared = sub.add_parser("train-noah-kahan-shared-review-models")
    train_noah_shared.add_argument("--review-sample-root", type=Path)
    train_noah_shared.add_argument("--artist-name", default="Noah Kahan")
    train_noah_shared.add_argument("--label", action="append", choices=["kick", "snare"])
    train_noah_shared.add_argument("--kick-initial-model", type=Path)
    train_noah_shared.add_argument("--snare-initial-model", type=Path)
    train_noah_shared.add_argument("--no-warm-start", action="store_true")

    review_sample_doctor = sub.add_parser("review-sample-doctor")
    review_sample_doctor.add_argument("review_sample_root", type=Path)
    review_sample_doctor.add_argument("--output-root", type=Path)
    review_sample_doctor.add_argument(
        "--label",
        action="append",
        help="Limit the doctor run to one class folder. Repeat for multiple labels.",
    )
    review_sample_doctor.add_argument(
        "--conflict-policy",
        choices=["quarantine", "latest-review-wins", "latest_review_wins"],
        default="quarantine",
        help=(
            "How exact-content conflicts are handled. quarantine keeps the "
            "conservative default; latest-review-wins recovers the newest "
            "reviewed row into the clean pool and quarantines superseded rows."
        ),
    )

    review_sample_reexport = sub.add_parser("reexport-ez-review-samples")
    review_sample_reexport.add_argument("project_path", type=Path, nargs="+")
    review_sample_reexport.add_argument("--output-root", type=Path, required=True)
    review_sample_reexport.add_argument(
        "--label",
        action="append",
        help="Limit the re-export to one reviewed class. Repeat for multiple labels.",
    )
    review_sample_reexport.add_argument(
        "--include-promoted-events",
        action="store_true",
        help="Also export promoted model-detected events that have not been explicitly reviewed.",
    )
    review_sample_reexport.add_argument("--overwrite", action="store_true")

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

    install_runtime = sub.add_parser("install-runtime-bundle")
    install_runtime.add_argument("artifact_ref")
    install_runtime.add_argument("--label")
    install_runtime.add_argument("--bundle-name")
    install_runtime.add_argument("--models-dir", type=Path)

    val = sub.add_parser("validate-artifact")
    val.add_argument("artifact_id")

    sub.add_parser(
        "migrate-state", help="Explicitly migrate legacy foundry/state JSON to v1 envelopes"
    )

    sub.add_parser("ui", help="Launch standalone Foundry UI")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "migrate-state":
        result = migrate_foundry_state(args.root)
        print(json.dumps({"migrated": result}, indent=2))
        return 0

    if args.command == "review-sample-doctor":
        from echozero.foundry.services.review_sample_doctor_service import (
            ReviewSampleDoctorService,
        )

        result = ReviewSampleDoctorService().audit_and_repair(
            args.review_sample_root,
            output_root=args.output_root,
            labels=tuple(args.label) if args.label else None,
            conflict_policy=args.conflict_policy,
        )
        print(json.dumps(result.to_payload(), indent=2))
        return 0

    if args.command == "reexport-ez-review-samples":
        from echozero.foundry.services.ez_review_sample_reexport_service import (
            EzReviewSampleReexportService,
        )

        result = EzReviewSampleReexportService().reexport(
            list(args.project_path),
            output_root=args.output_root,
            labels=tuple(args.label) if args.label else None,
            include_promoted_events=args.include_promoted_events,
            overwrite=args.overwrite,
        )
        print(json.dumps(result.to_payload(), indent=2))
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

    if args.command == "record-sample-library":
        records = app.record_sample_library_version(
            args.version_id,
            state=LibrarySampleState(args.state),
        )
        print(
            json.dumps(
                {
                    "version_id": args.version_id,
                    "recorded_count": len(records),
                    "state": args.state,
                    "library_sample_ids": [record.id for record in records],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "sample-library-summary":
        print(json.dumps(app.summarize_sample_library(), indent=2))
        return 0

    if args.command == "train-sample-library":
        refreshed_sample_count = 0
        if args.refresh_version_id:
            version = app.datasets.get_version(args.refresh_version_id)
            if version is None:
                raise ValueError(f"DatasetVersion not found: {args.refresh_version_id}")
            refreshed_sample_count = len(version.samples)
        run = app.kickoff_sample_library_run(
            name=args.name,
            epochs=args.epochs,
            scope=args.scope,
            refresh_version_id=args.refresh_version_id,
            refresh_state=LibrarySampleState(args.refresh_state),
        )
        eval_reports = EvalReportRepository(args.root).list_for_run(run.id)
        artifacts = ModelArtifactRepository(args.root).list_for_run(run.id)
        print(
            json.dumps(
                {
                    "run_id": run.id,
                    "dataset_version_id": run.dataset_version_id,
                    "status": run.status.value,
                    "scope": args.scope,
                    "refresh_version_id": args.refresh_version_id,
                    "refresh_state": args.refresh_state if args.refresh_version_id else None,
                    "refreshed_sample_count": refreshed_sample_count,
                    "eval_report_ids": [report.id for report in eval_reports],
                    "artifact_ids": [artifact.id for artifact in artifacts],
                    "exports_dir": str(run.exports_dir(args.root)),
                },
                indent=2,
            )
        )
        return 0 if run.status.value == "completed" else 1

    if args.command == "import-review-session":
        session = app.reviews.import_session_file(
            args.items_path,
            name=args.name,
            session_id=args.session_id,
        )
        _print_review_session_summary(session)
        return 0

    if args.command == "create-project-review-session":
        session = app.reviews.create_project_session(
            args.project_path,
            name=args.name,
            session_id=args.session_id,
            song_id=args.song_id,
            song_version_id=args.song_version_id,
            layer_id=args.layer_id,
            questionable_score_threshold=args.questionable_score_threshold,
            item_limit=args.item_limit,
        )
        _print_review_session_summary(session)
        return 0

    if args.command == "import-review-folder":
        session = app.reviews.import_session_folder(
            args.folder,
            name=args.name,
            session_id=args.session_id,
            target_class=args.target_class,
        )
        _print_review_session_summary(session)
        return 0

    if args.command == "serve-review-session":
        return serve_review_session(
            args.root,
            args.session_id,
            host=args.host,
            port=args.port,
        )

    if args.command == "extract-project-review-dataset":
        version = app.extract_project_review_dataset(
            args.project_path,
            project_ref=args.project_ref,
            song_id=args.song_id,
            song_version_id=args.song_version_id,
            layer_id=args.layer_id,
            queue_source_kind=args.queue_source_kind,
        )
        print(
            json.dumps(
                {
                    "operation": "review_extraction",
                    "dataset_id": version.dataset_id,
                    "version_id": version.id,
                    "version": version.version,
                    "sample_count": int(version.stats.get("sample_count", len(version.samples))),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "extract-review-signal":
        payload = app.extract_review_signal(
            session_id=args.session_id,
            signal_id=args.signal_id,
        )
        payload["operation"] = "review_extraction"
        print(json.dumps(payload, indent=2))
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
        dataset = app.datasets.create_dataset(
            args.name, source_ref=str(Path(args.folder).resolve())
        )
        version = app.datasets.ingest_from_folder(
            dataset.id, args.folder, sample_rate=args.sample_rate
        )
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
                gate_per_class_recall_floors=_parse_per_class_recall_floors(
                    args.gate_per_class_recall_floor
                ),
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

    if args.command == "train-grouped-binary-models":
        payloads: list[dict[str, object]] = []
        for target_label, source_labels in _parse_grouped_model_specs(args.model):
            derived = app.datasets.derive_binary_dataset_version(
                args.source_version_id,
                positive_label=target_label,
                positive_aliases=tuple(source_labels),
            )
            if not derived.split_plan.get("assignments"):
                app.plan_version(
                    derived.id,
                    validation_split=0.15,
                    test_split=0.10,
                    seed=42,
                    balance_strategy="none",
                )
                refreshed = app.datasets.get_version(derived.id)
                if refreshed is None:
                    raise ValueError(f"Derived dataset version disappeared: {derived.id}")
                derived = refreshed
            run = app.create_run(derived.id, _monster_binary_run_spec(derived))
            run = app.start_run(run.id)
            eval_reports = EvalReportRepository(args.root).list_for_run(run.id)
            artifacts = ModelArtifactRepository(args.root).list_for_run(run.id)
            artifact_ids = [artifact.id for artifact in artifacts]
            installed_bundle = None
            if args.install_runtime and artifacts:
                latest_artifact = sorted(artifacts, key=lambda artifact: artifact.created_at)[-1]
                bundle = app.runtime_bundles.install_binary_drum_artifact(
                    latest_artifact.id,
                    bundle_label=target_label,
                    bundle_name=ProjectSpecializedModelService._bundle_name(
                        label=target_label,
                        artifact_id=latest_artifact.id,
                    ),
                    models_dir=args.models_dir,
                )
                installed_bundle = {
                    "label": bundle.label,
                    "bundle_name": bundle.bundle_name,
                    "bundle_dir": str(bundle.bundle_dir),
                    "manifest_path": str(bundle.manifest_path),
                    "weights_path": str(bundle.weights_path),
                }
            payloads.append(
                {
                    "target_label": target_label,
                    "source_labels": source_labels,
                    "dataset_version_id": derived.id,
                    "run_id": run.id,
                    "status": run.status.value,
                    "eval_report_ids": [report.id for report in eval_reports],
                    "artifact_ids": artifact_ids,
                    "exports_dir": str(run.exports_dir(args.root)),
                    "installed_bundle": installed_bundle,
                }
            )
        print(
            json.dumps(
                {
                    "source_version_id": args.source_version_id,
                    "models": payloads,
                },
                indent=2,
            )
        )
        return 0 if all(item["status"] == "completed" for item in payloads) else 1

    if args.command == "train-noah-kahan-shared-review-models":
        labels = tuple(args.label or ["kick", "snare"])
        initial_model_paths = {
            label: path
            for label, path in {
                "kick": args.kick_initial_model,
                "snare": args.snare_initial_model,
            }.items()
            if path is not None
        }
        service = SharedReviewSpecializedModelService(args.root)
        result = service.create_artist_drum_models(
            artist_name=args.artist_name,
            review_sample_root=args.review_sample_root,
            labels=labels,
            source_labels=None,
            initial_model_paths=initial_model_paths,
            warm_start=not args.no_warm_start,
        )
        print(
            json.dumps(
                {
                    "artist_name": result.artist_name,
                    "source_dataset_id": result.source_dataset_id,
                    "source_dataset_version_id": result.source_dataset_version_id,
                    "review_sample_root": str(result.review_sample_root),
                    "promotions": [
                        {
                            "label": promotion.label,
                            "dataset_version_id": promotion.dataset_version_id,
                            "run_id": promotion.run_id,
                            "artifact_id": promotion.artifact_id,
                            "manifest_path": str(promotion.manifest_path),
                            "weights_path": str(promotion.weights_path),
                            "initial_model_path": (
                                None
                                if promotion.initial_model_path is None
                                else str(promotion.initial_model_path)
                            ),
                        }
                        for promotion in result.promotions
                    ],
                },
                indent=2,
            )
        )
        return 0

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

    if args.command == "install-runtime-bundle":
        bundle = app.runtime_bundles.install_binary_drum_artifact(
            args.artifact_ref,
            bundle_label=args.label,
            bundle_name=args.bundle_name,
            models_dir=args.models_dir,
        )
        print(
            json.dumps(
                {
                    "label": bundle.label,
                    "bundle_name": bundle.bundle_name,
                    "bundle_dir": str(bundle.bundle_dir),
                    "manifest_path": str(bundle.manifest_path),
                    "weights_path": str(bundle.weights_path),
                    "artifact_id": bundle.artifact_id,
                    "run_id": bundle.run_id,
                },
                indent=2,
            )
        )
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


def _monster_binary_run_spec(version) -> dict[str, object]:
    return {
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "binary",
        "model": {"type": "crnn"},
        "data": {
            "datasetVersionId": version.id,
            "sampleRate": version.sample_rate,
            "maxLength": version.sample_rate,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {
            "epochs": 12,
            "batchSize": 4,
            "learningRate": 0.001,
            "seed": 42,
            "trainerProfile": "stronger_v1",
            "optimizer": "adamw",
            "regularizationAlpha": 0.00005,
            "weightDecay": 0.0001,
            "averageWeights": True,
            "earlyStoppingPatience": 4,
            "minEpochs": 4,
            "classWeighting": "balanced",
            "rebalanceStrategy": "oversample",
            "augmentTrain": True,
            "augmentNoiseStd": 0.03,
            "augmentGainJitter": 0.15,
            "augmentCopies": 2,
            "syntheticMix": {"enabled": True, "ratio": 0.35, "cap": 400},
        },
    }


def _parse_grouped_model_specs(raw_specs: list[str]) -> list[tuple[str, list[str]]]:
    specs: list[tuple[str, list[str]]] = []
    for raw_spec in raw_specs:
        spec = str(raw_spec).strip()
        if not spec:
            raise ValueError("Grouped model spec must be non-empty.")
        if "=" not in spec:
            label = spec.lower()
            specs.append((label, [label]))
            continue
        target_raw, labels_raw = spec.split("=", 1)
        target_label = target_raw.strip().lower()
        source_labels = [label.strip().lower() for label in labels_raw.split(",") if label.strip()]
        if not target_label:
            raise ValueError(f"Grouped model spec '{raw_spec}' is missing a target label.")
        if not source_labels:
            raise ValueError(
                f"Grouped model spec '{raw_spec}' must include at least one source label."
            )
        specs.append((target_label, source_labels))
    return specs


def _print_review_session_summary(session) -> None:
    print(
        json.dumps(
            {
                "session_id": session.id,
                "name": session.name,
                "items": len(session.items),
                "classes": session.class_map,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
