from __future__ import annotations

import argparse
import json
from pathlib import Path

from echozero.foundry import FoundryApp


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

    if args.command == "create-run":
        run = app.create_run(args.dataset_version_id, json.loads(args.spec_json))
        print(json.dumps({"run_id": run.id, "status": run.status.value}, indent=2))
        return 0

    if args.command == "start-run":
        run = app.start_run(args.run_id)
        print(json.dumps({"run_id": run.id, "status": run.status.value}, indent=2))
        return 0

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

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
