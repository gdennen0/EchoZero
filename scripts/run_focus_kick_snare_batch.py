import json
import subprocess
from datetime import datetime, UTC
from pathlib import Path

from run_beefy_batch import (
    ROOT,
    BATCH_DIR,
    migrate_foundry_state,
    FoundryApp,
    DatasetVersionRepository,
    load_versions,
    select_latest_planned_multiclass,
    make_ovr_version,
    build_spec,
    write_batch_status,
)


def select_base_version_fallback(repo: DatasetVersionRepository):
    versions = load_versions(repo)
    try:
        return select_latest_planned_multiclass(versions)
    except Exception:
        pass

    # Fallback: any version with deterministic split plan and >=2 classes.
    candidates = []
    for v in versions:
        sp = v.split_plan or {}
        if sp.get("train_ids") and sp.get("val_ids") and len(v.class_map) >= 2:
            candidates.append(v)
    if not candidates:
        raise RuntimeError("No eligible dataset version found for focus batch")
    candidates.sort(key=lambda x: x.created_at, reverse=True)
    return candidates[0]


def notify(text: str):
    try:
        subprocess.run(["openclaw", "system", "event", "--text", text, "--mode", "now"], cwd=str(ROOT), check=False)
    except Exception:
        pass


def run_one(app, repo, base, label: str, model_type: str, seed: int):
    ovr = make_ovr_version(base, label, repo)
    spec = build_spec(ovr.id, seed=seed, gate_floor=0.90, model_type=model_type)
    # strengthen critical-class behavior
    spec.setdefault("training", {})["epochs"] = 60
    spec["training"]["earlyStoppingPatience"] = 10
    spec["training"]["minEpochs"] = 12
    spec["promotion"]["gate_policy"]["per_class_recall_floors"] = {label: 0.95}

    run = app.create_run(ovr.id, spec)
    run = app.start_run(run.id)
    arts = app.artifacts._artifact_repo.list_for_run(run.id)
    artifact_id = arts[0].id if arts else None
    validation = app.validate_artifact(artifact_id) if artifact_id else None
    return {
        "mode": "binary_ovr",
        "target_label": label,
        "model_type": model_type,
        "run_id": run.id,
        "status": run.status.value,
        "artifact_id": artifact_id,
        "artifact_ok": None if validation is None else bool(validation.ok),
        "artifact_errors": [] if validation is None else list(validation.errors),
        "dataset_version_id": ovr.id,
        "completed_at": datetime.now(UTC).isoformat(),
    }


def main():
    batch_id = f"focus_kick_snare_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    migrate_foundry_state(ROOT)
    app = FoundryApp(ROOT)
    repo = DatasetVersionRepository(ROOT)
    base = select_base_version_fallback(repo)

    status = {
        "batch_id": batch_id,
        "type": "focus_kick_snare",
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "base_dataset_version_id": base.id,
        "runs": [],
    }
    write_batch_status(batch_id, status)
    notify(f"🚀 Started focus batch {batch_id}: kick/snare priority.")

    for i, (label, model) in enumerate([("kick", "crnn"), ("snare", "crnn"), ("kick", "cnn"), ("snare", "cnn")], start=1):
        rec = run_one(app, repo, base, label=label, model_type=model, seed=3000 + i)
        status["runs"].append(rec)
        write_batch_status(batch_id, status)

    per_class = {"kick": 0.95, "snare": 0.95}
    m_spec = build_spec(base.id, seed=3999, gate_floor=0.90, model_type="crnn", per_class=per_class)
    m_spec["training"]["epochs"] = 64
    m_spec["training"]["earlyStoppingPatience"] = 10
    m_spec["training"]["minEpochs"] = 14
    run = app.create_run(base.id, m_spec)
    run = app.start_run(run.id)
    arts = app.artifacts._artifact_repo.list_for_run(run.id)
    artifact_id = arts[0].id if arts else None
    validation = app.validate_artifact(artifact_id) if artifact_id else None
    status["runs"].append({
        "mode": "multiclass",
        "model_type": "crnn",
        "run_id": run.id,
        "status": run.status.value,
        "artifact_id": artifact_id,
        "artifact_ok": None if validation is None else bool(validation.ok),
        "artifact_errors": [] if validation is None else list(validation.errors),
        "dataset_version_id": base.id,
        "completed_at": datetime.now(UTC).isoformat(),
    })

    status["status"] = "completed"
    status["finished_at"] = datetime.now(UTC).isoformat()
    write_batch_status(batch_id, status)
    ok = sum(1 for r in status["runs"] if r.get("status") == "completed")
    notify(f"✅ Focus batch done {batch_id}: {ok}/{len(status['runs'])} completed. File: {BATCH_DIR / (batch_id + '.json')}")


if __name__ == "__main__":
    main()
