import subprocess
from datetime import datetime, UTC

from run_focus_kick_snare_batch import (
    ROOT,
    BATCH_DIR,
    migrate_foundry_state,
    FoundryApp,
    DatasetVersionRepository,
    select_base_version_fallback,
    run_one,
    build_spec,
    write_batch_status,
)


def notify(text: str):
    try:
        subprocess.run(["openclaw", "system", "event", "--text", text, "--mode", "now"], cwd=str(ROOT), check=False)
    except Exception:
        pass


def main():
    batch_id = f"test_kick_snare_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    migrate_foundry_state(ROOT)
    app = FoundryApp(ROOT)
    repo = DatasetVersionRepository(ROOT)
    base = select_base_version_fallback(repo)

    status = {
        "batch_id": batch_id,
        "type": "test",
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "base_dataset_version_id": base.id,
        "runs": [],
    }
    write_batch_status(batch_id, status)
    notify(f"🚀 Started test batch {batch_id} (kick/snare + multiclass)")

    status["runs"].append(run_one(app, repo, base, label="kick", model_type="crnn", seed=5101))
    write_batch_status(batch_id, status)
    status["runs"].append(run_one(app, repo, base, label="snare", model_type="crnn", seed=5102))
    write_batch_status(batch_id, status)

    m_spec = build_spec(base.id, seed=5103, gate_floor=0.90, model_type="crnn", per_class={"kick": 0.95, "snare": 0.95})
    m_spec["training"]["epochs"] = 40
    m_spec["training"]["earlyStoppingPatience"] = 8
    m_spec["training"]["minEpochs"] = 10
    run = app.create_run(base.id, m_spec)
    run = app.start_run(run.id)
    arts = app.artifacts._artifact_repo.list_for_run(run.id)
    artifact_id = arts[0].id if arts else None
    validation = app.validate_artifact(artifact_id) if artifact_id else None
    status["runs"].append(
        {
            "mode": "multiclass",
            "model_type": "crnn",
            "run_id": run.id,
            "status": run.status.value,
            "artifact_id": artifact_id,
            "artifact_ok": None if validation is None else bool(validation.ok),
            "artifact_errors": [] if validation is None else list(validation.errors),
            "dataset_version_id": base.id,
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )

    status["status"] = "completed"
    status["finished_at"] = datetime.now(UTC).isoformat()
    write_batch_status(batch_id, status)
    ok = sum(1 for r in status["runs"] if r.get("status") == "completed")
    notify(f"✅ Test batch done {batch_id}: {ok}/{len(status['runs'])} completed. File: {BATCH_DIR / (batch_id + '.json')}")


if __name__ == "__main__":
    main()
