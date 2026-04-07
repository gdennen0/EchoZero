import json
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, UTC
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\griff\EchoZero")
sys.path.insert(0, str(ROOT))

from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import DatasetVersionRepository, migrate_foundry_state

BATCH_DIR = ROOT / "foundry" / "tracking" / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def load_versions(repo: DatasetVersionRepository):
    path = ROOT / "foundry" / "state" / "dataset_versions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("items", payload) if isinstance(payload, dict) else {}
    out = []
    for version_id in rows.keys():
        v = repo.get(version_id)
        if v is not None:
            out.append(v)
    return out


def select_latest_planned_multiclass(versions):
    valid = []
    for v in versions:
        sp = v.split_plan or {}
        if not sp.get("train_ids") or not sp.get("val_ids"):
            continue
        if v.taxonomy.get("namespace") != "percussion.one_shot":
            continue
        if len(v.class_map) < 2:
            continue
        valid.append(v)
    if not valid:
        raise RuntimeError("No planned dataset version found")
    valid.sort(key=lambda x: x.created_at, reverse=True)
    return valid[0]


def make_ovr_version(base, target_label, repo: DatasetVersionRepository):
    v = deepcopy(base)
    v.id = f"dsv_ovr_{target_label}_{uuid.uuid4().hex[:8]}"
    v.version = (base.version or 1) + 2000
    v.class_map = [target_label, "other"]
    v.label_policy = {
        "schema": "foundry.label_policy.v1",
        "classification_mode": "multiclass",
        "allowed_labels": [target_label, "other"],
        "unknown_label": None,
        "unit": "one_shot",
    }
    for s in v.samples:
        s.label = target_label if s.label == target_label else "other"
        prov = dict(s.source_provenance or {})
        prov["ovr_target_label"] = target_label
        s.source_provenance = prov
    counts = Counter(s.label for s in v.samples)
    v.stats = dict(v.stats or {})
    v.stats["class_counts"] = dict(counts)
    v.stats["sample_count"] = len(v.samples)
    v.manifest_hash = f"ovr-beefy:{target_label}:{base.manifest_hash}"
    v.lineage = {
        "derived_from_dataset_version_id": base.id,
        "mode": "one_vs_rest",
        "target_label": target_label,
        "generated_at": datetime.now(UTC).isoformat(),
        "batch": "beefy_v2",
    }
    return repo.save(v)


def build_spec(
    dsv_id: str,
    seed: int,
    gate_floor: float,
    *,
    model_type: str,
    per_class: dict | None = None,
):
    spec = {
        "model": {"type": model_type},
        "schema": "foundry.train_run_spec.v1",
        "classificationMode": "multiclass",
        "data": {
            "datasetVersionId": dsv_id,
            "sampleRate": 22050,
            "maxLength": 22050,
            "nFft": 2048,
            "hopLength": 512,
            "nMels": 128,
            "fmax": 8000,
        },
        "training": {
            "epochs": 48,
            "batchSize": 12,
            "learningRate": 0.004,
            "seed": seed,
            "trainerProfile": "stronger_v1",
            "optimizer": "adamw",
            "classWeighting": "balanced",
            "rebalanceStrategy": "oversample",
            "augmentTrain": True,
            "augmentNoiseStd": 0.04,
            "augmentGainJitter": 0.20,
            "augmentCopies": 4,
            "averageWeights": True,
            "regularizationAlpha": 0.0003,
            "weightDecay": 0.0002,
            "gradientClipNorm": 1.0,
            "earlyStoppingPatience": 6,
            "minEpochs": 8,
            "syntheticMix": {"enabled": True, "ratio": 0.30, "cap": None},
        },
        "promotion": {
            "gate_policy": {
                "macro_f1_floor": gate_floor,
                "max_real_vs_synth_gap": 0.08,
            }
        },
    }
    if per_class:
        spec["promotion"]["gate_policy"]["per_class_recall_floors"] = per_class
    return spec


def write_batch_status(batch_id: str, payload: dict):
    out = BATCH_DIR / f"{batch_id}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    batch_id = f"beefy_v2_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    migrate_foundry_state(ROOT)
    app = FoundryApp(ROOT)
    repo = DatasetVersionRepository(ROOT)
    base = select_latest_planned_multiclass(load_versions(repo))

    status = {
        "batch_id": batch_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "base_dataset_version_id": base.id,
        "class_map": list(base.class_map),
        "runs": [],
    }
    write_batch_status(batch_id, status)
    print(json.dumps({"milestone":"started","batch_id":batch_id,"planned_runs":len(base.class_map)+1}, indent=2), flush=True)

    failure_sent = False
    total = len(base.class_map) + 1
    done = 0

    for i, label in enumerate(base.class_map):
        ovr = make_ovr_version(base, label, repo)
        model_type = "cnn" if i % 2 == 0 else "crnn"
        spec = build_spec(ovr.id, seed=500+i, gate_floor=0.82, model_type=model_type)
        run = app.create_run(ovr.id, spec)
        run = app.start_run(run.id)
        artifacts = app.artifacts._artifact_repo.list_for_run(run.id)
        artifact_id = artifacts[0].id if artifacts else None
        validation = app.validate_artifact(artifact_id) if artifact_id else None
        done += 1
        rec = {
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
        status["runs"].append(rec)
        write_batch_status(batch_id, status)
        if run.status.value == "failed" and not failure_sent:
            print(json.dumps({"milestone":"first_failure","done":done,"total":total,"run":rec}, indent=2), flush=True)
            failure_sent = True
        if done == 1 or done % 3 == 0:
            print(json.dumps({"milestone":"progress","done":done,"total":total,"last":rec}, indent=2), flush=True)

    per_class = {c: 0.72 for c in base.class_map}
    m_spec = build_spec(base.id, seed=777, gate_floor=0.84, model_type="crnn", per_class=per_class)
    run = app.create_run(base.id, m_spec)
    run = app.start_run(run.id)
    artifacts = app.artifacts._artifact_repo.list_for_run(run.id)
    artifact_id = artifacts[0].id if artifacts else None
    validation = app.validate_artifact(artifact_id) if artifact_id else None
    done += 1
    rec = {
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
    status["runs"].append(rec)

    status["status"] = "completed"
    status["finished_at"] = datetime.now(UTC).isoformat()
    write_batch_status(batch_id, status)
    success = [r for r in status["runs"] if r.get("status") == "completed"]
    failed = [r for r in status["runs"] if r.get("status") != "completed"]
    valid = [r for r in status["runs"] if r.get("artifact_ok") is True]
    summary = {
        "milestone": "completed",
        "batch_id": batch_id,
        "done": done,
        "total": total,
        "completed": len(success),
        "failed": len(failed),
        "artifacts_valid": len(valid),
        "models_used": sorted({r.get("model_type") for r in status["runs"]}),
        "runs": [
            {
                "run_id": r.get("run_id"),
                "mode": r.get("mode"),
                "model_type": r.get("model_type"),
                "status": r.get("status"),
                "artifact_ok": r.get("artifact_ok"),
            }
            for r in status["runs"]
        ],
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
