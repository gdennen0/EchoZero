from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(r"C:\Users\griff\EchoZero")
RUNS_DIR = ROOT / "foundry" / "runs"
STATE_DIR = ROOT / "foundry" / "state"
TRACK_DIR = ROOT / "foundry" / "tracking"
MODEL_CARDS_DIR = TRACK_DIR / "model_cards"


def _load_state(name: str) -> dict:
    path = STATE_DIR / name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    return payload if isinstance(payload, dict) else {}


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _last_event(path: Path) -> dict:
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {}


def _metrics_from_exports(run_dir: Path) -> dict:
    metrics = _safe_read_json(run_dir / "exports" / "metrics.json")
    final_eval = metrics.get("finalEval", {})
    m = final_eval.get("metrics", {})
    return {
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "loss": m.get("loss"),
        "sample_count": m.get("sample_count"),
    }


def build_index() -> list[dict]:
    train_runs = _load_state("train_runs.json")
    artifacts = _load_state("artifacts.json")
    artifacts_by_run: dict[str, list[dict]] = {}
    for art in artifacts.values():
        artifacts_by_run.setdefault(art.get("run_id", ""), []).append(art)

    rows: list[dict] = []
    for run_dir in sorted([p for p in RUNS_DIR.glob("run_*") if p.is_dir()]):
        run_id = run_dir.name
        run_state = train_runs.get(run_id, {})
        spec = _safe_read_json(run_dir / "spec.json")
        last_event = _last_event(run_dir / "events.jsonl")
        metrics = _metrics_from_exports(run_dir)

        row = {
            "run_id": run_id,
            "status": run_state.get("status") or last_event.get("payload", {}).get("status") or "unknown",
            "dataset_version_id": run_state.get("dataset_version_id") or spec.get("data", {}).get("datasetVersionId"),
            "classification_mode": spec.get("classificationMode"),
            "trainer_profile": spec.get("training", {}).get("trainerProfile"),
            "optimizer": spec.get("training", {}).get("optimizer"),
            "epochs": spec.get("training", {}).get("epochs"),
            "created_at": run_state.get("created_at"),
            "updated_at": run_state.get("updated_at"),
            "last_event_type": last_event.get("type"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "loss": metrics.get("loss"),
            "sample_count": metrics.get("sample_count"),
            "artifact_ids": [a.get("id") for a in artifacts_by_run.get(run_id, []) if a.get("id")],
            "artifact_paths": [a.get("path") for a in artifacts_by_run.get(run_id, []) if a.get("path")],
            "run_dir": str(run_dir),
        }
        rows.append(row)

    rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return rows


def write_human_ledger(rows: list[dict]) -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Foundry Training Ledger")
    lines.append("")
    lines.append(f"Updated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append("## Latest Runs")
    lines.append("")
    lines.append("| Run ID | Status | Mode | Macro F1 | Accuracy | Artifacts |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in rows[:30]:
        mf1 = "-" if r.get("macro_f1") is None else f"{r['macro_f1']:.4f}"
        acc = "-" if r.get("accuracy") is None else f"{r['accuracy']:.4f}"
        lines.append(
            f"| {r['run_id']} | {r.get('status','-')} | {r.get('classification_mode','-')} | {mf1} | {acc} | {len(r.get('artifact_ids',[]))} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This file is human-readable summary.")
    lines.append("- Machine index lives in `training_index.json`.")
    lines.append("- Per-run cards live in `model_cards/`.")

    (TRACK_DIR / "TRAINING_LEDGER.md").write_text("\n".join(lines), encoding="utf-8")

    for r in rows:
        card = []
        card.append(f"# Model Card — {r['run_id']}")
        card.append("")
        card.append(f"- Status: `{r.get('status')}`")
        card.append(f"- Dataset Version: `{r.get('dataset_version_id')}`")
        card.append(f"- Classification Mode: `{r.get('classification_mode')}`")
        card.append(f"- Trainer Profile: `{r.get('trainer_profile')}`")
        card.append(f"- Optimizer: `{r.get('optimizer')}`")
        card.append(f"- Epochs: `{r.get('epochs')}`")
        card.append(f"- Macro F1: `{r.get('macro_f1')}`")
        card.append(f"- Accuracy: `{r.get('accuracy')}`")
        card.append(f"- Artifact IDs: `{', '.join(r.get('artifact_ids', [])) or '-'}`")
        card.append(f"- Run Dir: `{r.get('run_dir')}`")
        (MODEL_CARDS_DIR / f"{r['run_id']}.md").write_text("\n".join(card), encoding="utf-8")


def main() -> None:
    rows = build_index()
    TRACK_DIR.mkdir(parents=True, exist_ok=True)
    (TRACK_DIR / "training_index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_human_ledger(rows)
    print(json.dumps({
        "runs_indexed": len(rows),
        "ledger": str(TRACK_DIR / "TRAINING_LEDGER.md"),
        "index": str(TRACK_DIR / "training_index.json"),
        "cards_dir": str(MODEL_CARDS_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
