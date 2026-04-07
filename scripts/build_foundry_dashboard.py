from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(r"C:\Users\griff\EchoZero")
TRACK_DIR = ROOT / "foundry" / "tracking"
RUNS_DIR = ROOT / "foundry" / "runs"
STATE_DIR = ROOT / "foundry" / "state"


def _load_state(name: str) -> dict:
    path = STATE_DIR / name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "items" in payload:
        return payload["items"]
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def build_runs_payload() -> list[dict]:
    train_runs = _load_state("train_runs.json")
    artifacts = _load_state("artifacts.json")
    artifacts_by_run: dict[str, list[dict]] = {}
    for art in artifacts.values():
        artifacts_by_run.setdefault(art.get("run_id", ""), []).append(art)

    runs: list[dict] = []
    for run_dir in sorted([p for p in RUNS_DIR.glob("run_*") if p.is_dir()]):
        run_id = run_dir.name
        state = train_runs.get(run_id, {})
        spec = _read_json(run_dir / "spec.json")
        metrics = _read_json(run_dir / "exports" / "metrics.json")
        summary = _read_json(run_dir / "exports" / "run_summary.json")
        events = _read_events(run_dir / "events.jsonl")
        last_event = events[-1] if events else {}

        runs.append(
            {
                "run_id": run_id,
                "status": state.get("status") or last_event.get("payload", {}).get("status") or "unknown",
                "created_at": state.get("created_at"),
                "updated_at": state.get("updated_at"),
                "dataset_version_id": state.get("dataset_version_id") or spec.get("data", {}).get("datasetVersionId"),
                "classification_mode": spec.get("classificationMode"),
                "trainer_profile": spec.get("training", {}).get("trainerProfile"),
                "optimizer": spec.get("training", {}).get("optimizer"),
                "epochs": spec.get("training", {}).get("epochs"),
                "learning_rate": spec.get("training", {}).get("learningRate"),
                "final_metrics": metrics.get("finalEval", {}).get("metrics", {}),
                "per_class_metrics": metrics.get("finalEval", {}).get("per_class_metrics", {}),
                "confusion": metrics.get("finalEval", {}).get("confusion", {}),
                "checkpoints": metrics.get("checkpoints", []),
                "synthetic_eval": metrics.get("syntheticEval"),
                "run_summary": summary,
                "events_tail": events[-8:],
                "artifacts": [
                    {
                        "artifact_id": a.get("id"),
                        "path": a.get("path"),
                        "created_at": a.get("created_at"),
                        "artifact_version": a.get("artifact_version"),
                    }
                    for a in artifacts_by_run.get(run_id, [])
                ],
                "run_dir": str(run_dir),
            }
        )

    runs.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return runs


def write_dashboard(runs: list[dict]) -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)

    payload = {"updated_at": datetime.now(UTC).isoformat(), "count": len(runs), "runs": runs}
    (TRACK_DIR / "training_runs_full.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Foundry Training Dashboard</title>
  <style>
    body { font-family: Inter, Segoe UI, Arial, sans-serif; margin:0; background:#0f1115; color:#e8edf2; }
    .wrap { display:grid; grid-template-columns: 320px 1fr; height:100vh; }
    .left { border-right:1px solid #2a2f3a; overflow:auto; }
    .right { overflow:auto; padding:16px 20px; }
    .top { padding:12px; border-bottom:1px solid #2a2f3a; position:sticky; top:0; background:#0f1115; z-index:2; }
    .run { padding:10px 12px; border-bottom:1px solid #1e2330; cursor:pointer; }
    .run:hover { background:#151a22; }
    .run.active { background:#1b2230; }
    .muted { color:#9aa5b1; font-size:12px; }
    .pill { display:inline-block; font-size:11px; padding:2px 8px; border-radius:999px; margin-right:6px; }
    .ok { background:#173a27; color:#7de2a8; }
    .bad { background:#4a1f1f; color:#ff9a9a; }
    table { width:100%; border-collapse:collapse; margin-top:10px; }
    th, td { border-bottom:1px solid #2a2f3a; text-align:left; padding:8px; font-size:13px; }
    h2, h3 { margin:10px 0; }
    .grid { display:grid; grid-template-columns: repeat(4,minmax(120px,1fr)); gap:10px; }
    .card { background:#141923; border:1px solid #2a2f3a; border-radius:10px; padding:10px; }
    .num { font-size:20px; font-weight:700; }
    .controls button { margin-right:8px; }
    button { background:#1f2735; color:#e8edf2; border:1px solid #334159; padding:6px 10px; border-radius:8px; cursor:pointer; }
    .best-table td,.best-table th { font-size:12px; }
  </style>
</head>
<body>
<div class=\"wrap\">
  <div class=\"left\">
    <div class=\"top\">
      <div><strong>Foundry Runs</strong></div>
      <div class=\"muted\">Flip through full results</div>
      <div class=\"muted\" id=\"summaryStats\" style=\"margin-top:6px\"></div>
      <input id=\"search\" placeholder=\"Search run/class/status\" style=\"width:100%;margin-top:8px;padding:6px;border-radius:8px;border:1px solid #334159;background:#111722;color:#e8edf2;\" />
      <label class=\"muted\" style=\"display:block;margin-top:8px\"><input type=\"checkbox\" id=\"failedOnly\" /> Failed runs only</label>
    </div>
    <div id=\"runList\"></div>
  </div>
  <div class=\"right\">
    <div class=\"controls\"><button onclick=\"prevRun()\">← Prev</button><button onclick=\"nextRun()\">Next →</button><button onclick=\"location.href='training_brief.md'\">Open training_brief.md</button></div>
    <div id=\"bestByClass\" style=\"margin-top:10px\"></div>
    <div id=\"detail\"></div>
  </div>
</div>
<script>
const DATA_PATH = 'training_runs_full.json';
let data = null;
let filtered = [];
let idx = 0;

function fmt(x, d=4) { return (x===null || x===undefined || Number.isNaN(x)) ? '-' : Number(x).toFixed(d); }
function statusSummary(runs) {
  const complete = runs.filter(r => r.status === 'completed').length;
  const failed = runs.filter(r => r.status === 'failed').length;
  return `${runs.length} runs · ${complete} completed · ${failed} failed`;
}

function renderBestByClass() {
  const byClass = {};
  for (const r of data.runs) {
    if (r.status !== 'completed') continue;
    const pcs = r.per_class_metrics || {};
    for (const [cls, m] of Object.entries(pcs)) {
      const f1 = Number(m.f1 ?? -1);
      if (!byClass[cls] || f1 > byClass[cls].f1) {
        byClass[cls] = { className: cls, f1, runId: r.run_id, acc: Number(r.final_metrics?.accuracy ?? -1) };
      }
    }
  }
  const rows = Object.values(byClass)
    .sort((a,b)=>a.className.localeCompare(b.className))
    .map(x => `<tr><td>${x.className}</td><td>${fmt(x.f1)}</td><td>${x.runId}</td><td>${fmt(x.acc)}</td></tr>`)
    .join('');
  document.getElementById('bestByClass').innerHTML = `<h3>Best run per class</h3><table class=\"best-table\"><tr><th>Class</th><th>Best F1</th><th>Run</th><th>Run Acc</th></tr>${rows || '<tr><td colspan=\"4\">No completed runs with per-class metrics</td></tr>'}</table>`;
}

async function load() {
  const res = await fetch(DATA_PATH);
  data = await res.json();
  filtered = data.runs;
  document.getElementById('summaryStats').textContent = statusSummary(data.runs);
  renderBestByClass();
  renderList();
  renderDetail();
}

function renderList() {
  const q = (document.getElementById('search').value || '').toLowerCase().trim();
  const failedOnly = document.getElementById('failedOnly').checked;
  filtered = data.runs.filter(r => {
    if (failedOnly && r.status !== 'failed') return false;
    if (!q) return true;
    return JSON.stringify([r.run_id, r.status, r.classification_mode, r.dataset_version_id]).toLowerCase().includes(q);
  });
  if (idx >= filtered.length) idx = Math.max(0, filtered.length - 1);
  const box = document.getElementById('runList');
  box.innerHTML = filtered.map((r,i)=>`<div class=\"run ${i===idx?'active':''}\" onclick=\"pick(${i})\"><div><strong>${r.run_id}</strong></div><div class=\"muted\">${r.status} · ${r.classification_mode||'-'}</div><div class=\"muted\">F1 ${fmt(r.final_metrics?.macro_f1)} · Acc ${fmt(r.final_metrics?.accuracy)}</div></div>`).join('');
}

function pick(i) { idx = i; renderList(); renderDetail(); }
function prevRun() { if (idx>0) idx--; renderList(); renderDetail(); }
function nextRun() { if (idx<filtered.length-1) idx++; renderList(); renderDetail(); }

function renderDetail() {
  const r = filtered[idx];
  const el = document.getElementById('detail');
  if (!r) { el.innerHTML = '<p>No runs.</p>'; return; }
  const perClassRows = Object.entries(r.per_class_metrics||{}).map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v.precision)}</td><td>${fmt(v.recall)}</td><td>${fmt(v.f1)}</td><td>${v.support??'-'}</td></tr>`).join('');
  const ckRows = (r.checkpoints||[]).slice(-20).map(c=>`<tr><td>${c.epoch}</td><td>${fmt(c.train_macro_f1)}</td><td>${fmt(c.val_macro_f1)}</td><td>${fmt(c.train_accuracy)}</td><td>${fmt(c.val_accuracy)}</td></tr>`).join('');
  const eventRows = (r.events_tail||[]).map(e=>`<tr><td>${e.at||'-'}</td><td>${e.type||'-'}</td><td><code>${JSON.stringify(e.payload||{})}</code></td></tr>`).join('');
  el.innerHTML = `
    <h2>${r.run_id}</h2>
    <div>${r.status==='completed' ? '<span class="pill ok">completed</span>' : '<span class="pill bad">'+r.status+'</span>'} <span class=\"muted\">${r.dataset_version_id||'-'}</span></div>
    <div class=\"grid\" style=\"margin-top:12px\">
      <div class=\"card\"><div class=\"muted\">Macro F1</div><div class=\"num\">${fmt(r.final_metrics?.macro_f1)}</div></div>
      <div class=\"card\"><div class=\"muted\">Accuracy</div><div class=\"num\">${fmt(r.final_metrics?.accuracy)}</div></div>
      <div class=\"card\"><div class=\"muted\">Loss</div><div class=\"num\">${fmt(r.final_metrics?.loss)}</div></div>
      <div class=\"card\"><div class=\"muted\">Artifacts</div><div class=\"num\">${(r.artifacts||[]).length}</div></div>
    </div>
    <h3>Run Config</h3>
    <table><tr><th>Mode</th><th>Profile</th><th>Optimizer</th><th>Epochs</th><th>LR</th></tr><tr><td>${r.classification_mode||'-'}</td><td>${r.trainer_profile||'-'}</td><td>${r.optimizer||'-'}</td><td>${r.epochs??'-'}</td><td>${r.learning_rate??'-'}</td></tr></table>
    <h3>Per-Class Metrics</h3>
    <table><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>${perClassRows || '<tr><td colspan="5">No per-class metrics</td></tr>'}</table>
    <h3>Checkpoint Trend (last 20)</h3>
    <table><tr><th>Epoch</th><th>Train F1</th><th>Val F1</th><th>Train Acc</th><th>Val Acc</th></tr>${ckRows || '<tr><td colspan="5">No checkpoints</td></tr>'}</table>
    <h3>Artifacts</h3>
    <table><tr><th>ID</th><th>Path</th><th>Created</th></tr>${(r.artifacts||[]).map(a=>`<tr><td>${a.artifact_id||'-'}</td><td>${a.path||'-'}</td><td>${a.created_at||'-'}</td></tr>`).join('') || '<tr><td colspan="3">No artifacts</td></tr>'}</table>
    <h3>Recent Events</h3>
    <table><tr><th>At</th><th>Type</th><th>Payload</th></tr>${eventRows || '<tr><td colspan="3">No events</td></tr>'}</table>
  `;
}

document.getElementById('search').addEventListener('input', ()=>{ renderList(); renderDetail(); });
document.getElementById('failedOnly').addEventListener('change', ()=>{ idx = 0; renderList(); renderDetail(); });
load();
</script>
</body>
</html>
"""

    (TRACK_DIR / "dashboard.html").write_text(html, encoding="utf-8")

    completed = [r for r in runs if r.get("status") == "completed"]
    failed = [r for r in runs if r.get("status") == "failed"]
    top = sorted(
        [r for r in completed if r.get("final_metrics", {}).get("macro_f1") is not None],
        key=lambda x: x.get("final_metrics", {}).get("macro_f1", -1),
        reverse=True,
    )[:10]

    lines = [
        "# Foundry Training Brief",
        "",
        f"Updated: {datetime.now(UTC).isoformat()}",
        f"Total runs: {len(runs)}",
        f"Completed: {len(completed)}",
        f"Failed: {len(failed)}",
        "",
        "## Top 10 by Macro F1",
        "",
        "| Rank | Run | Macro F1 | Accuracy | Dataset Version |",
        "|---:|---|---:|---:|---|",
    ]
    for i, r in enumerate(top, start=1):
        m = r.get("final_metrics", {})
        lines.append(
            f"| {i} | {r.get('run_id')} | {m.get('macro_f1', 0):.4f} | {m.get('accuracy', 0):.4f} | {r.get('dataset_version_id','-')} |"
        )

    if failed:
        lines.extend(["", "## Failed Runs", "", "| Run | Dataset Version | Last Event |", "|---|---|---|"])
        for r in failed:
            last_type = (r.get("events_tail") or [{}])[-1].get("type", "-")
            lines.append(f"| {r.get('run_id')} | {r.get('dataset_version_id','-')} | {last_type} |")

    (TRACK_DIR / "training_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    runs = build_runs_payload()
    write_dashboard(runs)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "dashboard": str(TRACK_DIR / "dashboard.html"),
                "full_data": str(TRACK_DIR / "training_runs_full.json"),
                "brief": str(TRACK_DIR / "training_brief.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
