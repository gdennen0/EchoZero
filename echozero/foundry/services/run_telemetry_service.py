from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from echozero.foundry.domain import TrainRun


class RunTelemetryService:
    def __init__(self, root: Path):
        self._root = root

    def write_status_snapshot(self, run_id: str, *, status: str, event_type: str) -> None:
        tracking = self._root / "foundry" / "tracking" / "snapshots"
        tracking.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "status": status,
            "event_type": event_type,
            "at": datetime.now(UTC).isoformat(),
        }
        (tracking / f"{run_id}_latest_status.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_progress_snapshot(self, run_id: str, *, epoch: int, metric_snapshot: dict) -> None:
        tracking = self._root / "foundry" / "tracking" / "snapshots"
        tracking.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run_id,
            "epoch": epoch,
            "metric_snapshot": metric_snapshot,
            "at": datetime.now(UTC).isoformat(),
        }
        (tracking / f"{run_id}_latest_progress.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def append_run_telemetry(self, run: TrainRun, *, epoch: int, metric_snapshot: dict[str, Any]) -> None:
        run_dir = run.run_dir(self._root)
        telemetry_path = run_dir / "telemetry.jsonl"
        latest_path = run_dir / "telemetry.latest.json"

        loss = metric_snapshot.get("val_loss")
        if loss is None:
            loss = metric_snapshot.get("train_loss")

        payload: dict[str, Any] = {
            "at": datetime.now(UTC).isoformat(),
            "run_id": run.id,
            "status": run.status.value,
            "epoch": int(epoch),
            "loss": loss,
            "eta_seconds": metric_snapshot.get("eta_seconds"),
            "metrics": metric_snapshot,
        }
        payload.update(self.collect_system_stats())

        with telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def collect_system_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "cpu_percent": None,
            "ram_percent": None,
            "ram_used_mb": None,
            "gpu_vram_used_mb": None,
            "gpu_vram_total_mb": None,
        }

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                stats["cpu_percent"] = float(psutil.cpu_percent(interval=None))
                stats["ram_percent"] = float(vm.percent)
                stats["ram_used_mb"] = round(float(vm.used) / (1024 * 1024), 2)
            except Exception:
                pass

        if torch is not None:
            try:
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    used = float(torch.cuda.memory_allocated(device)) / (1024 * 1024)
                    total = float(torch.cuda.get_device_properties(device).total_memory) / (1024 * 1024)
                    stats["gpu_vram_used_mb"] = round(used, 2)
                    stats["gpu_vram_total_mb"] = round(total, 2)
            except Exception:
                pass

        return stats

    def refresh_tracking_artifacts(self, run_id: str, *, status: str) -> None:
        scripts = [
            self._root / "scripts" / "refresh_foundry_tracking.py",
            self._root / "scripts" / "build_foundry_dashboard.py",
        ]
        for script in scripts:
            if not script.exists():
                continue
            try:
                subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(self._root),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            except Exception:
                pass

        marker = self._root / "foundry" / "tracking" / "snapshots"
        marker.mkdir(parents=True, exist_ok=True)
        (marker / f"{run_id}_terminal.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "at": datetime.now(UTC).isoformat(),
                    "dashboard": str(self._root / "foundry" / "tracking" / "dashboard.html"),
                    "brief": str(self._root / "foundry" / "tracking" / "training_brief.md"),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
