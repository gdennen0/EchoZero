from __future__ import annotations

import json
import sys
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(r"C:\Users\griff\EchoZero")
sys.path.insert(0, str(ROOT))

from echozero.foundry.app import FoundryApp
from echozero.foundry.persistence import migrate_foundry_state
from run_focus_kick_snare_batch import run_one


def find_sample_root() -> Path:
    roots = sorted((ROOT / ".foundry-test-tmp").glob("*/samples"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in roots:
        if (r / "kick").exists() and (r / "snare").exists():
            return r
    raise RuntimeError("No suitable sample folder found under .foundry-test-tmp")


def main() -> None:
    migrate_foundry_state(ROOT)
    app = FoundryApp(ROOT)
    samples = find_sample_root()

    dataset = app.datasets.create_dataset("tmp-kick-snare-test", source_ref=str(samples))
    version = app.datasets.ingest_from_folder(dataset.id, samples)
    app.plan_version(
        version.id,
        validation_split=0.25,
        test_split=0.25,
        seed=42,
        balance_strategy="none",
    )
    version = app.datasets.get_version(version.id)
    assert version is not None

    rec1 = run_one(app, app.datasets._versions, version, label="kick", model_type="crnn", seed=6001)
    rec2 = run_one(app, app.datasets._versions, version, label="snare", model_type="cnn", seed=6002)

    report = {
        "batch": f"tmp_test_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        "dataset_version_id": version.id,
        "sample_root": str(samples),
        "runs": [rec1, rec2],
        "finished_at": datetime.now(UTC).isoformat(),
    }
    out = ROOT / "foundry" / "tracking" / "batches" / f"{report['batch']}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "batch": report["batch"], "status_file": str(out)}, indent=2))


if __name__ == "__main__":
    main()
