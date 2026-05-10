from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from echozero.testing.ma3.simulator import _SimulatedMA3OSCServer


def _load_dev_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "MA3" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ma3_datapool_documenter_writes_bundle(capsys, tmp_path: Path) -> None:
    module = _load_dev_module("ma3_datapool_documenter")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    output_dir = tmp_path / "datapool-docs"

    try:
        result = module.main(
            [
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "--output-dir",
                str(output_dir),
            ]
        )
    finally:
        server.stop()

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["object_count"] >= 7
    assert summary["source_of_truth"]["custom_api"] == "osc_lua_service_layer"
    assert summary["source_of_truth"]["raw_ma_authority"] == "ma_terminal_cli"

    snapshot_payload = json.loads((output_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snapshot_payload["object_count"] == summary["object_count"]
    assert snapshot_payload["source_of_truth"]["raw_ma_authority"] == "ma_terminal_cli"
    assert any(item["class"] == "Track" for item in snapshot_payload["objects"])
    root_object = next(item for item in snapshot_payload["objects"] if item["path"] == "")
    assert root_object["class"] == "DataPool"

    hierarchy_text = (output_dir / "hierarchy.md").read_text(encoding="utf-8")
    capture_plan_text = (output_dir / "terminal_capture_plan.md").read_text(encoding="utf-8")
    capture_targets = json.loads(
        (output_dir / "terminal_capture_targets.json").read_text(encoding="utf-8")
    )
    assert "`DataPool` [DataPool]" in hierarchy_text
    assert (
        "Rule: native MA introspection must be captured from the MA terminal/CLI interface."
        in capture_plan_text
    )
    assert capture_targets[0]["requires_terminal_native_capture"] is True
