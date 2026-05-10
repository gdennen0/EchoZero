from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from echozero.testing.ma3.simulator import _SimulatedMA3OSCServer


def _load_dev_module(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "MA3" / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ma3_app_path_smoke_opens_pull_workspace_against_live_bridge_shape(capsys) -> None:
    module = _load_dev_module("ma3_app_path_smoke")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sync_state"]["connected"] is True
    assert payload["pull_workspace"]["workspace_active"] is True
    assert payload["pull_workspace"]["timecode_count"] >= 1
    assert payload["pull_workspace"]["track_count"] >= 1


def test_ma3_app_path_push_smoke_sends_one_bounded_event_against_live_bridge_shape(
    capsys,
) -> None:
    module = _load_dev_module("ma3_app_path_push_smoke")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "--target-track-coord",
                "tc1_tg2_tr3",
                "--cue-number",
                "901",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sync_state"]["connected"] is True
    assert payload["push_target"]["track_coord"] == "tc1_tg2_tr3"
    assert payload["push_target"]["sequence_before"] == 12
    assert payload["push_target"]["sequence_after"] == 12
    assert payload["push_target"]["event_count_before"] == 2
    assert payload["push_target"]["event_count_after"] == 3
    assert payload["push_event"]["present_after_push"] is True
    assert payload["push_event"]["remote_snapshot"]["cue_number"] == 901
    assert payload["push_operation"]["status"] == "success"
    assert payload["push_operation"]["saved_route"] == "tc1_tg2_tr3"


def test_ma3_app_path_push_smoke_blocks_sequence_less_target_without_opt_in() -> None:
    module = _load_dev_module("ma3_app_path_push_smoke")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        with pytest.raises(SystemExit, match="has no assigned sequence"):
            module.main(
                [
                    "--json",
                    "--ma3-host",
                    "127.0.0.1",
                    "--ma3-port",
                    str(server.endpoint[1]),
                    "--listen-host",
                    "127.0.0.1",
                    "--target-track-coord",
                    "tc1_tg2_tr4",
                ]
            )
    finally:
        server.stop()
