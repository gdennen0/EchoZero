from __future__ import annotations

import importlib.util
import json
import sys
import threading
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


def test_ma3_harness_cli_ping_json_round_trip(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "ping",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "ping"
    assert payload["result"]["status"] == "ok"


def test_ma3_harness_cli_smoke_emits_expected_sections(capsys, tmp_path: Path) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    transcript_path = tmp_path / "ma3-smoke-transcript.json"

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
                "--transcript-out",
                str(transcript_path),
                "smoke",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    smoke = payload["result"]
    assert smoke["ping"]["status"] == "ok"
    assert str(smoke["version"]["ez_version"]) == "2.0"
    assert smoke["health"]["hitmaker_loaded"] is True
    assert smoke["browse"]["timecodes"][0]["number"] == 1
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["target"]["ma3_host"] == "127.0.0.1"
    assert any(message["key"] == "plugin.version" for message in transcript["messages"])


def test_ma3_harness_cli_health_check_compares_against_expected_root(
    capsys, tmp_path: Path
) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    expected_root = tmp_path / "plugins"
    ez_dir = expected_root / "EZ"
    hitmaker_dir = expected_root / "HitMaker"
    ez_dir.mkdir(parents=True)
    hitmaker_dir.mkdir(parents=True)
    (ez_dir / "ez_core.lua").write_text(
        'EZ._version = "2.0"\nEZ._build = EZ._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )
    (hitmaker_dir / "main.lua").write_text(
        'HitMaker._version = HitMaker._version or "1.1.0"\n'
        'HitMaker._build = HitMaker._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )

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
                "health-check",
                "--expected-root",
                str(expected_root),
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "health-check"
    assert payload["result"]["status"] == "pass"
    assert payload["result"]["failures"] == []


def test_ma3_harness_cli_validation_report_writes_artifacts(capsys, tmp_path: Path) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    expected_root = tmp_path / "plugins"
    output_dir = tmp_path / "report"
    ez_dir = expected_root / "EZ"
    hitmaker_dir = expected_root / "HitMaker"
    ez_dir.mkdir(parents=True)
    hitmaker_dir.mkdir(parents=True)
    (ez_dir / "ez_core.lua").write_text(
        'EZ._version = "2.0"\nEZ._build = EZ._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )
    (hitmaker_dir / "main.lua").write_text(
        'HitMaker._version = HitMaker._version or "1.1.0"\n'
        'HitMaker._build = HitMaker._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )

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
                "validation-report",
                "--expected-root",
                str(expected_root),
                "--output-dir",
                str(output_dir),
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "validation-report"
    assert payload["result"]["status"] == "pass"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    transcript = output_dir / "transcript.json"
    assert summary_json.exists()
    assert summary_md.exists()
    assert transcript.exists()
    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary_payload["summary"]["timecode_count"] >= 1
    assert "MA3 Hardware Validation Report" in summary_md.read_text(encoding="utf-8")


def test_ma3_harness_cli_validation_report_can_include_receive_capture(
    capsys, tmp_path: Path
) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    expected_root = tmp_path / "plugins"
    output_dir = tmp_path / "report-with-receive"
    ez_dir = expected_root / "EZ"
    hitmaker_dir = expected_root / "HitMaker"
    ez_dir.mkdir(parents=True)
    hitmaker_dir.mkdir(parents=True)
    (ez_dir / "ez_core.lua").write_text(
        'EZ._version = "2.0"\nEZ._build = EZ._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )
    (hitmaker_dir / "main.lua").write_text(
        'HitMaker._version = HitMaker._version or "1.1.0"\n'
        'HitMaker._build = HitMaker._build or "2026-04-30.hitmaker-health-1"\n',
        encoding="utf-8",
    )

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
                "validation-report",
                "--expected-root",
                str(expected_root),
                "--output-dir",
                str(output_dir),
                "--receive-duration-seconds",
                "0.2",
                "--receive-trigger-command",
                "EZ.Ping()",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    receive_capture = payload["result"]["receive_capture"]
    assert receive_capture["trigger_command"] == "EZ.Ping()"
    assert receive_capture["message_count"] >= 1
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["receive_capture"]["message_count"] >= 1


def test_ma3_harness_cli_receive_capture_records_inbound_transport(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    def _emit_transport() -> None:
        server._send_message("transport", "scrubbed", {"to_seconds": 12.5})  # noqa: SLF001

    timer = threading.Timer(0.4, _emit_transport)
    timer.start()
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
                "receive-capture",
                "--duration-seconds",
                "1.0",
                "--ping-first",
            ]
        )
    finally:
        timer.cancel()
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "receive-capture"
    assert payload["result"]["message_count"] >= 1
    assert payload["result"]["transport_update_count"] >= 1
    assert payload["result"]["latest_transport_update"]["to_seconds"] == 12.5


def test_ma3_harness_cli_receive_capture_can_trigger_command(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "receive-capture",
                "--duration-seconds",
                "0.2",
                "--trigger-command",
                "EZ.Ping()",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "receive-capture"
    assert payload["result"]["trigger_command"] == "EZ.Ping()"
    assert payload["result"]["message_count"] >= 1
    assert "connection.ping" in payload["result"]["message_keys"]


def test_ma3_harness_cli_stream_emits_ndjson_messages(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        result = module.main(
            [
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "stream",
                "--duration-seconds",
                "0.3",
                "--trigger-command",
                "EZ.Version()",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines
    payload = json.loads(lines[0])
    assert payload["key"] == "plugin.version"
    assert payload["message_type"] == "plugin"
    assert payload["fields"]["ez_build"]


def test_ma3_harness_cli_create_timecode_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-timecode",
                "--name",
                "Song B",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-timecode"
    assert payload["result"] == {"number": 2, "name": "Song B"}


def test_ma3_harness_cli_create_track_group_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-track-group",
                "--timecode-no",
                "1",
                "--name",
                "FX",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-track-group"
    assert payload["result"] == {"number": 1, "name": "FX", "track_count": 0}


def test_ma3_harness_cli_create_track_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    server._handle_CreateTrackGroup(1, "FX")  # noqa: SLF001

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
                "create-track",
                "--timecode-no",
                "1",
                "--track-group-no",
                "1",
                "--name",
                "Laser",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-track"
    assert payload["result"]["coord"] == "tc1_tg1_tr1"
    assert payload["result"]["name"] == "Laser"
    assert payload["result"]["number"] == 1
    assert payload["result"]["event_count"] == 0
    assert payload["result"]["sequence_no"] is None


def test_ma3_harness_cli_create_sequence_next_available_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-sequence-next-available",
                "--name",
                "Lead Next",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-sequence-next-available"
    assert payload["result"] == {"number": 16, "name": "Lead Next", "cue_count": None}


def test_ma3_harness_cli_create_sequence_in_current_song_range_emits_structured_result(
    capsys,
) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-sequence-in-current-song-range",
                "--name",
                "Song A - Lead",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-sequence-in-current-song-range"
    assert payload["result"] == {"number": 13, "name": "Song A - Lead", "cue_count": None}


def test_ma3_harness_cli_datapool_children_returns_structured_entries(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "datapool-children",
                "--path",
                "Timecodes/1/2",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    children = payload["result"]
    assert payload["command"] == "datapool-children"
    assert children[0]["path"] == "Timecodes/1/2/3"
    assert children[0]["class"] == "Track"
    assert children[0]["no"] == 3


def test_ma3_harness_cli_datapool_object_stays_hierarchy_only(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "datapool-object",
                "--path",
                "Timecodes/1/2/3",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    result_payload = payload["result"]
    assert payload["command"] == "datapool-object"
    assert result_payload["path"] == "Timecodes/1/2/3"
    assert result_payload["class"] == "Track"
    assert result_payload["preview_children"][0]["class"] == "CmdEvent"
    assert "dump" not in result_payload
    assert "property_items" not in result_payload


def test_ma3_harness_cli_datapool_report_renders_plain_text_for_preset_ref(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        create_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "create-phaser-fixture-set",
                "--selection-command",
                "Fixture 1 Thru 4",
            ]
        )
        assert create_result == 0
        capsys.readouterr()

        report_result = module.main(
            [
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "datapool-report",
                "--preset-ref",
                "21.221",
                "--depth",
                "1",
            ]
        )
    finally:
        server.stop()

    assert report_result == 0
    output = capsys.readouterr().out
    assert "path: PresetPools/21/221" in output
    assert "class: Preset" in output
    assert "name: Mixed Type Phaser" in output
    assert "PRESETMODE = Selective" in output
    assert "children:" in output
    assert "path: PresetPools/21/221/Recipe 1" in output
    assert "STEP_1 = 1.1+4.1" in output
    assert "SPEEDFROMX = 96.0" in output


def test_ma3_harness_cli_describe_preset_uses_explicit_osc_preset_api(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        create_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "create-phaser-fixture-set",
                "--selection-command",
                "Fixture 1 Thru 4",
            ]
        )
        assert create_result == 0
        capsys.readouterr()

        describe_result = module.main(
            [
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "describe-preset",
                "--preset-ref",
                "21.221",
            ]
        )
    finally:
        server.stop()

    assert describe_result == 0
    output = capsys.readouterr().out
    assert "path: PresetPools/21/221" in output
    assert "class: Preset" in output
    assert "name: Mixed Type Phaser" in output
    assert "STEP_1 = 1.1+4.1" in output
    assert "SPEEDFROMX = 96.0" in output


def test_ma3_harness_cli_list_presets_uses_explicit_osc_preset_api(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        create_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "create-phaser-fixture-set",
                "--selection-command",
                "Fixture 1 Thru 4",
            ]
        )
        assert create_result == 0
        capsys.readouterr()

        list_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "list-presets",
                "--preset-type",
                "21",
            ]
        )
    finally:
        server.stop()

    assert list_result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "list-presets"
    assert payload["result"]["preset_type"] == 21
    assert payload["result"]["count"] == 1
    assert payload["result"]["presets"][0]["number"] == 221
    assert payload["result"]["presets"][0]["kind"] == "phaser"


def test_ma3_harness_cli_preview_and_apply_replace_preset_when_group(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        preview_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "preview-replace-preset-when-group",
                "--preset-type",
                "21",
                "--source-preset-ref",
                "Preset 21.221",
                "--dest-preset-ref",
                "Preset 21.222",
                "--group-filter",
                "Drums",
                "--sequence-numbers",
                "12",
            ]
        )
        assert preview_result == 0
        preview_payload = json.loads(capsys.readouterr().out)

        apply_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "replace-preset-when-group",
                "--preset-type",
                "21",
                "--source-preset-ref",
                "Preset 21.221",
                "--dest-preset-ref",
                "Preset 21.222",
                "--group-filter",
                "Drums",
                "--sequence-numbers",
                "12",
            ]
        )
        assert apply_result == 0
        apply_payload = json.loads(capsys.readouterr().out)
    finally:
        server.stop()

    assert preview_payload["command"] == "preview-replace-preset-when-group"
    assert preview_payload["result"]["count"] == 2
    assert preview_payload["result"]["findings"][0]["matched_group"] == "Drums"
    assert apply_payload["command"] == "replace-preset-when-group"
    assert apply_payload["result"]["replaced_count"] == 2


def test_ma3_harness_cli_analyze_cue_recipe_state_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "analyze-cue-recipe-state",
                "--sequence-no",
                "12",
                "--cue-no",
                "2",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "analyze-cue-recipe-state"
    assert payload["result"]["status"] == "ready"
    assert payload["result"]["local_line_count"] == 3
    assert payload["result"]["contributor_count"] == 3


def test_ma3_harness_cli_analyze_cue_recipe_state_can_attach_terminal_feedback(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()
    transcript_path = tmp_path / "analyze-terminal-feedback.json"

    terminal_calls: list[tuple[str, float, str]] = []

    class _FakeTerminalSession:
        def __init__(
            self, *, host: str, timeout_seconds: float, quiet_period_seconds: float = 0.6
        ) -> None:
            del quiet_period_seconds
            self._host = host
            self._timeout_seconds = timeout_seconds

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def send_command(self, command: str) -> str:
            terminal_calls.append((self._host, self._timeout_seconds, command))
            return "[EZ HARNESS] analyze seq=12 cue=2 status=ready supported=true local=3 contributors=3"

    monkeypatch.setattr(module, "MA3TerminalSession", _FakeTerminalSession)

    try:
        result = module.main(
            [
                "--json",
                "--terminal-feedback",
                "--transcript-out",
                str(transcript_path),
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "analyze-cue-recipe-state",
                "--sequence-no",
                "12",
                "--cue-no",
                "2",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["terminal_feedback"]["status"] == "ok"
    assert payload["terminal_feedback"]["probe"] == "analyze-cue-recipe-state"
    assert 'EZ.AnalyzeCueRecipeState(12, "2")' in payload["terminal_feedback"]["command"]
    assert "status=ready" in payload["terminal_feedback"]["output"]
    assert terminal_calls == [
        (
            "127.0.0.1",
            10.0,
            payload["terminal_feedback"]["command"],
        )
    ]
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["terminal_feedback"]["probe"] == "analyze-cue-recipe-state"
    assert "contributors=3" in transcript["terminal_feedback"]["output"]


def test_ma3_harness_cli_preview_recipe_cue_only_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "preview-recipe-cue-only",
                "--sequence-no",
                "12",
                "--source-cue-no",
                "4",
                "--target-cue-no",
                "2",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "preview-recipe-cue-only"
    assert payload["result"]["status"] == "ready"
    assert payload["result"]["changed_keys"] == ["Drums:Beam", "Drums:Color", "Drums:Dimmer"]
    assert len(payload["result"]["stored_lines"]) == 1


def test_ma3_harness_cli_preview_copy_cue_with_status_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "preview-copy-cue-with-status",
                "--sequence-no",
                "12",
                "--source-cue-no",
                "3",
                "--dest-cue-no",
                "5",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "preview-copy-cue-with-status"
    assert payload["result"]["status"] == "ready"
    assert payload["result"]["copied_line_count"] == 3
    assert payload["result"]["local_line_count"] == 0


def test_ma3_harness_cli_recipe_cue_state_and_copy_modes(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
    server = _SimulatedMA3OSCServer(host="127.0.0.1", port=0).start()

    try:
        analysis_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "analyze-cue-recipe-state",
                "--sequence-no",
                "12",
                "--cue-no",
                "2",
            ]
        )
        assert analysis_result == 0
        analysis_payload = json.loads(capsys.readouterr().out)

        preview_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "preview-recipe-cue-only",
                "--sequence-no",
                "12",
                "--source-cue-no",
                "4",
                "--target-cue-no",
                "2",
            ]
        )
        assert preview_result == 0
        preview_payload = json.loads(capsys.readouterr().out)

        apply_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "apply-recipe-cue-only",
                "--sequence-no",
                "12",
                "--source-cue-no",
                "4",
                "--target-cue-no",
                "2",
            ]
        )
        assert apply_result == 0
        apply_payload = json.loads(capsys.readouterr().out)

        copy_result = module.main(
            [
                "--json",
                "--ma3-host",
                "127.0.0.1",
                "--ma3-port",
                str(server.endpoint[1]),
                "--listen-host",
                "127.0.0.1",
                "copy-cue-with-status",
                "--sequence-no",
                "12",
                "--source-cue-no",
                "3",
                "--dest-cue-no",
                "5",
            ]
        )
        assert copy_result == 0
        copy_payload = json.loads(capsys.readouterr().out)
    finally:
        server.stop()

    assert analysis_payload["command"] == "analyze-cue-recipe-state"
    assert analysis_payload["result"]["contributor_count"] == 3
    assert preview_payload["command"] == "preview-recipe-cue-only"
    assert preview_payload["result"]["changed_keys"] == [
        "Drums:Beam",
        "Drums:Color",
        "Drums:Dimmer",
    ]
    assert apply_payload["command"] == "apply-recipe-cue-only"
    assert {row["preset_ref"] for row in apply_payload["result"]["restore_lines"]} == {
        "Preset 21.222",
        "Preset 4.44",
        "Preset 5.23",
    }
    assert copy_payload["command"] == "copy-cue-with-status"
    assert copy_payload["result"]["copied_line_count"] == 3


def test_ma3_harness_cli_create_static_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-static-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "101",
                "--store-mode",
                "Global",
                "--name",
                "Deep Red",
                "--selection-command",
                "Fixture 1 Thru 4",
                "--value-command",
                "At Preset 4.1",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-static-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 101
    assert payload["result"]["name"] == "Deep Red"
    assert payload["result"]["kind"] == "static"


def test_ma3_harness_cli_create_phaser_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-phaser-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "102",
                "--store-mode",
                "Global",
                "--name",
                "Red Blue Chase",
                "--selection-command",
                "Fixture 1 Thru 4",
                "--step",
                "4.1",
                "--step",
                "4.2",
                "--step",
                "4.3+2.5",
                "--speed-bpm",
                "120",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-phaser-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 102
    assert payload["result"]["name"] == "Red Blue Chase"
    assert payload["result"]["kind"] == "phaser"
    assert payload["result"]["step_count"] == 3


def test_ma3_harness_cli_create_phaser_fixture_set_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-phaser-fixture-set",
                "--selection-command",
                "Fixture 1 Thru 4",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    result_payload = payload["result"]
    assert payload["command"] == "create-phaser-fixture-set"
    assert [item["preset_type"] for item in result_payload["per_type_phasers"]] == [
        1,
        2,
        4,
        5,
        6,
        22,
    ]
    assert [item["number"] for item in result_payload["per_type_phasers"]] == [
        201,
        202,
        204,
        205,
        206,
        222,
    ]
    assert [item["name"] for item in result_payload["per_type_phasers"]] == [
        "Dimmer Chase",
        "Position Chase",
        "Color Chase",
        "Beam Chase",
        "Focus Chase",
        "Optical Chase",
    ]
    assert all(item["kind"] == "phaser" for item in result_payload["per_type_phasers"])
    assert all(item["step_count"] == 3 for item in result_payload["per_type_phasers"])
    assert result_payload["look_21_phaser"] == {
        "preset_type": 21,
        "number": 221,
        "name": "Mixed Type Phaser",
        "store_mode": "Selective",
        "kind": "phaser",
        "step_count": 3,
    }


def test_ma3_harness_cli_create_recipe_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "create-recipe-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "103",
                "--store-mode",
                "Selective",
                "--name",
                "Base Recipe",
                "--selection-command",
                "Fixture 1 Thru 4",
                "--source-preset-ref",
                "4.1",
                "--selection-mode",
                "Strict",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "create-recipe-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 103
    assert payload["result"]["name"] == "Base Recipe"
    assert payload["result"]["kind"] == "recipe"
    assert payload["result"]["step_count"] == 1


def test_ma3_harness_cli_edit_phaser_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "edit-phaser-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "104",
                "--store-mode",
                "Global",
                "--name",
                "Wide Chase",
                "--selection-command",
                "Fixture 1 Thru 4",
                "--step",
                "4.1",
                "--step",
                "4.3",
                "--speed-bpm",
                "90",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "edit-phaser-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 104
    assert payload["result"]["name"] == "Wide Chase"
    assert payload["result"]["kind"] == "phaser"
    assert payload["result"]["step_count"] == 2


def test_ma3_harness_cli_edit_static_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "edit-static-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "104",
                "--store-mode",
                "Global",
                "--name",
                "New Blue",
                "--selection-command",
                "Fixture 5 Thru 8",
                "--value-command",
                "At Preset 4.2",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "edit-static-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 104
    assert payload["result"]["name"] == "New Blue"
    assert payload["result"]["kind"] == "static"
    assert payload["result"]["step_count"] == 1


def test_ma3_harness_cli_edit_recipe_preset_emits_structured_result(capsys) -> None:
    module = _load_dev_module("ma3_harness_cli")
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
                "edit-recipe-preset",
                "--preset-type",
                "4",
                "--preset-no",
                "106",
                "--store-mode",
                "Selective",
                "--name",
                "Updated Recipe",
                "--selection-command",
                "Fixture 9 Thru 12",
                "--source-preset-ref",
                "4.9",
                "--selection-mode",
                "Strict",
            ]
        )
    finally:
        server.stop()

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "edit-recipe-preset"
    assert payload["result"]["preset_type"] == 4
    assert payload["result"]["number"] == 106
    assert payload["result"]["name"] == "Updated Recipe"
    assert payload["result"]["kind"] == "recipe"
    assert payload["result"]["step_count"] == 1
