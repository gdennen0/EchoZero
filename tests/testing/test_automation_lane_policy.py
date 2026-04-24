from __future__ import annotations

from pathlib import Path

from ui_automation import EchoZeroAutomationProvider, LiveEchoZeroAutomationProvider

from echozero.testing.run import LANES


def test_humanflow_all_excludes_simulated_gui_lane_members():
    humanflow = LANES["humanflow-all"]

    assert "tests/testing/test_gui_dsl.py" not in humanflow
    assert "tests/testing/test_gui_lane_b.py" not in humanflow


def test_ui_automation_lane_covers_canonical_automation_surfaces():
    ui_automation = LANES["ui-automation"]

    assert ui_automation == [
        "tests/ui_automation/test_session.py",
        "tests/ui_automation/test_echozero_backend.py",
        "tests/ui_automation/test_bridge_server.py",
    ]


def test_public_echozero_provider_name_points_to_live_client():
    assert EchoZeroAutomationProvider is LiveEchoZeroAutomationProvider


def test_canonical_runtime_sources_do_not_import_timeline_support_surfaces():
    runtime_sources = [
        Path("/Users/march/Documents/GitHub/EchoZero/run_echozero.py"),
        Path("/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/launcher_surface.py"),
        Path("/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/app_shell.py"),
    ]
    forbidden_imports = (
        "timeline.demo_app",
        "timeline.fixture_loader",
        "timeline.real_data_fixture",
        "timeline.test_harness",
    )

    for source_path in runtime_sources:
        source = source_path.read_text(encoding="utf-8")
        for forbidden_import in forbidden_imports:
            assert forbidden_import not in source
