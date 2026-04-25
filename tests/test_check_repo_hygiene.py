"""
Repo hygiene script tests: prove size guardrails on cleaned public entrypoints.
Exists because the hygiene script is now part of the cleanup enforcement path.
Connects pure helper checks to targeted pytest proof instead of shell-only coverage.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_hygiene_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_repo_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_module_size_failures_accepts_guarded_file_within_limit(tmp_path: Path) -> None:
    module = _load_hygiene_module()
    guarded_file = tmp_path / "echozero" / "ui" / "qt" / "app_shell.py"
    guarded_file.parent.mkdir(parents=True, exist_ok=True)
    guarded_file.write_text('"""\nExists to.\nConnects.\n"""\n', encoding="utf-8")

    failures = module.module_size_failures(tmp_path, rules={"echozero/ui/qt/app_shell.py": 10})

    assert failures == []


def test_module_size_failures_reports_guarded_file_over_limit(tmp_path: Path) -> None:
    module = _load_hygiene_module()
    guarded_file = tmp_path / "tests" / "ui" / "test_timeline_shell.py"
    guarded_file.parent.mkdir(parents=True, exist_ok=True)
    guarded_file.write_text("\n".join(f"line {index}" for index in range(30)), encoding="utf-8")

    failures = module.module_size_failures(tmp_path, rules={"tests/ui/test_timeline_shell.py": 25})

    assert failures == [
        "tests/ui/test_timeline_shell.py: exceeds max line count 25 (30 lines)"
    ]


def test_tracked_ez_failures_allow_explicit_allowlist_entries() -> None:
    module = _load_hygiene_module()

    failures = module.tracked_ez_failures(
        ["fixtures/template.ez", "docs/STATUS.md"],
        allowlist=("fixtures/template.ez",),
    )

    assert failures == []


def test_tracked_ez_failures_report_non_allowlisted_archives() -> None:
    module = _load_hygiene_module()

    failures = module.tracked_ez_failures(
        ["project.ez", "fixtures/template.ez"],
        allowlist=("fixtures/template.ez",),
    )

    assert failures == [
        "tracked .ez archives must be explicitly allowlisted:",
        "  - project.ez",
    ]


def test_canonical_runtime_support_import_failures_accept_clean_runtime_sources(tmp_path: Path) -> None:
    module = _load_hygiene_module()
    runtime_file = tmp_path / "echozero" / "ui" / "qt" / "app_shell.py"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text(
        'from echozero.ui.qt.app_shell_runtime_services import build_runtime_timeline_application\n',
        encoding="utf-8",
    )

    failures = module.canonical_runtime_support_import_failures(
        tmp_path,
        rules={
            "echozero/ui/qt/app_shell.py": (
                "echozero.ui.qt.timeline.demo_app",
                "echozero.ui.qt.timeline.fixture_loader",
            )
        },
    )

    assert failures == []


def test_canonical_runtime_support_import_failures_report_support_surface_imports(tmp_path: Path) -> None:
    module = _load_hygiene_module()
    runtime_file = tmp_path / "run_echozero.py"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text(
        'from echozero.ui.qt.timeline.fixture_loader import load_realistic_timeline_fixture\n',
        encoding="utf-8",
    )

    failures = module.canonical_runtime_support_import_failures(
        tmp_path,
        rules={
            "run_echozero.py": (
                "echozero.ui.qt.timeline.demo_app",
                "echozero.ui.qt.timeline.fixture_loader",
            )
        },
    )

    assert failures == [
        "run_echozero.py: imports support-only timeline surfaces "
        "echozero.ui.qt.timeline.fixture_loader"
    ]


def test_recent_split_roots_have_default_size_guardrails() -> None:
    module = _load_hygiene_module()

    assert module.MODULE_MAX_LINE_RULES["echozero/ui/qt/timeline/object_info_panel.py"] == 325
    assert module.MODULE_MAX_LINE_RULES["echozero/ui/qt/timeline/widget_canvas.py"] == 225
    assert module.MODULE_MAX_LINE_RULES["echozero/foundry/ui/main_window_run_mixin.py"] == 40
    assert module.MODULE_MAX_LINE_RULES["echozero/application/timeline/orchestrator_selection_mixin.py"] == 60
    assert module.MODULE_MAX_LINE_RULES["echozero/application/timeline/object_action_settings_service.py"] == 500
    assert module.MODULE_MAX_LINE_RULES["echozero/persistence/session.py"] == 430
    assert module.MODULE_MAX_LINE_RULES["echozero/models/provider.py"] == 360
    assert module.MODULE_MAX_LINE_RULES["echozero/foundry/services/baseline_trainer.py"] == 150
    assert module.MODULE_MAX_LINE_RULES["echozero/foundry/ui/main_window_workspace_mixin.py"] == 120
    assert module.MODULE_MAX_LINE_RULES["echozero/ui/qt/timeline/widget_action_transfer_mixin.py"] == 220
    assert module.MODULE_MAX_LINE_RULES["tests/ui/timeline_shell_support.py"] == 30
    assert module.MODULE_MAX_LINE_RULES["tests/ui/app_shell_runtime_flow_support.py"] == 25
    assert module.MODULE_MAX_LINE_RULES["tests/persistence_support.py"] == 20
    assert module.MODULE_MAX_LINE_RULES["tests/audio_engine_support.py"] == 20
    assert module.MODULE_MAX_LINE_RULES["tests/session_support.py"] == 20
    assert module.MODULE_MAX_LINE_RULES["tests/ui/runtime_audio_support.py"] == 20


def test_recent_split_helpers_have_default_header_guardrails() -> None:
    module = _load_hygiene_module()
    expected_paths = (
        "echozero/application/presentation/inspector_contract_context_actions.py",
        "echozero/application/presentation/inspector_contract_lookup.py",
        "echozero/application/presentation/inspector_contract_preview.py",
        "echozero/application/presentation/inspector_contract_support.py",
        "echozero/application/presentation/inspector_contract_types.py",
        "echozero/ui/qt/timeline/object_info_panel.py",
        "echozero/ui/qt/timeline/object_info_panel_actions_mixin.py",
        "echozero/ui/qt/timeline/object_info_panel_preview.py",
        "echozero/ui/qt/timeline/object_info_panel_text.py",
        "echozero/ui/qt/timeline/widget_canvas.py",
        "echozero/ui/qt/timeline/widget_canvas_interaction_mixin.py",
        "echozero/ui/qt/timeline/widget_canvas_paint_mixin.py",
        "echozero/ui/qt/timeline/widget_canvas_types.py",
        "echozero/foundry/ui/main_window_run_mixin.py",
        "echozero/foundry/ui/main_window_run_actions_mixin.py",
        "echozero/foundry/ui/main_window_run_build_mixin.py",
        "echozero/foundry/ui/main_window_run_summary_mixin.py",
        "echozero/foundry/ui/main_window_workspace_mixin.py",
        "echozero/foundry/ui/main_window_workspace_build_mixin.py",
        "echozero/foundry/ui/main_window_workspace_state_mixin.py",
        "echozero/foundry/ui/main_window_worker.py",
        "echozero/foundry/ui/main_window_types.py",
        "echozero/foundry/services/baseline_trainer_runtime.py",
        "echozero/models/provider_shared.py",
        "echozero/models/provider_sources.py",
        "echozero/persistence/session_versioning_mixin.py",
        "echozero/persistence/session_runtime_mixin.py",
        "echozero/ui/qt/timeline/widget_action_transfer_mixin.py",
        "echozero/ui/qt/timeline/widget_action_transfer_workspace_mixin.py",
        "tests/ui/timeline_shell_support.py",
        "tests/ui/timeline_shell_shared_support.py",
        "tests/ui/timeline_shell_layout_support.py",
        "tests/ui/timeline_shell_contract_actions_support.py",
        "tests/ui/timeline_shell_object_info_support.py",
        "tests/ui/timeline_shell_transfer_support.py",
        "tests/ui/timeline_shell_interactions_support.py",
        "tests/ui/app_shell_runtime_flow_support.py",
        "tests/ui/app_shell_runtime_flow_shared_support.py",
        "tests/ui/app_shell_runtime_flow_project_support.py",
        "tests/ui/app_shell_runtime_flow_settings_support.py",
        "tests/ui/app_shell_runtime_flow_pipeline_support.py",
        "tests/ui/app_shell_runtime_flow_audio_support.py",
        "tests/persistence_support.py",
        "tests/persistence_shared_support.py",
        "tests/persistence_core_support.py",
        "tests/persistence_layers_support.py",
        "tests/persistence_roundtrip_support.py",
        "tests/persistence_integrity_support.py",
        "tests/audio_engine_support.py",
        "tests/audio_engine_shared_support.py",
        "tests/audio_engine_clock_transport_support.py",
        "tests/audio_engine_layers_support.py",
        "tests/audio_engine_integration_support.py",
        "tests/audio_engine_regressions_support.py",
        "tests/session_support.py",
        "tests/session_shared_support.py",
        "tests/session_dirty_create_support.py",
        "tests/session_save_graph_support.py",
        "tests/session_lifecycle_support.py",
        "tests/session_edge_cases_support.py",
        "tests/ui/runtime_audio_support.py",
        "tests/ui/runtime_audio_shared_support.py",
        "tests/ui/runtime_audio_controller_support.py",
        "tests/ui/runtime_audio_widget_support.py",
    )

    for relative_path in expected_paths:
        assert module.MODULE_HEADER_RULES[relative_path] == ('"""', "Exists to", "Connects")
