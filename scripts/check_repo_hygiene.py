#!/usr/bin/env python
"""Repository hygiene guardrails.
Exists to keep generated output, legacy surfaces, and boundary drift out of Git.
Connects the cleanup and LLM-readability rules to one fast CI/local check.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_PREFIXES = (
    "artifacts/",
    "foundry/runs/",
    "foundry/state/",
    "foundry/tracking/",
)
FORBIDDEN_LEGACY_PREFIXES = (
    "src/",
    "ui/",
)
FORBIDDEN_LEGACY_FILES = (
    "main.py",
    "main_qt.py",
)
REQUIRED_FILES = (
    "docs/STATUS.md",
    "docs/LLM-CLEANUP-BOARD.md",
    "echozero/application/timeline/README.md",
    "echozero/application/presentation/README.md",
    "echozero/ui/qt/timeline/README.md",
    "echozero/foundry/README.md",
    "run_echozero.py",
)
MODULE_HEADER_RULES: dict[str, tuple[str, ...]] = {
    "run_echozero.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/app_shell.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/app_shell_project_timeline.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/app_shell_runtime_services.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/app_shell_timeline_state.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/app_shell_layer_storage.py": ('"""', "Exists to", "Connects"),
    "echozero/application/timeline/app.py": ('"""', "Exists to", "Connects"),
    "echozero/application/timeline/orchestrator.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract_context_actions.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract_lookup.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract_preview.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract_support.py": ('"""', "Exists to", "Connects"),
    "echozero/application/presentation/inspector_contract_types.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_actions.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/object_info_panel.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/object_info_panel_actions_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/object_info_panel_preview.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/object_info_panel_text.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_canvas.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_canvas_interaction_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_canvas_paint_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_canvas_types.py": ('"""', "Exists to", "Connects"),
    "echozero/audio/engine.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/persistence/repositories.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/services/baseline_trainer.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/services/cnn_trainer.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_run_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_run_actions_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_run_build_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_run_summary_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_workspace_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_workspace_build_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_workspace_state_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_worker.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/ui/main_window_types.py": ('"""', "Exists to", "Connects"),
    "echozero/foundry/services/baseline_trainer_runtime.py": ('"""', "Exists to", "Connects"),
    "echozero/infrastructure/sync/ma3_osc.py": ('"""', "Exists to", "Connects"),
    "echozero/models/provider.py": ('"""', "Exists to", "Connects"),
    "echozero/models/provider_shared.py": ('"""', "Exists to", "Connects"),
    "echozero/models/provider_sources.py": ('"""', "Exists to", "Connects"),
    "echozero/persistence/entities.py": ('"""', "Exists to", "Connects"),
    "echozero/persistence/session.py": ('"""', "Exists to", "Connects"),
    "echozero/persistence/session_versioning_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/persistence/session_runtime_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/processors/separate_audio.py": ('"""', "Exists to", "Connects"),
    "echozero/services/orchestrator.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_action_transfer_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/widget_action_transfer_workspace_mixin.py": ('"""', "Exists to", "Connects"),
    "echozero/ui/qt/timeline/demo_app.py": ('"""', "Exists to", "Never", "canonical"),
    "echozero/ui/qt/timeline/fixture_loader.py": ('"""', "Exists to", "Never", "canonical"),
    "echozero/ui/qt/timeline/test_harness.py": ('"""', "Exists to", "Never", "canonical"),
    "echozero/testing/gui_lane_b.py": ('"""', "Exists to", "Never", "canonical"),
    "echozero/testing/demo_suite_scenarios.py": ('"""', "Exists to", "Never", "canonical"),
    "echozero/testing/ma3/simulator.py": ('"""', "Exists to", "Never", "canonical"),
    "tests/ui/timeline_shell_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_layout_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_contract_actions_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_object_info_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_transfer_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/timeline_shell_interactions_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_project_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_settings_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_pipeline_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/app_shell_runtime_flow_audio_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_core_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_layers_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_roundtrip_support.py": ('"""', "Exists to", "Connects"),
    "tests/persistence_integrity_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_clock_transport_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_layers_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_integration_support.py": ('"""', "Exists to", "Connects"),
    "tests/audio_engine_regressions_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_dirty_create_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_save_graph_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_lifecycle_support.py": ('"""', "Exists to", "Connects"),
    "tests/session_edge_cases_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/runtime_audio_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/runtime_audio_shared_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/runtime_audio_controller_support.py": ('"""', "Exists to", "Connects"),
    "tests/ui/runtime_audio_widget_support.py": ('"""', "Exists to", "Connects"),
}
MODULE_MAX_LINE_RULES: dict[str, int] = {
    # Regression ceilings for public roots already reduced by the cleanup pass.
    # These intentionally leave some slack so the guardrail preserves the split
    # without blocking follow-up cleanup that is still in flight.
    "run_echozero.py": 200,
    "echozero/ui/qt/app_shell.py": 325,
    "echozero/ui/qt/app_shell_project_timeline.py": 260,
    "echozero/ui/qt/timeline/widget.py": 260,
    "echozero/ui/qt/timeline/widget_actions.py": 275,
    "echozero/ui/qt/timeline/object_info_panel.py": 325,
    "echozero/ui/qt/timeline/widget_canvas.py": 225,
    "echozero/application/timeline/orchestrator.py": 900,
    "echozero/application/timeline/orchestrator_selection_mixin.py": 60,
    "echozero/application/timeline/object_action_settings_service.py": 500,
    "echozero/application/timeline/assembler.py": 900,
    "echozero/application/presentation/inspector_contract.py": 500,
    "echozero/foundry/services/baseline_trainer.py": 150,
    "echozero/foundry/ui/main_window.py": 150,
    "echozero/foundry/ui/main_window_run_mixin.py": 40,
    "echozero/foundry/ui/main_window_workspace_mixin.py": 120,
    "echozero/models/provider.py": 360,
    "echozero/persistence/session.py": 430,
    "echozero/ui/qt/timeline/widget_action_transfer_mixin.py": 220,
    "echozero/ui/qt/timeline/demo_app.py": 150,
    "echozero/testing/gui_lane_b.py": 450,
    "tests/ui/test_timeline_shell.py": 25,
    "tests/ui/test_app_shell_runtime_flow.py": 25,
    "tests/ui/test_runtime_audio.py": 25,
    "tests/test_persistence.py": 25,
    "tests/test_audio_engine.py": 25,
    "tests/test_session.py": 25,
    "tests/ui/timeline_shell_support.py": 30,
    "tests/ui/app_shell_runtime_flow_support.py": 25,
    "tests/persistence_support.py": 20,
    "tests/audio_engine_support.py": 20,
    "tests/session_support.py": 20,
    "tests/ui/runtime_audio_support.py": 20,
}
CANONICAL_RUNTIME_SUPPORT_IMPORT_RULES: dict[str, tuple[str, ...]] = {
    "run_echozero.py": (
        "echozero.ui.qt.timeline.demo_app",
        "echozero.ui.qt.timeline.fixture_loader",
        "echozero.ui.qt.timeline.real_data_fixture",
        "echozero.ui.qt.timeline.test_harness",
    ),
    "echozero/ui/qt/launcher_surface.py": (
        "echozero.ui.qt.timeline.demo_app",
        "echozero.ui.qt.timeline.fixture_loader",
        "echozero.ui.qt.timeline.real_data_fixture",
        "echozero.ui.qt.timeline.test_harness",
    ),
    "echozero/ui/qt/app_shell.py": (
        "echozero.ui.qt.timeline.demo_app",
        "echozero.ui.qt.timeline.fixture_loader",
        "echozero.ui.qt.timeline.real_data_fixture",
        "echozero.ui.qt.timeline.test_harness",
    ),
}


def _git_ls_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _read_header(path: Path, *, max_lines: int = 8) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[:max_lines])


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def tracked_path_failures(tracked: Iterable[str]) -> list[str]:
    offenders: list[str] = []
    for path in tracked:
        normalized = path.replace("\\", "/")
        if normalized.startswith(FORBIDDEN_PREFIXES):
            offenders.append(normalized)
        if normalized.startswith(FORBIDDEN_LEGACY_PREFIXES):
            offenders.append(normalized)
        if normalized in FORBIDDEN_LEGACY_FILES:
            offenders.append(normalized)

    failures: list[str] = []
    if offenders:
        failures.append("generated/runtime or forbidden legacy paths are tracked:")
        failures.extend(f"  - {item}" for item in offenders)
    return failures


def required_file_failures(repo_root: Path, *, required_files: Iterable[str] = REQUIRED_FILES) -> list[str]:
    failures: list[str] = []
    for relative_path in required_files:
        if not (repo_root / relative_path).exists():
            failures.append(f"missing required current-truth or subsystem file: {relative_path}")
    return failures


def module_header_failures(
    repo_root: Path,
    *,
    rules: dict[str, tuple[str, ...]] = MODULE_HEADER_RULES,
) -> list[str]:
    failures: list[str] = []
    for relative_path, required_fragments in rules.items():
        path = repo_root / relative_path
        if not path.exists():
            failures.append(f"missing guarded module for hygiene checks: {relative_path}")
            continue
        header = _read_header(path)
        missing = [fragment for fragment in required_fragments if fragment not in header]
        if missing:
            missing_text = ", ".join(repr(fragment) for fragment in missing)
            failures.append(f"{relative_path}: missing required header fragments {missing_text}")
    return failures


def module_size_failures(
    repo_root: Path,
    *,
    rules: dict[str, int] = MODULE_MAX_LINE_RULES,
) -> list[str]:
    failures: list[str] = []
    for relative_path, max_lines in rules.items():
        path = repo_root / relative_path
        if not path.exists():
            failures.append(f"missing guarded module for hygiene checks: {relative_path}")
            continue
        line_count = len(_read_lines(path))
        if line_count > max_lines:
            failures.append(
                f"{relative_path}: exceeds max line count {max_lines} "
                f"({line_count} lines)"
            )
    return failures


def canonical_runtime_support_import_failures(
    repo_root: Path,
    *,
    rules: dict[str, tuple[str, ...]] = CANONICAL_RUNTIME_SUPPORT_IMPORT_RULES,
) -> list[str]:
    failures: list[str] = []
    for relative_path, forbidden_imports in rules.items():
        path = repo_root / relative_path
        if not path.exists():
            failures.append(f"missing guarded module for hygiene checks: {relative_path}")
            continue
        source = path.read_text(encoding="utf-8")
        found = [name for name in forbidden_imports if name in source]
        if found:
            failures.append(
                f"{relative_path}: imports support-only timeline surfaces {', '.join(found)}"
            )
    return failures


def collect_hygiene_failures(repo_root: Path, *, tracked: list[str] | None = None) -> list[str]:
    tracked_files = tracked if tracked is not None else _git_ls_files(repo_root)
    failures: list[str] = []
    failures.extend(tracked_path_failures(tracked_files))
    failures.extend(required_file_failures(repo_root))
    failures.extend(module_header_failures(repo_root))
    failures.extend(module_size_failures(repo_root))
    failures.extend(canonical_runtime_support_import_failures(repo_root))
    return failures


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    failures = collect_hygiene_failures(repo_root)

    if failures:
        print("Repo hygiene check failed:")
        for failure in failures:
            print(f"  - {failure}")
        print(
            "\nKeep generated outputs local, preserve canonical docs, "
            "and maintain boundary headers plus module-size guardrails."
        )
        return 1

    print("Repo hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
