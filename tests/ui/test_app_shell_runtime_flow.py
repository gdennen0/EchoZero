from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from echozero.application.timeline.intents import Seek, ToggleLayerExpanded
from echozero.application.presentation.inspector_contract import (
    TimelineInspectorHitTarget,
    build_timeline_inspector_contract,
)
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from echozero.ui.qt.timeline.demo_app import DemoTimelineApp


def _repo_local_temp_root() -> Path:
    root = Path("C:/Users/griff/.codex/memories/test_app_shell_runtime_flow") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_app_shell_runtime_new_save_open_reopen_flow():
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"
    save_path = temp_root / "runtime-flow.ez"

    runtime = build_app_shell(
        working_dir_root=working_root,
        initial_project_name="Runtime Flow",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        assert runtime.project_storage.project.name == "Runtime Flow"
        assert runtime.project_path is None
        assert runtime.is_dirty is False

        runtime.dispatch(Seek(1.25))
        assert runtime.is_dirty is False

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True

        runtime.new_project("Second Runtime Flow")
        assert runtime.project_storage.project.name == "Second Runtime Flow"
        assert runtime.project_path is None
        assert runtime.is_dirty is False

        layer_id = runtime.presentation().layers[0].layer_id
        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True

        returned_path = runtime.save_project_as(save_path)
        assert returned_path == save_path
        assert save_path.exists()
        assert runtime.project_path == save_path
        assert runtime.is_dirty is False

        runtime.dispatch(ToggleLayerExpanded(layer_id))
        assert runtime.is_dirty is True
        saved_path = runtime.save_project()
        assert saved_path == save_path
        assert runtime.is_dirty is False

        runtime.open_project(save_path)
        assert runtime.project_path == save_path
        assert runtime.project_storage.working_dir.exists()
        assert runtime.project_storage.project.name == "Second Runtime Flow"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert runtime.session.active_timeline_id == runtime.presentation().timeline_id
        assert runtime.presentation().title == "Second Runtime Flow"
        assert runtime.is_dirty is False

        runtime.open_project(save_path)
        assert runtime.project_path == save_path
        assert runtime.project_storage.working_dir.exists()
        assert Path(runtime.project_storage.working_dir).is_dir()
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_exposes_transfer_surface_actions():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        presentation = runtime.presentation()
        first_layer = presentation.layers[0]
        contract = build_timeline_inspector_contract(
            presentation,
            hit_target=TimelineInspectorHitTarget(kind="layer", layer_id=first_layer.layer_id),
        )

        action_ids = {
            action.action_id
            for section in contract.context_sections
            for action in section.actions
        }

        assert "push_to_ma3" in action_ids
        assert "pull_from_ma3" in action_ids
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_canonical_build_does_not_depend_on_fixture_loader(monkeypatch):
    temp_root = _repo_local_temp_root()
    working_root = temp_root / "working"

    def fail_fixture_load():
        raise AssertionError("fixture loader should not be called for canonical app shell")

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell.load_realistic_timeline_fixture",
        fail_fixture_load,
    )

    runtime = build_app_shell(
        use_demo_fixture=False,
        working_dir_root=working_root,
        initial_project_name="Native Baseline",
    )

    assert isinstance(runtime, AppShellRuntime)

    try:
        presentation = runtime.presentation()

        assert presentation.title == "Native Baseline"
        assert presentation.timeline_id == f"timeline_{runtime.project_storage.project.id}"
        assert runtime.session.project_id == runtime.project_storage.project.id
        assert runtime.session.active_song_id is None
        assert runtime.session.active_song_version_id is None
        assert runtime.session.active_timeline_id == presentation.timeline_id
        assert len(presentation.layers) == 1
        assert presentation.layers[0].events == []
        assert presentation.layers[0].takes == []
        assert all(not layer.events for layer in presentation.layers)
        assert all(not layer.takes for layer in presentation.layers)
        assert presentation.playhead == 0.0
        assert presentation.current_time_label == "00:00.00"
        assert presentation.end_time_label == "00:00.00"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_build_app_shell_demo_fixture_mode_still_returns_demo_runtime():
    runtime = build_app_shell(use_demo_fixture=True)

    assert isinstance(runtime, DemoTimelineApp)
    assert not isinstance(runtime, AppShellRuntime)
