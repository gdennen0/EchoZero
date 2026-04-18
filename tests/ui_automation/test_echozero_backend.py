from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from ui_automation import AutomationSession, HarnessEchoZeroAutomationProvider

from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_model, write_test_wav

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_echozero_automation_backend")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def test_echozero_backend_imports_song_and_exposes_timeline_targets():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-import.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        snapshot = session.invoke(
            "add_song_from_path",
            params={"title": "Automation Song", "audio_path": str(audio_path)},
        )

        source_layer = session.find_target("timeline.layer:source_audio")

        assert source_layer is not None
        assert source_layer.label == "Automation Song"
        assert any(action.action_id == "extract_stems" for action in snapshot.actions)
        assert session.screenshot(target_id="shell.timeline").startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_drives_stems_events_and_classification_flow():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-flow.wav")
    model_path = write_test_model(temp_root / "fixtures" / "automation-model.pth")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Automation Flow Song", "audio_path": str(audio_path)},
        )
        session.invoke("extract_stems", target_id="timeline.layer:source_audio")

        drums_target = session.find_target("Drums")
        assert drums_target is not None

        session.invoke("extract_drum_events", target_id=drums_target.target_id)
        after_events = session.snapshot()
        assert any(target.kind == "event" for target in after_events.targets)

        session.invoke(
            "classify_drum_events",
            target_id=drums_target.target_id,
            params={"model_path": str(model_path)},
        )

        classified_layer = session.find_target("Drum_Classified_Events")
        assert classified_layer is not None
        assert classified_layer.kind == "layer"
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_drags_selected_event_and_scrolls_timeline():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-drag.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            window_width=320,
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Automation Drag Song", "audio_path": str(audio_path)},
        )
        session.invoke("extract_stems", target_id="timeline.layer:source_audio")

        drums_target = session.find_target("Drums")
        assert drums_target is not None

        session.invoke("extract_drum_events", target_id=drums_target.target_id)
        event_target = next(target for target in session.snapshot().targets if target.kind == "event")
        before_time = event_target.time_seconds
        assert before_time is not None

        session.click(event_target.target_id)
        after_drag = session.drag(event_target.target_id, {"dx": 100, "dy": 0})
        moved_event = next(target for target in after_drag.targets if target.target_id == event_target.target_id)
        assert moved_event.time_seconds is not None
        assert moved_event.time_seconds > before_time

        before_scroll = session.find_target("shell.timeline")
        assert before_scroll is not None
        before_scroll_x = float(before_scroll.metadata["scroll_x"])

        after_scroll = session.scroll("shell.timeline", dx=240)
        timeline_target = next(target for target in after_scroll.targets if target.target_id == "shell.timeline")
        assert float(timeline_target.metadata["scroll_x"]) > before_scroll_x
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_drives_transport_actions():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-project.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Lifecycle Song", "audio_path": str(audio_path)},
        )

        after_play = session.invoke("transport.play")
        assert any(action.action_id == "transport.pause" for action in after_play.actions)

        after_pause = session.invoke("transport.pause")
        assert any(action.action_id == "transport.play" for action in after_pause.actions)

        after_stop = session.invoke("transport.stop")
        assert any(action.action_id == "transport.play" for action in after_stop.actions)

    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_tracks_pointer_hover_and_double_click():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-pointer.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Pointer Song", "audio_path": str(audio_path)},
        )
        session.invoke("extract_stems", target_id="timeline.layer:source_audio")
        drums_target = session.find_target("Drums")
        assert drums_target is not None

        hovered = session.hover(drums_target.target_id)
        assert hovered.artifacts["pointer_target_id"] == drums_target.target_id
        assert hovered.artifacts["pointer_position"] is not None
        assert hovered.focused_target_id == "timeline.layer:source_audio"

        moved = session.move_pointer("shell.transport.play")
        assert moved.artifacts["pointer_target_id"] == "shell.transport.play"
        assert moved.focused_target_id == "timeline.layer:source_audio"

        clicked = session.double_click(drums_target.target_id)
        assert clicked.artifacts["pointer_target_id"] == drums_target.target_id
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_exposes_selected_object_capabilities():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-object.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Object Song", "audio_path": str(audio_path)},
        )
        snapshot = session.click("timeline.layer:source_audio")

        assert snapshot.objects
        selected_object = snapshot.objects[0]
        assert selected_object.object_id == "source_audio"
        assert selected_object.object_type == "layer"
        assert selected_object.target_id == "timeline.layer:source_audio"
        assert snapshot.focused_target_id == "timeline.layer:source_audio"
        assert snapshot.focused_object_id == "source_audio"
        assert any(fact.label == "kind" for fact in selected_object.facts)
        assert any(action.action_id == "extract_stems" for action in selected_object.actions)
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_echozero_backend_drives_sync_enable_disable_flow():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "automation-sync.wav")
    session = AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
            simulate_ma3=True,
        )
    )

    try:
        session.invoke(
            "add_song_from_path",
            params={"title": "Sync Flow Song", "audio_path": str(audio_path)},
        )

        sync_on = session.invoke("sync.enable")
        assert sync_on.sync["connected"] is True
        assert sync_on.sync["mode"] == "ma3"

        sync_off = session.invoke("sync.disable")
        assert sync_off.sync["connected"] is False
        assert sync_off.sync["mode"] == "none"
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)
