from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from echozero.application.timeline.object_actions.descriptors import (
    resolve_action_id,
)
import echozero.application.timeline.object_actions.descriptors as action_descriptors
from echozero.testing.analysis_mocks import build_mock_analysis_service, write_test_wav
from ui_automation import AutomationSession, HarnessEchoZeroAutomationProvider

_TEST_TEMP_ROOT = Path("C:/Users/griff/.codex/memories/test_primitive_conformance")


def _repo_local_temp_root() -> Path:
    root = _TEST_TEMP_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _build_session(temp_root: Path) -> AutomationSession:
    return AutomationSession.attach(
        HarnessEchoZeroAutomationProvider(
            working_dir_root=temp_root / "working",
            analysis_service=build_mock_analysis_service(),
        )
    )


def test_automation_snapshot_emits_canonical_action_ids():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "primitive-song.wav")
    session = _build_session(temp_root)

    try:
        session.invoke(
            "song.add",
            params={"title": "Primitive Song", "audio_path": str(audio_path)},
        )
        snapshot = session.click("timeline.layer:source_audio")
        for action in snapshot.actions:
            assert resolve_action_id(action.action_id) == action.action_id
        for obj in snapshot.objects:
            for action in obj.actions:
                assert resolve_action_id(action.action_id) == action.action_id
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_legacy_alias_resolution_warns_once(caplog: pytest.LogCaptureFixture):
    action_descriptors._ALIAS_WARNED_IDS.clear()  # type: ignore[attr-defined]
    caplog.set_level("WARNING", logger=action_descriptors.__name__)

    assert resolve_action_id("open_push_surface", warn_on_alias=True) == "transfer.workspace_open"
    assert resolve_action_id("open_push_surface", warn_on_alias=True) == "transfer.workspace_open"

    warnings = [
        record.message for record in caplog.records if "Deprecated action alias" in record.message
    ]
    assert len(warnings) == 1
    assert "open_push_surface" in warnings[0]
    assert "transfer.workspace_open" in warnings[0]


def test_legacy_nudge_alias_executes_through_canonical_dispatch():
    temp_root = _repo_local_temp_root()
    audio_path = write_test_wav(temp_root / "fixtures" / "primitive-nudge.wav")
    session = _build_session(temp_root)

    try:
        session.invoke(
            "song.add",
            params={"title": "Primitive Nudge", "audio_path": str(audio_path)},
        )
        session.invoke("timeline.extract_stems", target_id="timeline.layer:source_audio")
        drums_target = session.find_target("Drums")
        assert drums_target is not None
        session.invoke("timeline.extract_drum_events", target_id=drums_target.target_id)

        first_event = next(target for target in session.snapshot().targets if target.kind == "event")
        session.click(first_event.target_id)

        # Legacy alias should resolve to timeline.nudge_selection and execute.
        snapshot = session.invoke("nudge_left", params={"steps": 1})
        assert snapshot.targets
    finally:
        session.close()
        shutil.rmtree(temp_root, ignore_errors=True)
