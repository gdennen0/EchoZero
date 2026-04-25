from __future__ import annotations

import threading
import time

import pytest

from echozero.application.session.models import Session
from echozero.application.timeline.pipeline_run_service import (
    PipelineRunService,
    PreparedPipelineRun,
)
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.result import err, ok
from echozero.services.orchestrator import AnalysisResult


def _wait_until(predicate, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _BlockingOrchestrator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def execute(self, _session, config_id, runtime_bindings=None, on_progress=None):
        self.calls.append((config_id, runtime_bindings))
        if on_progress is not None:
            on_progress("Loading configuration", 0.0)
            on_progress("Preparing pipeline", 0.1)
            on_progress("Executing pipeline", 0.2)
        self.started.set()
        self.release.wait(timeout=5.0)
        if on_progress is not None:
            on_progress("Persisting results", 0.8)
            on_progress("Complete", 1.0)
        return ok(
            AnalysisResult(
                song_version_id="version_1",
                pipeline_id="stem_separation",
                layer_ids=["layer_output"],
                take_ids=["take_output"],
                duration_ms=12.5,
            )
        )


class _FailingOrchestrator:
    def execute(self, _session, _config_id, runtime_bindings=None, on_progress=None):
        if on_progress is not None:
            on_progress("Executing pipeline", 0.2)
        return err(RuntimeError(f"boom: {runtime_bindings}"))


def _build_service(*, analysis_service, persisted_calls: list[tuple[object, object]]) -> tuple[PipelineRunService, Session]:
    session = Session(id="session_1", project_id="project_1")

    def _prepare_run(action_id, params, object_id, object_type, persist_scope):
        suffix = str(object_id or "object")
        return PreparedPipelineRun(
            action_id=action_id,
            workflow_id=f"workflow:{action_id}",
            pipeline_template_id="stem_separation",
            config_id=f"config:{suffix}:{persist_scope}",
            display_label="Extract Stems",
            object_id=str(object_id or ""),
            object_type=str(object_type or "object"),
            source_layer_id=str(object_id or "") or None,
            runtime_bindings=dict(params or {}),
        )

    def _persist_generated_source_layer_id(*, analysis_result, source_layer_id):
        persisted_calls.append((analysis_result, source_layer_id))

    service = PipelineRunService(
        project_storage_getter=lambda: object(),
        session_getter=lambda: session,
        analysis_service=analysis_service,
        prepare_run=_prepare_run,
        persist_generated_source_layer_id=_persist_generated_source_layer_id,
    )
    return service, session


def test_pipeline_run_service_request_run_returns_immediately_and_completes():
    persisted_calls: list[tuple[object, object]] = []
    analysis_service = _BlockingOrchestrator()
    service, _session = _build_service(
        analysis_service=analysis_service,
        persisted_calls=persisted_calls,
    )

    try:
        started_at = time.monotonic()
        run_id = service.request_run(
            "timeline.extract_stems",
            {"audio_file": "/tmp/song.wav"},
            object_id="layer_source",
            object_type="layer",
        )
        elapsed = time.monotonic() - started_at

        assert elapsed < 0.2
        assert _wait_until(
            lambda: service.get_run(run_id) is not None
            and service.get_run(run_id).status in {"resolving", "running"},
        )
        visible = service.visible_run_for(
            action_id="timeline.extract_stems",
            object_id="layer_source",
            object_type="layer",
        )
        assert visible is not None
        assert visible.run_id == run_id
        assert visible.message in {"Loading configuration", "Preparing pipeline", "Executing pipeline"}

        analysis_service.release.set()
        final_state = service.wait_for_run(run_id, timeout=5.0)

        assert final_state.status == "completed"
        assert final_state.output_layer_ids == ("layer_output",)
        assert persisted_calls and persisted_calls[-1][1] == "layer_source"
        notification = service.consume_updates_since(0)
        assert notification is not None
        assert notification.refresh_presentation is True
    finally:
        service.shutdown()


def test_pipeline_run_service_blocks_only_conflicting_subjects():
    persisted_calls: list[tuple[object, object]] = []
    analysis_service = _BlockingOrchestrator()
    service, _session = _build_service(
        analysis_service=analysis_service,
        persisted_calls=persisted_calls,
    )

    try:
        first_run_id = service.request_run(
            "timeline.extract_stems",
            object_id="layer_a",
            object_type="layer",
        )
        assert _wait_until(lambda: analysis_service.started.is_set())

        with pytest.raises(RuntimeError, match="already running"):
            service.request_run(
                "timeline.extract_stems",
                object_id="layer_a",
                object_type="layer",
            )

        second_run_id = service.request_run(
            "timeline.extract_stems",
            object_id="layer_b",
            object_type="layer",
        )

        assert first_run_id != second_run_id
        analysis_service.release.set()
        assert service.wait_for_run(first_run_id, timeout=5.0).status == "completed"
        assert service.wait_for_run(second_run_id, timeout=5.0).status == "completed"
    finally:
        service.shutdown()


def test_pipeline_run_service_scopes_subjects_and_visibility_by_song_version():
    persisted_calls: list[tuple[object, object]] = []
    analysis_service = _BlockingOrchestrator()
    service, session = _build_service(
        analysis_service=analysis_service,
        persisted_calls=persisted_calls,
    )
    session.active_song_id = SongId("song_a")
    session.active_song_version_id = SongVersionId("version_a")

    try:
        first_run_id = service.request_run(
            "timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
        )
        assert _wait_until(lambda: analysis_service.started.is_set())

        session.active_song_id = SongId("song_b")
        session.active_song_version_id = SongVersionId("version_b")
        second_run_id = service.request_run(
            "timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
        )

        first_visible = service.visible_run_for(
            action_id="timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
            song_version_id="version_a",
        )
        second_visible = service.visible_run_for(
            action_id="timeline.extract_stems",
            object_id="source_audio",
            object_type="layer",
            song_version_id="version_b",
        )

        assert first_visible is not None
        assert second_visible is not None
        assert first_visible.run_id == first_run_id
        assert second_visible.run_id == second_run_id

        analysis_service.release.set()
        assert service.wait_for_run(first_run_id, timeout=5.0).status == "completed"
        assert service.wait_for_run(second_run_id, timeout=5.0).status == "completed"
    finally:
        service.shutdown()


def test_pipeline_run_service_failed_runs_remain_observable():
    persisted_calls: list[tuple[object, object]] = []
    service, _session = _build_service(
        analysis_service=_FailingOrchestrator(),
        persisted_calls=persisted_calls,
    )

    try:
        run_id = service.request_run(
            "timeline.extract_stems",
            {"audio_file": "/tmp/song.wav"},
            object_id="layer_source",
            object_type="layer",
        )
        final_state = service.wait_for_run(run_id, timeout=5.0)

        assert final_state.status == "failed"
        assert "boom" in (final_state.error or "")
        assert not persisted_calls
        visible = service.visible_run_for(
            action_id="timeline.extract_stems",
            object_id="layer_source",
            object_type="layer",
        )
        assert visible is not None
        assert visible.status == "failed"
        assert "boom" in (visible.error or "")
        notification = service.consume_updates_since(0)
        assert notification is not None
        assert notification.refresh_presentation is False
    finally:
        service.shutdown()
