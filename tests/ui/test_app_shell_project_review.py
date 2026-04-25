"""App-shell project-review tests for the canonical Stage Zero runtime.
Exists to keep project-backed review coverage separate from launcher chrome and timeline flows.
Connects AppShellRuntime review actions to the current ProjectStorage working dir.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from echozero.foundry.domain.review import ReviewPolarity, ReviewSession
from echozero.foundry.review_server_controller import (
    ReviewServerController,
    ReviewServerLaunch,
)
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from tests.ui.app_shell_runtime_flow_shared_support import _repo_local_temp_root


def test_review_server_controller_builds_stable_desktop_and_phone_urls(monkeypatch):
    class _FakeServer:
        def __init__(self) -> None:
            self.server_address = ("0.0.0.0", 8421)
            self.default_session_id = ""
            self.shutdown_calls = 0
            self.server_close_calls = 0

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_calls += 1

        def server_close(self) -> None:
            self.server_close_calls += 1

    fake_server = _FakeServer()
    monkeypatch.setattr(
        "echozero.foundry.review_server_controller.create_review_http_server",
        lambda root, session_id, *, host, port: fake_server,
    )
    monkeypatch.setattr(
        "echozero.foundry.review_server_controller._detect_lan_ipv4_address",
        lambda: "192.168.50.25",
    )
    controller = ReviewServerController()
    controller.enable()

    try:
        launch = controller.build_session_launch(Path("/tmp/review"), "rev demo")
        assert controller.build_session_url(Path("/tmp/review"), "rev demo") == launch.url
    finally:
        controller.stop()

    assert launch == ReviewServerLaunch(
        url="http://127.0.0.1:8421/?sessionId=rev%20demo",
        desktop_url="http://127.0.0.1:8421/?sessionId=rev%20demo",
        phone_url="http://192.168.50.25:8421/?sessionId=rev%20demo",
        bind_host="0.0.0.0",
        port=8421,
    )


def test_app_shell_runtime_open_project_review_session_uses_project_working_dir(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    captured: dict[str, object] = {}

    class _FakeReviewService:
        def __init__(self, root: Path):
            captured["root"] = root

        def create_project_session(self, project_path: Path, **kwargs) -> ReviewSession:
            captured["project_path"] = project_path
            captured["kwargs"] = kwargs
            return ReviewSession(id="rev_demo", name="Demo Review", items=[])

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_review.ReviewSessionService",
        _FakeReviewService,
    )
    monkeypatch.setattr(
        runtime._review_server_controller,
        "build_session_launch",
        lambda root, session_id: captured.update({"url_root": root, "session_id": session_id})
        or ReviewServerLaunch(
            url="http://127.0.0.1:8421/?sessionId=rev_demo",
            desktop_url="http://127.0.0.1:8421/?sessionId=rev_demo",
            phone_url="http://192.168.1.44:8421/?sessionId=rev_demo",
            bind_host="0.0.0.0",
            port=8421,
        ),
    )
    runtime.enable_phone_review_service()

    try:
        launch = runtime.open_project_review_session(
            song_version_id="ver_active",
            review_mode="all_events",
            questionable_score_threshold=0.8,
            item_limit=25,
        )

        assert launch.session_id == "rev_demo"
        assert launch.session_name == "Demo Review"
        assert launch.item_count == 0
        assert launch.url == "http://127.0.0.1:8421/?sessionId=rev_demo"
        assert launch.desktop_url == "http://127.0.0.1:8421/?sessionId=rev_demo"
        assert launch.phone_url == "http://192.168.1.44:8421/?sessionId=rev_demo"
        assert launch.bind_host == "0.0.0.0"
        assert launch.port == 8421
        assert captured["root"] == runtime.project_storage.working_dir.resolve()
        assert captured["project_path"] == runtime.project_storage.working_dir.resolve()
        assert captured["url_root"] == runtime.project_storage.working_dir.resolve()
        assert captured["session_id"] == "rev_demo"
        assert captured["kwargs"] == {
            "name": None,
            "song_id": None,
            "song_version_id": "ver_active",
            "layer_id": None,
            "polarity": ReviewPolarity.POSITIVE,
            "review_mode": "all_events",
            "questionable_score_threshold": 0.8,
            "item_limit": 25,
            "application_session": {
                "sessionId": str(runtime.session.id),
                "projectId": str(runtime.session.project_id),
                "activeSongId": None,
                "activeSongVersionId": None,
                "activeTimelineId": str(runtime.session.active_timeline_id),
                "activeSongVersionMa3TimecodePoolNo": None,
                "uiPrefsRef": None,
            },
        }
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_project_review_requires_phone_service_enablement(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    class _FakeReviewService:
        def __init__(self, _root: Path):
            return None

        def create_project_session(self, _project_path: Path, **_kwargs) -> ReviewSession:
            return ReviewSession(id="rev_demo", name="Demo Review", items=[])

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_review.ReviewSessionService",
        _FakeReviewService,
    )

    try:
        assert runtime.is_phone_review_service_enabled() is False
        try:
            runtime.open_project_review_session(song_version_id="ver_active")
        except ValueError as exc:
            assert "Phone review service is disabled" in str(exc)
        else:
            raise AssertionError("Expected phone review service gating to raise a ValueError.")
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)
