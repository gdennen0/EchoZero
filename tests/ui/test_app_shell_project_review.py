"""App-shell project-review tests for the canonical Stage Zero runtime.
Exists to keep project-backed review coverage separate from launcher chrome and timeline flows.
Connects AppShellRuntime review actions to the current ProjectStorage working dir.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from echozero.foundry.app import FoundryApp
from echozero.foundry.domain import Dataset, DatasetSample, DatasetVersion
from echozero.foundry.domain.review import ReviewOutcome, ReviewPolarity, ReviewSession
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository, ReviewSignalRepository
from echozero.foundry.review_server_controller import (
    ReviewServerController,
    ReviewServerLaunch,
)
from echozero.foundry.services.project_review_queue_builder import ProjectReviewQueueBuilder
from echozero.foundry.services.review_session_service import ReviewSessionService
from echozero.persistence.session import ProjectStorage
from echozero.ui.qt.app_shell import AppShellRuntime, build_app_shell
from tests.foundry.test_review_project_queue_builder import _build_project_review_fixture
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
        lambda root, session_id, *, host, port, application_session=None: fake_server,
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
        url="http://127.0.0.1:8421/",
        desktop_url="http://127.0.0.1:8421/",
        phone_url="http://192.168.50.25:8421/",
        bind_host="0.0.0.0",
        port=8421,
    )


def test_review_server_controller_retargets_enabled_server_to_runtime_root(monkeypatch):
    created: list[tuple[Path, str | None, dict[str, object] | None, object]] = []

    class _FakeServer:
        def __init__(self, root: Path, session_id: str | None, application_session: dict[str, object] | None) -> None:
            self.server_address = ("0.0.0.0", 8421)
            self.default_session_id = str(session_id or "").strip()
            self.current_application_session = application_session
            self.shutdown_calls = 0
            self.server_close_calls = 0
            created.append((root, session_id, application_session, self))

        def serve_forever(self) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_calls += 1

        def server_close(self) -> None:
            self.server_close_calls += 1

    monkeypatch.setattr(
        "echozero.foundry.review_server_controller.create_review_http_server",
        lambda root, session_id, *, host, port, application_session=None: _FakeServer(
            root,
            session_id,
            application_session,
        ),
    )
    controller = ReviewServerController()
    controller.set_runtime_context(
        Path("/tmp/project_a"),
        application_session={"projectName": "Project A"},
        clear_active_session=True,
    )

    controller.enable()
    controller.build_session_launch(Path("/tmp/project_a"), "rev_demo")
    controller.set_runtime_context(
        Path("/tmp/project_b"),
        application_session={"projectName": "Project B"},
        clear_active_session=True,
    )

    first_root, first_session_id, first_application_session, first_server = created[0]
    second_root, second_session_id, second_application_session, _second_server = created[1]

    assert first_root == Path("/tmp/project_a").resolve()
    assert first_session_id is None
    assert first_application_session == {"projectName": "Project A"}
    assert first_server.default_session_id == "rev_demo"
    assert second_root == Path("/tmp/project_b").resolve()
    assert second_session_id is None
    assert second_application_session == {"projectName": "Project B"}
    assert controller.last_session_id is None


def test_app_shell_runtime_open_project_publishes_phone_review_runtime_context(monkeypatch, tmp_path: Path):
    ez_path, _working_dir, _refs = _build_project_review_fixture(tmp_path)
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    captured: list[tuple[Path, dict[str, object] | None, bool]] = []
    monkeypatch.setattr(
        runtime._review_server_controller,
        "set_runtime_context",
        lambda root, *, application_session=None, clear_active_session=False: captured.append(
            (Path(root).resolve(), application_session, clear_active_session)
        )
        or None,
    )

    try:
        runtime.open_project(ez_path)

        assert captured
        root, application_session, clear_active_session = captured[-1]
        assert root == runtime.project_storage.working_dir.resolve()
        assert clear_active_session is True
        assert application_session is not None
        assert application_session["projectName"] == runtime.project_storage.project.name
        assert application_session["projectRef"] == f"project:{runtime.project_storage.project.id}"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


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
            url="http://127.0.0.1:8421/",
            desktop_url="http://127.0.0.1:8421/",
            phone_url="http://192.168.1.44:8421/",
            bind_host="0.0.0.0",
            port=8421,
        ),
    )
    monkeypatch.setattr(
        runtime._review_server_controller,
        "set_runtime_context",
        lambda root, application_session=None, clear_active_session=False: None,
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
        assert launch.url == "http://127.0.0.1:8421/"
        assert launch.desktop_url == "http://127.0.0.1:8421/"
        assert launch.phone_url == "http://192.168.1.44:8421/"
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
                "projectName": runtime.project_storage.project.name,
                "projectRef": f"project:{runtime.project_storage.project.id}",
                "activeSongId": None,
                "activeSongRef": None,
                "activeSongTitle": None,
                "activeSongVersionId": None,
                "activeSongVersionRef": None,
                "activeSongVersionLabel": None,
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


def test_app_shell_runtime_enable_phone_review_service_binds_current_project_root(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    captured: list[Path] = []
    monkeypatch.setattr(
        runtime._review_server_controller,
        "set_runtime_context",
        lambda root, application_session=None, clear_active_session=False: captured.append(
            Path(root).resolve()
        ),
    )

    try:
        runtime.enable_phone_review_service()

        assert captured == [runtime.project_storage.working_dir.resolve()]
        assert runtime.is_phone_review_service_enabled() is True
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_open_project_rebinds_phone_review_root_when_enabled(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)
    original_root = runtime.project_storage.working_dir.resolve()

    alternate_storage = ProjectStorage.create_new(
        name="Olivia Scratch 3",
        working_dir_root=temp_root / "other-working",
    )
    alternate_path = temp_root / "Olivia Scratch 3.ez"
    alternate_storage.save_as(alternate_path)
    alternate_storage.close()

    captured: list[Path] = []
    monkeypatch.setattr(
        runtime._review_server_controller,
        "set_runtime_context",
        lambda root, application_session=None, clear_active_session=False: captured.append(
            Path(root).resolve()
        ),
    )

    try:
        runtime.enable_phone_review_service()
        runtime.open_project(alternate_path)

        assert captured[0] == original_root
        assert captured[-1] == runtime.project_storage.working_dir.resolve()
        assert len(captured) >= 2
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_reload_phone_review_status_uses_last_session(monkeypatch):
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    captured: dict[str, object] = {}

    class _FakeReviewService:
        def __init__(self, root: Path):
            captured["root"] = root

        def get_session(self, session_id: str) -> ReviewSession | None:
            captured["session_id"] = session_id
            return ReviewSession(id=session_id, name="Demo Review", items=[])

    monkeypatch.setattr(
        "echozero.ui.qt.app_shell_project_review.ReviewSessionService",
        _FakeReviewService,
    )
    monkeypatch.setattr(
        runtime._review_server_controller,
        "reload_status",
        lambda root, session_id: captured.update({"url_root": root, "reload_session_id": session_id})
        or ReviewServerLaunch(
            url="http://127.0.0.1:8421/",
            desktop_url="http://127.0.0.1:8421/",
            phone_url="http://192.168.1.44:8421/",
            bind_host="0.0.0.0",
            port=8421,
        ),
    )
    monkeypatch.setattr(
        runtime._review_server_controller,
        "set_runtime_context",
        lambda root, application_session=None, clear_active_session=False: None,
    )
    runtime.enable_phone_review_service()
    runtime._review_server_controller._last_session_id = "rev_demo"

    try:
        launch = runtime.reload_phone_review_status()

        assert launch.session_id == "rev_demo"
        assert launch.session_name == "Demo Review"
        assert launch.item_count == 0
        assert launch.url == "http://127.0.0.1:8421/"
        assert launch.phone_url == "http://192.168.1.44:8421/"
        assert captured["root"] == runtime.project_storage.working_dir.resolve()
        assert captured["session_id"] == "rev_demo"
        assert captured["url_root"] == runtime.project_storage.working_dir.resolve()
        assert captured["reload_session_id"] == "rev_demo"
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_exposes_latest_project_review_dataset_paths():
    temp_root = _repo_local_temp_root()
    runtime = build_app_shell(working_dir_root=temp_root / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        cache_dir = (
            runtime.project_storage.working_dir
            / "foundry"
            / "cache"
            / "review_projects"
            / "fixture"
            / "clips"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        matching_clip = cache_dir / "kick.wav"
        foreign_clip = cache_dir / "foreign.wav"
        matching_clip.write_bytes(b"matching")
        foreign_clip.write_bytes(b"foreign")
        project_ref = f"project:{runtime.project_storage.project.id}"
        dataset_repo = DatasetRepository(runtime.project_storage.working_dir)
        version_repo = DatasetVersionRepository(runtime.project_storage.working_dir)
        dataset_repo.save(
            Dataset(
                id="ds_active_review",
                name="Review Samples - Active Project",
                source_kind="project_review_export",
                source_ref=str(runtime.project_storage.working_dir),
                metadata={
                    "schema": "foundry.project_review_dataset.v1",
                    "review_dataset_key": f"ez_project:{project_ref}",
                    "queue_source_kind": "ez_project",
                    "project_ref": project_ref,
                },
                created_at=datetime.now(UTC),
            )
        )
        dataset_repo.save(
            Dataset(
                id="ds_foreign_review",
                name="Review Samples - Foreign Project",
                source_kind="project_review_export",
                source_ref=str(runtime.project_storage.working_dir),
                metadata={
                    "schema": "foundry.project_review_dataset.v1",
                    "review_dataset_key": "ez_project:project:foreign",
                    "queue_source_kind": "ez_project",
                    "project_ref": "project:foreign",
                },
                created_at=datetime.now(UTC),
            )
        )
        version_repo.save(
            DatasetVersion(
                id="dsv_active_review",
                dataset_id="ds_active_review",
                version=1,
                manifest_hash="hash-active",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick"],
                samples=[DatasetSample(sample_id="sm_active", audio_ref=str(matching_clip), label="kick")],
                manifest={"schema": "foundry.review_dataset_manifest.v1"},
                created_at=datetime.now(UTC),
            )
        )
        version_repo.save(
            DatasetVersion(
                id="dsv_foreign_review",
                dataset_id="ds_foreign_review",
                version=1,
                manifest_hash="hash-foreign",
                sample_rate=22050,
                audio_standard="mono_wav_pcm16",
                class_map=["kick"],
                samples=[DatasetSample(sample_id="sm_foreign", audio_ref=str(foreign_clip), label="kick")],
                manifest={"schema": "foundry.review_dataset_manifest.v1"},
                created_at=datetime.now(UTC),
            )
        )

        dataset_versions = runtime.list_project_review_dataset_versions()
        latest = runtime.get_latest_project_review_dataset_version()

        assert len(dataset_versions) == 1
        assert latest is not None
        assert latest.dataset_id == "ds_active_review"
        assert latest.version_id == "dsv_active_review"
        assert latest.folder_path == cache_dir.resolve()
        assert latest.version_artifact_path == (
            runtime.project_storage.working_dir / "foundry" / "state" / "dataset_versions.json"
        ).resolve()
        assert runtime.latest_project_review_dataset_folder() == cache_dir.resolve()
        assert runtime.latest_project_review_dataset_artifact_path() == (
            runtime.project_storage.working_dir / "foundry" / "state" / "dataset_versions.json"
        ).resolve()
    finally:
        runtime.shutdown()
        shutil.rmtree(temp_root, ignore_errors=True)


def test_app_shell_runtime_phone_review_correct_writes_back_via_runtime_bridge(tmp_path: Path):
    ez_path, _working_dir, refs = _build_project_review_fixture(tmp_path)
    runtime = build_app_shell(working_dir_root=tmp_path / "working")

    assert isinstance(runtime, AppShellRuntime)

    try:
        runtime.open_project(ez_path)
        runtime._review_server_controller = ReviewServerController(port=0)
        runtime.enable_phone_review_service()
        launch = runtime.open_project_review_session(
            song_id=refs["alpha_song_id"],
            layer_id="layer_alpha_kick",
            review_mode="all_events",
            item_limit=2,
        )

        service = ReviewSessionService(runtime.project_storage.working_dir)
        session = service.get_session(launch.session_id)
        assert session is not None

        service.set_item_review(
            launch.session_id,
            session.items[0].item_id,
            outcome=ReviewOutcome.CORRECT,
            corrected_label=None,
            review_note="phone verified via runtime bridge",
        )
        signal = ReviewSignalRepository(runtime.project_storage.working_dir).list()[-1]
        queue = ProjectReviewQueueBuilder(runtime.project_storage.working_dir).build_queue(
            runtime.project_storage.working_dir,
            song_id=refs["alpha_song_id"],
            layer_id="layer_alpha_kick",
            review_mode="all_events",
            item_limit=2,
        )
        exported = FoundryApp(runtime.project_storage.working_dir).export_project_review_dataset(
            runtime.project_storage.working_dir,
            project_ref=refs["project_ref"],
            song_id=refs["alpha_song_id"],
            layer_id="layer_alpha_kick",
        )

        assert signal.review_decision is not None
        assert signal.review_decision.kind.value == "verified"
        assert signal.source_provenance["project_writeback"]["status"] == "applied_via_runtime_bridge"
        assert queue.items[0].review_outcome == ReviewOutcome.CORRECT
        assert queue.items[0].review_decision is not None
        assert queue.items[0].review_decision.kind.value == "verified"
        assert exported.stats["review_positive_count"] == 1
        assert exported.stats["review_negative_count"] == 0
    finally:
        runtime.shutdown()
