"""Contract-action timeline-shell support cases.
Exists to isolate runtime action routing coverage from layout, transfer, and interaction support.
Connects the compatibility wrapper to the bounded contract-action support slice.
"""

from pathlib import Path

from tests.ui.timeline_shell_shared_support import *  # noqa: F401,F403

from echozero.ui.qt.song_browser_drop import SongBrowserAudioDrop

def test_contract_add_song_action_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))

        assert runtime.calls == [("import", "C:/audio/import.wav")]
        assert widget.presentation.title == "import"
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_action_passes_import_pipeline_kwargs_when_runtime_supports_them(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-contract-add-song-pipeline-kwargs.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, bool | None, tuple[str, ...] | None]] = []
            self.pipeline_calls: list[tuple[str, str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(
            self,
            title: str,
            audio_path: str,
            *,
            run_import_pipeline: bool | None = None,
            import_pipeline_action_ids: tuple[str, ...] | None = None,
        ):
            self.calls.append(
                (title, audio_path, run_import_pipeline, import_pipeline_action_ids)
            )
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

        def run_object_action(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ):
            self.pipeline_calls.append((action_id, str(object_id), object_type))
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))

        assert runtime.calls == [
            (
                "import",
                "C:/audio/import.wav",
                True,
                ("timeline.extract_stems",),
            )
        ]
        assert runtime.pipeline_calls == []
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_action_queues_import_pipeline_actions_when_request_path_is_available(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-contract-add-song-pipeline-queued.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _PipelineState:
        def __init__(self, status: str) -> None:
            self.status = status

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, bool | None, tuple[str, ...] | None]] = []
            self.switch_calls: list[str] = []
            self.request_calls: list[tuple[str, str, str]] = []
            self._run_states: dict[str, _PipelineState] = {}
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(
            self,
            title: str,
            audio_path: str,
            *,
            run_import_pipeline: bool | None = None,
            import_pipeline_action_ids: tuple[str, ...] | None = None,
        ):
            self.calls.append(
                (title, audio_path, run_import_pipeline, import_pipeline_action_ids)
            )
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id="song_added",
                active_song_title=title,
                active_song_version_id="song_version_added",
            )
            return self._presentation

        def switch_song_version(self, song_version_id: str):
            self.switch_calls.append(song_version_id)
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
            )
            return self._presentation

        def request_object_action_run(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ) -> str:
            del params
            run_id = f"run_{len(self.request_calls) + 1}"
            self.request_calls.append(
                (self._presentation.active_song_version_id, action_id, str(object_id))
            )
            self._run_states[run_id] = _PipelineState(status="completed")
            return run_id

        def get_operation_state(self, run_id: str):
            return self._run_states.get(run_id)

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))
        for _ in range(6):
            widget._action_router._advance_import_pipeline_queue()
            app.processEvents()

        assert runtime.calls == [
            (
                "import",
                "C:/audio/import.wav",
                False,
                None,
            )
        ]
        assert runtime.switch_calls == ["song_version_added"]
        assert runtime.request_calls == [
            ("song_version_added", "timeline.extract_stems", "layer_song")
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_action_legacy_runtime_runs_configured_import_pipeline_actions(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-contract-add-song-pipeline-legacy.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self.pipeline_calls: list[tuple[str, str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

        def run_object_action(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ):
            self.pipeline_calls.append((action_id, str(object_id), object_type))
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))

        assert runtime.calls == [("import", "C:/audio/import.wav")]
        assert runtime.pipeline_calls == [
            ("timeline.extract_stems", "layer_song", "layer")
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_legacy_alias_still_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="add_song_from_path", label="Add Song From Path")
        )

        assert runtime.calls == [("import", "C:/audio/import.wav")]
        assert widget.presentation.title == "import"
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_action_reopens_picker_in_last_used_directory(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = _audio_pipeline_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(self._presentation, title=title)
            return self._presentation

    runtime = _Runtime()
    picker_directories: list[str] = []
    selected_paths = iter(
        [
            ("C:/audio/import.wav", "Audio Files"),
            ("D:/shows/intro.wav", "Audio Files"),
        ]
    )

    def _open_file_dialog(*args, **kwargs):
        if len(args) >= 3:
            picker_directories.append(str(args[2]))
        else:
            picker_directories.append(str(kwargs.get("directory", "")))
        return next(selected_paths)

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        _open_file_dialog,
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))

        assert picker_directories == ["", "C:/audio"]
        assert runtime.calls == [
            ("import", "C:/audio/import.wav"),
            ("intro", "D:/shows/intro.wav"),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_select_song_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[str] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def select_song(self, song_id: str):
            self.calls.append(song_id)
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_title="Beta Song",
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.select"
        )

        widget._trigger_contract_action(
            replace(action, params={**action.params, "song_id": "song_beta"})
        )

        assert runtime.calls == ["song_beta"]
        assert widget.presentation.active_song_id == "song_beta"
        assert widget.presentation.active_song_title == "Beta Song"
    finally:
        widget.close()
        app.processEvents()


def test_contract_switch_song_version_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[str] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def switch_song_version(self, song_version_id: str):
            self.calls.append(song_version_id)
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
                active_song_version_label="Original",
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.switch"
        )

        widget._trigger_contract_action(
            replace(action, params={**action.params, "song_version_id": "song_version_original"})
        )

        assert runtime.calls == ["song_version_original"]
        assert widget.presentation.active_song_version_id == "song_version_original"
        assert widget.presentation.active_song_version_label == "Original"
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_version_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.calls.append((song_id, audio_path, label))
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_version_id="song_version_deluxe",
                active_song_version_label=label or "v3",
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.add"
        )

        widget._trigger_contract_action(
            replace(
                action,
                params={
                    **action.params,
                    "audio_path": "C:/audio/version-3.wav",
                    "label": "Deluxe Edit",
                },
            )
        )

        assert runtime.calls == [
            ("song_alpha", "C:/audio/version-3.wav", "Deluxe Edit")
        ]
        assert widget.presentation.active_song_version_id == "song_version_deluxe"
        assert widget.presentation.active_song_version_label == "Deluxe Edit"
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_version_action_passes_import_pipeline_kwargs_when_runtime_supports_them():
    app = QApplication.instance() or QApplication([])
    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-contract-add-version-pipeline-kwargs.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, str | None, bool | None, tuple[str, ...] | None]] = []
            self.pipeline_calls: list[tuple[str, str, str]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_version(
            self,
            song_id: str,
            audio_path: str,
            *,
            label: str | None = None,
            transfer_layers: bool = False,
            transfer_layer_ids: list[str] | None = None,
            run_import_pipeline: bool | None = None,
            import_pipeline_action_ids: tuple[str, ...] | None = None,
        ):
            self.calls.append(
                (
                    song_id,
                    audio_path,
                    label,
                    run_import_pipeline,
                    import_pipeline_action_ids,
                )
            )
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_version_id="song_version_deluxe",
                active_song_version_label=label or "v3",
            )
            return self._presentation

        def run_object_action(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ):
            self.pipeline_calls.append((action_id, str(object_id), object_type))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.add"
        )

        widget._trigger_contract_action(
            replace(
                action,
                params={
                    **action.params,
                    "audio_path": "C:/audio/version-3.wav",
                    "label": "Deluxe Edit",
                },
            )
        )

        assert runtime.calls == [
            (
                "song_alpha",
                "C:/audio/version-3.wav",
                "Deluxe Edit",
                True,
                ("timeline.extract_stems",),
            )
        ]
        assert runtime.pipeline_calls == []
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_version_action_legacy_runtime_runs_configured_import_pipeline_actions():
    app = QApplication.instance() or QApplication([])
    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-contract-add-version-pipeline-legacy.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, str | None]] = []
            self.pipeline_calls: list[tuple[str, str, str]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.calls.append((song_id, audio_path, label))
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_version_id="song_version_new",
                active_song_version_label=label or "v3",
            )
            return self._presentation

        def run_object_action(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ):
            self.pipeline_calls.append((action_id, str(object_id), object_type))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.add"
        )

        widget._trigger_contract_action(
            replace(
                action,
                params={
                    **action.params,
                    "audio_path": "C:/audio/version-3.wav",
                    "label": "Legacy Edit",
                },
            )
        )

        assert runtime.calls == [
            ("song_alpha", "C:/audio/version-3.wav", "Legacy Edit")
        ]
        assert runtime.pipeline_calls == [
            ("timeline.extract_stems", "layer_song", "layer")
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_song_version_action_passes_layer_transfer_params():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, str | None, bool, list[str] | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_version(
            self,
            song_id: str,
            audio_path: str,
            *,
            label: str | None = None,
            transfer_layers: bool = False,
            transfer_layer_ids: list[str] | None = None,
        ):
            self.calls.append(
                (song_id, audio_path, label, transfer_layers, transfer_layer_ids)
            )
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_version_id="song_version_transfer",
                active_song_version_label=label or "v3",
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="song.version.add",
                label="Add Version",
                params={
                    "song_id": "song_alpha",
                    "audio_path": "C:/audio/version-3.wav",
                    "label": "Transfer Edit",
                    "transfer_layers": True,
                    "transfer_layer_ids": ["layer_drums", "layer_bass"],
                },
            )
        )

        assert runtime.calls == [
            (
                "song_alpha",
                "C:/audio/version-3.wav",
                "Transfer Edit",
                True,
                ["layer_drums", "layer_bass"],
            )
        ]
        assert widget.presentation.active_song_version_id == "song_version_transfer"
    finally:
        widget.close()
        app.processEvents()


def test_contract_delete_song_action_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[str] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def delete_song(self, song_id: str):
            self.calls.append(song_id)
            self._presentation = replace(
                self._presentation,
                active_song_id="song_beta",
                active_song_title="Beta Song",
            )
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="song.delete", label="Delete Song", params={"song_id": "song_alpha"})
        )

        assert runtime.calls == ["song_alpha"]
        assert widget.presentation.active_song_id == "song_beta"
    finally:
        widget.close()
        app.processEvents()


def test_contract_delete_song_version_action_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[str] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def delete_song_version(self, song_version_id: str):
            self.calls.append(song_version_id)
            self._presentation = replace(
                self._presentation,
                active_song_version_id="song_version_original",
                active_song_version_label="Original",
            )
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="song.version.delete",
                label="Delete Version",
                params={"song_version_id": "song_version_festival"},
            )
        )

        assert runtime.calls == ["song_version_festival"]
        assert widget.presentation.active_song_version_id == "song_version_original"
    finally:
        widget.close()
        app.processEvents()


def test_contract_set_song_version_ma3_timecode_pool_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, int | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def list_ma3_timecode_pools(self):
            return [(101, "Intro"), (113, "Festival")]

        def set_song_version_ma3_timecode_pool(
            self,
            song_version_id: str,
            timecode_pool_no: int | None,
        ):
            self.calls.append((song_version_id, timecode_pool_no))
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
                active_song_version_ma3_timecode_pool_no=timecode_pool_no,
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.set_ma3_timecode_pool"
        )

        widget._trigger_contract_action(
            replace(
                action,
                params={
                    **action.params,
                    "song_version_id": "song_version_festival",
                    "timecode_pool_no": 113,
                },
            )
        )

        assert runtime.calls == [("song_version_festival", 113)]
        assert widget.presentation.active_song_version_ma3_timecode_pool_no == 113
    finally:
        widget.close()
        app.processEvents()


def test_contract_set_song_version_ma3_timecode_pool_allows_manual_entry_without_discovered_pools(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, int | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def list_ma3_timecode_pools(self):
            return []

        def set_song_version_ma3_timecode_pool(
            self,
            song_version_id: str,
            timecode_pool_no: int | None,
        ):
            self.calls.append((song_version_id, timecode_pool_no))
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
                active_song_version_ma3_timecode_pool_no=timecode_pool_no,
            )
            return self._presentation

    prompt_capture: dict[str, object] = {}

    def _manual_pool_dialog(*args, **kwargs):
        del kwargs
        prompt_capture["options"] = list(args[3])
        prompt_capture["editable"] = bool(args[5])
        return "TC211", True

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        _manual_pool_dialog,
    )

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        action = next(
            action
            for section in build_timeline_inspector_contract(widget.presentation).context_sections
            for action in section.actions
            if action.action_id == "song.version.set_ma3_timecode_pool"
        )

        widget._trigger_contract_action(
            replace(
                action,
                params={
                    **action.params,
                    "song_version_id": "song_version_festival",
                },
            )
        )

        assert prompt_capture == {
            "options": ["None (Unconfigured)"],
            "editable": True,
        }
        assert runtime.calls == [("song_version_festival", 211)]
        assert widget.presentation.active_song_version_ma3_timecode_pool_no == 211
    finally:
        widget.close()
        app.processEvents()


def test_contract_set_project_ma3_push_offset_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[float] = []
            self._offset_seconds = -1.0
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def get_project_ma3_push_offset_seconds(self) -> float:
            return self._offset_seconds

        def set_project_ma3_push_offset_seconds(self, offset_seconds: float):
            self.calls.append(float(offset_seconds))
            self._offset_seconds = float(offset_seconds)
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="project.settings.set_ma3_push_offset",
                label="Set Global MA3 Push Offset",
                params={"offset_seconds": -0.25},
            )
        )

        assert runtime.calls == [-0.25]
        assert runtime.get_project_ma3_push_offset_seconds() == -0.25
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_adds_song_when_no_active_song(tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_drop_empty"),
                title="Empty",
                layers=[],
                end_time_label="00:00.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.calls.append((title, audio_path))
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id="song_drop",
                active_song_title=title,
            )
            return self._presentation

    runtime = _Runtime()
    audio_path = tmp_path / "drop-song.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(audio_path),))

        assert handled is True
        assert runtime.calls == [("drop-song", str(audio_path))]
        assert widget.presentation.active_song_id == "song_drop"
        assert widget.presentation.active_song_title == "drop-song"
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_runs_configured_import_pipeline_actions(tmp_path):
    app = QApplication.instance() or QApplication([])

    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-drop-import-pipeline-actions.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(run_extract_stems=True)
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.pipeline_calls: list[tuple[str, str, str]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_drop_pipeline"),
                title="Empty",
                layers=[],
                end_time_label="00:00.00",
            )
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id="song_drop",
                active_song_title=title,
                active_song_version_id="song_version_drop",
                active_song_version_label="Original",
            )
            return self._presentation

        def run_object_action(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ):
            self.pipeline_calls.append((action_id, str(object_id), object_type))
            return self._presentation

    runtime = _Runtime()
    audio_path = tmp_path / "drop-song.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(audio_path),))

        assert handled is True
        assert runtime.add_song_calls == [("drop-song", str(audio_path))]
        assert runtime.pipeline_calls == [
            ("timeline.extract_stems", "source_audio", "layer")
        ]
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_offers_new_version_when_song_is_loaded(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.add_version_calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.add_version_calls.append((song_id, audio_path, label))
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_version_id="song_version_new",
                active_song_version_label="v3",
            )
            return self._presentation

    prompts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda _parent, title, message, *_args: prompts.append((title, message))
        or QMessageBox.StandardButton.Yes,
    )

    runtime = _Runtime()
    audio_path = tmp_path / "festival-edit.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(audio_path),))

        assert handled is True
        assert prompts == [
            (
                "Create New Version",
                'This timeline already has a source song loaded.\n\n'
                'Create a new version of "Alpha Song" from "festival-edit.wav"?',
            )
        ]
        assert runtime.add_song_calls == []
        assert runtime.add_version_calls == [("song_alpha", str(audio_path), None)]
        assert widget.presentation.active_song_version_id == "song_version_new"
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_does_nothing_when_new_version_is_declined(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.add_version_calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.add_version_calls.append((song_id, audio_path, label))
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    runtime = _Runtime()
    audio_path = tmp_path / "festival-edit.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(audio_path),))

        assert handled is True
        assert runtime.add_song_calls == []
        assert runtime.add_version_calls == []
        assert widget.presentation.active_song_version_id == "song_version_festival"
    finally:
        widget.close()
        app.processEvents()


def test_timeline_song_browser_targeted_drop_prompts_for_hovered_song(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.add_version_calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.add_version_calls.append((song_id, audio_path, label))
            return self._presentation

    prompts: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda _parent, title, message, *_args: prompts.append((title, message))
        or QMessageBox.StandardButton.Yes,
    )

    runtime = _Runtime()
    audio_path = tmp_path / "beta-edit.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._handle_song_browser_drop(
            SongBrowserAudioDrop(
                audio_paths=(str(audio_path),),
                target_song_id="song_beta",
                target_song_title="Beta Song",
            )
        )

        assert prompts == [
            (
                "Add to Existing Song",
                'Add "beta-edit.wav" to "Beta Song" as a new version?\n\n'
                "Choose No to import it as a new song instead.",
            )
        ]
        assert runtime.add_song_calls == []
        assert runtime.add_version_calls == [("song_beta", str(audio_path), None)]
    finally:
        widget.close()
        app.processEvents()


def test_timeline_song_browser_targeted_drop_can_import_new_song(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.add_version_calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.add_version_calls.append((song_id, audio_path, label))
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )

    runtime = _Runtime()
    audio_path = tmp_path / "beta-edit.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._handle_song_browser_drop(
            SongBrowserAudioDrop(
                audio_paths=(str(audio_path),),
                target_song_id="song_beta",
                target_song_title="Beta Song",
            )
        )

        assert runtime.add_song_calls == [("beta-edit", str(audio_path))]
        assert runtime.add_version_calls == []
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_multiple_files_prompts_and_imports_in_natural_order(
    monkeypatch,
    tmp_path,
):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_drop_batch"),
                title="Batch",
                layers=[],
                end_time_label="00:00.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id=f"song_{len(self.add_song_calls)}",
                active_song_title=title,
            )
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *_args, **_kwargs: ("Import as new songs (append to end)", True),
    )

    runtime = _Runtime()
    path_10 = tmp_path / "Song 10.wav"
    path_2 = tmp_path / "Song 2.wav"
    path_1 = tmp_path / "Song 1.wav"
    for path in (path_10, path_2, path_1):
        path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(path_10), str(path_2), str(path_1)))

        assert handled is True
        assert runtime.add_song_calls == [
            ("Song 1", str(path_1)),
            ("Song 2", str(path_2)),
            ("Song 10", str(path_10)),
        ]
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_import_multiple_files_can_cancel_after_partial_progress(
    monkeypatch,
    tmp_path,
):
    app = QApplication.instance() or QApplication([])

    class _FakeProgressDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            self._canceled = False

        def setWindowTitle(self, _title: str) -> None:
            pass

        def setWindowModality(self, _modality: object) -> None:
            pass

        def setMinimumDuration(self, _duration: int) -> None:
            pass

        def setAutoClose(self, _enabled: bool) -> None:
            pass

        def setAutoReset(self, _enabled: bool) -> None:
            pass

        def setValue(self, value: int) -> None:
            if value >= 1:
                self._canceled = True

        def show(self) -> None:
            pass

        def setLabelText(self, _text: str) -> None:
            pass

        def wasCanceled(self) -> bool:
            return self._canceled

        def close(self) -> None:
            pass

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_drop_cancel"),
                title="Batch",
                layers=[],
                end_time_label="00:00.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id=f"song_{len(self.add_song_calls)}",
                active_song_title=title,
            )
            return self._presentation

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *_args, **_kwargs: ("Import as new songs (append to end)", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_actions.QProgressDialog",
        _FakeProgressDialog,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    runtime = _Runtime()
    path_1 = tmp_path / "Song 1.wav"
    path_2 = tmp_path / "Song 2.wav"
    for path in (path_1, path_2):
        path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(path_1), str(path_2)))

        assert handled is True
        assert runtime.add_song_calls == [("Song 1", str(path_1))]
    finally:
        widget.close()
        app.processEvents()


def test_timeline_drop_batch_import_queues_pipeline_runs_in_context_menu_path_order(
    monkeypatch,
    tmp_path,
):
    app = QApplication.instance() or QApplication([])

    from echozero.application.settings import (
        AppPreferences,
        AppSettingsService,
        SongImportPreferences,
    )

    class _MemorySettingsStore:
        path = Path("/tmp/echozero-test-drop-import-queued-pipeline.json")

        def __init__(self) -> None:
            self._preferences = AppPreferences(
                song_import=SongImportPreferences(
                    pipeline_action_ids=("timeline.extract_stems",)
                )
            )

        def load(self) -> AppPreferences:
            return self._preferences

        def save(self, preferences: AppPreferences) -> None:
            self._preferences = preferences

    class _PipelineState:
        def __init__(self, status: str) -> None:
            self.status = status

    class _Runtime:
        def __init__(self):
            self.operations: list[str] = []
            self.add_song_calls: list[tuple[str, str, bool | None, tuple[str, ...] | None]] = []
            self.switch_calls: list[str] = []
            self.request_calls: list[tuple[str, str, str]] = []
            self._run_states: dict[str, _PipelineState] = {}
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_drop_queue"),
                title="Batch",
                layers=[],
                end_time_label="00:00.00",
            )
            self.runtime_audio = None
            self.app_settings_service = AppSettingsService(_MemorySettingsStore())

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(
            self,
            title: str,
            audio_path: str,
            *,
            run_import_pipeline: bool | None = None,
            import_pipeline_action_ids: tuple[str, ...] | None = None,
        ):
            self.operations.append(f"add:{Path(audio_path).name}")
            self.add_song_calls.append(
                (
                    title,
                    audio_path,
                    run_import_pipeline,
                    import_pipeline_action_ids,
                )
            )
            next_index = len(self.add_song_calls)
            self._presentation = replace(
                self._presentation,
                title=title,
                active_song_id=f"song_{next_index}",
                active_song_title=title,
                active_song_version_id=f"song_version_{next_index}",
                active_song_version_label="Original",
            )
            return self._presentation

        def switch_song_version(self, song_version_id: str):
            self.switch_calls.append(song_version_id)
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
            )
            return self._presentation

        def request_object_action_run(
            self,
            action_id: str,
            params: dict[str, object],
            *,
            object_id: LayerId,
            object_type: str,
        ) -> str:
            del params
            self.operations.append(f"run:{action_id}")
            run_id = f"run_{len(self.request_calls) + 1}"
            self.request_calls.append(
                (self._presentation.active_song_version_id, action_id, str(object_id))
            )
            self._run_states[run_id] = _PipelineState(status="completed")
            return run_id

        def get_operation_state(self, run_id: str):
            return self._run_states.get(run_id)

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *_args, **_kwargs: ("Import as new songs (append to end)", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    runtime = _Runtime()
    path_1 = tmp_path / "Song 1.wav"
    path_2 = tmp_path / "Song 2.wav"
    for path in (path_1, path_2):
        path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_song_drop((str(path_1), str(path_2)))
        for _ in range(6):
            widget._action_router._advance_import_pipeline_queue()
            app.processEvents()

        assert handled is True
        assert runtime.add_song_calls == [
            ("Song 1", str(path_1), False, None),
            ("Song 2", str(path_2), False, None),
        ]
        assert runtime.switch_calls == ["song_version_1", "song_version_2"]
        assert runtime.request_calls == [
            ("song_version_1", "timeline.extract_stems", "source_audio"),
            ("song_version_2", "timeline.extract_stems", "source_audio"),
        ]
        assert runtime.operations == [
            "add:Song 1.wav",
            "add:Song 2.wav",
            "run:timeline.extract_stems",
            "run:timeline.extract_stems",
        ]
    finally:
        widget.close()
        app.processEvents()


def test_song_browser_move_actions_call_runtime_setlist_reorder():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.move_calls: list[tuple[str, int]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def move_song(self, song_id: str, *, steps: int):
            self.move_calls.append((song_id, steps))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._song_browser_panel.move_song_up_requested.emit("song_beta")
        widget._song_browser_panel.move_song_down_requested.emit("song_beta")

        assert runtime.move_calls == [("song_beta", -1), ("song_beta", 1)]
    finally:
        widget.close()
        app.processEvents()


def test_song_browser_selects_song_and_version_through_widget_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.song_calls: list[str] = []
            self.version_calls: list[str] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def select_song(self, song_id: str):
            self.song_calls.append(song_id)
            self._presentation = replace(
                self._presentation,
                active_song_id=song_id,
                active_song_title="Beta Song",
            )
            return self._presentation

        def switch_song_version(self, song_version_id: str):
            self.version_calls.append(song_version_id)
            self._presentation = replace(
                self._presentation,
                active_song_version_id=song_version_id,
                active_song_version_label="Original",
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        version_item = widget._song_browser_panel._version_list.item(0)
        widget._song_browser_panel._handle_version_clicked(version_item)
        song_item = widget._song_browser_panel._songs_tree.topLevelItem(1)
        widget._song_browser_panel._handle_song_item_clicked(song_item, 0)

        assert runtime.version_calls == ["song_version_original"]
        assert runtime.song_calls == ["song_beta"]
    finally:
        widget.close()
        app.processEvents()


def test_song_browser_batch_actions_call_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )

    class _Runtime:
        def __init__(self):
            self.reorder_calls: list[list[str]] = []
            self.delete_calls: list[str] = []
            presentation = _song_switching_presentation()
            self._presentation = replace(
                presentation,
                available_songs=[
                    *presentation.available_songs,
                    SongOptionPresentation(
                        song_id="song_gamma",
                        title="Gamma Song",
                        active_version_id="song_version_gamma",
                        active_version_label="Original",
                        version_count=1,
                        versions=[
                            SongVersionOptionPresentation(
                                song_version_id="song_version_gamma",
                                label="Original",
                                is_active=True,
                            )
                        ],
                    ),
                ],
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def reorder_songs(self, song_ids: list[str]):
            self.reorder_calls.append(song_ids)
            return self._presentation

        def delete_song(self, song_id: str):
            self.delete_calls.append(song_id)
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        selected_song_ids = ("song_alpha", "song_gamma")
        widget._song_browser_panel.batch_move_songs_to_top_requested.emit(selected_song_ids)
        widget._song_browser_panel.batch_move_songs_to_bottom_requested.emit(selected_song_ids)
        widget._song_browser_panel.songs_reordered_requested.emit(
            ("song_beta", "song_alpha", "song_gamma")
        )
        widget._song_browser_panel.batch_delete_songs_requested.emit(selected_song_ids)

        assert runtime.reorder_calls == [
            ["song_alpha", "song_gamma", "song_beta"],
            ["song_beta", "song_alpha", "song_gamma"],
            ["song_beta", "song_alpha", "song_gamma"],
        ]
        assert runtime.delete_calls == ["song_alpha", "song_gamma"]
    finally:
        widget.close()
        app.processEvents()


def test_song_browser_drop_adds_song_when_song_is_loaded(tmp_path):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.add_song_calls: list[tuple[str, str]] = []
            self.add_version_calls: list[tuple[str, str, str | None]] = []
            self._presentation = _song_switching_presentation()
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_song_from_path(self, title: str, audio_path: str):
            self.add_song_calls.append((title, audio_path))
            self._presentation = replace(
                self._presentation,
                active_song_id="song_gamma",
                active_song_title=title,
            )
            return self._presentation

        def add_song_version(self, song_id: str, audio_path: str, *, label: str | None = None):
            self.add_version_calls.append((song_id, audio_path, label))
            return self._presentation

    runtime = _Runtime()
    audio_path = tmp_path / "new-song.wav"
    audio_path.write_bytes(b"RIFF")
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._song_browser_panel.audio_paths_dropped.emit((str(audio_path),))

        assert runtime.add_song_calls == [("new-song", str(audio_path))]
        assert runtime.add_version_calls == []
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_event_layer_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[LayerKind] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_layer(self, kind: LayerKind):
            self.calls.append(kind)
            self._presentation = replace(
                self._presentation,
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_event"),
                        title="Event Layer",
                        main_take_id=None,
                        kind=kind,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_event"),
                selected_layer_ids=[LayerId("layer_event")],
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="add_event_layer", label="Add Event Layer")
        )

        assert runtime.calls == [LayerKind.EVENT]
        assert [layer.title for layer in widget.presentation.layers] == ["Event Layer"]
        assert widget.presentation.layers[0].kind is LayerKind.EVENT
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_section_layer_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[LayerKind] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_layer(self, kind: LayerKind):
            self.calls.append(kind)
            self._presentation = replace(
                self._presentation,
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_section"),
                        title="Section Layer",
                        main_take_id=None,
                        kind=kind,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_section"),
                selected_layer_ids=[LayerId("layer_section")],
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="add_section_layer", label="Add Section Layer")
        )

        assert runtime.calls == [LayerKind.SECTION]
        assert [layer.title for layer in widget.presentation.layers] == ["Section Layer"]
        assert widget.presentation.layers[0].kind is LayerKind.SECTION
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_smpte_layer_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[LayerKind, str | None]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_layer(self, kind: LayerKind, title: str | None = None):
            self.calls.append((kind, title))
            self._presentation = replace(
                self._presentation,
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_smpte"),
                        title=title or "Audio Layer",
                        main_take_id=None,
                        kind=kind,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_smpte"),
                selected_layer_ids=[LayerId("layer_smpte")],
            )
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="add_smpte_layer", label="Add SMPTE Layer")
        )

        assert runtime.calls == [(LayerKind.AUDIO, "SMPTE Layer")]
        assert [layer.title for layer in widget.presentation.layers] == ["SMPTE Layer"]
        assert widget.presentation.layers[0].kind is LayerKind.AUDIO
    finally:
        widget.close()
        app.processEvents()


def test_contract_add_smpte_layer_from_import_split_action_calls_runtime():
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls = 0
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def add_smpte_layer_from_import_split(self):
            self.calls += 1
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="add_smpte_layer_from_import_split",
                label="Add SMPTE Layer from Import Split",
            )
        )

        assert runtime.calls == 1
    finally:
        widget.close()
        app.processEvents()


def test_contract_import_smpte_audio_to_layer_action_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_smpte"),
                        title="SMPTE Layer",
                        main_take_id=None,
                        kind=LayerKind.AUDIO,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_smpte"),
                selected_layer_ids=[LayerId("layer_smpte")],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def import_smpte_audio_to_layer(self, layer_id: str, audio_path: str):
            self.calls.append((layer_id, audio_path))
            return self._presentation

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/smpte-print.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="import_smpte_audio_to_layer",
                label="Import SMPTE Audio",
                params={"layer_id": "layer_smpte"},
            )
        )

        assert runtime.calls == [("layer_smpte", "C:/audio/smpte-print.wav")]
    finally:
        widget.close()
        app.processEvents()


def test_contract_import_smpte_audio_to_layer_prompts_for_uncertain_ltc_channel(
    monkeypatch,
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, bool, str | None]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_smpte"),
                        title="SMPTE Layer",
                        main_take_id=None,
                        kind=LayerKind.AUDIO,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_smpte"),
                selected_layer_ids=[LayerId("layer_smpte")],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def import_smpte_audio_to_layer(
            self,
            layer_id: str,
            audio_path: str,
            *,
            strip_ltc_timecode: bool = True,
            ltc_channel_override: str | None = None,
        ):
            self.calls.append(
                (layer_id, audio_path, strip_ltc_timecode, ltc_channel_override)
            )
            return self._presentation

    source_path = tmp_path / "printed-dual-track.wav"
    source_path.write_bytes(b"RIFF" + b"\x00" * 128)

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(source_path), "Audio Files"),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_contract_mixin.scan_audio_metadata",
        lambda _path: type("_Metadata", (), {"channel_count": 2})(),
    )

    def _fake_detect(_path: Path, *, mode: str = "strict") -> str | None:
        if mode == "aggressive":
            return "right"
        return None

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_contract_mixin.detect_ltc_channel",
        _fake_detect,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: (args[3][1], True),
    )

    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="import_smpte_audio_to_layer",
                label="Import SMPTE Audio",
                params={"layer_id": "layer_smpte"},
            )
        )

        assert runtime.calls == [
            ("layer_smpte", str(source_path), True, "left")
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_import_smpte_audio_to_layer_can_import_stereo_as_is_when_prompted(
    monkeypatch,
    tmp_path: Path,
):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.calls: list[tuple[str, str, bool, str | None]] = []
            self._presentation = TimelinePresentation(
                timeline_id=TimelineId("timeline_empty"),
                title="Empty",
                layers=[
                    LayerPresentation(
                        layer_id=LayerId("layer_smpte"),
                        title="SMPTE Layer",
                        main_take_id=None,
                        kind=LayerKind.AUDIO,
                        status=LayerStatusPresentation(),
                    )
                ],
                selected_layer_id=LayerId("layer_smpte"),
                selected_layer_ids=[LayerId("layer_smpte")],
                end_time_label="00:05.00",
            )
            self.runtime_audio = None

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def import_smpte_audio_to_layer(
            self,
            layer_id: str,
            audio_path: str,
            *,
            strip_ltc_timecode: bool = True,
            ltc_channel_override: str | None = None,
        ):
            self.calls.append(
                (layer_id, audio_path, strip_ltc_timecode, ltc_channel_override)
            )
            return self._presentation

    source_path = tmp_path / "printed-stereo.wav"
    source_path.write_bytes(b"RIFF" + b"\x00" * 256)

    runtime = _Runtime()
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(source_path), "Audio Files"),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_contract_mixin.scan_audio_metadata",
        lambda _path: type("_Metadata", (), {"channel_count": 2})(),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget_action_contract_mixin.detect_ltc_channel",
        lambda _path, *, mode="strict": None,
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QInputDialog.getItem",
        lambda *args, **kwargs: (args[3][2], True),
    )

    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(
                action_id="import_smpte_audio_to_layer",
                label="Import SMPTE Audio",
                params={"layer_id": "layer_smpte"},
            )
        )

        assert runtime.calls == [
            ("layer_smpte", str(source_path), False, None)
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_extract_pipeline_action_warns_when_not_implemented(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def run_object_action(self, action_id, params, *, object_id=None, object_type=None):
            raise NotImplementedError(f"{action_id} pending for {params['layer_id']}")

    warnings: list[str] = []
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QMessageBox.warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "timeline.extract_stems",
            {"layer_id": LayerId("layer_song")},
        )

        assert handled is True
        assert warnings == ["timeline.extract_stems pending for layer_song"]
    finally:
        widget.close()
        app.processEvents()


def test_contract_classify_pipeline_action_prompts_for_model_and_calls_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()
            self.calls: list[tuple[str, dict[str, object]]] = []

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def run_object_action(self, action_id, params, *, object_id=None, object_type=None):
            self.calls.append((action_id, params))
            return self._presentation

    runtime = _Runtime()
    recorded_args: list[tuple[object, ...]] = []

    def _pick(*args, **kwargs):
        recorded_args.append(args)
        return ("C:/models/art_demo.manifest.json", "Artifact Manifests")

    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.ensure_installed_models_dir",
        lambda: Path("C:/Users/griff/.echozero/models"),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        _pick,
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "timeline.classify_drum_events",
            {"layer_id": LayerId("layer_drums")},
        )

        assert handled is True
        assert runtime.calls == [
            (
                "timeline.classify_drum_events",
                {
                    "layer_id": LayerId("layer_drums"),
                    "model_path": "C:/models/art_demo.manifest.json",
                },
            )
        ]
        assert recorded_args
        assert recorded_args[0][2] == "C:/Users/griff/.echozero/models"
    finally:
        widget.close()
        app.processEvents()


def test_contract_extract_classified_drums_calls_runtime_without_picker(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()
            self.calls: list[tuple[str, dict[str, object]]] = []

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def run_object_action(self, action_id, params, *, object_id=None, object_type=None):
            self.calls.append((action_id, params))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "timeline.extract_classified_drums",
            {"layer_id": LayerId("layer_drums")},
        )

        assert handled is True
        assert runtime.calls == [
            (
                "timeline.extract_classified_drums",
                {"layer_id": LayerId("layer_drums")},
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_contract_extract_song_drum_events_calls_runtime_without_picker(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()
            self.calls: list[tuple[str, dict[str, object]]] = []

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def run_object_action(self, action_id, params, *, object_id=None, object_type=None):
            self.calls.append((action_id, params))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._handle_runtime_pipeline_action(
            "timeline.extract_song_drum_events",
            {"layer_id": LayerId("layer_song")},
        )

        assert handled is True
        assert runtime.calls == [
            (
                "timeline.extract_song_drum_events",
                {"layer_id": LayerId("layer_song")},
            )
        ]
    finally:
        widget.close()
        app.processEvents()


def test_unknown_registered_object_action_routes_through_runtime(monkeypatch):
    app = QApplication.instance() or QApplication([])

    class _Runtime:
        def __init__(self):
            self.runtime_audio = None
            self._presentation = _audio_pipeline_presentation()
            self.calls: list[tuple[str, dict[str, object]]] = []

        def presentation(self):
            return self._presentation

        def dispatch(self, intent):
            return self._presentation

        def run_object_action(self, action_id, params, *, object_id=None, object_type=None):
            self.calls.append((action_id, params))
            return self._presentation

    runtime = _Runtime()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        handled = widget._action_router.handle_transfer_action(
            "timeline.extract_drum_events",
            {"layer_id": LayerId("layer_drums")},
        )

        assert handled is True
        assert runtime.calls == [
            (
                "timeline.extract_drum_events",
                {"layer_id": LayerId("layer_drums")},
            )
        ]
    finally:
        widget.close()
        app.processEvents()



__all__ = [name for name in globals() if name.startswith("test_")]
