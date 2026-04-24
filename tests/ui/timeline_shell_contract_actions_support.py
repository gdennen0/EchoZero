"""Contract-action timeline-shell support cases.
Exists to isolate runtime action routing coverage from layout, transfer, and interaction support.
Connects the compatibility wrapper to the bounded contract-action support slice.
"""

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
        "echozero.ui.qt.timeline.widget.QInputDialog.getText",
        lambda *args, **kwargs: ("Imported Song", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(InspectorAction(action_id="song.add", label="Add Song"))

        assert runtime.calls == [("Imported Song", "C:/audio/import.wav")]
        assert widget.presentation.title == "Imported Song"
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
        "echozero.ui.qt.timeline.widget.QInputDialog.getText",
        lambda *args, **kwargs: ("Imported Song", True),
    )
    monkeypatch.setattr(
        "echozero.ui.qt.timeline.widget.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: ("C:/audio/import.wav", "Audio Files"),
    )
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch)
    try:
        widget._trigger_contract_action(
            InspectorAction(action_id="add_song_from_path", label="Add Song From Path")
        )

        assert runtime.calls == [("Imported Song", "C:/audio/import.wav")]
        assert widget.presentation.title == "Imported Song"
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
        version_item = widget._song_browser_panel._tree.topLevelItem(0).child(0)
        widget._song_browser_panel._handle_item_clicked(version_item, 0)
        song_item = widget._song_browser_panel._tree.topLevelItem(1)
        widget._song_browser_panel._handle_item_clicked(song_item, 0)

        assert runtime.version_calls == ["song_version_original"]
        assert runtime.song_calls == ["song_beta"]
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
