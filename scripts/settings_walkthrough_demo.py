"""Narrated Qt walkthrough for the new EchoZero settings and object-action flows.
Exists because recorded demonstrations should use one repeatable sequence instead of ad hoc manual capture.
Connects the live timeline shell, settings dialogs, and copy/apply runtime flows for OBS recording.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QLabel, QPlainTextEdit, QVBoxLayout

from echozero.application.presentation.inspector_contract import build_timeline_inspector_contract
from echozero.ui.qt.app_shell import build_app_shell
from echozero.ui.qt.settings_dialog import ActionSettingsDialog
from echozero.ui.qt.timeline.widget import TimelineWidget


def _app() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


def _speak(text: str) -> None:
    subprocess.Popen(["say", text])


def _select_layer(widget: TimelineWidget, title: str) -> None:
    current = widget.presentation
    target = next(layer for layer in current.layers if layer.title == title)
    widget.set_presentation(
        replace(
            current,
            selected_layer_id=target.layer_id,
            selected_layer_ids=[target.layer_id],
            selected_event_ids=[],
        )
    )


def _find_action(widget: TimelineWidget, action_id: str):
    contract = build_timeline_inspector_contract(widget.presentation)
    for section in contract.context_sections:
        for action in section.actions:
            if action.action_id == action_id:
                return action
    raise ValueError(f"Action not found: {action_id}")


class _PreviewDialog(QDialog):
    def __init__(self, title: str, payload: dict[str, object], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 360)
        layout = QVBoxLayout(self)
        header = QLabel("Settings copy preview", self)
        layout.addWidget(header)
        body = QPlainTextEdit(self)
        body.setReadOnly(True)
        lines = []
        for change in payload.get("changes", []):
            lines.append(f"{change['key']}: source={change['from']} target={change['to']} apply={change['apply']}")
        body.setPlainText("\n".join(lines) or "No changes.")
        layout.addWidget(body)


def _show_dialog(dialog: QDialog) -> None:
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()


def _create_runtime(tmp_root: Path, *, audio_path: Path):
    runtime = build_app_shell(
        working_dir_root=tmp_root / "working",
    )
    runtime.add_song_from_path("Settings Demo Song", audio_path)
    return runtime


def _run_extract_stems(runtime, widget: TimelineWidget) -> None:
    runtime.extract_stems("source_audio")
    widget.set_presentation(runtime.presentation())


def _run_extract_classified_drums(runtime, widget: TimelineWidget) -> None:
    drums_layer_id = next(layer.layer_id for layer in runtime.presentation().layers if layer.title == "Drums")
    runtime.extract_classified_drums(drums_layer_id)
    widget.set_presentation(runtime.presentation())


def run_part(part: int) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-path", type=Path, required=True)
    args = parser.parse_args()
    root = Path("artifacts") / "settings-demo-runtime" / f"part-{part}"
    root.mkdir(parents=True, exist_ok=True)
    runtime = _create_runtime(root, audio_path=args.audio_path)

    app = _app()
    widget = TimelineWidget(runtime.presentation(), on_intent=runtime.dispatch, runtime_audio=runtime.runtime_audio)
    widget.setWindowTitle(f"EchoZero Settings Demo Part {part}")
    widget.resize(1540, 980)
    widget.show()
    widget.raise_()
    widget.activateWindow()

    dialogs: list[QDialog] = []

    def open_action_dialog(action_id: str, *, scope: str = "version", title_suffix: str = "") -> None:
        action = _find_action(widget, action_id)
        plan = runtime.describe_object_action(
            action_id,
            action.params,
            object_id=action.params.get("layer_id"),
            object_type="layer",
            scope=scope,
        )
        dialog = ActionSettingsDialog(plan, parent=widget)
        if title_suffix:
            dialog.setWindowTitle(f"{plan.title} Settings {title_suffix}")
        dialogs.append(dialog)
        _show_dialog(dialog)

    def close_dialogs() -> None:
        while dialogs:
            dialogs.pop().close()

    steps: list[tuple[int, callable]] = []

    if part == 1:
        steps = [
            (500, lambda: (_select_layer(widget, "Settings Demo Song"), _speak("Part one. Starting on the source audio layer. This layer exposes the stem separation object action and its pipeline settings."))),
            (5000, lambda: open_action_dialog("timeline.extract_stems", title_suffix="This Version")),
            (11000, lambda: (close_dialogs(), _speak("Now running extract stems through the persisted version settings."))),
            (13000, lambda: _run_extract_stems(runtime, widget)),
            (17000, lambda: (_select_layer(widget, "Drums"), _speak("After stem separation, the drums layer exposes multiple pipeline actions. Extract classified drums, extract drum events, and classify drum events."))),
            (22000, lambda: open_action_dialog("timeline.classify_drum_events", title_suffix="This Version")),
            (29000, lambda: (close_dialogs(), _speak("Now running extract classified drums from the drums layer to create kick and snare outputs."))),
            (31000, lambda: _run_extract_classified_drums(runtime, widget)),
            (36000, lambda: (_select_layer(widget, "Kick"), _speak("The generated kick layer is now visible in the timeline. This completes the object action and pipeline action flow from source audio to derived outputs."))),
            (43000, lambda: app.quit()),
        ]
    elif part == 2:
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q", "device": "cpu"},
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "htdemucs_ft", "device": "mps"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )
        widget.set_presentation(runtime.presentation())
        steps = [
            (500, lambda: (_select_layer(widget, "Settings Demo Song"), _speak("Part two. This shows separate settings scopes. First, version owned effective settings."))),
            (4500, lambda: open_action_dialog("timeline.extract_stems", scope="version", title_suffix="This Version")),
            (11000, lambda: (close_dialogs(), _speak("Now the song default settings for the same action. These are separate from the current version."))),
            (13500, lambda: open_action_dialog("timeline.extract_stems", scope="song_default", title_suffix="Song Default")),
            (20000, lambda: (close_dialogs(), _speak("Reopening the version scope shows that the version kept its own values."))),
            (22500, lambda: open_action_dialog("timeline.extract_stems", scope="version", title_suffix="This Version")),
            (29000, lambda: app.quit()),
        ]
    else:
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "mdx_extra_q", "device": "cpu"},
            object_id="source_audio",
            object_type="layer",
            scope="version",
        )
        runtime.save_object_action_settings(
            "timeline.extract_stems",
            {"layer_id": "source_audio", "model": "htdemucs_ft", "device": "mps"},
            object_id="source_audio",
            object_type="layer",
            scope="song_default",
        )
        widget.set_presentation(runtime.presentation())
        preview_holder: list[QDialog] = []

        def show_preview() -> None:
            payload = runtime.preview_object_action_settings_copy(
                "timeline.extract_stems",
                source_scope="song_default",
                target_scope="version",
                keys=["model", "device"],
            )
            dialog = _PreviewDialog("Copy Settings From Song Default", payload, parent=widget)
            preview_holder.append(dialog)
            _show_dialog(dialog)

        def close_preview() -> None:
            while preview_holder:
                preview_holder.pop().close()

        steps = [
            (500, lambda: (_select_layer(widget, "Settings Demo Song"), _speak("Part three. This shows copy and apply from song default to this version."))),
            (4500, lambda: open_action_dialog("timeline.extract_stems", scope="version", title_suffix="This Version")),
            (10000, lambda: (close_dialogs(), open_action_dialog("timeline.extract_stems", scope="song_default", title_suffix="Song Default"), _speak("The version and song default currently differ."))),
            (16000, lambda: (close_dialogs(), show_preview(), _speak("This preview shows the partial settings transfer before apply."))),
            (23000, lambda: (close_preview(), runtime.apply_object_action_settings_copy("timeline.extract_stems", source_scope="song_default", target_scope="version", keys=["model", "device"]), _speak("Applying the copy now updates only the selected version settings."))),
            (26000, lambda: open_action_dialog("timeline.extract_stems", scope="version", title_suffix="This Version After Apply")),
            (33000, lambda: app.quit()),
        ]

    for delay_ms, callback in steps:
        QTimer.singleShot(delay_ms, callback)

    app.exec()
    close_dialogs()
    runtime.shutdown()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--audio-path", type=Path, required=True)
    args = parser.parse_args()
    # Re-parse inside run_part keeps the script simple for direct subprocess usage.
    import sys
    sys.argv = [sys.argv[0], "--audio-path", str(args.audio_path)]
    return run_part(args.part)


if __name__ == "__main__":
    raise SystemExit(main())
