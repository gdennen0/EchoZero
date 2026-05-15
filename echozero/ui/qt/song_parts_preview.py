"""
Song-parts confirmation preview panel.
Exists to give the song-section detection flow a stable explanatory surface before run.
Connects object-action sessions to lightweight Qt preview text.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from echozero.application.timeline.object_actions import ObjectActionSettingsSession


class SongPartsPreviewPanel(QWidget):
    """Small preview/explanation panel shown for song-part detection settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("songPartsPreviewPanel")
        self.setProperty("section", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self._title = QLabel("Song Parts Preview", self)
        self._title.setObjectName("songPartsPreviewTitle")
        self._summary = QLabel(self)
        self._summary.setObjectName("songPartsPreviewSummary")
        self._summary.setWordWrap(True)
        layout.addWidget(self._title)
        layout.addWidget(self._summary)
        self.hide()

    def set_session(self, session: ObjectActionSettingsSession) -> None:
        """Show this panel only for song-section detection sessions."""

        if session.action_id != "timeline.extract_song_sections":
            self.hide()
            return
        self._summary.setText(_summary_text(session))
        self.show()


def _summary_text(session: ObjectActionSettingsSession) -> str:
    method = _field_value(session, "detect_method") or "mir_self_similarity"
    method_label = {
        "mir_self_similarity": "MIR part-boundary detection",
        "mfcc_sequence_pooling": "MFCC sequence pooling",
        "determine_sections_style": "legacy section estimation",
    }.get(str(method), str(method).replace("_", " ").title())
    source = next(
        (str(value) for key, value in session.plan.locked_bindings if key == "audio_file"),
        "the selected audio layer",
    )
    return (
        f"EchoZero will analyze {source} using {method_label}, then create a section layer "
        "showing where parts change for you to review before syncing or editing."
    )


def _field_value(session: ObjectActionSettingsSession, key: str) -> object | None:
    for state in session.scope_states:
        if state.scope != session.scope:
            continue
        for field_value in state.field_values:
            if field_value.key == key:
                return field_value.draft_value
    for field in tuple(session.plan.editable_fields) + tuple(session.plan.advanced_fields):
        if field.key == key:
            return field.value
    return None


__all__ = ["SongPartsPreviewPanel"]
