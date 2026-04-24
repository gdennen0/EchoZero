"""Manual push route dialog for MA3 target selection.
Exists to keep push routing selection cohesive and out of widget action mixins.
Connects push-flow presentation options to one popup that selects timecode, group, and track.
"""

from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QLabel, QMessageBox, QVBoxLayout


class ManualPushRouteDialog(QDialog):
    """Popup for selecting MA3 timecode, track group, and track for push routing."""

    timecode_selected = pyqtSignal(int)
    track_group_selected = pyqtSignal(int)
    create_timecode_requested = pyqtSignal()
    create_track_group_requested = pyqtSignal()
    create_track_requested = pyqtSignal()

    _CREATE_TIMECODE_SENTINEL = "__manual_push_route__:create_timecode"
    _CREATE_TRACK_GROUP_SENTINEL = "__manual_push_route__:create_track_group"
    _CREATE_TRACK_SENTINEL = "__manual_push_route__:create_track"

    SEQUENCE_MODE_NONE = "none"
    SEQUENCE_MODE_ASSIGN_EXISTING = "assign_existing"
    SEQUENCE_MODE_CREATE_NEXT_AVAILABLE = "create_next_available"
    SEQUENCE_MODE_CREATE_CURRENT_SONG = "create_current_song"

    def __init__(self, *, title: str, prompt: str, parent=None) -> None:
        super().__init__(parent)
        self._syncing = False
        self._show_sequence_controls = False
        self._show_apply_mode_controls = False
        self._track_by_coord: dict[str, object] = {}
        self._available_sequence_items: list[tuple[int, str]] = []
        self._sequence_range_available = False
        self._sequence_range_summary = "Current song range: unavailable"

        self.setWindowTitle(title)
        self.resize(540, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        intro = QLabel(
            f"{prompt}: pick MA3 timecode, track group, and track in one place.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._timecode_combo = QComboBox(self)
        self._timecode_combo.setObjectName("manualPushRouteTimecodeCombo")
        layout.addWidget(QLabel("Timecode pool", self))
        layout.addWidget(self._timecode_combo)

        self._track_group_combo = QComboBox(self)
        self._track_group_combo.setObjectName("manualPushRouteTrackGroupCombo")
        layout.addWidget(QLabel("Track group", self))
        layout.addWidget(self._track_group_combo)

        self._track_combo = QComboBox(self)
        self._track_combo.setObjectName("manualPushRouteTrackCombo")
        layout.addWidget(QLabel("Track", self))
        layout.addWidget(self._track_combo)

        self._sequence_mode_label = QLabel("Sequence", self)
        self._sequence_mode_combo = QComboBox(self)
        self._sequence_mode_combo.setObjectName("manualPushRouteSequenceModeCombo")
        self._sequence_combo_label = QLabel("Existing sequence", self)
        self._sequence_combo = QComboBox(self)
        self._sequence_combo.setObjectName("manualPushRouteSequenceCombo")
        self._sequence_hint = QLabel("", self)
        self._sequence_hint.setWordWrap(True)
        layout.addWidget(self._sequence_mode_label)
        layout.addWidget(self._sequence_mode_combo)
        layout.addWidget(self._sequence_combo_label)
        layout.addWidget(self._sequence_combo)
        layout.addWidget(self._sequence_hint)

        self._apply_mode_label = QLabel("Write mode", self)
        self._apply_mode_combo = QComboBox(self)
        self._apply_mode_combo.setObjectName("manualPushRouteApplyModeCombo")
        self._apply_mode_combo.addItem("Merge", "merge")
        self._apply_mode_combo.addItem("Overwrite", "overwrite")
        self._apply_mode_hint = QLabel("", self)
        self._apply_mode_hint.setWordWrap(True)
        layout.addWidget(self._apply_mode_label)
        layout.addWidget(self._apply_mode_combo)
        layout.addWidget(self._apply_mode_hint)

        self._summary = QLabel("Target: Select an MA3 track", self)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Use Route")
        layout.addWidget(self._buttons)

        self._timecode_combo.currentIndexChanged.connect(self._handle_timecode_changed)
        self._track_group_combo.currentIndexChanged.connect(self._handle_track_group_changed)
        self._track_combo.currentIndexChanged.connect(self._handle_track_changed)
        self._sequence_mode_combo.currentIndexChanged.connect(self._sync_sequence_controls)

        self.configure_sheet(
            show_sequence_controls=False,
            show_apply_mode_controls=False,
            default_apply_mode="merge",
        )

    def configure_sheet(
        self,
        *,
        show_sequence_controls: bool,
        show_apply_mode_controls: bool,
        default_apply_mode: str = "merge",
    ) -> None:
        self._show_sequence_controls = bool(show_sequence_controls)
        self._show_apply_mode_controls = bool(show_apply_mode_controls)

        self._set_section_visible(
            self._sequence_mode_label,
            self._show_sequence_controls,
        )
        self._set_section_visible(
            self._sequence_mode_combo,
            self._show_sequence_controls,
        )
        self._set_section_visible(
            self._sequence_combo_label,
            self._show_sequence_controls,
        )
        self._set_section_visible(
            self._sequence_combo,
            self._show_sequence_controls,
        )
        self._set_section_visible(
            self._sequence_hint,
            self._show_sequence_controls,
        )

        self._set_section_visible(
            self._apply_mode_label,
            self._show_apply_mode_controls,
        )
        self._set_section_visible(
            self._apply_mode_combo,
            self._show_apply_mode_controls,
        )
        self._set_section_visible(
            self._apply_mode_hint,
            self._show_apply_mode_controls,
        )

        self._set_combo_value(
            self._apply_mode_combo,
            str(default_apply_mode or "merge").strip().lower() or "merge",
            default_index=0,
        )

    def set_flow(self, flow, *, preferred_track_coord: str | None = None) -> None:
        self._syncing = True
        try:
            self._track_by_coord = {
                str(track.coord): track for track in flow.available_tracks
            }
            self._available_sequence_items = [
                (int(sequence.number), self._sequence_label(sequence.number, sequence.name))
                for sequence in flow.available_sequences
            ]
            sequence_range = flow.current_song_sequence_range
            self._sequence_range_available = sequence_range is not None
            if sequence_range is None:
                self._sequence_range_summary = "Current song range: unavailable"
            elif sequence_range.song_label:
                self._sequence_range_summary = (
                    f"Current song range: {sequence_range.song_label} "
                    f"({sequence_range.start}-{sequence_range.end})"
                )
            else:
                self._sequence_range_summary = (
                    f"Current song range: {sequence_range.start}-{sequence_range.end}"
                )

            with QSignalBlocker(self._timecode_combo):
                self._timecode_combo.clear()
                self._timecode_combo.addItem(
                    "+ Create New Timecode...",
                    self._CREATE_TIMECODE_SENTINEL,
                )
                for timecode in flow.available_timecodes:
                    self._timecode_combo.addItem(
                        self._timecode_label(timecode.number, timecode.name),
                        int(timecode.number),
                    )
                self._set_combo_value(
                    self._timecode_combo,
                    flow.selected_timecode_no,
                    default_index=1 if self._timecode_combo.count() > 1 else 0,
                )

            with QSignalBlocker(self._track_group_combo):
                self._track_group_combo.clear()
                self._track_group_combo.addItem(
                    "+ Create New Track Group...",
                    self._CREATE_TRACK_GROUP_SENTINEL,
                )
                for group in flow.available_track_groups:
                    self._track_group_combo.addItem(
                        self._track_group_label(group.number, group.name, group.track_count),
                        int(group.number),
                    )
                self._set_combo_value(
                    self._track_group_combo,
                    flow.selected_track_group_no,
                    default_index=1 if self._track_group_combo.count() > 1 else 0,
                )

            with QSignalBlocker(self._track_combo):
                self._track_combo.clear()
                self._track_combo.addItem(
                    "+ Create New Track...",
                    self._CREATE_TRACK_SENTINEL,
                )
                for track in flow.available_tracks:
                    self._track_combo.addItem(self._track_label(track), str(track.coord))
                initial_coord = (
                    str(preferred_track_coord).strip()
                    if preferred_track_coord not in {None, ""}
                    else None
                )
                if initial_coord not in {None, ""}:
                    selected_coord = initial_coord
                else:
                    selected_coord = (
                        flow.target_track_coord
                        if flow.target_track_coord in {
                            str(track.coord) for track in flow.available_tracks
                        }
                        else None
                    )
                if selected_coord is None and self._track_combo.count() > 1:
                    selected_coord = str(self._track_combo.itemData(1))
                self._set_combo_value(
                    self._track_combo,
                    selected_coord,
                    default_index=1 if self._track_combo.count() > 1 else 0,
                )
        finally:
            self._syncing = False

        self._sync_sequence_controls()
        self._sync_apply_mode_controls()
        self._refresh_summary()

    def selected_timecode_no(self) -> int | None:
        value = self._timecode_combo.currentData()
        if self._is_create_sentinel(value):
            return None
        return int(value) if value not in {None, ""} else None

    def selected_track_group_no(self) -> int | None:
        value = self._track_group_combo.currentData()
        if self._is_create_sentinel(value):
            return None
        return int(value) if value not in {None, ""} else None

    def selected_track_coord(self) -> str | None:
        value = self._track_combo.currentData()
        if self._is_create_sentinel(value):
            return None
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def selected_sequence_mode(self) -> str | None:
        if not self._show_sequence_controls:
            return None
        value = self._sequence_mode_combo.currentData()
        if value in {None, ""}:
            return None
        text = str(value).strip()
        return text or None

    def selected_sequence_no(self) -> int | None:
        if self.selected_sequence_mode() != self.SEQUENCE_MODE_ASSIGN_EXISTING:
            return None
        value = self._sequence_combo.currentData()
        return int(value) if value not in {None, ""} else None

    def selected_apply_mode(self) -> str | None:
        if not self._show_apply_mode_controls:
            return None
        if not self._apply_mode_combo.isEnabled():
            return "merge"
        value = self._apply_mode_combo.currentData()
        if value in {None, ""}:
            return "merge"
        text = str(value).strip().lower()
        return text or "merge"

    def accept(self) -> None:
        selected_track_coord = self.selected_track_coord()
        if selected_track_coord is None:
            QMessageBox.warning(self, self.windowTitle(), "Select an MA3 track first.")
            return

        selected_track = self._track_by_coord.get(selected_track_coord)
        if self._show_sequence_controls and selected_track is not None:
            sequence_no = getattr(selected_track, "sequence_no", None)
            if sequence_no is None:
                sequence_mode = self.selected_sequence_mode()
                if sequence_mode == self.SEQUENCE_MODE_ASSIGN_EXISTING:
                    if self.selected_sequence_no() is None:
                        QMessageBox.warning(
                            self,
                            self.windowTitle(),
                            "Select an existing MA3 sequence first.",
                        )
                        return
                elif sequence_mode == self.SEQUENCE_MODE_CREATE_CURRENT_SONG:
                    if not self._sequence_range_available:
                        QMessageBox.warning(
                            self,
                            self.windowTitle(),
                            "Current song MA3 sequence range is not available right now.",
                        )
                        return

        super().accept()

    def _handle_timecode_changed(self) -> None:
        if self._syncing:
            return
        if self._timecode_combo.currentData() == self._CREATE_TIMECODE_SENTINEL:
            self.create_timecode_requested.emit()
            return
        selected = self.selected_timecode_no()
        if selected is not None:
            self.timecode_selected.emit(selected)

    def _handle_track_group_changed(self) -> None:
        if self._syncing:
            return
        if self._track_group_combo.currentData() == self._CREATE_TRACK_GROUP_SENTINEL:
            self.create_track_group_requested.emit()
            return
        selected = self.selected_track_group_no()
        if selected is not None:
            self.track_group_selected.emit(selected)

    def _handle_track_changed(self) -> None:
        if not self._syncing and self._track_combo.currentData() == self._CREATE_TRACK_SENTINEL:
            self.create_track_requested.emit()
        self._sync_sequence_controls()
        self._sync_apply_mode_controls()
        self._refresh_summary()

    def _sync_sequence_controls(self) -> None:
        if not self._show_sequence_controls:
            return

        selected_track = self._selected_track()
        previous_mode = str(self._sequence_mode_combo.currentData() or "").strip().lower()

        with QSignalBlocker(self._sequence_mode_combo), QSignalBlocker(self._sequence_combo):
            self._sequence_mode_combo.clear()
            self._sequence_combo.clear()

            if selected_track is None:
                self._sequence_mode_combo.addItem("Select a track first", self.SEQUENCE_MODE_NONE)
                self._sequence_mode_combo.setEnabled(False)
                self._sequence_combo.setEnabled(False)
                self._sequence_combo_label.setVisible(False)
                self._sequence_combo.setVisible(False)
                self._sequence_hint.setText("Sequence prep appears after you choose a track.")
                return

            sequence_no = getattr(selected_track, "sequence_no", None)
            if sequence_no is not None:
                self._sequence_mode_combo.addItem(
                    f"Assigned sequence: {int(sequence_no)}",
                    self.SEQUENCE_MODE_NONE,
                )
                self._sequence_mode_combo.setEnabled(False)
                self._sequence_combo.setEnabled(False)
                self._sequence_combo_label.setVisible(False)
                self._sequence_combo.setVisible(False)
                self._sequence_hint.setText(
                    "This track already has an MA3 sequence. No prep needed."
                )
                return

            self._sequence_mode_combo.setEnabled(True)
            self._sequence_mode_combo.addItem(
                "Assign existing sequence",
                self.SEQUENCE_MODE_ASSIGN_EXISTING,
            )
            self._sequence_mode_combo.addItem(
                "Create next available sequence",
                self.SEQUENCE_MODE_CREATE_NEXT_AVAILABLE,
            )
            self._sequence_mode_combo.addItem(
                "Create sequence in current song range",
                self.SEQUENCE_MODE_CREATE_CURRENT_SONG,
            )

            if previous_mode in {
                self.SEQUENCE_MODE_ASSIGN_EXISTING,
                self.SEQUENCE_MODE_CREATE_NEXT_AVAILABLE,
                self.SEQUENCE_MODE_CREATE_CURRENT_SONG,
            }:
                self._set_combo_value(self._sequence_mode_combo, previous_mode, default_index=0)
            else:
                default_mode = (
                    self.SEQUENCE_MODE_ASSIGN_EXISTING
                    if self._available_sequence_items
                    else self.SEQUENCE_MODE_CREATE_NEXT_AVAILABLE
                )
                self._set_combo_value(self._sequence_mode_combo, default_mode, default_index=0)

            selected_mode = str(self._sequence_mode_combo.currentData() or "")
            if selected_mode == self.SEQUENCE_MODE_ASSIGN_EXISTING:
                for sequence_no, label in self._available_sequence_items:
                    self._sequence_combo.addItem(label, int(sequence_no))
                has_sequences = self._sequence_combo.count() > 0
                self._sequence_combo.setEnabled(has_sequences)
                self._sequence_combo_label.setVisible(True)
                self._sequence_combo.setVisible(True)
                self._sequence_hint.setText(
                    "No MA3 sequences available right now."
                    if not has_sequences
                    else self._sequence_range_summary
                )
            elif selected_mode == self.SEQUENCE_MODE_CREATE_CURRENT_SONG:
                self._sequence_combo_label.setVisible(False)
                self._sequence_combo.setVisible(False)
                self._sequence_combo.setEnabled(False)
                self._sequence_hint.setText(self._sequence_range_summary)
            else:
                self._sequence_combo_label.setVisible(False)
                self._sequence_combo.setVisible(False)
                self._sequence_combo.setEnabled(False)
                self._sequence_hint.setText(self._sequence_range_summary)

    def _sync_apply_mode_controls(self) -> None:
        if not self._show_apply_mode_controls:
            return

        selected_track = self._selected_track()
        if selected_track is None:
            self._apply_mode_combo.setEnabled(False)
            self._set_combo_value(self._apply_mode_combo, "merge", default_index=0)
            self._apply_mode_hint.setText("Choose a track first.")
            return

        event_count = getattr(selected_track, "event_count", None)
        try:
            is_empty_or_new = event_count in {None, ""} or int(event_count) <= 0
        except (TypeError, ValueError):
            is_empty_or_new = True
        if is_empty_or_new:
            self._apply_mode_combo.setEnabled(False)
            self._set_combo_value(self._apply_mode_combo, "merge", default_index=0)
            self._apply_mode_hint.setText(
                "New or empty track selected: write mode is automatic."
            )
            return

        self._apply_mode_combo.setEnabled(True)
        self._apply_mode_hint.setText(
            "Choose how incoming events should apply to existing MA3 events."
        )

    def _refresh_summary(self) -> None:
        track_label = self._track_combo.currentText().strip()
        if self._track_combo.currentData() == self._CREATE_TRACK_SENTINEL:
            self._summary.setText("Target: Create a new MA3 track in the selected group")
            return
        if not track_label:
            self._summary.setText("Target: Select an MA3 track")
            return

        if self._show_apply_mode_controls:
            selected_mode = self.selected_apply_mode()
            if selected_mode == "overwrite" and self._apply_mode_combo.isEnabled():
                self._summary.setText(f"Target: {track_label} · Overwrite")
                return
            self._summary.setText(f"Target: {track_label} · Merge")
            return

        self._summary.setText(f"Target: {track_label}")

    def _selected_track(self):
        selected_coord = self.selected_track_coord()
        if selected_coord is None:
            return None
        return self._track_by_coord.get(selected_coord)

    @staticmethod
    def _set_section_visible(widget, visible: bool) -> None:
        widget.setVisible(bool(visible))

    @staticmethod
    def _set_combo_value(combo: QComboBox, value, *, default_index: int = 0) -> None:
        if combo.count() == 0:
            return
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(min(max(int(default_index), 0), combo.count() - 1))

    @classmethod
    def _is_create_sentinel(cls, value: object) -> bool:
        return value in {
            cls._CREATE_TIMECODE_SENTINEL,
            cls._CREATE_TRACK_GROUP_SENTINEL,
            cls._CREATE_TRACK_SENTINEL,
        }

    @staticmethod
    def _sequence_label(number: int, name: str) -> str:
        return f"{int(number)} - {str(name or number)}"

    @staticmethod
    def _timecode_label(number: int, name: str | None) -> str:
        if name:
            return f"TC{number} {name}"
        return f"TC{number}"

    @staticmethod
    def _track_group_label(number: int, name: str, track_count: int | None) -> str:
        if track_count is None:
            return f"TG{number} {name}"
        noun = "track" if int(track_count) == 1 else "tracks"
        return f"TG{number} {name} · {track_count} {noun}"

    @staticmethod
    def _track_label(track) -> str:
        title_prefix = f"TR{track.number}" if track.number is not None else "TR?"
        title = str(track.name or title_prefix).strip() or title_prefix
        details = [str(track.coord)]
        if track.note:
            details.append(str(track.note))
        if track.event_count is not None:
            noun = "event" if int(track.event_count) == 1 else "events"
            details.append(f"{track.event_count} {noun}")
        return f"{title_prefix} {title} · " + " · ".join(details)
