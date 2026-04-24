"""Manual pull source browser widgets for MA3 timecode and track selection.
Exists to keep the pull workspace focused on operator-friendly source browsing instead of combo chains.
Connects manual pull presentation data to lightweight Qt controls that emit canonical selection signals.
"""

from __future__ import annotations

from types import SimpleNamespace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ManualPullTimecodePicker(QWidget):
    """Compact segmented picker for MA3 timecode pools."""

    timecode_selected = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._buttons_by_number: dict[int, QPushButton] = {}
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)

    def set_timecodes(self, timecodes, selected_timecode_no: int | None) -> None:
        self._clear_layout()
        self._buttons_by_number = {}
        for timecode in timecodes:
            number = int(timecode.number)
            button = QPushButton(self._timecode_label(timecode), self)
            button.setCheckable(True)
            button.setChecked(number == selected_timecode_no)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setObjectName(f"manualPullTimecodeButton{number}")
            button.setStyleSheet(
                "QPushButton { padding: 7px 12px; border-radius: 14px; text-align: center; }"
            )
            button.clicked.connect(
                lambda _checked=False, timecode_no=number: self.timecode_selected.emit(
                    int(timecode_no)
                )
            )
            self._button_group.addButton(button)
            self._buttons_by_number[number] = button
            self._layout.addWidget(button)
        self._layout.addStretch(1)

    def _clear_layout(self) -> None:
        self._button_group.deleteLater()
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _timecode_label(timecode) -> str:
        if timecode.name:
            return f"TC{timecode.number} {timecode.name}"
        return f"TC{timecode.number}"


class ManualPullSourceBrowser(QWidget):
    """Grouped clickable browser for MA3 track groups and source tracks."""

    track_group_selected = pyqtSignal(int)
    track_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._group_buttons: dict[int, QPushButton] = {}
        self._track_buttons: dict[str, QPushButton] = {}
        self._container_layout = QVBoxLayout(self)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(10)
        self._container_layout.addStretch(1)

    def set_source_options(
        self,
        *,
        track_groups,
        tracks,
        selected_track_group_no: int | None,
        active_track_coord: str | None,
        selected_track_coords: list[str] | tuple[str, ...],
    ) -> None:
        self._clear_layout()
        self._group_buttons = {}
        self._track_buttons = {}

        tracks_by_group: dict[int, list] = {}
        for track in sorted(tracks, key=self._track_sort_key):
            group_no = self._track_group_no(track.coord)
            if group_no is None:
                continue
            tracks_by_group.setdefault(group_no, []).append(track)

        if not track_groups and not tracks:
            empty_label = QLabel("No MA3 track groups are available for this timecode pool.", self)
            empty_label.setWordWrap(True)
            self._container_layout.addWidget(empty_label)
            self._container_layout.addStretch(1)
            return

        selected_coords = {str(coord) for coord in selected_track_coords}
        groups = list(track_groups)
        rendered_group_numbers = {int(group.number) for group in groups}
        for group_no in sorted(set(tracks_by_group) - rendered_group_numbers):
            groups.append(
                SimpleNamespace(
                    number=group_no,
                    name=f"Group {group_no}",
                    track_count=len(tracks_by_group[group_no]),
                )
            )
        groups.sort(key=lambda value: int(value.number))

        for group in groups:
            group_no = int(group.number)
            section = QFrame(self)
            section.setFrameShape(QFrame.Shape.StyledPanel)
            section.setObjectName(f"manualPullGroupSection{group_no}")
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(12, 12, 12, 12)
            section_layout.setSpacing(8)

            header_button = QPushButton(self._group_label(group), section)
            header_button.setCheckable(True)
            header_button.setChecked(group_no == selected_track_group_no)
            header_button.setCursor(Qt.CursorShape.PointingHandCursor)
            header_button.setObjectName(f"manualPullGroupButton{group_no}")
            header_button.setStyleSheet(
                "QPushButton { padding: 8px 10px; border-radius: 12px; text-align: left; font-weight: 600; }"
            )
            header_button.clicked.connect(
                lambda _checked=False, next_group_no=group_no: self.track_group_selected.emit(
                    int(next_group_no)
                )
            )
            section_layout.addWidget(header_button)
            self._group_buttons[group_no] = header_button

            group_tracks = tracks_by_group.get(group_no, [])
            if not group_tracks:
                empty_label = QLabel("No tracks in this group.", section)
                empty_label.setObjectName(f"manualPullGroupEmpty{group_no}")
                section_layout.addWidget(empty_label)
                self._container_layout.addWidget(section)
                continue

            tracks_layout = QVBoxLayout()
            tracks_layout.setContentsMargins(0, 0, 0, 0)
            tracks_layout.setSpacing(6)
            for track in group_tracks:
                track_button = QPushButton(
                    self._track_label(
                        track,
                        is_active=track.coord == active_track_coord,
                        is_selected=track.coord in selected_coords,
                    ),
                    section,
                )
                track_button.setCheckable(True)
                track_button.setChecked(track.coord == active_track_coord)
                track_button.setCursor(Qt.CursorShape.PointingHandCursor)
                track_button.setMinimumHeight(54)
                track_button.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Fixed,
                )
                track_button.setObjectName(
                    "manualPullTrackButton_" + self._safe_name(track.coord)
                )
                track_button.setStyleSheet(
                    "QPushButton { padding: 10px 12px; border-radius: 12px; text-align: left; }"
                )
                track_button.clicked.connect(
                    lambda _checked=False, coord=str(track.coord): self.track_selected.emit(coord)
                )
                tracks_layout.addWidget(track_button)
                self._track_buttons[str(track.coord)] = track_button
            section_layout.addLayout(tracks_layout)
            self._container_layout.addWidget(section)

        self._container_layout.addStretch(1)

    def _clear_layout(self) -> None:
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    child_widget = child_item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    @staticmethod
    def _group_label(track_group) -> str:
        track_count = track_group.track_count
        suffix = ""
        if track_count is not None:
            noun = "track" if int(track_count) == 1 else "tracks"
            suffix = f" · {track_count} {noun}"
        name = str(track_group.name or f"Group {track_group.number}").strip()
        return f"TG{track_group.number} {name}{suffix}"

    @staticmethod
    def _track_label(track, *, is_active: bool, is_selected: bool) -> str:
        track_no = track.number
        title_prefix = f"TR{track_no}" if track_no is not None else "TR?"
        title = str(track.name or title_prefix).strip() or title_prefix
        meta_parts = [str(track.coord)]
        if track.note:
            meta_parts.append(str(track.note))
        if track.event_count is not None:
            noun = "event" if int(track.event_count) == 1 else "events"
            meta_parts.append(f"{track.event_count} {noun}")
        if is_active:
            meta_parts.append("Previewing")
        elif is_selected:
            meta_parts.append("Queued")
        return f"{title_prefix} {title}\n" + " · ".join(meta_parts)

    @staticmethod
    def _track_group_no(raw_coord: str | None) -> int | None:
        coord = str(raw_coord or "").strip().lower()
        if "_tg" not in coord:
            return None
        group_text = coord.split("_tg", 1)[1].split("_", 1)[0]
        try:
            parsed = int(group_text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

    @staticmethod
    def _track_sort_key(track) -> tuple[int, str]:
        track_no = track.number if track.number is not None else 999999
        return int(track_no), str(track.coord)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(character if character.isalnum() else "_" for character in str(value))
