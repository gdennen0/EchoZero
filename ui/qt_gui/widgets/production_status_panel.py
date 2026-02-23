"""
Production Status Panel

Simplified progress dashboard for production mode.
Shows high-level song-by-song processing status with overall progress.

Subscribes to EventBus events from SetlistProcessingService and presents
them in a user-friendly format without block-level execution details.
"""
from typing import Dict, Optional, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QScrollArea, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor

from ui.qt_gui.design_system import Colors, Spacing, Typography, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class _SongStatusRow(QFrame):
    """Single row representing one song's processing status."""

    def __init__(self, song_name: str, parent=None):
        super().__init__(parent)
        self._song_name = song_name
        self._status = "pending"

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_MEDIUM.name()};
                border-radius: {border_radius()}px;
                padding: 4px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._status_dot = QLabel()
        self._status_dot.setFixedSize(12, 12)
        self._update_dot_color()
        layout.addWidget(self._status_dot)

        self._name_label = QLabel(song_name)
        self._name_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 13px;")
        layout.addWidget(self._name_label, 1)

        self._progress = QProgressBar()
        self._progress.setMinimum(0)
        self._progress.setMaximum(100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedWidth(180)
        self._progress.setFixedHeight(18)
        layout.addWidget(self._progress)

        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        self._detail_label.setFixedWidth(160)
        layout.addWidget(self._detail_label)

    def set_status(self, status: str, detail: str = "", percentage: int = -1):
        self._status = status
        self._update_dot_color()
        self._detail_label.setText(detail)
        if percentage >= 0:
            self._progress.setValue(percentage)
        if status == "completed":
            self._progress.setValue(100)
            self._progress.setFormat("Complete")
        elif status == "failed":
            self._progress.setFormat("Failed")

    def _update_dot_color(self):
        color_map = {
            "pending": Colors.TEXT_SECONDARY,
            "processing": Colors.ACCENT_BLUE,
            "completed": Colors.ACCENT_GREEN,
            "failed": Colors.ACCENT_RED,
        }
        color = color_map.get(self._status, Colors.TEXT_SECONDARY)
        self._status_dot.setStyleSheet(f"""
            background-color: {color.name()};
            border-radius: 6px;
            min-width: 12px; min-height: 12px;
            max-width: 12px; max-height: 12px;
        """)


class ProductionStatusPanel(QWidget):
    """
    High-level processing dashboard for production mode.

    Shows:
    - Overall progress bar with current song name
    - Song list with status dots (pending/processing/done/failed)
    - Error summary with retry option
    """

    def __init__(self, facade=None, parent=None):
        super().__init__(parent)
        self._facade = facade
        self._song_rows: Dict[str, _SongStatusRow] = {}
        self._song_ids: List[str] = []
        self._setup_ui()
        if facade and facade.event_bus:
            self._subscribe_events(facade.event_bus)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.SM)

        # Header
        header = QLabel("Processing Status")
        header.setFont(Typography.heading_font())
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 16px; font-weight: bold;")
        layout.addWidget(header)

        # Overall progress
        self._overall_label = QLabel("Idle")
        self._overall_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 13px;")
        layout.addWidget(self._overall_label)

        self._overall_progress = QProgressBar()
        self._overall_progress.setMinimum(0)
        self._overall_progress.setMaximum(100)
        self._overall_progress.setValue(0)
        self._overall_progress.setTextVisible(True)
        self._overall_progress.setFixedHeight(22)
        layout.addWidget(self._overall_progress)

        # Scrollable song list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._songs_container = QWidget()
        self._songs_layout = QVBoxLayout(self._songs_container)
        self._songs_layout.setContentsMargins(0, 0, 0, 0)
        self._songs_layout.setSpacing(4)
        self._songs_layout.addStretch()

        scroll.setWidget(self._songs_container)
        layout.addWidget(scroll, 1)

        # Error summary area
        self._error_frame = QFrame()
        self._error_frame.setVisible(False)
        self._error_frame.setStyleSheet(f"""
            QFrame {{
                background: {Colors.BG_DARK.name()};
                border: 1px solid {Colors.ACCENT_RED.name()};
                border-radius: {border_radius()}px;
                padding: 8px;
            }}
        """)
        error_layout = QVBoxLayout(self._error_frame)
        error_layout.setContentsMargins(8, 8, 8, 8)
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-size: 12px;")
        error_layout.addWidget(self._error_label)
        layout.addWidget(self._error_frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_songs(self, songs: list):
        """Populate the song list.  Each item needs .id and .audio_path."""
        self.clear()
        from pathlib import Path
        for song in songs:
            name = Path(song.audio_path).stem if hasattr(song, 'audio_path') else str(song)
            song_id = song.id if hasattr(song, 'id') else str(song)
            row = _SongStatusRow(name)
            self._song_rows[song_id] = row
            self._song_ids.append(song_id)
            idx = self._songs_layout.count() - 1  # before the stretch
            self._songs_layout.insertWidget(idx, row)

    def clear(self):
        """Remove all song rows."""
        for row in self._song_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._song_rows.clear()
        self._song_ids.clear()
        self._error_frame.setVisible(False)
        self._overall_progress.setValue(0)
        self._overall_label.setText("Idle")

    def update_song_status(self, song_id: str, status: str, detail: str = "", percentage: int = -1):
        row = self._song_rows.get(song_id)
        if row:
            row.set_status(status, detail, percentage)
        self._update_overall()

    def show_error(self, message: str):
        self._error_label.setText(message)
        self._error_frame.setVisible(True)

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------

    def _subscribe_events(self, event_bus):
        event_bus.subscribe("SetlistProcessingStarted", self._on_processing_started)
        event_bus.subscribe("SetlistSongStarted", self._on_song_started)
        event_bus.subscribe("SetlistSongCompleted", self._on_song_completed)
        event_bus.subscribe("SetlistSongFailed", self._on_song_failed)
        event_bus.subscribe("SetlistProcessingCompleted", self._on_processing_completed)
        event_bus.subscribe("SubprocessProgress", self._on_subprocess_progress)

    def _on_processing_started(self, event):
        data = getattr(event, 'data', {}) or {}
        total = data.get('total_songs', 0)
        self._overall_label.setText(f"Processing {total} song(s)...")
        self._overall_progress.setValue(0)
        self._error_frame.setVisible(False)

    def _on_song_started(self, event):
        data = getattr(event, 'data', {}) or {}
        song_id = data.get('song_id', '')
        self.update_song_status(song_id, "processing", "Starting...")

    def _on_song_completed(self, event):
        data = getattr(event, 'data', {}) or {}
        song_id = data.get('song_id', '')
        self.update_song_status(song_id, "completed")

    def _on_song_failed(self, event):
        data = getattr(event, 'data', {}) or {}
        song_id = data.get('song_id', '')
        error = data.get('error', 'Unknown error')
        self.update_song_status(song_id, "failed", error[:60])
        self.show_error(f"Song failed: {error}")

    def _on_processing_completed(self, event):
        self._overall_label.setText("Processing complete")
        self._overall_progress.setValue(100)

    def _on_subprocess_progress(self, event):
        data = getattr(event, 'data', {}) or {}
        message = data.get('message', '')
        percentage = data.get('percentage', 0)
        # Update overall progress with sub-block detail
        self._overall_label.setText(message[:80] if message else "Processing...")
        if self._song_ids:
            # Find the currently-processing song and update its percentage
            for sid, row in self._song_rows.items():
                if row._status == "processing":
                    row.set_status("processing", message[:40], percentage)
                    break

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_overall(self):
        total = len(self._song_rows)
        if total == 0:
            return
        completed = sum(1 for r in self._song_rows.values() if r._status in ("completed", "failed"))
        pct = int(100 * completed / total)
        self._overall_progress.setValue(pct)
        self._overall_label.setText(f"{completed} of {total} songs processed")
