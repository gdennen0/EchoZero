"""
Setlist Error Summary Panel

Compact error display for setlist processing failures.
Shows full error messages inline, optimized for narrow sidebar layout.
"""
import subprocess
import sys
from typing import List, Dict, Any
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class SetlistErrorSummaryPanel(ThemeAwareMixin, QWidget):
    """
    Compact error summary for setlist processing.

    Shows each failed song with its full error message inline.
    Provides access to the log folder for deeper investigation.
    """

    retry_requested = pyqtSignal(str)  # song_id (kept for backward compat)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.errors: List[Dict[str, Any]] = []
        self._setup_ui()
        self._init_theme_aware()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, Spacing.SM, 0, 0)
        layout.setSpacing(Spacing.XS)

        # Header row: red accent bar with count
        header = QHBoxLayout()
        header.setSpacing(Spacing.SM)

        self._header_label = QLabel("0 errors")
        self._header_label.setStyleSheet(
            f"color: {Colors.ACCENT_RED.name()}; "
            f"font-size: 12px; font-weight: bold;"
        )
        header.addWidget(self._header_label)

        header.addStretch()

        # Open log folder button
        log_btn = QPushButton("Open Logs")
        log_btn.setStyleSheet(StyleFactory.button("small"))
        log_btn.setToolTip("Open the application log folder")
        log_btn.clicked.connect(self._on_open_log_folder)
        header.addWidget(log_btn)

        layout.addLayout(header)

        # Scrollable error list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        self._error_list_widget = QWidget()
        self._error_list_layout = QVBoxLayout(self._error_list_widget)
        self._error_list_layout.setContentsMargins(0, 0, 0, 0)
        self._error_list_layout.setSpacing(Spacing.XS)
        self._error_list_layout.addStretch()

        scroll.setWidget(self._error_list_widget)
        layout.addWidget(scroll, 1)

        # Export button (compact, at bottom)
        export_btn = QPushButton("Export Error Report")
        export_btn.setStyleSheet(StyleFactory.button("small"))
        export_btn.clicked.connect(self._on_export)
        layout.addWidget(export_btn)

        self.setVisible(False)

    def set_errors(self, errors: List[Dict[str, Any]]):
        """Set errors to display."""
        self.errors = errors
        self._rebuild_error_list()
        self.setVisible(len(errors) > 0)

        count = len(errors)
        self._header_label.setText(
            f"{count} error{'s' if count != 1 else ''}"
        )

    def _rebuild_error_list(self):
        """Rebuild the inline error list from current errors."""
        # Clear existing error cards
        while self._error_list_layout.count() > 1:
            item = self._error_list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for error in self.errors:
            song_path = error.get("song", "Unknown")
            error_msg = error.get("error", "Unknown error")
            card = self._create_error_card(song_path, error_msg)
            # Insert before the stretch
            self._error_list_layout.insertWidget(
                self._error_list_layout.count() - 1, card
            )

    def _create_error_card(self, song_path: str, error_msg: str) -> QFrame:
        """Create a compact error card showing song name and full error."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.ACCENT_RED.darker(130).name()};
                border-left: 3px solid {Colors.ACCENT_RED.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.XS}px;
            }}
        """)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(
            Spacing.SM, Spacing.XS, Spacing.XS, Spacing.XS
        )
        card_layout.setSpacing(2)

        # Song name
        name_label = QLabel(Path(song_path).name)
        name_label.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY.name()}; "
            f"font-weight: bold; font-size: 11px; "
            f"border: none; background: transparent;"
        )
        name_label.setToolTip(song_path)
        card_layout.addWidget(name_label)

        # Full error message (monospace, wrapped, selectable)
        error_label = QLabel(error_msg)
        error_label.setStyleSheet(
            f"color: {Colors.ACCENT_RED.lighter(120).name()}; "
            f"font-size: 10px; font-family: monospace; "
            f"border: none; background: transparent;"
        )
        error_label.setWordWrap(True)
        error_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        error_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        card_layout.addWidget(error_label)

        return card

    def _on_open_log_folder(self):
        """Open the application log folder in the system file manager."""
        try:
            from src.utils.paths import get_logs_dir
            logs_dir = get_logs_dir()

            if not logs_dir.exists():
                logs_dir.mkdir(parents=True, exist_ok=True)

            if sys.platform == "darwin":
                subprocess.Popen(["open", str(logs_dir)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(logs_dir)])
            else:
                subprocess.Popen(["xdg-open", str(logs_dir)])

            Log.info(f"SetlistErrorSummaryPanel: Opened log folder: {logs_dir}")
        except Exception as e:
            Log.error(f"SetlistErrorSummaryPanel: Failed to open log folder: {e}")
            QMessageBox.warning(
                self, "Could Not Open Logs",
                f"Failed to open log folder:\n{e}"
            )

    def _on_export(self):
        """Export error report to a text file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Error Report",
            "setlist_errors.txt",
            "Text Files (*.txt);;All Files (*)",
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Setlist Processing Error Report\n")
                    f.write("=" * 50 + "\n\n")
                    for i, error in enumerate(self.errors, 1):
                        song = error.get("song", "Unknown")
                        msg = error.get("error", "Unknown error")
                        f.write(f"{i}. {Path(song).name}\n")
                        f.write(f"   Path: {song}\n")
                        f.write(f"   Error:\n")
                        for line in msg.splitlines():
                            f.write(f"     {line}\n")
                        f.write("\n")

                    from src.utils.paths import get_logs_dir
                    f.write(f"Log folder: {get_logs_dir()}\n")

                QMessageBox.information(
                    self, "Export Complete",
                    f"Error report saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.warning(
                    self, "Export Failed",
                    f"Failed to export error report:\n{e}"
                )
