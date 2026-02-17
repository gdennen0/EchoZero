"""
Setlist Error Summary Panel

Displays errors that occurred during setlist processing.
Allows users to see failed songs and retry them.
"""
from typing import List, Dict, Any
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QAbstractItemView, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class SetlistErrorSummaryPanel(ThemeAwareMixin, QWidget):
    """
    Error summary panel for setlist processing.
    
    Shows failed songs with error messages and provides
    options to retry failed songs or export error report.
    """
    
    # Signal emitted when retry is requested
    retry_requested = pyqtSignal(str)  # song_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.errors: List[Dict[str, Any]] = []
        self._setup_ui()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup the UI layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        layout.setSpacing(Spacing.XS)
        
        # Title
        title = QLabel("Error Summary")
        title.setFont(Typography.heading_font())
        title.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Error count
        self.error_count_label = QLabel("0 errors")
        self.error_count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.error_count_label)
        
        # Errors table
        self.errors_table = QTableWidget()
        self.errors_table.setColumnCount(4)
        self.errors_table.setHorizontalHeaderLabels(["Song", "Error", "", ""])
        self.errors_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.errors_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.errors_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.errors_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.errors_table.setColumnWidth(2, 60)  # Details button
        self.errors_table.setColumnWidth(3, 60)  # Retry button
        self.errors_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.errors_table.setAlternatingRowColors(True)
        self.errors_table.setStyleSheet(StyleFactory.table())
        layout.addWidget(self.errors_table)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(Spacing.SM)
        
        self.retry_all_btn = QPushButton("Retry All Failed")
        self.retry_all_btn.setStyleSheet(StyleFactory.button("primary"))
        self.retry_all_btn.clicked.connect(self._on_retry_all)
        buttons_layout.addWidget(self.retry_all_btn)
        
        self.export_btn = QPushButton("Export Error Report")
        self.export_btn.setStyleSheet(StyleFactory.button())
        self.export_btn.clicked.connect(self._on_export)
        buttons_layout.addWidget(self.export_btn)
        
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Initially hidden
        self.setVisible(False)
    
    def set_errors(self, errors: List[Dict[str, Any]]):
        """Set errors to display"""
        self.errors = errors
        self._update_table()
        
        # Show/hide panel based on errors
        self.setVisible(len(errors) > 0)
        
        # Update count
        count = len(errors)
        self.error_count_label.setText(f"{count} error{'s' if count != 1 else ''}")
    
    def _update_table(self):
        """Update errors table"""
        self.errors_table.setRowCount(len(self.errors))
        
        for row, error in enumerate(self.errors):
            song_path = error.get("song", "Unknown")
            error_msg = error.get("error", "Unknown error")
            song_id = error.get("song_id")
            
            # Song name
            song_item = QTableWidgetItem(Path(song_path).name)
            song_item.setData(Qt.ItemDataRole.UserRole, song_id)  # Store song_id
            self.errors_table.setItem(row, 0, song_item)
            
            # Error message (truncated for display)
            display_msg = error_msg if len(error_msg) <= 60 else error_msg[:57] + "..."
            error_item = QTableWidgetItem(display_msg)
            error_item.setForeground(Colors.ACCENT_RED)
            error_item.setToolTip(error_msg)  # Full error on hover
            self.errors_table.setItem(row, 1, error_item)
            
            # Details button to show full error
            details_btn = QPushButton("Details")
            details_btn.setStyleSheet(StyleFactory.button("small"))
            details_btn.clicked.connect(lambda checked, s=song_path, e=error_msg: self._show_error_details(s, e))
            self.errors_table.setCellWidget(row, 2, details_btn)
            
            # Retry button
            retry_btn = QPushButton("Retry")
            retry_btn.setStyleSheet(StyleFactory.button("small"))
            if song_id:
                retry_btn.clicked.connect(lambda checked, sid=song_id: self._on_retry_song(sid))
            self.errors_table.setCellWidget(row, 3, retry_btn)
    
    def _on_retry_song(self, song_id: str):
        """Retry a specific song"""
        self.retry_requested.emit(song_id)
    
    def _show_error_details(self, song_path: str, error_msg: str):
        """Show full error details in a dialog"""
        from PyQt6.QtWidgets import QDialog, QTextEdit, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Error Details - {Path(song_path).name}")
        dialog.setMinimumSize(500, 300)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(Spacing.SM)
        
        # Song info
        song_label = QLabel(f"Song: {Path(song_path).name}")
        song_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: bold;")
        layout.addWidget(song_label)
        
        path_label = QLabel(f"Path: {song_path}")
        path_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        
        # Error message in a text edit for easy copying
        error_label = QLabel("Error Message:")
        error_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(error_label)
        
        error_text = QTextEdit()
        error_text.setPlainText(error_msg)
        error_text.setReadOnly(True)
        error_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                color: {Colors.ACCENT_RED.name()};
                padding: 8px;
                font-family: monospace;
            }}
        """)
        layout.addWidget(error_text, 1)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        
        # Add copy button
        copy_btn = QPushButton("Copy Error")
        copy_btn.clicked.connect(lambda: self._copy_to_clipboard(error_msg))
        button_box.addButton(copy_btn, QDialogButtonBox.ButtonRole.ActionRole)
        
        layout.addWidget(button_box)
        
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
            }}
        """)
        
        dialog.exec()
    
    def _copy_to_clipboard(self, text: str):
        """Copy text to clipboard"""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        Log.info("SetlistErrorSummaryPanel: Copied error to clipboard")
    
    def _on_retry_all(self):
        """Retry all failed songs"""
        if not self.errors:
            return
        
        reply = QMessageBox.question(
            self, "Retry All Failed Songs",
            f"Retry processing for {len(self.errors)} failed song(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for error in self.errors:
                song_id = error.get("song_id")
                if song_id:
                    self.retry_requested.emit(song_id)
    
    def _on_export(self):
        """Export error report"""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Error Report",
            "setlist_errors.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Setlist Processing Error Report\n")
                    f.write("=" * 50 + "\n\n")
                    for i, error in enumerate(self.errors, 1):
                        f.write(f"{i}. {Path(error.get('song', 'Unknown')).name}\n")
                        f.write(f"   Error: {error.get('error', 'Unknown error')}\n")
                        f.write(f"   Path: {error.get('song', 'Unknown')}\n\n")
                QMessageBox.information(self, "Export Complete", f"Error report saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Failed", f"Failed to export error report:\n{e}")
    
    # Style helpers