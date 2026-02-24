"""
ExportMA2 block panel.

Provides UI for configuring GrandMA2 timecode export settings.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QPushButton, QFileDialog, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.application.settings.export_ma2_settings import ExportMA2SettingsManager
from src.utils.message import Log
from src.utils.settings import app_settings
from pathlib import Path


@register_block_panel("ExportMA2")
class ExportMA2Panel(BlockPanelBase):
    """Panel for ExportMA2 block configuration."""

    def __init__(self, block_id: str, facade, parent=None):
        super().__init__(block_id, facade, parent)

        self._settings_manager = ExportMA2SettingsManager(facade, block_id, parent=self)
        self._settings_manager.settings_changed.connect(self._on_setting_changed)

        if self.block:
            self.refresh()

    def create_content_widget(self) -> QWidget:
        """Create ExportMA2-specific UI."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)

        # Output file group
        file_group = QGroupBox("Output File")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(Spacing.SM)

        self.file_path_label = QLabel("No output file selected")
        self.file_path_label.setWordWrap(True)
        self.file_path_label.setStyleSheet(f"""
            QLabel {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
                color: {Colors.TEXT_SECONDARY.name()};
                font-family: monospace;
                font-size: 11px;
            }}
        """)
        file_layout.addWidget(self.file_path_label)

        browse_btn = QPushButton("Browse for Output File...")
        browse_btn.clicked.connect(self._on_browse_file)
        file_layout.addWidget(browse_btn)

        layout.addWidget(file_group)

        # Input requirements group
        input_group = QGroupBox("Input Requirements")
        input_layout = QVBoxLayout(input_group)
        input_layout.setSpacing(Spacing.SM)

        events_label = QLabel(
            "This block requires an 'events' input connection.\n"
            "Connect a block that produces event data (e.g., Detect Onsets, "
            "Editor, or a classify block) to the events input port."
        )
        events_label.setWordWrap(True)
        events_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        input_layout.addWidget(events_label)

        layout.addWidget(input_group)

        # Info note
        info_label = QLabel(
            "Note: MA2 timecode export is not yet implemented.\n"
            "This block will export event timing data to GrandMA2-compatible "
            "timecode format once the export logic is complete."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.ACCENT_YELLOW.name()}; font-size: 10pt;")
        layout.addWidget(info_label)

        # Port filters
        self.add_port_filter_sections(layout)

        layout.addStretch()

        return widget

    def refresh(self):
        """Update UI with current settings from settings manager."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return

        if not self.block or not self._settings_manager.is_loaded():
            return

        try:
            output_path = self._settings_manager.output_path
        except Exception as e:
            Log.error(f"ExportMA2Panel: Failed to load settings: {e}")
            return

        if output_path:
            path = Path(output_path)
            parent_exists = path.parent.exists()
            if parent_exists:
                self.file_path_label.setText(str(path))
                self.file_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_GREEN.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.TEXT_PRIMARY.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
                self.set_status_message(f"Output: {path.name}")
            else:
                self.file_path_label.setText(f"{output_path}\n(Parent directory not found)")
                self.file_path_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: {Colors.BG_LIGHT.name()};
                        border: 1px solid {Colors.ACCENT_RED.name()};
                        border-radius: {border_radius(4)};
                        padding: {Spacing.SM}px;
                        color: {Colors.ACCENT_RED.name()};
                        font-family: monospace;
                        font-size: 11px;
                    }}
                """)
                self.set_status_message("Parent directory not found", error=True)
        else:
            self.file_path_label.setText("No output file selected")
            self.set_status_message("No output file set")

    def _on_browse_file(self):
        """Open dialog to select output file path (undoable via settings manager)."""
        current_path = self._settings_manager.output_path or ""
        start_dir = str(Path(current_path).parent) if current_path else app_settings.get_dialog_path("export_ma2")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select MA2 Output File",
            start_dir,
            "XML Files (*.xml);;All Files (*)"
        )

        if file_path:
            app_settings.set_dialog_path("export_ma2", str(Path(file_path).parent))
            try:
                self._settings_manager.output_path = file_path
                self.set_status_message("Output file set", error=False)
            except ValueError as e:
                self.set_status_message(str(e), error=True)
                self.refresh()

    def _on_setting_changed(self, setting_name: str):
        """React to settings changes from this panel's settings manager."""
        if setting_name == 'output_path':
            self.refresh()

    def refresh_for_undo(self):
        """Refresh panel after undo/redo operation."""
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        self.refresh()

    def _on_block_updated_base(self, event):
        """Route EventBus BlockUpdated to ExportMA2Panel's handler."""
        self._on_block_updated(event)

    def _on_block_updated(self, event):
        """Handle block update event - reload settings and refresh UI."""
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            if self._is_saving:
                return

            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                return

            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()

            QTimer.singleShot(0, self.refresh)
