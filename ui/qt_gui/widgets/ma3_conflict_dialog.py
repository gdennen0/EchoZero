"""
MA3 Conflict Resolution Dialog

UI for resolving conflicts between MA3 and EchoZero event versions.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QTextEdit,
    QButtonGroup, QRadioButton, QDialogButtonBox
)
from PyQt6.QtCore import Qt
from typing import List, Dict, Any, Optional

from src.features.ma3.domain.ma3_sync_state import ConflictRecord, ConflictResolution
from ui.qt_gui.design_system import Colors, Spacing, border_radius


class MA3ConflictDialog(QDialog):
    """
    Dialog for resolving MA3/EchoZero sync conflicts.
    
    Shows side-by-side comparison of conflicting event versions
    and allows user to choose resolution strategy.
    """
    
    def __init__(self, conflicts: List[ConflictRecord], parent=None):
        super().__init__(parent)
        
        self.conflicts = conflicts
        self.current_index = 0
        self.resolutions: Dict[str, ConflictResolution] = {}
        
        self.setWindowTitle("Resolve MA3 Sync Conflicts")
        self.setMinimumSize(800, 600)
        
        self._setup_ui()
        self._show_conflict(0)
    
    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        
        # Header
        header_label = QLabel(f"Resolve {len(self.conflicts)} Conflict(s)")
        header_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(header_label)
        
        # Progress
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.progress_label)
        
        # Conflict details
        details_group = QGroupBox("Conflict Details")
        details_layout = QVBoxLayout(details_group)
        
        self.event_id_label = QLabel()
        self.event_id_label.setStyleSheet(f"font-family: monospace; color: {Colors.TEXT_PRIMARY.name()};")
        details_layout.addWidget(self.event_id_label)
        
        self.differences_label = QLabel()
        self.differences_label.setWordWrap(True)
        self.differences_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        details_layout.addWidget(self.differences_label)
        
        layout.addWidget(details_group)
        
        # Side-by-side comparison
        comparison_group = QGroupBox("Version Comparison")
        comparison_layout = QHBoxLayout(comparison_group)
        
        # MA3 version
        ma3_layout = QVBoxLayout()
        ma3_header = QLabel("MA3 Version")
        ma3_header.setStyleSheet(f"font-weight: bold; color: {Colors.ACCENT_BLUE.name()};")
        ma3_layout.addWidget(ma3_header)
        
        self.ma3_text = QTextEdit()
        self.ma3_text.setReadOnly(True)
        self.ma3_text.setMaximumHeight(200)
        self.ma3_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                font-family: monospace;
                font-size: 10px;
                border: 2px solid {Colors.ACCENT_BLUE.name()};
                border-radius: {border_radius(4)};
            }}
        """)
        ma3_layout.addWidget(self.ma3_text)
        comparison_layout.addLayout(ma3_layout)
        
        # EchoZero version
        ez_layout = QVBoxLayout()
        ez_header = QLabel("EchoZero Version")
        ez_header.setStyleSheet(f"font-weight: bold; color: {Colors.ACCENT_GREEN.name()};")
        ez_layout.addWidget(ez_header)
        
        self.ez_text = QTextEdit()
        self.ez_text.setReadOnly(True)
        self.ez_text.setMaximumHeight(200)
        self.ez_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                font-family: monospace;
                font-size: 10px;
                border: 2px solid {Colors.ACCENT_GREEN.name()};
                border-radius: {border_radius(4)};
            }}
        """)
        ez_layout.addWidget(self.ez_text)
        comparison_layout.addLayout(ez_layout)
        
        layout.addWidget(comparison_group)
        
        # Resolution options
        resolution_group = QGroupBox("Resolution")
        resolution_layout = QVBoxLayout(resolution_group)
        
        self.resolution_group = QButtonGroup(self)
        
        self.use_ma3_radio = QRadioButton("Use MA3 Version")
        self.use_ma3_radio.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        self.resolution_group.addButton(self.use_ma3_radio, ConflictResolution.USE_MA3.value)
        resolution_layout.addWidget(self.use_ma3_radio)
        
        self.use_ez_radio = QRadioButton("Use EchoZero Version")
        self.use_ez_radio.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        self.resolution_group.addButton(self.use_ez_radio, ConflictResolution.USE_EZ.value)
        resolution_layout.addWidget(self.use_ez_radio)
        
        self.merge_radio = QRadioButton("Merge (MA3 time + EZ classification)")
        self.merge_radio.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        self.resolution_group.addButton(self.merge_radio, ConflictResolution.MERGE.value)
        resolution_layout.addWidget(self.merge_radio)
        
        self.skip_radio = QRadioButton("Skip (keep as-is)")
        self.skip_radio.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        self.resolution_group.addButton(self.skip_radio, ConflictResolution.SKIP.value)
        resolution_layout.addWidget(self.skip_radio)
        
        # Default to USE_MA3
        self.use_ma3_radio.setChecked(True)
        
        layout.addWidget(resolution_group)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self._on_previous)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self._on_next)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        # Batch resolution
        self.apply_all_btn = QPushButton("Apply to All Remaining")
        self.apply_all_btn.clicked.connect(self._on_apply_all)
        nav_layout.addWidget(self.apply_all_btn)
        
        layout.addLayout(nav_layout)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def _show_conflict(self, index: int):
        """Display conflict at given index."""
        if index < 0 or index >= len(self.conflicts):
            return
        
        self.current_index = index
        conflict = self.conflicts[index]
        
        # Update progress
        self.progress_label.setText(f"Conflict {index + 1} of {len(self.conflicts)}")
        
        # Update event ID
        self.event_id_label.setText(f"Event ID: {conflict.event_id}")
        
        # Update differences
        if conflict.differences:
            diff_text = "Differences: " + ", ".join(conflict.differences)
        else:
            diff_text = "Both versions modified since last sync"
        self.differences_label.setText(diff_text)
        
        # Update MA3 version
        if conflict.ma3_version:
            ma3_lines = []
            for key, value in conflict.ma3_version.items():
                if key not in ('metadata', 'source'):
                    ma3_lines.append(f"{key}: {value}")
            self.ma3_text.setPlainText("\n".join(ma3_lines))
        else:
            self.ma3_text.setPlainText("(deleted in MA3)")
        
        # Update EZ version
        if conflict.ez_version:
            ez_lines = []
            for key, value in conflict.ez_version.items():
                if key not in ('user_data',):
                    ez_lines.append(f"{key}: {value}")
            self.ez_text.setPlainText("\n".join(ez_lines))
        else:
            self.ez_text.setPlainText("(deleted in EchoZero)")
        
        # Restore previous resolution if exists
        if conflict.event_id in self.resolutions:
            resolution = self.resolutions[conflict.event_id]
            button = self.resolution_group.button(resolution.value)
            if button:
                button.setChecked(True)
        
        # Update navigation buttons
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.conflicts) - 1)
    
    def _save_current_resolution(self):
        """Save the current conflict's resolution."""
        if self.current_index < 0 or self.current_index >= len(self.conflicts):
            return
        
        conflict = self.conflicts[self.current_index]
        button_id = self.resolution_group.checkedId()
        resolution = ConflictResolution(button_id)
        self.resolutions[conflict.event_id] = resolution
    
    def _on_previous(self):
        """Go to previous conflict."""
        self._save_current_resolution()
        self._show_conflict(self.current_index - 1)
    
    def _on_next(self):
        """Go to next conflict."""
        self._save_current_resolution()
        self._show_conflict(self.current_index + 1)
    
    def _on_apply_all(self):
        """Apply current resolution to all remaining conflicts."""
        self._save_current_resolution()
        
        current_resolution = self.resolutions.get(self.conflicts[self.current_index].event_id)
        if not current_resolution:
            return
        
        # Apply to all remaining
        for i in range(self.current_index + 1, len(self.conflicts)):
            conflict = self.conflicts[i]
            self.resolutions[conflict.event_id] = current_resolution
        
        # Show confirmation
        remaining = len(self.conflicts) - self.current_index - 1
        if remaining > 0:
            self.progress_label.setText(
                f"Applied {current_resolution.name} to {remaining} remaining conflicts"
            )
    
    def get_resolutions(self) -> Dict[str, ConflictResolution]:
        """
        Get all conflict resolutions.
        
        Returns:
            Dict mapping event_id to ConflictResolution
        """
        # Save current before returning
        self._save_current_resolution()
        return self.resolutions.copy()
