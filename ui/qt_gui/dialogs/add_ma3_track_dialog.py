"""
Add MA3 Track Dialog

Dialog for selecting an MA3 track to add to synced layers.
Shows available MA3 tracks (excluding already synced ones).
Each item has a + button to add it.
"""

from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from src.utils.message import Log


class AddMA3TrackDialog(ThemeAwareMixin, QDialog):
    """
    Dialog for adding MA3 tracks to synced layers.
    
    Shows available MA3 tracks (not already synced) with + button per item.
    """
    
    track_added = pyqtSignal(str, int, int, int, str)  # Emits (coord, timecode_no, track_group, track, name) when track is added
    
    def __init__(
        self,
        available_tracks: List[Dict[str, Any]],  # List of track dicts from controller.get_available_ma3_tracks()
        parent=None
    ):
        super().__init__(parent)
        self.available_tracks = available_tracks
        self.selected_track: Optional[Dict[str, Any]] = None
        
        self.setWindowTitle("Add MA3 Track")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        self.setModal(True)
        
        self._setup_ui()
        self._init_theme_aware()
    
    def _setup_ui(self):
        """Setup dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Header
        header_label = QLabel("Select an MA3 track to sync:")
        header_label.setFont(Typography.heading_font())
        header_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; padding-bottom: {Spacing.SM}px;")
        layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel("Available tracks that are not yet synced:")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding-bottom: {Spacing.MD}px;")
        layout.addWidget(desc_label)
        
        # Scroll area with track list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background-color: transparent; border: none;")
        
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(Spacing.SM)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add track rows
        if self.available_tracks:
            for track_data in self.available_tracks:
                row = self._create_track_row(track_data)
                container_layout.addWidget(row)
        else:
            # No tracks available
            empty_label = QLabel("No available MA3 tracks.")
            empty_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: {Spacing.MD}px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(empty_label)
        
        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.setMinimumWidth(100)
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def _create_track_row(self, track_data: Dict[str, Any]) -> QWidget:
        """Create a row widget for a track"""
        row = QFrame()
        row.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: {Spacing.SM}px;
            }}
        """)
        
        row_layout = QHBoxLayout(row)
        row_layout.setSpacing(Spacing.MD)
        row_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        # Track display name (from display_name if available, otherwise construct from coord)
        display_name = track_data.get('display_name', '')
        if not display_name:
            # Construct display name
            coord = track_data.get('coord', '')
            name = track_data.get('name', '')
            if name:
                display_name = f"{coord} ({name})"
            else:
                display_name = coord
        
        track_label = QLabel(display_name)
        track_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        row_layout.addWidget(track_label, 1)
        
        # + button
        add_button = QPushButton("+")
        add_button.setFixedWidth(30)
        add_button.setFixedHeight(30)
        add_button.clicked.connect(lambda: self._on_add_track(track_data))
        add_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(110).name()};
            }}
        """)
        row_layout.addWidget(add_button)
        
        return row
    
    def _on_add_track(self, track_data: Dict[str, Any]):
        """Handle + button click"""
        coord = track_data.get('coord', '')
        timecode_no = track_data.get('timecode_no', 0)
        track_group = track_data.get('track_group', 0)
        track = track_data.get('track', 0)
        name = track_data.get('name', '')
        
        if coord and timecode_no:
            self.selected_track = track_data
            self.track_added.emit(coord, timecode_no, track_group, track, name)
            self.accept()  # Close dialog after adding
