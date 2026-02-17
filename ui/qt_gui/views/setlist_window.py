"""
Setlist Window

Simple container for setlist processing view.
Kept minimal to ensure docking works correctly.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QSizePolicy
from PyQt6.QtCore import Qt

from src.application.api.application_facade import ApplicationFacade
from src.utils.message import Log
from ui.qt_gui.views.setlist_view import SetlistView


class SetlistWindow(QWidget):
    """
    Setlist window - simple container for SetlistView.
    
    Uses a scroll area to allow the content to be viewed in smaller spaces,
    ensuring the dock can resize freely.
    """
    
    def __init__(self, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.facade = facade
        self.setlist_view = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI with scroll area for flexibility"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create scroll area so the content can be smaller than its preferred size
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # Create the setlist view
        self.setlist_view = SetlistView(self.facade)
        scroll.setWidget(self.setlist_view)
        
        layout.addWidget(scroll)
        
        # Set flexible size policy so dock can resize freely
        # But don't stretch vertically - content should justify to top
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    
    def refresh(self):
        """Refresh the view"""
        if self.setlist_view and hasattr(self.setlist_view, 'refresh'):
            self.setlist_view.refresh()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.setlist_view and hasattr(self.setlist_view, 'cleanup'):
            self.setlist_view.cleanup()

