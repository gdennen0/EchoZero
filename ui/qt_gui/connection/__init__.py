"""
Connection UI module

Provides UI components for creating connections between blocks:
- ConnectionHelper: Shared logic for port enumeration and connection creation
- ConnectionDialog: Dialog-based connection creation
- Drag-to-connect: Handled in NodeScene
"""

from ui.qt_gui.connection.connection_helper import ConnectionHelper
from ui.qt_gui.connection.connection_dialog import ConnectionDialog

__all__ = ['ConnectionHelper', 'ConnectionDialog']


