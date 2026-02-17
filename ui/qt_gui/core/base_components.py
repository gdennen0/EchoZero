"""
Base UI Components

Reusable, standardized UI building blocks.
All custom widgets should follow these patterns.
"""
from typing import Optional, Callable
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGraphicsItem
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from ui.qt_gui.design_system import Colors, Spacing, Typography, Sizes, border_radius


class Panel(QWidget):
    """
    Base panel widget with consistent styling.
    Use for all container widgets.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            Panel {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-radius: {border_radius(6)};
            }}
        """)


class Section(QWidget):
    """
    Section within a panel - groups related controls.
    """
    
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(Spacing.SM)
        self.layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        
        # Title
        if title:
            title_label = QLabel(title)
            title_label.setFont(Typography.heading_font())
            title_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
            self.layout.addWidget(title_label)


class Button(QPushButton):
    """
    Standardized button with consistent styling.
    """
    
    def __init__(self, text: str, variant: str = "primary", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self._apply_style(variant)
    
    def _apply_style(self, variant: str):
        if variant == "primary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 8px 16px;
                    font-weight: 500;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.ACCENT_BLUE.darker(110).name()};
                }}
                QPushButton:disabled {{
                    background-color: {Colors.BG_LIGHT.name()};
                    color: {Colors.TEXT_DISABLED.name()};
                }}
            """)
        elif variant == "secondary":
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(4)};
                    padding: 8px 16px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.HOVER.name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.SELECTED.name()};
                }}
            """)


class BaseGraphicsNode(QGraphicsItem):
    """
    Base class for all graphics items in the node editor.
    Provides consistent interface and behavior.
    """
    
    # Signals (would be on a QObject wrapper in practice)
    
    def __init__(self):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        
        self._hovered = False
    
    def hoverEnterEvent(self, event):
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.update()
        super().hoverLeaveEvent(event)
    
    def get_color(self) -> QColor:
        """Override to provide item-specific color"""
        return Colors.BG_MEDIUM
    
    def draw_shadow(self, painter: QPainter):
        """Draw standard drop shadow"""
        # Simple shadow effect (could be enhanced with QGraphicsDropShadowEffect)
        pass


def create_horizontal_layout(spacing: int = Spacing.MD, margins: tuple = (0, 0, 0, 0)) -> QHBoxLayout:
    """Create standardized horizontal layout"""
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


def create_vertical_layout(spacing: int = Spacing.MD, margins: tuple = (0, 0, 0, 0)) -> QVBoxLayout:
    """Create standardized vertical layout"""
    layout = QVBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    return layout


class ItemSelectionDialog:
    """
    Standardized dialog for selecting items to delete/modify.
    Reusable pattern for any type of item management.
    """
    
    @staticmethod
    def show_delete_dialog(parent, title: str, items: list, item_name_getter: Callable = None) -> Optional[list]:
        """
        Show a dialog to select items for deletion.
        
        Args:
            parent: Parent widget
            title: Dialog title
            items: List of items to choose from
            item_name_getter: Function to get display name from item (default: str())
        
        Returns:
            List of selected items, or None if cancelled
        """
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel
        from PyQt6.QtCore import Qt
        
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(300)
        
        layout = create_vertical_layout(margins=(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD))
        dialog.setLayout(layout)
        
        # Instructions
        label = QLabel("Select items to delete (Ctrl/Cmd+Click for multiple):")
        label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 13px;")
        layout.addWidget(label)
        
        # List widget
        list_widget = QListWidget()
        list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {Colors.BG_DARK.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-radius: {border_radius(3)};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QListWidget::item:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QListWidget::item:selected {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """)
        
        # Add items
        if item_name_getter is None:
            item_name_getter = str
        
        for item in items:
            display_name = item_name_getter(item)
            list_widget.addItem(display_name)
            list_item = list_widget.item(list_widget.count() - 1)
            list_item.setData(Qt.ItemDataRole.UserRole, item)
        
        layout.addWidget(list_widget)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        button_box.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 16px;
                border-radius: {border_radius(4)};
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
            }}
        """)
        layout.addWidget(button_box)
        
        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = []
            for item in list_widget.selectedItems():
                selected_items.append(item.data(Qt.ItemDataRole.UserRole))
            return selected_items if selected_items else None
        
        return None

