"""
Style Factory

Centralized style builder for all common widget types.
Every method reads from the current Colors and border_radius() at call time,
so returned stylesheets always reflect the active theme.

Usage:
    from ui.qt_gui.style_factory import StyleFactory

    button.setStyleSheet(StyleFactory.button())
    table.setStyleSheet(StyleFactory.table())
    combo.setStyleSheet(StyleFactory.combo())
"""
from ui.qt_gui.design_system import Colors, Spacing, border_radius


class StyleFactory:
    """
    Reusable stylesheet builders for common widget types.

    All methods are static and return stylesheet strings using the current
    theme values from Colors / border_radius().  Call them at the point of
    use -- never cache the return value across theme changes.
    """

    # =====================================================================
    # Buttons
    # =====================================================================

    @staticmethod
    def button(variant: str = "default") -> str:
        """
        QPushButton stylesheet.

        Variants:
            default  -- neutral background, bordered
            primary  -- accent-blue background, bold
            danger   -- neutral background, red on hover
            small    -- compact padding, smaller font
        """
        if variant == "primary":
            return f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 6px 14px;
                    color: {Colors.TEXT_PRIMARY.name()};
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.ACCENT_BLUE.darker(110).name()};
                }}
                QPushButton:disabled {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    color: {Colors.TEXT_DISABLED.name()};
                }}
            """

        if variant == "danger":
            return f"""
                QPushButton {{
                    background-color: {Colors.BG_LIGHT.name()};
                    color: {Colors.TEXT_SECONDARY.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(4)};
                    padding: 6px 12px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.HOVER.name()};
                    color: {Colors.ACCENT_RED.name()};
                    border-color: {Colors.ACCENT_RED.name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    color: {Colors.ACCENT_RED.name()};
                }}
                QPushButton:disabled {{
                    background-color: {Colors.BG_DARK.name()};
                    color: {Colors.TEXT_DISABLED.name()};
                }}
            """

        if variant == "small":
            return f"""
                QPushButton {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(3)};
                    padding: 2px 6px;
                    color: {Colors.TEXT_PRIMARY.name()};
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.HOVER.name()};
                    border-color: {Colors.HOVER.name()};
                }}
                QPushButton:pressed {{
                    background-color: {Colors.BORDER.name()};
                }}
            """

        # default
        return f"""
            QPushButton {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 6px 12px;
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QPushButton:hover {{
                background-color: {Colors.HOVER.name()};
                border-color: {Colors.HOVER.name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.SELECTED.name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_DISABLED.name()};
                border-color: {Colors.BG_LIGHT.name()};
            }}
        """

    # =====================================================================
    # Tables
    # =====================================================================

    @staticmethod
    def table() -> str:
        """QTableWidget / QTableView stylesheet."""
        return f"""
            QTableWidget, QTableView {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: 0px;
                color: {Colors.TEXT_PRIMARY.name()};
                gridline-color: {Colors.BORDER.name()};
                font-size: 12px;
            }}
            QTableWidget::item, QTableView::item {{
                padding: 4px;
            }}
            QTableWidget::item:selected, QTableView::item:selected {{
                background-color: {Colors.SELECTED.name()};
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_SECONDARY.name()};
                border: none;
                border-bottom: 1px solid {Colors.BORDER.name()};
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }}
        """

    # =====================================================================
    # Combo Boxes
    # =====================================================================

    @staticmethod
    def combo(variant: str = "default") -> str:
        """
        QComboBox stylesheet.

        Variants:
            default  -- standard combo box
            detailed -- includes custom arrow, dropdown styling, item view
        """
        if variant == "detailed":
            return f"""
                QComboBox {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(3)};
                    padding: 5px 8px;
                    padding-right: 24px;
                    color: {Colors.TEXT_PRIMARY.name()};
                    font-size: 11px;
                }}
                QComboBox:hover {{
                    background-color: {Colors.HOVER.name()};
                    border-color: {Colors.HOVER.name()};
                }}
                QComboBox:focus {{
                    border: 2px solid {Colors.ACCENT_BLUE.name()};
                    outline: none;
                }}
                QComboBox:disabled {{
                    color: {Colors.TEXT_DISABLED.name()};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 20px;
                    background-color: transparent;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 5px solid {Colors.TEXT_SECONDARY.name()};
                    width: 0px;
                    height: 0px;
                    margin-right: 4px;
                }}
                QComboBox::down-arrow:hover {{
                    border-top-color: {Colors.TEXT_PRIMARY.name()};
                }}
                QComboBox QAbstractItemView {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    selection-background-color: {Colors.ACCENT_BLUE.name()};
                    selection-color: {Colors.TEXT_PRIMARY.name()};
                    outline: none;
                }}
                QComboBox QAbstractItemView::item {{
                    padding: 6px 8px;
                    min-height: 24px;
                    border: none;
                    background-color: {Colors.BG_MEDIUM.name()};
                }}
                QComboBox QAbstractItemView::item:selected {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                }}
                QComboBox QAbstractItemView::item:hover {{
                    background-color: {Colors.HOVER.name()};
                }}
            """

        # default
        return f"""
            QComboBox {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QComboBox:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
        """

    # =====================================================================
    # Inputs
    # =====================================================================

    @staticmethod
    def input(variant: str = "default") -> str:
        """
        QLineEdit / QSpinBox / QDoubleSpinBox stylesheet.

        Variants:
            default  -- standard input
            detailed -- includes focus ring, spin-box buttons
        """
        if variant == "detailed":
            return f"""
                QLineEdit, QDoubleSpinBox, QSpinBox {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    border: 1px solid {Colors.BORDER.name()};
                    border-radius: {border_radius(3)};
                    padding: 5px 8px;
                    color: {Colors.TEXT_PRIMARY.name()};
                    font-size: 11px;
                    selection-background-color: {Colors.ACCENT_BLUE.name()};
                    selection-color: {Colors.TEXT_PRIMARY.name()};
                }}
                QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
                    border: 2px solid {Colors.ACCENT_BLUE.name()};
                    outline: none;
                }}
                QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
                    border-color: {Colors.HOVER.name()};
                }}
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
                QSpinBox::up-button, QSpinBox::down-button {{
                    border: none;
                    background-color: {Colors.BG_MEDIUM.name()};
                    width: 16px;
                }}
                QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
                QSpinBox::up-button:hover, QSpinBox::down-button:hover {{
                    background-color: {Colors.HOVER.name()};
                }}
            """

        # default
        return f"""
            QLineEdit {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QLineEdit:focus {{
                border-color: {Colors.ACCENT_BLUE.name()};
            }}
        """

    # =====================================================================
    # Group Boxes
    # =====================================================================

    @staticmethod
    def group_box() -> str:
        """QGroupBox stylesheet."""
        return f"""
            QGroupBox {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                margin-top: 8px;
                padding-top: 14px;
                font-weight: bold;
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {Colors.TEXT_SECONDARY.name()};
            }}
        """

    # =====================================================================
    # Progress Bars
    # =====================================================================

    @staticmethod
    def progress_bar(color: str = None, compact: bool = False) -> str:
        """
        QProgressBar stylesheet.

        Args:
            color: Override chunk color (hex or color name). Defaults to ACCENT_BLUE.
            compact: If True, uses smaller font and reduced padding.
        """
        chunk_color = color or Colors.ACCENT_BLUE.name()
        font_size = "11px" if compact else "12px"
        bg = Colors.BG_DARK.name() if compact else Colors.BG_MEDIUM.name()
        return f"""
            QProgressBar {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                text-align: center;
                background-color: {bg};
                color: {Colors.TEXT_PRIMARY.name()};
                font-size: {font_size};
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: {border_radius(3)};
            }}
        """

    # =====================================================================
    # Tree Widgets
    # =====================================================================

    @staticmethod
    def tree() -> str:
        """QTreeWidget / QTreeView stylesheet."""
        return f"""
            QTreeWidget, QTreeView {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QTreeWidget::item, QTreeView::item {{
                padding: 4px;
            }}
            QTreeWidget::item:hover, QTreeView::item:hover {{
                background-color: {Colors.HOVER.name()};
            }}
            QTreeWidget::item:selected, QTreeView::item:selected {{
                background-color: {Colors.SELECTED.name()};
            }}
        """

    # =====================================================================
    # Containers & Layout
    # =====================================================================

    @staticmethod
    def container() -> str:
        """Generic bordered container widget."""
        return f"""
            QWidget {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
            }}
        """

    @staticmethod
    def scroll_area() -> str:
        """QScrollArea -- transparent with no border."""
        return f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
        """

    @staticmethod
    def splitter() -> str:
        """QSplitter handle styling."""
        return f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER.name()};
                width: 1px;
            }}
            QSplitter::handle:hover {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """

    # =====================================================================
    # Tabs
    # =====================================================================

    @staticmethod
    def tabs() -> str:
        """QTabWidget + QTabBar stylesheet."""
        return f"""
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                background-color: {Colors.BG_MEDIUM.name()};
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_SECONDARY.name()};
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid {Colors.BORDER.name()};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border-bottom: 2px solid {Colors.ACCENT_BLUE.name()};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.HOVER.name()};
            }}
        """

    # =====================================================================
    # Text Editors
    # =====================================================================

    @staticmethod
    def text_edit() -> str:
        """QTextEdit stylesheet."""
        return f"""
            QTextEdit {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(3)};
                padding: 6px;
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """

    # =====================================================================
    # Menus
    # =====================================================================

    @staticmethod
    def menu() -> str:
        """QMenu popup stylesheet."""
        return f"""
            QMenu {{
                background-color: {Colors.BG_MEDIUM.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 8px;
                border-radius: {border_radius(2)};
            }}
            QMenu::item:selected {{
                background-color: {Colors.HOVER.name()};
            }}
        """

    # =====================================================================
    # Checkboxes
    # =====================================================================

    @staticmethod
    def checkbox() -> str:
        """QCheckBox stylesheet with custom indicator."""
        return f"""
            QCheckBox {{
                color: {Colors.TEXT_PRIMARY.name()};
                font-size: 11px;
                padding: 2px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(2)};
                background-color: {Colors.BG_LIGHT.name()};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.ACCENT_BLUE.name()};
                border-color: {Colors.ACCENT_BLUE.name()};
            }}
        """

    # =====================================================================
    # Toolbars / Control Bars
    # =====================================================================

    @staticmethod
    def toolbar() -> str:
        """Control bar / toolbar QFrame (background with bottom border)."""
        return f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-bottom: 1px solid {Colors.BORDER.name()};
            }}
        """

    @staticmethod
    def status_bar() -> str:
        """Status bar QFrame (background with top border)."""
        return f"""
            QFrame {{
                background-color: {Colors.BG_MEDIUM.name()};
                border-top: 1px solid {Colors.BORDER.name()};
            }}
        """

    # =====================================================================
    # Dock Tab Overrides
    # =====================================================================

    @staticmethod
    def dock_tabs() -> str:
        """Dock widget tab bar overrides (compact, 20px height)."""
        return f"""
            QTabBar {{
                background-color: {Colors.BG_DARK.name()};
                border: none;
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_SECONDARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-bottom: none;
                padding: 4px 16px;
                min-height: 20px;
                max-height: 20px;
                font-size: 12px;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border-color: {Colors.ACCENT_BLUE.name()};
                border-bottom: 2px solid {Colors.ACCENT_BLUE.name()};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.HOVER.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """
