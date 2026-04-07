from __future__ import annotations

from echozero.ui.style.tokens import SHELL_TOKENS, ShellTokens


def build_object_info_panel_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    return f"""
        QWidget#objectInfoPanel {{
            background: {tokens.panel_bg};
            border-left: {scales.border_width}px solid {tokens.panel_border};
        }}
        QLabel#objectPaletteHeader {{
            color: {tokens.text_primary};
            font-size: 15px;
            font-weight: 700;
            padding: 0 0 2px 0;
        }}
        QFrame[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        QLabel[sectionTitle='true'] {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            padding: 0;
        }}
        QLabel#selectionPrimaryLabel {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 600;
        }}
        QLabel#selectionSecondaryLabel, QLabel#selectionMetaLabel, QLabel#gainLabel {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        QPushButton {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
        }}
        QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        QPushButton[active='true'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
        }}
        QSlider::groove:horizontal {{
            background: {tokens.panel_border};
            height: {scales.slider_groove_height}px;
            border-radius: {max(1, scales.slider_groove_height // 2)}px;
        }}
        QSlider::handle:horizontal {{
            background: {tokens.slider_handle};
            width: {scales.slider_handle_width}px;
            margin: {scales.slider_handle_margin}px 0;
            border-radius: {scales.slider_handle_width // 2}px;
        }}
    """


def build_foundry_shell_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    return f"""
        QMainWindow {{
            background: {tokens.window_bg};
        }}
        QWidget#foundryRoot, QWidget#foundryHeader, QWidget#foundryWorkspacePanel, QTabWidget::pane {{
            background: {tokens.window_bg};
            color: {tokens.text_primary};
        }}
        QLabel {{
            color: {tokens.text_primary};
        }}
        QLabel#foundryStatusLine {{
            color: {tokens.text_secondary};
            padding: 0 0 {scales.compact_gap}px 0;
        }}
        QGroupBox {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            margin-top: 12px;
            padding: 12px;
            color: {tokens.text_primary};
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }}
        QPushButton, QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
        }}
        QPushButton {{
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
        }}
        QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        QLineEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {{
            selection-background-color: {tokens.control_bg_active};
            selection-color: {tokens.text_primary};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
        }}
        QListWidget::item:selected {{
            background: {tokens.control_bg_active};
        }}
        QTabBar::tab {{
            background: {tokens.panel_alt_bg};
            color: {tokens.text_secondary};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
            border: {scales.border_width}px solid {tokens.section_border};
            border-bottom: none;
            border-top-left-radius: {scales.button_radius}px;
            border-top-right-radius: {scales.button_radius}px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {tokens.control_bg};
            color: {tokens.text_primary};
        }}
    """
