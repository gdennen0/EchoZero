from __future__ import annotations

from echozero.ui.FEEL import TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX
from echozero.ui.style.tokens import SHELL_TOKENS, ShellTokens


def build_object_info_panel_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#objectInfoPanel"
    return f"""
        {root} {{
            background: {tokens.panel_bg};
            border-left: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root} QFrame#timeline_object_info_summary[section='true'],
        {root} QFrame#timeline_object_info_event_preview[section='true'],
        {root} QFrame#timeline_object_info_layer_controls[section='true'],
        {root} QFrame#timeline_object_info_action_row[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QFrame#timeline_object_info_event_preview_waveform {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QWidget#timeline_object_info_action_buttons {{
            background: transparent;
        }}
        {root} QSplitter#timeline_object_info_splitter::handle:vertical {{
            background: {tokens.section_border};
            border-radius: {max(1, scales.panel_radius // 2)}px;
            margin: 1px 120px;
        }}
        {root} QScrollArea#timeline_object_info_scroll {{
            background: transparent;
            border: none;
        }}
        {root} QScrollArea#timeline_object_info_scroll > QWidget > QWidget {{
            background: transparent;
        }}
        {root} QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            margin: 2px 0 2px 0;
        }}
        {root} QScrollBar::handle:vertical {{
            background: {tokens.control_border};
            min-height: 24px;
            border-radius: 5px;
        }}
        {root} QScrollBar::add-line:vertical,
        {root} QScrollBar::sub-line:vertical,
        {root} QScrollBar::add-page:vertical,
        {root} QScrollBar::sub-page:vertical {{
            background: transparent;
            border: none;
            height: 0px;
        }}
        {root} QLabel#objectPaletteHeader {{
            color: {tokens.text_primary};
            font-size: 15px;
            font-weight: 700;
            padding: 0 0 2px 0;
        }}
        {root} QLabel#timeline_object_info_section {{
            color: {tokens.text_secondary};
            font-size: 11px;
            font-weight: 700;
            padding: 0 0 2px 0;
        }}
        {root} QLabel[sectionTitle='true'] {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            padding: 0;
        }}
        {root} QLabel#timeline_object_info_kind {{
            color: {tokens.text_primary};
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border_active};
            border-radius: 11px;
            padding: 2px 10px;
            font-size: 11px;
            font-weight: 600;
        }}
        {root} QLabel#selectionPrimaryLabel {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 600;
        }}
        {root} QLabel#selectionSecondaryLabel, {root} QLabel#selectionMetaLabel, {root} QLabel#gainLabel {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        {root} QPlainTextEdit#selectionSecondaryLabel {{
            background: transparent;
            border: none;
            color: {tokens.text_secondary};
            font-size: 11px;
            padding: 0;
        }}
        {root} QPushButton {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
            min-height: 28px;
            font-weight: 600;
        }}
        {root} QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[appearance='primary'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton[appearance='subtle'] {{
            background: {tokens.panel_bg};
            border-color: {tokens.panel_border};
            color: {tokens.control_text};
        }}
        {root} QPushButton[active='true'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
        }}
        {root} QPushButton:focus, {root} QDoubleSpinBox:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root} QDoubleSpinBox {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
            min-height: 26px;
        }}
        {root} QSlider::groove:horizontal {{
            background: {tokens.panel_border};
            height: {scales.slider_groove_height}px;
            border-radius: {max(1, scales.slider_groove_height // 2)}px;
        }}
        {root} QSlider::handle:horizontal {{
            background: {tokens.slider_handle};
            width: {scales.slider_handle_width}px;
            margin: {scales.slider_handle_margin}px 0;
            border-radius: {scales.slider_handle_width // 2}px;
        }}
    """


def build_timeline_editor_bar_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#timelineEditorModeBar"
    compact_padding_v = max(4, scales.field_padding_v - 2)
    return f"""
        {root} {{
            background: {tokens.panel_bg};
            border-top: {scales.border_width}px solid {tokens.panel_border};
            border-bottom: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root} QWidget#timelineEditorModeGroup,
        {root} QWidget#timelineEditorAssistGroup,
        {root} QWidget#timelineEditorShellGroup {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QLabel[timelineToolbarLabel='true'] {{
            color: {tokens.text_secondary};
            font-size: 11px;
            font-weight: 600;
            padding: {compact_padding_v}px 2px {compact_padding_v}px 0;
            min-height: {TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX}px;
        }}
        {root} QPushButton[timelineModeButton='true'],
        {root} QPushButton#timelineEditorSnapButton,
        {root} QPushButton#timelineEditorGridButton,
        {root} QPushButton#timelineEditorSettingsButton {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            padding: {compact_padding_v}px {scales.field_padding_h}px;
            min-height: {TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX}px;
            font-weight: 600;
        }}
        {root} QPushButton[timelineModeButton='true'] {{
            min-width: 72px;
        }}
        {root} QPushButton#timelineEditorGridButton {{
            min-width: 90px;
        }}
        {root} QPushButton#timelineEditorSettingsButton {{
            min-width: 88px;
        }}
        {root} QPushButton[timelineModeButton='true']:checked,
        {root} QPushButton#timelineEditorSnapButton:checked {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton[timelineModeButton='true']:disabled,
        {root} QPushButton#timelineEditorSnapButton:disabled,
        {root} QPushButton#timelineEditorGridButton:disabled,
        {root} QPushButton#timelineEditorSettingsButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[timelineModeButton='true']:focus,
        {root} QPushButton#timelineEditorSnapButton:focus,
        {root} QPushButton#timelineEditorGridButton:focus,
        {root} QPushButton#timelineEditorSettingsButton:focus {{
            border-color: {tokens.control_border_active};
        }}
    """


def build_timeline_pipeline_status_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QFrame#timelinePipelineStatus"
    return f"""
        {root} {{
            background: {tokens.panel_alt_bg};
            border-top: {scales.border_width}px solid {tokens.panel_border};
            border-bottom: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root}[tone='running'] {{
            background: {tokens.control_bg_active};
            border-top-color: {tokens.control_border_active};
            border-bottom-color: {tokens.control_border_active};
        }}
        {root}[tone='error'] {{
            background: #3a1f24;
            border-top-color: #8c3947;
            border-bottom-color: #8c3947;
        }}
        {root} QLabel#timelinePipelineStatusLabel {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 600;
        }}
    """


def build_timeline_splitter_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    return f"""
        QSplitter#timelineShellSplitter::handle,
        QSplitter#timelineMainSplitter::handle {{
            background: {tokens.window_bg};
        }}
        QSplitter#timelineShellSplitter::handle:horizontal,
        QSplitter#timelineMainSplitter::handle:horizontal {{
            width: 6px;
            margin: 0 1px;
            border-left: {scales.border_width}px solid {tokens.panel_border};
            border-right: {scales.border_width}px solid {tokens.panel_border};
        }}
    """


def build_action_settings_dialog_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QDialog#actionSettingsDialog"
    return f"""
        {root} {{
            background: {tokens.window_bg};
        }}
        {root} QFrame#actionSettingsDialogHeader[section='true'] {{
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QLabel#actionSettingsDialogEyebrow {{
            color: {tokens.text_secondary};
            font-size: 11px;
            font-weight: 700;
            padding: 0;
        }}
        {root} QLabel#actionSettingsDialogTitle {{
            color: {tokens.text_primary};
            font-size: 16px;
            font-weight: 700;
        }}
        {root} QLabel#actionSettingsDialogContext,
        {root} QLabel#actionSettingsDialogHint {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        {root} QLabel#actionSettingsCopyPreview {{
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.panel_radius}px;
            color: {tokens.text_secondary};
            padding: {scales.section_padding}px;
        }}
        {root} QGroupBox[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            margin-top: 14px;
            padding: 14px;
            color: {tokens.text_primary};
            font-weight: 600;
        }}
        {root} QGroupBox[section='true']::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }}
        {root} QLabel {{
            color: {tokens.text_primary};
        }}
        {root} QCheckBox {{
            color: {tokens.text_secondary};
            spacing: 6px;
        }}
        {root} QScrollArea,
        {root} QScrollArea > QWidget > QWidget {{
            background: transparent;
            border: none;
        }}
        {root} QPushButton,
        {root} QLineEdit,
        {root} QPlainTextEdit,
        {root} QListWidget,
        {root} QComboBox,
        {root} QSpinBox,
        {root} QDoubleSpinBox {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
        }}
        {root} QPushButton {{
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
            min-height: 28px;
            font-weight: 600;
        }}
        {root} QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[appearance='primary'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton[appearance='subtle'] {{
            background: {tokens.panel_bg};
            border-color: {tokens.panel_border};
            color: {tokens.control_text};
        }}
        {root} QLineEdit,
        {root} QPlainTextEdit,
        {root} QListWidget,
        {root} QComboBox,
        {root} QSpinBox,
        {root} QDoubleSpinBox {{
            selection-background-color: {tokens.control_bg_active};
            selection-color: {tokens.text_primary};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
            min-height: 28px;
        }}
        {root} QPushButton:focus,
        {root} QLineEdit:focus,
        {root} QPlainTextEdit:focus,
        {root} QListWidget:focus,
        {root} QComboBox:focus,
        {root} QSpinBox:focus,
        {root} QDoubleSpinBox:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root} QDialogButtonBox#actionSettingsButtons {{
            border-top: {scales.border_width}px solid {tokens.panel_border};
            padding-top: {scales.layout_gap}px;
        }}
    """


def build_song_browser_panel_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#songBrowserPanel"
    return f"""
        {root} {{
            background: {tokens.panel_bg};
            border-right: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root}[collapsed=true] {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
        }}
        {root} QLabel#songBrowserTitle {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 700;
        }}
        {root} QStackedWidget#songBrowserContent {{
            background: transparent;
        }}
        {root} QLabel#songBrowserEmptyState {{
            color: {tokens.text_secondary};
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px dashed {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            padding: 22px 16px;
        }}
        {root} QPushButton#songBrowserQuickAddButton,
        {root} QToolButton#songBrowserCollapseButton {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            font-weight: 600;
        }}
        {root} QPushButton#songBrowserQuickAddButton,
        {root} QToolButton#songBrowserCollapseButton {{
            padding: 0;
        }}
        {root} QPushButton#songBrowserQuickAddButton {{
            min-width: 28px;
            max-width: 28px;
            min-height: 28px;
        }}
        {root} QToolButton#songBrowserCollapseButton {{
            min-width: 24px;
            max-width: 24px;
            min-height: 24px;
            font-size: 12px;
            font-weight: 700;
        }}
        {root}[collapsed=true] QToolButton#songBrowserCollapseButton {{
            background: {tokens.panel_bg};
            border-color: {tokens.control_border};
            color: {tokens.text_primary};
        }}
        {root} QPushButton:focus,
        {root} QToolButton:focus,
        {root} QTreeWidget:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root} QTreeWidget#songBrowserTree {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            color: {tokens.text_primary};
            outline: none;
            padding: 6px 0;
        }}
        {root} QTreeWidget#songBrowserTree::item {{
            padding: 6px 10px;
            border-radius: {max(4, scales.button_radius - 2)}px;
            margin: 1px 6px;
        }}
        {root} QTreeWidget#songBrowserTree::item:selected {{
            background: transparent;
            color: {tokens.text_primary};
        }}
        {root} QTreeWidget#songBrowserTree::branch {{
            background: transparent;
        }}
    """


def build_timeline_scroll_area_qss(background_hex: str) -> str:
    return f"""
        QScrollArea#timelineCanvasScrollArea {{
            background: {background_hex};
            border: none;
        }}
    """


def build_echozero_app_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    from echozero.ui.qt.timeline.style import TIMELINE_STYLE

    parts = (
        build_echozero_shell_qss(tokens),
        build_foundry_surface_qss(tokens),
        build_object_info_panel_qss(tokens),
        build_timeline_editor_bar_qss(tokens),
        build_timeline_pipeline_status_qss(tokens),
        build_timeline_splitter_qss(tokens),
        build_action_settings_dialog_qss(tokens),
        build_song_browser_panel_qss(tokens),
        build_timeline_scroll_area_qss(TIMELINE_STYLE.scroll_area_background_hex),
    )
    return "\n".join(part.strip() for part in parts if part.strip())


def build_echozero_shell_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    return f"""
        QMainWindow, QDialog {{
            background: {tokens.window_bg};
        }}
        QLabel {{
            color: {tokens.text_primary};
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
        QTabWidget::pane {{
            background: {tokens.window_bg};
            color: {tokens.text_primary};
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


def build_foundry_surface_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    return f"""
        QWidget#foundryRoot, QWidget#foundryHeader, QWidget#foundryWorkspacePanel {{
            background: {tokens.window_bg};
            color: {tokens.text_primary};
        }}
        QLabel#foundryStatusLine {{
            color: {tokens.text_secondary};
            padding: 0 0 {scales.compact_gap}px 0;
        }}
    """
