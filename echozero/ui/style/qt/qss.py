from __future__ import annotations

from echozero.ui.FEEL import TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX
from echozero.ui.style.tokens import SHELL_TOKENS, ShellTokens


def build_object_info_panel_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#objectInfoPanel"
    compact_field_padding_h = max(6, scales.field_padding_h - 2)
    compact_field_padding_v = max(4, scales.field_padding_v - 1)
    splitter_handle_margin_h = max(56, scales.panel_padding * 4)
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
            margin: 1px {splitter_handle_margin_h}px;
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
            padding: 2px {compact_field_padding_h}px;
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
            padding: {compact_field_padding_v}px {compact_field_padding_h}px;
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
            padding: {compact_field_padding_v}px {compact_field_padding_h}px;
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
    compact_padding_v = max(2, scales.field_padding_v - 4)
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
            font-size: 10px;
            font-weight: 600;
            padding: {compact_padding_v}px 2px {compact_padding_v}px 0;
            min-height: {TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX}px;
        }}
        {root} QPushButton[timelineModeButton='true'],
        {root} QPushButton#timelineEditorSnapButton,
        {root} QPushButton#timelineEditorGridButton,
        {root} QPushButton#timelineEditorSettingsButton,
        {root} QPushButton#timelineEditorRegionsButton,
        {root} QPushButton#timelineEditorFixRemoveButton,
        {root} QPushButton#timelineEditorFixSelectButton,
        {root} QPushButton#timelineEditorFixPromoteButton,
        {root} QPushButton#timelineEditorFixDemotedNavButton {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            padding: {compact_padding_v}px {scales.field_padding_h}px;
            min-height: {TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX}px;
            font-weight: 600;
            font-size: 10px;
        }}
        {root} QPushButton[timelineModeButton='true'] {{
            min-width: 72px;
        }}
        {root} QPushButton#timelineEditorGridButton {{
            min-width: 96px;
        }}
        {root} QPushButton#timelineEditorSettingsButton,
        {root} QPushButton#timelineEditorRegionsButton {{
            min-width: 98px;
        }}
        {root} QPushButton#timelineEditorFixRemoveButton,
        {root} QPushButton#timelineEditorFixPromoteButton {{
            min-width: 30px;
            padding-left: {max(4, scales.field_padding_h - 3)}px;
            padding-right: {max(4, scales.field_padding_h - 3)}px;
        }}
        {root} QPushButton#timelineEditorFixSelectButton {{
            min-width: 34px;
        }}
        {root} QPushButton[timelineModeButton='true']:checked,
        {root} QPushButton#timelineEditorSnapButton:checked,
        {root} QPushButton#timelineEditorFixDemotedNavButton:checked {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton[timelineModeButton='true']:disabled,
        {root} QPushButton#timelineEditorSnapButton:disabled,
        {root} QPushButton#timelineEditorGridButton:disabled,
        {root} QPushButton#timelineEditorSettingsButton:disabled,
        {root} QPushButton#timelineEditorRegionsButton:disabled,
        {root} QPushButton#timelineEditorFixRemoveButton:disabled,
        {root} QPushButton#timelineEditorFixSelectButton:disabled,
        {root} QPushButton#timelineEditorFixPromoteButton:disabled,
        {root} QPushButton#timelineEditorFixDemotedNavButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[timelineModeButton='true']:focus,
        {root} QPushButton#timelineEditorSnapButton:focus,
        {root} QPushButton#timelineEditorGridButton:focus,
        {root} QPushButton#timelineEditorSettingsButton:focus,
        {root} QPushButton#timelineEditorRegionsButton:focus,
        {root} QPushButton#timelineEditorFixRemoveButton:focus,
        {root} QPushButton#timelineEditorFixSelectButton:focus,
        {root} QPushButton#timelineEditorFixPromoteButton:focus,
        {root} QPushButton#timelineEditorFixDemotedNavButton:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root}[compact='true'] QPushButton[timelineModeButton='true'],
        {root}[compact='true'] QPushButton#timelineEditorGridButton,
        {root}[compact='true'] QPushButton#timelineEditorSettingsButton,
        {root}[compact='true'] QPushButton#timelineEditorRegionsButton,
        {root}[compact='true'] QPushButton#timelineEditorFixSelectButton,
        {root}[compact='true'] QPushButton#timelineEditorFixDemotedNavButton {{
            min-width: 28px;
            padding-left: {max(4, scales.field_padding_h - 4)}px;
            padding-right: {max(4, scales.field_padding_h - 4)}px;
            font-size: 9px;
        }}
        {root}[compact='true'] QPushButton#timelineEditorSnapButton {{
            min-width: 32px;
            font-size: 9px;
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
        {root} QLabel#actionSettingsDialogTitle {{
            color: {tokens.text_primary};
            font-size: 13px;
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
            padding: 8px 10px;
        }}
        {root} QGroupBox[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            margin-top: 10px;
            padding: 10px;
            color: {tokens.text_primary};
            font-weight: 600;
        }}
        {root} QGroupBox[section='true'][compact='true'] {{
            margin-top: 8px;
            padding: 8px;
        }}
        {root} QGroupBox[section='true']::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 2px;
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
            min-height: 24px;
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
            min-height: 24px;
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
            padding-top: 6px;
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
            border-radius: {scales.panel_radius}px;
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
        {root} QPushButton#songBrowserQuickAddButton {{
            min-width: 28px;
            max-width: 28px;
            min-height: 28px;
            padding: 0;
        }}
        {root} QToolButton#songBrowserCollapseButton {{
            min-width: 24px;
            max-width: 24px;
            min-height: 24px;
            font-size: 12px;
            font-weight: 700;
            padding: 0;
        }}
        {root}[collapsed=true] QToolButton#songBrowserCollapseButton {{
            background: {tokens.panel_bg};
            border-color: {tokens.control_border};
            color: {tokens.text_primary};
        }}
        {root} QWidget#songBrowserActiveCard,
        {root} QWidget#songBrowserBatchBar {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QLabel#songBrowserActiveCaption {{
            color: {tokens.text_secondary};
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        {root} QLabel#songBrowserActiveSongTitle {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 700;
        }}
        {root} QLabel#songBrowserActiveSongVersion,
        {root} QLabel#songBrowserSongsMeta,
        {root} QLabel#songBrowserBatchMeta {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        {root} QLabel#songBrowserSectionTitle {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        {root} QPushButton#songBrowserAddVersionButton,
        {root} QPushButton#songBrowserBatchSelectAll,
        {root} QPushButton#songBrowserBatchClear,
        {root} QPushButton#songBrowserBatchMoveTop,
        {root} QPushButton#songBrowserBatchMoveBottom,
        {root} QPushButton#songBrowserBatchDelete {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            font-size: 11px;
            font-weight: 600;
            min-height: 24px;
            padding: 0 8px;
        }}
        {root} QPushButton#songBrowserBatchDelete {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton:focus,
        {root} QToolButton:focus,
        {root} QTreeWidget:focus,
        {root} QListWidget:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root} QTreeWidget#songBrowserSongList,
        {root} QListWidget#songBrowserVersionList {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            color: {tokens.text_primary};
            outline: none;
            padding: 4px 0;
        }}
        {root} QTreeWidget#songBrowserSongList::item,
        {root} QListWidget#songBrowserVersionList::item {{
            padding: 6px 10px;
            border-radius: {max(4, scales.button_radius - 2)}px;
            margin: 1px 6px;
        }}
        {root} QTreeWidget#songBrowserSongList::item:selected,
        {root} QListWidget#songBrowserVersionList::item:selected {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
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
