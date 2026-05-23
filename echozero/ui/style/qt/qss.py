from __future__ import annotations

from pathlib import Path

from echozero.ui.FEEL import (
    TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX,
    TIMELINE_LAUNCHER_LOGO_CONTAINER_HEIGHT_PX,
    TIMELINE_LAUNCHER_SUBMENU_ARROW_RIGHT_PADDING_PX,
    TIMELINE_LAUNCHER_SUBMENU_ARROW_SIZE_PX,
)
from echozero.ui.style.tokens import SHELL_TOKENS, ShellTokens

COMBOBOX_CHEVRONS_ICON_PATH = (
    Path(__file__).resolve().parent / "assets" / "combobox_chevrons.svg"
).as_posix()


def build_object_info_panel_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#objectInfoPanel"
    compact_field_padding_h = max(4, scales.field_padding_h - 1)
    compact_field_padding_v = max(2, scales.field_padding_v - 1)
    compact_action_padding_h = max(3, compact_field_padding_h - 1)
    compact_action_padding_v = max(1, compact_field_padding_v - 1)
    compact_combo_padding_left = compact_field_padding_h + 4
    compact_combo_padding_right = max(18, compact_field_padding_h + 14)
    splitter_handle_margin_h = max(56, scales.panel_padding * 4)
    return f"""
        {root} {{
            background: {tokens.panel_bg};
            border-left: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root}[collapsed=true] {{
            background: transparent;
            border: none;
        }}
        {root} QFrame#timeline_object_info_summary[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        {root} QFrame#timeline_object_info_event_preview[section='true'],
        {root} QFrame#timeline_object_info_layer_controls[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {max(2, scales.panel_radius - 1)}px;
        }}
        {root} QFrame#timeline_object_info_action_row {{
            background: transparent;
            border: none;
            border-radius: 0;
        }}
        {root} QWidget#timeline_object_info_action_buttons {{
            background: transparent;
        }}
        {root} QSplitter#timeline_object_info_splitter::handle:vertical {{
            background: {tokens.section_border};
            border-radius: {max(1, scales.panel_radius // 2)}px;
            margin: 1px {max(40, splitter_handle_margin_h - 20)}px;
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
            width: 8px;
            margin: 1px 0 1px 0;
        }}
        {root} QScrollBar::handle:vertical {{
            background: {tokens.control_border};
            min-height: 20px;
            border-radius: 1px;
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
            font-size: 13px;
            font-weight: 700;
            padding: 0 0 2px 0;
        }}
        {root} QToolButton#objectInfoCollapseButton {{
            background: transparent;
            border: none;
            border-radius: 0;
            color: {tokens.text_primary};
            min-width: 18px;
            max-width: 18px;
            min-height: 18px;
            max-height: 18px;
            font-size: 9px;
            font-weight: 900;
            padding: 0;
        }}
        {root} QToolButton#objectInfoCollapseButton:hover {{
            color: {tokens.control_text};
        }}
        {root}[collapsed=true] QToolButton#objectInfoCollapseButton {{
            background: transparent;
            border: none;
            border-radius: 0;
            color: {tokens.text_primary};
        }}
        {root} QLabel#timeline_object_info_section {{
            color: {tokens.text_secondary};
            font-size: 11px;
            font-weight: 700;
            padding: 0 0 2px 0;
        }}
        {root} QToolButton#timeline_object_info_section_toggle {{
            background: transparent;
            border: none;
            color: {tokens.text_secondary};
            font-size: 10px;
            font-weight: 700;
            text-align: left;
            padding: 2px 0;
            min-height: 22px;
        }}
        {root} QToolButton#timeline_object_info_section_toggle:hover {{
            color: {tokens.text_primary};
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
            border: {scales.border_width}px solid {tokens.control_border};
            border-left: 2px solid #CC8844;
            border-radius: 1px;
            padding: 2px {compact_field_padding_h}px;
            font-size: 11px;
            font-weight: 600;
        }}
        {root} QLabel#selectionPrimaryLabel {{
            color: {tokens.text_primary};
            font-size: 12px;
            font-weight: 600;
        }}
        {root} QLabel#timeline_object_info_action_label {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 500;
        }}
        {root} QLabel#selectionSecondaryLabel, {root} QLabel#selectionMetaLabel, {root} QLabel#gainLabel {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        {root} QLabel#inspectorBusState {{
            color: {tokens.text_secondary};
            background: {tokens.window_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.button_radius}px;
            padding: 1px 5px;
            min-height: 18px;
            font-size: 10px;
            font-weight: 700;
        }}
        {root} QLabel#inspectorBusState[tone='mute'] {{
            color: #e5aa9e;
            border-color: {tokens.danger_border};
            background: {tokens.danger_bg};
        }}
        {root} QLabel#inspectorBusState[tone='solo'] {{
            color: {tokens.text_primary};
            border-color: {tokens.control_border_active};
            background: {tokens.control_bg_active};
        }}
        {root} QCheckBox[inspectorCheckbox='true'] {{
            color: {tokens.text_secondary};
            spacing: 5px;
            font-size: 10px;
            font-weight: 700;
        }}
        {root} QCheckBox[inspectorCheckbox='true']::indicator {{
            width: 11px;
            height: 11px;
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: 2px;
        }}
        {root} QCheckBox[inspectorCheckbox='true']::indicator:checked {{
            background: {tokens.control_bg_active};
            border-color: #CC8844;
        }}
        {root} QCheckBox[inspectorCheckbox='true']::indicator:disabled {{
            background: {tokens.control_bg_disabled};
            border-color: {tokens.panel_border};
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
            padding: 1px 5px;
            min-height: 20px;
            font-weight: 700;
            font-size: 10px;
        }}
        {root} QPushButton[compact='true'] {{
            padding: 1px 4px;
            min-height: 22px;
        }}
        {root} QPushButton[statusButton='true'] {{
            min-width: 26px;
            max-width: 26px;
            min-height: 22px;
            padding: 0;
            font-size: 10px;
            font-weight: 800;
        }}
        {root} QComboBox {{
            min-height: 20px;
            padding: {compact_field_padding_v}px {compact_combo_padding_right}px {compact_field_padding_v}px {compact_combo_padding_left}px;
        }}
        {root} QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 16px;
            border: none;
        }}
        {root} QComboBox::down-arrow {{
            image: url({COMBOBOX_CHEVRONS_ICON_PATH});
            width: 10px;
            height: 14px;
        }}
        {root} QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[appearance='primary'] {{
            background: {tokens.control_bg_active};
            border-color: #CC8844;
            color: {tokens.text_primary};
        }}
        {root} QPushButton[appearance='subtle'] {{
            background: {tokens.panel_bg};
            border-color: {tokens.control_border};
            color: {tokens.control_text};
        }}
        {root} QPushButton[appearance='danger'] {{
            background: {tokens.danger_bg};
            border-color: {tokens.danger_border};
            color: {tokens.text_primary};
        }}
        {root} QPushButton[active='true'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        {root} QPushButton#inspectorMuteButton[active='true'] {{
            background: {tokens.danger_bg};
            border-color: {tokens.danger_border};
            color: #e5aa9e;
        }}
        {root} QPushButton#inspectorSoloButton[active='true'] {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
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
            min-height: 22px;
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
            border-radius: {scales.button_radius}px;
        }}
    """


def build_timeline_editor_bar_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QWidget#timelineEditorModeBar"
    compact_padding_v = max(2, scales.field_padding_v - 4)
    return f"""
        {root} {{
            background: {tokens.panel_bg};
            border-bottom: {scales.border_width}px solid {tokens.panel_border};
        }}
        {root} QFrame#timelineEditorToolbarContainer {{
            background: #27262a;
            border: {scales.border_width}px solid #4b474f;
            border-radius: {scales.button_radius}px;
        }}
        {root} QWidget#timelineEditorModeGroup,
        {root} QWidget#timelineEditorAssistGroup,
        {root} QWidget#timelineEditorShellGroup {{
            background: transparent;
            border: none;
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
        {root} QPushButton#timelineEditorOscSettingsButton,
        {root} QPushButton#timelineEditorPipelineSettingsButton,
        {root} QPushButton#timelineEditorRegionsButton,
        {root} QPushButton#timelineEditorFitAllButton,
        {root} QPushButton#timelineEditorAddAtPlayheadButton,
        {root} QPushButton#timelineEditorFixRemoveButton,
        {root} QPushButton#timelineEditorFixSelectButton,
        {root} QPushButton#timelineEditorFixPromoteButton,
        {root} QPushButton#timelineEditorFixDemotedNavButton {{
            background: #242328;
            border: {scales.border_width}px solid #5d5962;
            border-radius: {scales.button_radius}px;
            color: #ded8d2;
            padding: 0;
            min-height: 18px;
            max-height: 18px;
            min-width: 26px;
            max-width: 26px;
            font-weight: 700;
            font-size: 10px;
        }}
        {root} QPushButton[timelineModeButton='true']:checked,
        {root} QPushButton#timelineEditorSnapButton:checked,
        {root} QPushButton#timelineEditorFixDemotedNavButton:checked {{
            background: #333138;
            border-color: #8a848f;
            color: #f3eee8;
        }}
        {root} QPushButton[timelineModeButton='true']:disabled,
        {root} QPushButton#timelineEditorSnapButton:disabled,
        {root} QPushButton#timelineEditorGridButton:disabled,
        {root} QPushButton#timelineEditorSettingsButton:disabled,
        {root} QPushButton#timelineEditorOscSettingsButton:disabled,
        {root} QPushButton#timelineEditorPipelineSettingsButton:disabled,
        {root} QPushButton#timelineEditorRegionsButton:disabled,
        {root} QPushButton#timelineEditorFitAllButton:disabled,
        {root} QPushButton#timelineEditorAddAtPlayheadButton:disabled,
        {root} QPushButton#timelineEditorFixRemoveButton:disabled,
        {root} QPushButton#timelineEditorFixSelectButton:disabled,
        {root} QPushButton#timelineEditorFixPromoteButton:disabled,
        {root} QPushButton#timelineEditorFixDemotedNavButton:disabled {{
            color: #77727a;
            border-color: #3b383f;
            background: #1c1b1f;
        }}
        {root} QPushButton[timelineModeButton='true']:focus,
        {root} QPushButton#timelineEditorSnapButton:focus,
        {root} QPushButton#timelineEditorGridButton:focus,
        {root} QPushButton#timelineEditorSettingsButton:focus,
        {root} QPushButton#timelineEditorOscSettingsButton:focus,
        {root} QPushButton#timelineEditorPipelineSettingsButton:focus,
        {root} QPushButton#timelineEditorRegionsButton:focus,
        {root} QPushButton#timelineEditorFitAllButton:focus,
        {root} QPushButton#timelineEditorAddAtPlayheadButton:focus,
        {root} QPushButton#timelineEditorFixRemoveButton:focus,
        {root} QPushButton#timelineEditorFixSelectButton:focus,
        {root} QPushButton#timelineEditorFixPromoteButton:focus,
        {root} QPushButton#timelineEditorFixDemotedNavButton:focus {{
            border-color: #a9a0ad;
        }}
        {root}[compact='true'] QPushButton[timelineModeButton='true'],
        {root}[compact='true'] QPushButton#timelineEditorGridButton,
        {root}[compact='true'] QPushButton#timelineEditorSettingsButton,
        {root}[compact='true'] QPushButton#timelineEditorOscSettingsButton,
        {root}[compact='true'] QPushButton#timelineEditorPipelineSettingsButton,
        {root}[compact='true'] QPushButton#timelineEditorRegionsButton,
        {root}[compact='true'] QPushButton#timelineEditorFitAllButton,
        {root}[compact='true'] QPushButton#timelineEditorAddAtPlayheadButton,
        {root}[compact='true'] QPushButton#timelineEditorFixSelectButton,
        {root}[compact='true'] QPushButton#timelineEditorFixDemotedNavButton {{
            min-width: 24px;
            max-width: 24px;
            padding: 0;
            font-size: 9px;
        }}
        {root}[compact='true'] QPushButton#timelineEditorSnapButton {{
            min-width: 24px;
            max-width: 24px;
            padding: 0;
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
            background: {tokens.danger_bg};
            border-top-color: {tokens.danger_border};
            border-bottom-color: {tokens.danger_border};
        }}
        {root}[tone='success'] {{
            background: {tokens.success_bg};
            border-top-color: {tokens.success_border};
            border-bottom-color: {tokens.success_border};
        }}
        {root} QLabel#timelinePipelineStatusLabel {{
            color: {tokens.text_primary};
            font-size: 11px;
            font-weight: 600;
        }}
        {root} QPushButton#timelinePipelineStatusCloseButton {{
            color: {tokens.text_primary};
            background: transparent;
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {max(2, scales.panel_radius - 1)}px;
            font-size: 10px;
            font-weight: 700;
            padding: 0;
        }}
        {root} QPushButton#timelinePipelineStatusCloseButton:hover {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
        }}
        {root}[tone='error'] QPushButton#timelinePipelineStatusCloseButton {{
            border-color: {tokens.danger_border};
        }}
        {root}[tone='success'] QPushButton#timelinePipelineStatusCloseButton {{
            border-color: {tokens.success_border};
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
            width: 4px;
            margin: 0;
            border-left: {scales.border_width}px solid {tokens.panel_border};
            border-right: {scales.border_width}px solid {tokens.panel_border};
        }}
    """


def build_action_settings_dialog_qss(tokens: ShellTokens = SHELL_TOKENS) -> str:
    scales = tokens.scales
    root = "QDialog#actionSettingsDialog"
    roots = (
        "QDialog#actionSettingsDialog, "
        "QDialog#pipelineSettingsBrowserDialog, "
        "QDialog#oscSettingsDialog"
    )
    return f"""
        {roots} {{
            background: {tokens.window_bg};
        }}
        QDialog#pipelineSettingsBrowserDialog QFrame#pipelineSettingsBrowserHeader[section='true'],
        QDialog#pipelineSettingsBrowserDialog QFrame#pipelineSettingsBrowserLeft[section='true'],
        QDialog#pipelineSettingsBrowserDialog QFrame#pipelineSettingsBrowserRight[section='true'] {{
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.panel_radius}px;
        }}
        QDialog#pipelineSettingsBrowserDialog QLabel#pipelineSettingsBrowserTitle {{
            color: {tokens.text_primary};
            font-size: 13px;
            font-weight: 800;
        }}
        QDialog#pipelineSettingsBrowserDialog QLabel#pipelineSettingsBrowserSummary,
        QDialog#pipelineSettingsBrowserDialog QLabel#pipelineSettingsBrowserContext {{
            color: {tokens.text_secondary};
            font-size: 10px;
        }}
        QDialog#pipelineSettingsBrowserDialog QLabel#pipelineSettingsBrowserLeftLabel {{
            color: {tokens.text_secondary};
            font-size: 10px;
            font-weight: 800;
            padding: 0;
        }}
        QDialog#pipelineSettingsBrowserDialog QListWidget#pipelineSettingsBrowserActionList {{
            background: {tokens.window_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.button_radius}px;
            padding: 1px;
        }}
        QDialog#pipelineSettingsBrowserDialog QListWidget#pipelineSettingsBrowserActionList::item {{
            padding: 4px 6px;
            margin: 1px;
            border: {scales.border_width}px solid transparent;
            border-radius: 1px;
        }}
        QDialog#pipelineSettingsBrowserDialog QListWidget#pipelineSettingsBrowserActionList::item:selected {{
            background: {tokens.control_bg_active};
            border-left: 2px solid #CC8844;
            color: {tokens.text_primary};
        }}
        QDialog#pipelineSettingsBrowserDialog QLabel#pipelineSettingsBrowserCopyPreview {{
            background: {tokens.window_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.text_secondary};
            padding: 3px 5px;
            font-size: 10px;
        }}
        QDialog#pipelineSettingsBrowserDialog QGroupBox[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.button_radius}px;
            margin-top: 5px;
            padding: 4px;
            color: {tokens.text_primary};
            font-weight: 700;
            font-size: 10px;
        }}
        QDialog#pipelineSettingsBrowserDialog QGroupBox[section='true']::title {{
            subcontrol-origin: margin;
            left: 5px;
            padding: 0 1px;
        }}
        QDialog#pipelineSettingsBrowserDialog QDialogButtonBox#pipelineSettingsBrowserButtons {{
            border-top: {scales.border_width}px solid {tokens.panel_border};
            padding-top: 4px;
        }}
        QDialog#pipelineSettingsBrowserDialog QSplitter::handle {{
            background: {tokens.panel_border};
            margin: 28px 1px;
        }}
        {roots} QWidget#settingsPageForm QScrollArea,
        {roots} QWidget#settingsPageForm QScrollArea > QWidget > QWidget {{
            background: transparent;
            border: none;
        }}
        {roots} QWidget#settingsPageForm QLineEdit,
        {roots} QWidget#settingsPageForm QComboBox,
        {roots} QWidget#settingsPageForm QSpinBox,
        {roots} QWidget#settingsPageForm QDoubleSpinBox {{
            min-height: 20px;
            padding: 2px 4px;
            font-size: 10px;
        }}
        {roots} QWidget#settingsPageForm QCheckBox {{
            color: {tokens.text_secondary};
            spacing: 4px;
            min-height: 18px;
            font-size: 10px;
            font-weight: 600;
        }}
        {roots} QWidget#settingsPageForm QCheckBox::indicator {{
            width: 10px;
            height: 10px;
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: 1px;
        }}
        {roots} QWidget#settingsPageForm QCheckBox::indicator:checked {{
            background: #885A2D;
            border-color: #CC8844;
        }}
        {roots} QWidget#settingsPageForm QCheckBox[settingsRole='advancedToggle'] {{
            color: {tokens.control_text};
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            padding: 1px 5px;
            max-width: 76px;
        }}
        {roots} QWidget#settingsPageForm QCheckBox[settingsRole='advancedToggle']::indicator {{
            width: 0px;
            height: 0px;
            border: none;
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
            padding: 5px 6px;
        }}
        {root} QGroupBox[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            margin-top: 6px;
            padding: 6px;
            color: {tokens.text_primary};
            font-weight: 600;
        }}
        {root} QGroupBox[section='true'][compact='true'] {{
            margin-top: 5px;
            padding: 5px;
        }}
        {root} QGroupBox[section='true']::title {{
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 1px;
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
            min-height: 20px;
            font-weight: 600;
        }}
        {root} QPushButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        {root} QPushButton[appearance='primary'] {{
            background: {tokens.control_bg_active};
            border-color: #CC8844;
            color: {tokens.text_primary};
        }}
        {root} QPushButton[appearance='subtle'] {{
            background: {tokens.panel_bg};
            border-color: {tokens.panel_border};
            color: {tokens.control_text};
        }}
        {root} QPushButton[appearance='danger'] {{
            background: {tokens.danger_bg};
            border-color: {tokens.danger_border};
            color: {tokens.text_primary};
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
            min-height: 20px;
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
            padding-top: 4px;
        }}
        QDialog#oscSettingsDialog QFrame#oscSettingsDialogHeader[section='true'] {{
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.panel_radius}px;
        }}
        QDialog#oscSettingsDialog QLabel#oscSettingsDialogEyebrow {{
            color: {tokens.text_secondary};
            font-size: 9px;
            font-weight: 800;
            letter-spacing: 0px;
        }}
        QDialog#oscSettingsDialog QLabel#oscSettingsDialogTitle {{
            color: {tokens.text_primary};
            font-size: 15px;
            font-weight: 800;
        }}
        QDialog#oscSettingsDialog QLabel#oscSettingsDialogSummary,
        QDialog#oscSettingsDialog QLabel#oscSettingsDialogStorePath {{
            color: {tokens.text_secondary};
            font-size: 10px;
        }}
        QDialog#oscSettingsDialog QLabel#oscSettingsDialogWarnings {{
            color: #ffb84d;
            font-size: 10px;
            font-weight: 700;
        }}
        QDialog#oscSettingsDialog QWidget#settingsPageForm QLabel[settingsRole='sectionTitle'] {{
            font-size: 10px;
            padding-top: 3px;
        }}
        QDialog#oscSettingsDialog QWidget#settingsPageForm QLabel[settingsRole='fieldLabel'] {{
            min-width: 0px;
        }}
        QDialog#oscSettingsDialog QGroupBox[section='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.button_radius}px;
            margin-top: 7px;
            padding: 7px;
            color: {tokens.text_primary};
            font-weight: 800;
            font-size: 10px;
        }}
        QDialog#oscSettingsDialog QGroupBox[section='true'][compact='true'] {{
            padding: 8px;
        }}
        QDialog#oscSettingsDialog QGroupBox[section='true']::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 2px;
        }}
        QDialog#oscSettingsDialog QPlainTextEdit {{
            background: {tokens.control_bg_disabled};
            border-color: {tokens.control_border};
            font-size: 10px;
        }}
        QDialog#oscSettingsDialog QSplitter#oscSettingsDialogSplitter::handle {{
            background: transparent;
            margin: 0;
        }}
        QDialog#oscSettingsDialog QLabel[statusLabel='true'] {{
            background: {tokens.window_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            border-radius: {scales.button_radius}px;
            padding: 2px 6px;
        }}
        QDialog#oscSettingsDialog QDialogButtonBox#oscSettingsDialogButtons {{
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
            background: transparent;
            border: none;
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
            padding: 12px 8px;
        }}
        {root} QPushButton#songBrowserQuickAddButton {{
            background: transparent;
            border: none;
            border-radius: 0;
            color: {tokens.text_primary};
            font-weight: 700;
            min-width: 22px;
            max-width: 22px;
            min-height: 22px;
            max-height: 22px;
            padding: 0;
        }}
        {root} QPushButton#songBrowserQuickAddButton:hover {{
            color: {tokens.control_text};
        }}
        {root} QToolButton#songBrowserCollapseButton {{
            background: transparent;
            border: none;
            border-radius: 0;
            color: {tokens.text_primary};
            min-width: 22px;
            max-width: 22px;
            min-height: 22px;
            max-height: 22px;
            font-size: 9px;
            font-weight: 900;
            padding: 0;
        }}
        {root} QToolButton#songBrowserCollapseButton:hover {{
            color: {tokens.control_text};
        }}
        {root}[collapsed=true] QToolButton#songBrowserCollapseButton {{
            background: transparent;
            border: none;
            border-radius: 0;
            color: {tokens.text_primary};
        }}
        {root} QWidget#songBrowserActiveCard {{
            background: transparent;
            border: none;
            border-radius: 0;
        }}
        {root} QWidget#songBrowserBatchBar {{
            background: transparent;
            border-top: {scales.border_width}px solid {tokens.panel_border};
            border-radius: 0;
        }}
        {root} QLabel#songBrowserActiveCaption {{
            color: {tokens.text_secondary};
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        {root} QLabel#songBrowserActiveSongTitle {{
            color: {tokens.text_primary};
            font-size: 14px;
            font-weight: 700;
        }}
        {root} QLabel#songBrowserActiveSongVersion,
        {root} QLabel#songBrowserSongsMeta,
        {root} QLabel#songBrowserBatchMeta {{
            color: {tokens.text_secondary};
            font-size: 11px;
        }}
        {root} QLabel#songBrowserSongsMeta {{
            padding: 0;
        }}
        {root} QLabel#songBrowserBatchMeta {{
            padding: 1px 0;
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
            min-height: 20px;
            padding: 0 4px;
        }}
        {root} QPushButton:focus,
        {root} QToolButton:focus,
        {root} QTreeWidget:focus,
        {root} QListWidget:focus {{
            border-color: {tokens.control_border_active};
        }}
        {root} QTreeWidget#songBrowserSongList {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: 0;
            color: {tokens.text_primary};
            outline: none;
            padding: 0;
        }}
        {root} QListWidget#songBrowserVersionList {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            color: {tokens.text_primary};
            outline: none;
            padding: 1px 0;
        }}
        {root} QTreeWidget#songBrowserSongList::item {{
            padding: 2px 0;
            border-radius: 0;
            margin: 0;
            border-bottom: {scales.border_width}px solid {tokens.section_border};
        }}
        {root} QListWidget#songBrowserVersionList::item {{
            padding: 3px 6px;
            border-radius: {scales.button_radius}px;
            margin: 0 2px;
        }}
        {root} QTreeWidget#songBrowserSongList::item:selected {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
            border-left: 2px solid #CC8844;
        }}
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
    combo_padding_left = scales.field_padding_h + 4
    combo_padding_right = max(18, scales.field_padding_h + 14)
    return f"""
        QMainWindow, QDialog {{
            background: {tokens.window_bg};
        }}
        QLabel {{
            color: {tokens.text_primary};
        }}
        QLabel[tone='ok'], QLabel[tone='success'] {{
            color: #7fd1ae;
        }}
        QLabel[tone='warn'] {{
            color: #ffb84d;
        }}
        QLabel[tone='error'] {{
            color: #c86f5f;
        }}
        QLabel[tone='unknown'], QLabel[tone='neutral'] {{
            color: {tokens.text_secondary};
        }}
        QLabel[statusLabel='true'] {{
            font-weight: 700;
        }}
        QGroupBox {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
            margin-top: 7px;
            padding: 7px;
            color: {tokens.text_primary};
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 2px;
        }}
        QFrame[section='true'], QFrame[card='true'] {{
            background: {tokens.panel_alt_bg};
            border: {scales.border_width}px solid {tokens.section_border};
            border-radius: {scales.panel_radius}px;
        }}
        QWidget#settingsPageForm QLabel[settingsRole='sectionTitle'] {{
            color: {tokens.text_primary};
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            padding: 2px 0 0 0;
        }}
        QWidget#settingsPageForm QLabel[settingsRole='sectionDescription'] {{
            color: {tokens.text_secondary};
            font-size: 10px;
            padding: 0;
        }}
        QWidget#settingsPageForm QLabel[settingsRole='fieldLabel'] {{
            color: {tokens.text_secondary};
            font-size: 10px;
            font-weight: 600;
            padding: 0;
        }}
        QWidget#settingsPageForm QLabel[settingsRole='worksheetHeader'] {{
            color: {tokens.text_secondary};
            font-size: 9px;
            font-weight: 800;
            text-transform: uppercase;
        }}
        QWidget#settingsPageForm QCheckBox[settingsRole='fieldToggle'],
        QWidget#settingsPageForm QCheckBox[settingsRole='checkboxOption'],
        QWidget#settingsPageForm QCheckBox[settingsRole='worksheetToggle'] {{
            color: {tokens.control_text};
            font-size: 10px;
            font-weight: 700;
            spacing: 4px;
            min-height: 20px;
        }}
        QPushButton, QToolButton,
        QLineEdit, QTextEdit, QPlainTextEdit,
        QListWidget, QTreeWidget, QTableWidget,
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {tokens.control_bg};
            border: {scales.border_width}px solid {tokens.control_border};
            border-radius: {scales.button_radius}px;
            color: {tokens.control_text};
            outline: none;
        }}
        QPushButton, QToolButton {{
            padding: 1px 5px;
            min-height: 20px;
            font-weight: 700;
            font-size: 10px;
        }}
        QToolButton {{
            min-width: 20px;
        }}
        QPushButton[compact='true'], QToolButton[compact='true'] {{
            padding: 1px 4px;
            min-height: 18px;
        }}
        QPushButton:checked, QToolButton:checked {{
            background: {tokens.control_bg_active};
            border-color: {tokens.control_border_active};
            color: {tokens.text_primary};
        }}
        QPushButton:disabled, QToolButton:disabled {{
            color: {tokens.control_text_disabled};
            border-color: {tokens.panel_border};
            background: {tokens.control_bg_disabled};
        }}
        QPushButton[appearance='primary'], QToolButton[appearance='primary'] {{
            background: {tokens.control_bg_active};
            border-color: #CC8844;
            color: {tokens.text_primary};
        }}
        QPushButton[appearance='subtle'], QToolButton[appearance='subtle'] {{
            background: {tokens.panel_bg};
            border-color: {tokens.control_border};
            color: {tokens.control_text};
        }}
        QPushButton[appearance='danger'], QToolButton[appearance='danger'] {{
            background: {tokens.danger_bg};
            border-color: {tokens.danger_border};
            color: {tokens.text_primary};
        }}
        QLineEdit, QTextEdit, QPlainTextEdit,
        QListWidget, QTreeWidget, QTableWidget,
        QComboBox, QSpinBox, QDoubleSpinBox {{
            selection-background-color: {tokens.control_bg_active};
            selection-color: {tokens.text_primary};
            padding: {scales.field_padding_v}px {scales.field_padding_h}px;
        }}
        QComboBox {{
            padding: {scales.field_padding_v}px {combo_padding_right}px {scales.field_padding_v}px {combo_padding_left}px;
        }}
        QComboBox::drop-down,
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background: {tokens.control_bg_disabled};
            border-left: {scales.border_width}px solid {tokens.panel_border};
            border-top-right-radius: {max(2, scales.button_radius - 1)}px;
            border-bottom-right-radius: {max(2, scales.button_radius - 1)}px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 16px;
        }}
        QComboBox::down-arrow {{
            image: url({COMBOBOX_CHEVRONS_ICON_PATH});
            width: 10px;
            height: 14px;
        }}
        QListWidget::item, QTreeWidget::item, QTableWidget::item {{
            padding: 2px 5px;
            border-radius: {max(2, scales.button_radius - 1)}px;
        }}
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
        }}
        QHeaderView::section {{
            background: {tokens.panel_alt_bg};
            color: {tokens.text_secondary};
            border: {scales.border_width}px solid {tokens.panel_border};
            padding: 2px 5px;
            font-weight: 700;
            font-size: 10px;
        }}
        QScrollArea {{
            background: {tokens.window_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
        }}
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent;
            border: none;
            margin: 0;
        }}
        QScrollBar:vertical {{ width: 8px; }}
        QScrollBar:horizontal {{ height: 8px; }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: {tokens.control_border};
            border-radius: 1px;
            min-height: 18px;
            min-width: 18px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line,
        QScrollBar::add-page, QScrollBar::sub-page {{
            background: transparent;
            border: none;
            width: 0;
            height: 0;
        }}
        QPushButton[manualPullRole='timecode'] {{
            padding: 1px 6px;
            text-align: center;
        }}
        QPushButton[manualPullRole='group'],
        QPushButton[manualPullRole='track'] {{
            padding: 2px 6px;
            text-align: left;
        }}
        QPushButton[manualPullRole='group'] {{
            font-weight: 800;
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
        QMenuBar#timelineLauncherMenuBar {{
            background: {tokens.panel_bg};
            border-bottom: {scales.border_width}px solid {tokens.panel_border};
            color: {tokens.text_primary};
            font-size: 13px;
            padding: 0 4px;
            margin: 0;
            min-height: {TIMELINE_LAUNCHER_LOGO_CONTAINER_HEIGHT_PX}px;
        }}
        QMenuBar#timelineLauncherMenuBar::item {{
            background: transparent;
            color: {tokens.text_secondary};
            padding: 3px 8px;
            margin: 0 1px;
            border-radius: {max(1, scales.button_radius - 1)}px;
        }}
        QMenuBar#timelineLauncherMenuBar::item:selected {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
        }}
        QMenuBar#timelineLauncherMenuBar::item:pressed {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
        }}
        QLabel#timelineLauncherMenuLogo {{
            background: transparent;
            padding: 0 7px;
        }}
        QMenu {{
            background: {tokens.panel_bg};
            border: {scales.border_width}px solid {tokens.panel_border};
            color: {tokens.text_primary};
        }}
        QMenu::item {{
            padding: 3px 12px 3px 8px;
            border-radius: {max(1, scales.button_radius - 1)}px;
        }}
        QMenu::item:selected {{
            background: {tokens.control_bg_active};
            color: {tokens.text_primary};
        }}
        QMenu::right-arrow {{
            width: {TIMELINE_LAUNCHER_SUBMENU_ARROW_SIZE_PX}px;
            height: {TIMELINE_LAUNCHER_SUBMENU_ARROW_SIZE_PX}px;
            right: {TIMELINE_LAUNCHER_SUBMENU_ARROW_RIGHT_PADDING_PX}px;
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
