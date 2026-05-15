"""Object info sidebar for the timeline shell.
Exists to render inspector contract text and expose object-scoped actions.
Connects timeline selection state to operator-visible controls without duplicating app logic.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract import InspectorAction, InspectorContract
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.object_actions import ObjectActionSettingsPlan
from echozero.output_routing import (
    MASTER_OUTPUT_BUS_TOKEN,
    NO_OUTPUT_BUS,
    output_bus_options,
    parse_output_bus_token,
)
from echozero.ui.FEEL import (
    TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX,
    TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX,
    TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX,
)
from echozero.ui.qt.timeline.object_info_panel_actions_mixin import (
    _ObjectInfoPanelActionsMixin,
)
from echozero.ui.qt.timeline.object_info_panel_preview import (
    EventPreviewWaveform as _EventPreviewWaveform,
    audio_event_preview_variant_label as _audio_event_preview_variant_label,
    audio_event_preview_variants as _audio_event_preview_variants,
    preview_meta_text as _preview_meta_text,
    preview_state_from_action as _preview_state_from_action,
)
from echozero.ui.qt.timeline.object_info_panel_text import (
    contract_detail_text as _contract_detail_text,
    contract_kind_label as _contract_kind_label,
    rendered_contract_text as _rendered_contract_text,
)
from echozero.ui.qt.timeline.style import TIMELINE_STYLE

_SECTION_CONTENT_MARGIN_PX = 6
_PANEL_COLLAPSED_WIDTH = 28
_PANEL_DEFAULT_EXPANDED_WIDTH = 296
_PANEL_COLLAPSED_GLYPH = "◀"
_PANEL_EXPANDED_GLYPH = "▶"


class ObjectInfoPanel(_ObjectInfoPanelActionsMixin, QFrame):
    """Sidebar panel that renders inspector facts and emits object actions."""

    action_requested = pyqtSignal(object)
    settings_requested = pyqtSignal(object)
    collapsed_changed = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        style = TIMELINE_STYLE.object_palette
        self._style = style
        self._collapsed = False
        self._expanded_width = max(
            style.min_width_px,
            min(style.max_width_px, _PANEL_DEFAULT_EXPANDED_WIDTH),
        )
        self.setObjectName(style.frame_object_name)
        self.setProperty("collapsed", False)
        self.setMinimumWidth(style.min_width_px)
        self.setMaximumWidth(style.max_width_px)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(
            style.content_padding.left,
            style.content_padding.top,
            style.content_padding.right,
            style.content_padding.bottom,
        )
        self._root_layout.setSpacing(style.section_spacing_px)

        header = QWidget(self)
        header.setObjectName("timelineObjectInfoHeader")
        self._header_layout = QHBoxLayout(header)
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.setSpacing(3)
        self._title = QLabel("Inspector", header)
        self._title.setObjectName(style.title_object_name)
        self._header_layout.addWidget(self._title, 1)
        self._collapse_button = QToolButton(header)
        self._collapse_button.setObjectName("objectInfoCollapseButton")
        self._collapse_button.setProperty("appearance", "subtle")
        self._collapse_button.setAutoRaise(True)
        self._collapse_button.setText(_PANEL_EXPANDED_GLYPH)
        self._collapse_button.setToolTip("Collapse Object Info")
        self._collapse_button.clicked.connect(self.toggle_collapsed)
        self._header_layout.addWidget(self._collapse_button)
        self._root_layout.addWidget(header)

        self._content_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._content_splitter.setObjectName("timeline_object_info_splitter")
        self._content_splitter.setChildrenCollapsible(False)
        self._content_splitter.setHandleWidth(TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX)
        self._root_layout.addWidget(self._content_splitter, 1)

        self._selection_card = QFrame(self)
        self._selection_card.setObjectName("timeline_object_info_summary")
        self._selection_card.setProperty("section", True)
        self._selection_card.setMinimumHeight(TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX)
        selection_layout = QVBoxLayout(self._selection_card)
        selection_layout.setContentsMargins(
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
        )
        selection_layout.setSpacing(3)

        selection_header = QHBoxLayout()
        selection_header.setContentsMargins(0, 0, 0, 0)
        selection_header.setSpacing(4)
        selection_section = QLabel("SELECTION", self._selection_card)
        selection_section.setObjectName("timeline_object_info_section")
        selection_header.addWidget(selection_section)
        selection_header.addStretch(1)

        self._kind = QLabel("None", self._selection_card)
        self._kind.setObjectName("timeline_object_info_kind")
        selection_header.addWidget(self._kind, 0, Qt.AlignmentFlag.AlignRight)
        selection_layout.addLayout(selection_header)

        self._selection_title = QLabel("No timeline object selected.", self._selection_card)
        self._selection_title.setObjectName("selectionPrimaryLabel")
        self._selection_title.setWordWrap(True)
        selection_layout.addWidget(self._selection_title)

        self._body = QPlainTextEdit(self._selection_card)
        self._body.setObjectName(style.body_object_name)
        self._body.setReadOnly(True)
        self._body.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._body.setMinimumHeight(TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX)
        self._body.document().setDocumentMargin(0)
        selection_layout.addWidget(self._body)
        self._content_splitter.addWidget(self._selection_card)

        self._details_container = QWidget(self._content_splitter)
        details_layout = QVBoxLayout(self._details_container)
        details_layout.setContentsMargins(0, 6, 0, 0)
        details_layout.setSpacing(max(5, style.section_spacing_px))
        self._content_splitter.addWidget(self._details_container)

        self._event_preview_card = QFrame(self)
        self._event_preview_card.setObjectName("timeline_object_info_event_preview")
        self._event_preview_card.setProperty("section", True)
        event_preview_layout = QVBoxLayout(self._event_preview_card)
        event_preview_layout.setContentsMargins(
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
        )
        event_preview_layout.setSpacing(3)
        event_preview_section = QLabel("EVENT PREVIEW", self._event_preview_card)
        event_preview_section.setObjectName("timeline_object_info_section")
        event_preview_layout.addWidget(event_preview_section)
        self._event_preview_meta = QLabel(self._event_preview_card)
        self._event_preview_meta.setObjectName("selectionSecondaryLabel")
        self._event_preview_meta.setWordWrap(True)
        event_preview_layout.addWidget(self._event_preview_meta)
        preview_variant_row = QHBoxLayout()
        preview_variant_row.setContentsMargins(0, 0, 0, 0)
        preview_variant_row.setSpacing(4)
        self._event_preview_variant_buttons: dict[str, QPushButton] = {}
        for variant in _audio_event_preview_variants():
            button = QPushButton(
                _audio_event_preview_variant_label(variant),
                self._event_preview_card,
            )
            button.setProperty("compact", True)
            self._set_button_appearance(button, "subtle")
            button.clicked.connect(
                lambda _checked=False, preview_variant=variant: self._set_event_preview_variant(
                    preview_variant
                )
            )
            preview_variant_row.addWidget(button, 1)
            self._event_preview_variant_buttons[variant] = button
        event_preview_layout.addLayout(preview_variant_row)
        self._event_preview_waveform = _EventPreviewWaveform(self._event_preview_card)
        event_preview_layout.addWidget(self._event_preview_waveform)
        self._event_preview_button = QPushButton("Play Clip", self._event_preview_card)
        self._set_button_appearance(self._event_preview_button, "primary")
        self._event_preview_button.clicked.connect(
            lambda _checked=False: self._emit_contract_action("preview_event_clip")
        )
        event_preview_layout.addWidget(self._event_preview_button)
        details_layout.addWidget(self._event_preview_card)

        self._contract = InspectorContract(title="No timeline object selected.")
        self._event_preview_variant = _audio_event_preview_variants()[0]
        self._sync_event_preview_variant_buttons()
        self._actions_scroll = QScrollArea(self._details_container)
        self._actions_scroll.setObjectName("timeline_object_info_scroll")
        self._actions_scroll.setWidgetResizable(True)
        self._actions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._actions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._actions_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        details_layout.addWidget(self._actions_scroll, 1)

        self._scroll_content = QWidget(self._actions_scroll)
        self._scroll_content_layout = QVBoxLayout(self._scroll_content)
        self._scroll_content_layout.setContentsMargins(0, 4, 0, 0)
        self._scroll_content_layout.setSpacing(max(4, style.section_spacing_px))
        self._scroll_content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll_content_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._actions_scroll.setWidget(self._scroll_content)

        self._action_sections = QWidget(self._scroll_content)
        self._action_sections.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        self._action_sections_layout = QVBoxLayout(self._action_sections)
        self._action_sections_layout.setContentsMargins(0, 0, 0, 0)
        self._action_sections_layout.setSpacing(max(4, style.section_spacing_px))
        self._action_sections_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._action_sections_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._scroll_content_layout.addWidget(self._action_sections)

        self._layer_controls = QFrame(self)
        self._layer_controls.setObjectName("timeline_object_info_layer_controls")
        self._layer_controls.setProperty("section", True)
        layer_controls_layout = QVBoxLayout(self._layer_controls)
        layer_controls_layout.setContentsMargins(
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
            max(4, _SECTION_CONTENT_MARGIN_PX - 3),
        )
        layer_controls_layout.setSpacing(3)
        playback_section = QLabel("BUS", self._layer_controls)
        playback_section.setObjectName("timeline_object_info_section")
        layer_controls_layout.addWidget(playback_section)

        self._layer_controls_title = QLabel("No layer selected.", self._layer_controls)
        self._layer_controls_title.setObjectName("selectionSecondaryLabel")
        self._layer_controls_title.setWordWrap(False)
        layer_controls_layout.addWidget(self._layer_controls_title)

        mix_row = QHBoxLayout()
        mix_row.setContentsMargins(0, 0, 0, 0)
        mix_row.setSpacing(3)
        self._bus_state = QLabel("BUS:LIVE", self._layer_controls)
        self._bus_state.setObjectName("inspectorBusState")
        mix_row.addWidget(self._bus_state, 1)
        self._panel_mute_btn = QPushButton("M", self._layer_controls)
        self._panel_mute_btn.setObjectName("inspectorMuteButton")
        self._panel_mute_btn.setProperty("statusButton", True)
        self._set_button_appearance(self._panel_mute_btn, "subtle")
        self._panel_solo_btn = QPushButton("S", self._layer_controls)
        self._panel_solo_btn.setObjectName("inspectorSoloButton")
        self._panel_solo_btn.setProperty("statusButton", True)
        self._set_button_appearance(self._panel_solo_btn, "subtle")
        mix_row.addWidget(self._panel_mute_btn, 0)
        mix_row.addWidget(self._panel_solo_btn, 0)
        layer_controls_layout.addLayout(mix_row)

        self._routing_output_channels = 2
        self._syncing_routing_controls = False

        routing_title = QLabel("OUTPUT ROUTING", self._layer_controls)
        routing_title.setProperty("sectionTitle", True)
        layer_controls_layout.addWidget(routing_title)

        routing_outputs_card = QFrame(self._layer_controls)
        routing_outputs_card.setObjectName("timeline_object_info_routing_outputs")
        routing_outputs_card.setProperty("section", True)
        routing_outputs_layout = QVBoxLayout(routing_outputs_card)
        routing_outputs_layout.setContentsMargins(6, 6, 6, 6)
        routing_outputs_layout.setSpacing(4)
        routing_outputs_header = QHBoxLayout()
        routing_outputs_header.setContentsMargins(0, 0, 0, 0)
        routing_outputs_header.setSpacing(4)
        routing_outputs_label = QLabel("Additional outputs", routing_outputs_card)
        routing_outputs_label.setObjectName("selectionSecondaryLabel")
        routing_outputs_header.addWidget(routing_outputs_label)
        routing_outputs_header.addStretch(1)
        self._route_to_master_checkbox = QCheckBox(
            "Send to master mix", routing_outputs_card
        )
        self._route_to_master_checkbox.setObjectName("inspectorRouteToMasterCheckbox")
        routing_outputs_header.addWidget(self._route_to_master_checkbox)
        routing_outputs_layout.addLayout(routing_outputs_header)

        self._routing_table = QTableWidget(0, 1, routing_outputs_card)
        self._routing_table.setObjectName("inspectorRoutingTable")
        self._routing_table.setHorizontalHeaderLabels(["Additional outputs"])
        self._routing_table.horizontalHeader().setVisible(False)
        self._routing_table.verticalHeader().setVisible(False)
        self._routing_table.horizontalHeader().setStretchLastSection(True)
        self._routing_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._routing_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._routing_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._routing_table.setShowGrid(False)
        self._routing_table.setMinimumHeight(56)
        self._routing_table.setMaximumHeight(96)
        routing_outputs_layout.addWidget(self._routing_table)
        layer_controls_layout.addWidget(routing_outputs_card)

        routing_edit_row = QHBoxLayout()
        routing_edit_row.setContentsMargins(0, 0, 0, 0)
        routing_edit_row.setSpacing(4)
        self._routing_add_btn = QPushButton("Add output", self._layer_controls)
        self._routing_remove_btn = QPushButton("Remove selected", self._layer_controls)
        self._routing_apply_btn = QPushButton("Save output routing", self._layer_controls)
        for button in (self._routing_add_btn, self._routing_remove_btn):
            self._set_button_appearance(button, "subtle")
            button.setProperty("compact", True)
        self._set_button_appearance(self._routing_apply_btn, "primary")
        self._routing_apply_btn.setProperty("compact", True)
        routing_edit_row.addWidget(self._routing_add_btn, 1)
        routing_edit_row.addWidget(self._routing_remove_btn, 1)
        layer_controls_layout.addLayout(routing_edit_row)
        layer_controls_layout.addWidget(self._routing_apply_btn)

        gain_title = QLabel("LAYER GAIN", self._layer_controls)
        gain_title.setProperty("sectionTitle", True)
        layer_controls_layout.addWidget(gain_title)
        gain_preset_row = QHBoxLayout()
        gain_preset_row.setContentsMargins(0, 0, 0, 0)
        gain_preset_row.setSpacing(4)
        self._gain_down_btn = QPushButton("-6 dB", self._layer_controls)
        self._set_button_appearance(self._gain_down_btn, "subtle")
        self._gain_unity_btn = QPushButton("0 dB", self._layer_controls)
        self._set_button_appearance(self._gain_unity_btn, "subtle")
        self._gain_up_btn = QPushButton("+6 dB", self._layer_controls)
        self._set_button_appearance(self._gain_up_btn, "subtle")
        gain_preset_row.addWidget(self._gain_down_btn, 1)
        gain_preset_row.addWidget(self._gain_unity_btn, 1)
        gain_preset_row.addWidget(self._gain_up_btn, 1)
        layer_controls_layout.addLayout(gain_preset_row)

        gain_custom_row = QGridLayout()
        gain_custom_row.setContentsMargins(0, 0, 0, 0)
        gain_custom_row.setHorizontalSpacing(4)
        gain_custom_row.setVerticalSpacing(4)
        gain_label = QLabel("Custom gain", self._layer_controls)
        gain_label.setObjectName("gainLabel")
        self._gain_spin = QDoubleSpinBox(self._layer_controls)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply gain", self._layer_controls)
        self._set_button_appearance(self._gain_apply_btn, "primary")
        gain_custom_row.addWidget(gain_label, 0, 0)
        gain_custom_row.addWidget(self._gain_spin, 1, 0)
        gain_custom_row.addWidget(self._gain_apply_btn, 1, 1)
        layer_controls_layout.addLayout(gain_custom_row)
        self._scroll_content_layout.addWidget(self._layer_controls)

        self._panel_mute_btn.clicked.connect(self._emit_toggle_mute_from_panel)
        self._panel_solo_btn.clicked.connect(self._emit_toggle_solo_from_panel)
        self._route_to_master_checkbox.toggled.connect(
            lambda _checked=False: self._on_routing_controls_changed()
        )
        self._routing_add_btn.clicked.connect(lambda _checked=False: self._add_routing_row())
        self._routing_remove_btn.clicked.connect(
            lambda _checked=False: self._remove_selected_routing_row()
        )
        self._routing_apply_btn.clicked.connect(lambda _checked=False: self._emit_apply_routing())
        self._gain_down_btn.clicked.connect(
            lambda _checked=False: self._emit_gain_preset("gain_down")
        )
        self._gain_unity_btn.clicked.connect(
            lambda _checked=False: self._emit_gain_preset("gain_unity")
        )
        self._gain_up_btn.clicked.connect(lambda _checked=False: self._emit_gain_preset("gain_up"))
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._action_buttons: dict[str, QPushButton] = {}
        self._settings_buttons: dict[str, QPushButton] = {}
        self._pipeline_action_plans: dict[str, ObjectActionSettingsPlan] = {}
        self._pipeline_action_rows: dict[str, QWidget] = {}
        self._action_section_expanded: dict[str, bool] = {}
        self._set_controls_enabled(has_layer=False)
        self._event_preview_card.setVisible(False)
        self._layer_controls.setVisible(False)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes([TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX, 320])
        self._apply_collapsed_state()

    @property
    def is_collapsed(self) -> bool:
        return self._collapsed

    @property
    def expanded_width(self) -> int:
        return self._expanded_width

    def target_width(self) -> int:
        return _PANEL_COLLAPSED_WIDTH if self._collapsed else self._expanded_width

    def toggle_collapsed(self) -> None:
        self._collapsed = not self._collapsed
        self._apply_collapsed_state()
        self.collapsed_changed.emit(self._collapsed)

    def remember_expanded_width(self, width: int) -> None:
        clamped_width = max(
            self._style.min_width_px,
            min(self._style.max_width_px, int(width)),
        )
        self._expanded_width = clamped_width
        if not self._collapsed:
            self.resize(self._expanded_width, self.height())
            self.updateGeometry()

    def _apply_collapsed_state(self) -> None:
        self._set_collapsed_style_state(self._collapsed)
        if self._collapsed:
            self._root_layout.setContentsMargins(2, 2, 2, 2)
            self._root_layout.setSpacing(0)
            self._header_layout.setSpacing(0)
            self.setMinimumWidth(_PANEL_COLLAPSED_WIDTH)
            self.setMaximumWidth(_PANEL_COLLAPSED_WIDTH)
        else:
            self._root_layout.setContentsMargins(
                self._style.content_padding.left,
                self._style.content_padding.top,
                self._style.content_padding.right,
                self._style.content_padding.bottom,
            )
            self._root_layout.setSpacing(self._style.section_spacing_px)
            self._header_layout.setSpacing(3)
            self.setMinimumWidth(self._style.min_width_px)
            self.setMaximumWidth(self._style.max_width_px)
            self.resize(self._expanded_width, self.height())
        self.updateGeometry()
        self._title.setVisible(not self._collapsed)
        self._content_splitter.setVisible(not self._collapsed)
        self._collapse_button.setText(
            _PANEL_COLLAPSED_GLYPH if self._collapsed else _PANEL_EXPANDED_GLYPH
        )
        self._collapse_button.setToolTip(
            "Expand Object Info" if self._collapsed else "Collapse Object Info"
        )

    def _set_collapsed_style_state(self, collapsed: bool) -> None:
        current = bool(self.property("collapsed"))
        if current == collapsed:
            return
        self.setProperty("collapsed", collapsed)
        for widget in (self, self._collapse_button):
            style = widget.style()
            if style is None:
                continue
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def _set_controls_enabled(self, *, has_layer: bool) -> None:
        self._panel_mute_btn.setEnabled(has_layer)
        self._panel_solo_btn.setEnabled(has_layer)
        self._route_to_master_checkbox.setEnabled(has_layer)
        self._routing_table.setEnabled(has_layer)
        self._routing_add_btn.setEnabled(has_layer)
        self._routing_remove_btn.setEnabled(has_layer and self._routing_table.rowCount() > 0)
        self._routing_apply_btn.setEnabled(has_layer)
        self._gain_down_btn.setEnabled(has_layer)
        self._gain_unity_btn.setEnabled(has_layer)
        self._gain_up_btn.setEnabled(has_layer)
        self._gain_spin.setEnabled(has_layer)
        self._gain_apply_btn.setEnabled(has_layer)

    @staticmethod
    def _set_button_appearance(button: QPushButton, appearance: str) -> None:
        button.setProperty("appearance", appearance)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def _sync_routing_controls(self, *, selected_layer: object | None) -> None:
        self._syncing_routing_controls = True
        try:
            self._routing_table.setRowCount(0)
            route_to_master, route_tokens = self._routing_state_from_layer(selected_layer)
            self._route_to_master_checkbox.setChecked(route_to_master)
            for route_token in route_tokens:
                self._add_routing_row(route_token)
        finally:
            self._syncing_routing_controls = False
        self._sync_routing_apply_state()

    def _routing_state_from_layer(self, selected_layer: object | None) -> tuple[bool, list[str]]:
        output_bus = getattr(selected_layer, "output_bus", None)
        if output_bus is None:
            return True, []
        tokens = [
            str(token or "").strip()
            for token in str(output_bus or "").split(",")
            if str(token or "").strip()
        ]
        if any(token.lower() == NO_OUTPUT_BUS for token in tokens):
            return False, []
        route_to_master = any(
            token.lower() in {MASTER_OUTPUT_BUS_TOKEN, "default"} for token in tokens
        )
        route_tokens: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lowered = token.lower()
            if lowered in {MASTER_OUTPUT_BUS_TOKEN, "default", NO_OUTPUT_BUS}:
                continue
            route = parse_output_bus_token(token)
            if route is None or route.token in seen:
                continue
            route_tokens.append(route.token)
            seen.add(route.token)
        return route_to_master, route_tokens

    def _add_routing_row(self, output_bus: str | None = None) -> None:
        row = self._routing_table.rowCount()
        self._routing_table.insertRow(row)
        combo = QComboBox(self._routing_table)
        combo.setObjectName("inspectorRoutingOutputCombo")
        route_options = list(output_bus_options(self._routing_output_channels))
        option_tokens = {route.token for route in route_options}
        selected_token = str(output_bus or "").strip()
        if selected_token and selected_token not in option_tokens:
            route = parse_output_bus_token(selected_token)
            if route is not None:
                route_options.append(route)
        for route in route_options:
            combo.addItem(route.label, route.token)
        if combo.count() == 0:
            combo.addItem("Output 1", "outputs_1_1")
        if selected_token:
            index = combo.findData(selected_token)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _index=0: self._on_routing_controls_changed())
        self._routing_table.setCellWidget(row, 0, combo)
        self._routing_table.selectRow(row)
        self._sync_routing_apply_state()

    def _remove_selected_routing_row(self) -> None:
        row = self._routing_table.currentRow()
        if row < 0 and self._routing_table.rowCount() > 0:
            row = self._routing_table.rowCount() - 1
        if row < 0:
            return
        self._routing_table.removeRow(row)
        if self._routing_table.rowCount() > 0:
            self._routing_table.selectRow(min(row, self._routing_table.rowCount() - 1))
        self._sync_routing_apply_state()
        self._emit_apply_routing_if_user_edit()

    def _routing_tokens_from_table(self) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()
        for row in range(self._routing_table.rowCount()):
            combo = self._routing_table.cellWidget(row, 0)
            if not isinstance(combo, QComboBox):
                continue
            token = str(combo.currentData() or "").strip()
            route = parse_output_bus_token(token)
            if route is None or route.token in seen:
                continue
            tokens.append(route.token)
            seen.add(route.token)
        return tokens

    def _output_bus_from_routing_controls(self) -> str | None:
        route_tokens = self._routing_tokens_from_table()
        if self._route_to_master_checkbox.isChecked():
            if not route_tokens:
                return None
            return ",".join((MASTER_OUTPUT_BUS_TOKEN, *route_tokens))
        if not route_tokens:
            return NO_OUTPUT_BUS
        return ",".join(route_tokens)

    def _on_routing_controls_changed(self) -> None:
        self._sync_routing_apply_state()
        self._emit_apply_routing_if_user_edit()

    def _emit_apply_routing_if_user_edit(self) -> None:
        if self._syncing_routing_controls:
            return
        if not self._layer_controls.isVisible():
            return
        self._emit_apply_routing()

    def _emit_apply_routing(self) -> None:
        layer_id = self._layer_id_for_controls()
        if layer_id is None:
            return
        self.action_requested.emit(
            InspectorAction(
                action_id="set_layer_output_bus_custom",
                label="Apply Routing",
                group="routing",
                params={
                    "layer_id": layer_id,
                    "output_bus": self._output_bus_from_routing_controls(),
                },
            )
        )

    def _sync_routing_apply_state(self) -> None:
        if self._syncing_routing_controls:
            return
        has_layer = self._layer_id_for_controls() is not None and self._layer_controls.isVisible()
        self._routing_remove_btn.setEnabled(has_layer and self._routing_table.rowCount() > 0)
        self._routing_apply_btn.setEnabled(has_layer)

    def set_contract(
        self, presentation: TimelinePresentation, contract: InspectorContract
    ) -> None:
        """Render a new inspector contract and refresh the enabled controls."""

        self._contract = contract
        self._selection_title.setText(contract.title)
        self._set_body_text(_contract_detail_text(contract))
        self._rebuild_action_sections()
        self._sync_event_preview(self._find_contract_action("preview_event_clip"))

        object_type = _contract_kind_label(contract)
        self._kind.setText(object_type)

        layer_id = self._layer_id_for_controls()
        selected_layer = None
        if layer_id is not None:
            selected_layer = next(
                (layer for layer in presentation.layers if layer.layer_id == layer_id),
                None,
            )
            if selected_layer is not None:
                self._layer_controls_title.setText(
                    f"{selected_layer.title}  // {selected_layer.layer_id}"
                )
            else:
                self._layer_controls_title.setText(f"LAYER // {layer_id}")
        else:
            self._layer_controls_title.setText("No layer selected.")

        self._routing_output_channels = max(
            1, min(16, int(presentation.playback_output_channels or 2))
        )
        show_audio_controls = self._has_lightweight_audio_controls(selected_layer=selected_layer)
        self._set_controls_enabled(has_layer=show_audio_controls)
        self._layer_controls.setVisible(show_audio_controls)
        self._sync_mute_solo_controls(selected_layer=selected_layer)
        self._sync_routing_controls(selected_layer=selected_layer)
        self._sync_gain_controls(selected_layer=selected_layer)

    def set_action_settings_plans(self, plans: tuple[ObjectActionSettingsPlan, ...]) -> None:
        """Attach inspector settings plans for pipeline-backed object actions."""

        self._pipeline_action_plans = {plan.action_id: plan for plan in plans}
        self._rebuild_action_sections()
        self._sync_event_preview(self._find_contract_action("preview_event_clip"))

    def contract(self) -> InspectorContract:
        """Return the currently rendered inspector contract."""

        return self._contract

    def text(self) -> str:
        """Return the currently rendered sidebar body text."""

        return _rendered_contract_text(self._contract, fallback=self._body.toPlainText())

    def _set_body_text(self, text: str) -> None:
        self._body.setPlainText(text)
        self._body.verticalScrollBar().setValue(0)

    def _sync_event_preview(self, action: InspectorAction | None) -> None:
        preview = _preview_state_from_action(action)
        is_visible = action is not None and action.enabled and preview is not None
        self._event_preview_card.setVisible(is_visible)
        self._action_buttons.pop("preview_event_clip", None)
        if not is_visible or preview is None or action is None:
            self._event_preview_meta.setText("")
            self._event_preview_waveform.set_preview(None)
            self._event_preview_button.setEnabled(False)
            self._sync_event_preview_variant_buttons(is_enabled=False)
            return
        self._event_preview_meta.setText(_preview_meta_text(preview))
        self._event_preview_waveform.set_preview(preview)
        self._event_preview_waveform.set_variant(self._event_preview_variant)
        self._event_preview_button.setText(action.label)
        self._event_preview_button.setEnabled(action.enabled)
        self._sync_event_preview_variant_buttons(is_enabled=True)
        self._action_buttons["preview_event_clip"] = self._event_preview_button

    def _set_event_preview_variant(self, variant: str) -> None:
        if variant not in self._event_preview_variant_buttons:
            return
        self._event_preview_variant = variant
        self._event_preview_waveform.set_variant(variant)
        self._sync_event_preview_variant_buttons(is_enabled=self._event_preview_card.isVisible())

    def _sync_event_preview_variant_buttons(self, *, is_enabled: bool = False) -> None:
        for variant, button in self._event_preview_variant_buttons.items():
            button.setProperty("active", variant == self._event_preview_variant)
            button.setEnabled(is_enabled)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()
