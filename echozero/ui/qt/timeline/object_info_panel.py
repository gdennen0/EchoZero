"""Object info sidebar for the timeline shell.
Exists to render inspector contract text and expose object-scoped actions.
Connects timeline selection state to operator-visible controls without duplicating app logic.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract import InspectorAction, InspectorContract
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.object_actions import ObjectActionSettingsPlan
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
    event_preview_from_action as _event_preview_from_action,
    event_preview_meta_text as _event_preview_meta_text,
)
from echozero.ui.qt.timeline.object_info_panel_text import (
    contract_detail_text as _contract_detail_text,
    contract_kind_label as _contract_kind_label,
    rendered_contract_text as _rendered_contract_text,
)
from echozero.ui.qt.timeline.style import TIMELINE_STYLE

_SECTION_CONTENT_MARGIN_PX = 8
_PANEL_COLLAPSED_WIDTH = 28
_PANEL_DEFAULT_EXPANDED_WIDTH = 320


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
        self._header_layout.setSpacing(6)
        self._title = QLabel("Inspector", header)
        self._title.setObjectName(style.title_object_name)
        self._header_layout.addWidget(self._title, 1)
        self._collapse_button = QToolButton(header)
        self._collapse_button.setObjectName("objectInfoCollapseButton")
        self._collapse_button.setProperty("appearance", "subtle")
        self._collapse_button.setAutoRaise(True)
        self._collapse_button.setText(">")
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
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
        )
        selection_layout.setSpacing(6)

        selection_header = QHBoxLayout()
        selection_header.setContentsMargins(0, 0, 0, 0)
        selection_header.setSpacing(6)
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
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(style.section_spacing_px)
        self._content_splitter.addWidget(self._details_container)

        self._event_preview_card = QFrame(self)
        self._event_preview_card.setObjectName("timeline_object_info_event_preview")
        self._event_preview_card.setProperty("section", True)
        event_preview_layout = QVBoxLayout(self._event_preview_card)
        event_preview_layout.setContentsMargins(
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
        )
        event_preview_layout.setSpacing(6)
        event_preview_section = QLabel("EVENT PREVIEW", self._event_preview_card)
        event_preview_section.setObjectName("timeline_object_info_section")
        event_preview_layout.addWidget(event_preview_section)
        self._event_preview_meta = QLabel(self._event_preview_card)
        self._event_preview_meta.setObjectName("selectionSecondaryLabel")
        self._event_preview_meta.setWordWrap(True)
        event_preview_layout.addWidget(self._event_preview_meta)
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
        self._actions_scroll = QScrollArea(self._details_container)
        self._actions_scroll.setObjectName("timeline_object_info_scroll")
        self._actions_scroll.setWidgetResizable(True)
        self._actions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._actions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._actions_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        details_layout.addWidget(self._actions_scroll, 1)

        self._action_sections = QWidget(self._actions_scroll)
        self._action_sections.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        self._action_sections_layout = QVBoxLayout(self._action_sections)
        self._action_sections_layout.setContentsMargins(0, 0, 0, 0)
        self._action_sections_layout.setSpacing(max(4, style.section_spacing_px - 6))
        self._action_sections_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._action_sections_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._actions_scroll.setWidget(self._action_sections)

        self._layer_controls = QFrame(self)
        self._layer_controls.setObjectName("timeline_object_info_layer_controls")
        self._layer_controls.setProperty("section", True)
        layer_controls_layout = QVBoxLayout(self._layer_controls)
        layer_controls_layout.setContentsMargins(
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
            _SECTION_CONTENT_MARGIN_PX,
        )
        layer_controls_layout.setSpacing(6)
        playback_section = QLabel("AUDIO", self._layer_controls)
        playback_section.setObjectName("timeline_object_info_section")
        layer_controls_layout.addWidget(playback_section)

        self._layer_controls_title = QLabel("No layer selected.", self._layer_controls)
        self._layer_controls_title.setObjectName("selectionSecondaryLabel")
        self._layer_controls_title.setWordWrap(True)
        layer_controls_layout.addWidget(self._layer_controls_title)

        route_row = QGridLayout()
        route_row.setContentsMargins(0, 0, 0, 0)
        route_row.setHorizontalSpacing(6)
        route_row.setVerticalSpacing(6)
        route_label = QLabel("Output Route", self._layer_controls)
        route_label.setObjectName("selectionMetaLabel")
        self._output_bus_combo = QComboBox(self._layer_controls)
        self._output_bus_apply_btn = QPushButton("Apply Route", self._layer_controls)
        self._set_button_appearance(self._output_bus_apply_btn, "primary")
        route_row.addWidget(route_label, 0, 0)
        route_row.addWidget(self._output_bus_combo, 1, 0)
        route_row.addWidget(self._output_bus_apply_btn, 1, 1)
        layer_controls_layout.addLayout(route_row)

        mix_row = QHBoxLayout()
        mix_row.setContentsMargins(0, 0, 0, 0)
        mix_row.setSpacing(6)
        self._panel_mute_btn = QPushButton("Mute", self._layer_controls)
        self._set_button_appearance(self._panel_mute_btn, "subtle")
        self._panel_solo_btn = QPushButton("Solo", self._layer_controls)
        self._set_button_appearance(self._panel_solo_btn, "subtle")
        mix_row.addWidget(self._panel_mute_btn, 1)
        mix_row.addWidget(self._panel_solo_btn, 1)
        layer_controls_layout.addLayout(mix_row)

        gain_preset_row = QHBoxLayout()
        gain_preset_row.setContentsMargins(0, 0, 0, 0)
        gain_preset_row.setSpacing(6)
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
        gain_custom_row.setHorizontalSpacing(6)
        gain_custom_row.setVerticalSpacing(6)
        gain_label = QLabel("Gain", self._layer_controls)
        gain_label.setObjectName("gainLabel")
        self._gain_spin = QDoubleSpinBox(self._layer_controls)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply Gain", self._layer_controls)
        self._set_button_appearance(self._gain_apply_btn, "primary")
        gain_custom_row.addWidget(gain_label, 0, 0)
        gain_custom_row.addWidget(self._gain_spin, 1, 0)
        gain_custom_row.addWidget(self._gain_apply_btn, 1, 1)
        layer_controls_layout.addLayout(gain_custom_row)
        details_layout.addWidget(self._layer_controls)

        self._output_bus_apply_btn.clicked.connect(self._emit_apply_output_bus)
        self._panel_mute_btn.clicked.connect(self._emit_toggle_mute_from_panel)
        self._panel_solo_btn.clicked.connect(self._emit_toggle_solo_from_panel)
        self._gain_down_btn.clicked.connect(
            lambda _checked=False: self._emit_gain_preset("gain_down")
        )
        self._gain_unity_btn.clicked.connect(
            lambda _checked=False: self._emit_gain_preset("gain_unity")
        )
        self._gain_up_btn.clicked.connect(
            lambda _checked=False: self._emit_gain_preset("gain_up")
        )
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._action_buttons: dict[str, QPushButton] = {}
        self._settings_buttons: dict[str, QPushButton] = {}
        self._pipeline_action_plans: dict[str, ObjectActionSettingsPlan] = {}
        self._pipeline_action_rows: dict[str, QWidget] = {}
        self._action_section_expanded: dict[str, bool] = {}
        self._output_bus_actions: tuple[InspectorAction, ...] = ()
        self._set_controls_enabled(has_layer=False)
        self._event_preview_card.setVisible(False)
        self._layer_controls.setVisible(False)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes(
            [TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX, 320]
        )
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
            self._header_layout.setSpacing(6)
            self.setMinimumWidth(self._style.min_width_px)
            self.setMaximumWidth(self._style.max_width_px)
            self.resize(self._expanded_width, self.height())
        self.updateGeometry()
        self._title.setVisible(not self._collapsed)
        self._content_splitter.setVisible(not self._collapsed)
        self._collapse_button.setText("<" if self._collapsed else ">")
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
        self._output_bus_combo.setEnabled(has_layer)
        self._output_bus_apply_btn.setEnabled(has_layer)
        self._panel_mute_btn.setEnabled(has_layer)
        self._panel_solo_btn.setEnabled(has_layer)
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

    def set_context(self, presentation: TimelinePresentation, text: str) -> None:
        """Set raw sidebar text for legacy callers during transition to full contracts."""

        del presentation
        self._contract = InspectorContract(title=text, empty_state=text)
        self._kind.setText("None")
        self._selection_title.setText("Selection")
        self._set_body_text(text)
        self._clear_action_sections()
        self._pipeline_action_plans = {}
        self._output_bus_actions = ()
        self._sync_event_preview(None)
        self._layer_controls.setVisible(False)
        self._layer_controls_title.setText("No layer selected.")
        self._panel_mute_btn.setText("Mute")
        self._panel_solo_btn.setText("Solo")
        self._set_button_active(self._panel_mute_btn, False)
        self._set_button_active(self._panel_solo_btn, False)
        self._set_button_active(self._gain_down_btn, False)
        self._set_button_active(self._gain_unity_btn, False)
        self._set_button_active(self._gain_up_btn, False)
        self._output_bus_combo.clear()
        self._output_bus_combo.setVisible(False)
        self._output_bus_apply_btn.setVisible(False)

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
        has_layer = layer_id is not None
        self._set_controls_enabled(has_layer=has_layer)
        self._layer_controls.setVisible(has_layer)

        route_layer_id = layer_id
        layer_action = None
        selected_layer = None
        if route_layer_id is not None:
            layer_action = next(
                (
                    action
                    for action in self._iter_contract_actions()
                    if action.params.get("layer_id") == route_layer_id
                ),
                None,
            )
        selected_output_bus = None
        if route_layer_id is not None:
            selected_layer = next(
                (layer for layer in presentation.layers if layer.layer_id == route_layer_id),
                None,
            )
            if selected_layer is not None:
                selected_output_bus = selected_layer.output_bus
                self._layer_controls_title.setText(
                    f"Layer: {selected_layer.title} ({selected_layer.layer_id})"
                )
            else:
                self._layer_controls_title.setText(f"Layer: {route_layer_id}")
        else:
            self._layer_controls_title.setText("No layer selected.")
        self._sync_output_bus_controls(
            layer_action=layer_action,
            selected_output_bus=selected_output_bus,
        )
        self._sync_mute_solo_controls(selected_layer=selected_layer)
        self._sync_gain_controls(layer_action, selected_layer=selected_layer)

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
        preview = _event_preview_from_action(action)
        is_visible = action is not None and action.enabled and preview is not None
        self._event_preview_card.setVisible(is_visible)
        self._action_buttons.pop("preview_event_clip", None)
        if not is_visible or preview is None or action is None:
            self._event_preview_meta.setText("")
            self._event_preview_waveform.set_preview(None)
            self._event_preview_button.setEnabled(False)
            return
        self._event_preview_meta.setText(_event_preview_meta_text(preview))
        self._event_preview_waveform.set_preview(preview)
        self._event_preview_button.setText(action.label)
        self._event_preview_button.setEnabled(action.enabled)
        self._action_buttons["preview_event_clip"] = self._event_preview_button
