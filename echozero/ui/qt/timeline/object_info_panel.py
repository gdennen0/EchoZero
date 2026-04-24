"""Object info sidebar for the timeline shell.
Exists to render inspector contract text and expose object-scoped actions.
Connects timeline selection state to operator-visible controls without duplicating app logic.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
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


class ObjectInfoPanel(_ObjectInfoPanelActionsMixin, QFrame):
    """Sidebar panel that renders inspector facts and emits object actions."""

    action_requested = pyqtSignal(object)
    settings_requested = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        style = TIMELINE_STYLE.object_palette
        self.setObjectName(style.frame_object_name)
        self.setMinimumWidth(style.min_width_px)
        self.setMaximumWidth(style.max_width_px)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            style.content_padding.left,
            style.content_padding.top,
            style.content_padding.right,
            style.content_padding.bottom,
        )
        layout.setSpacing(style.section_spacing_px)

        title = QLabel("Inspector", self)
        title.setObjectName(style.title_object_name)
        layout.addWidget(title)

        self._content_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._content_splitter.setObjectName("timeline_object_info_splitter")
        self._content_splitter.setChildrenCollapsible(False)
        self._content_splitter.setHandleWidth(TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX)
        layout.addWidget(self._content_splitter, 1)

        self._selection_card = QFrame(self)
        self._selection_card.setObjectName("timeline_object_info_summary")
        self._selection_card.setProperty("section", True)
        self._selection_card.setMinimumHeight(TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX)
        selection_layout = QVBoxLayout(self._selection_card)
        selection_layout.setContentsMargins(10, 10, 10, 10)
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
        event_preview_layout.setContentsMargins(10, 10, 10, 10)
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
        self._action_sections_layout.setSpacing(style.section_spacing_px)
        self._action_sections_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._action_sections_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self._actions_scroll.setWidget(self._action_sections)

        self._layer_controls = QFrame(self)
        self._layer_controls.setObjectName("timeline_object_info_layer_controls")
        self._layer_controls.setProperty("section", True)
        layer_controls_layout = QVBoxLayout(self._layer_controls)
        layer_controls_layout.setContentsMargins(10, 10, 10, 10)
        layer_controls_layout.setSpacing(6)
        playback_section = QLabel("PLAYBACK", self._layer_controls)
        playback_section.setObjectName("timeline_object_info_section")
        layer_controls_layout.addWidget(playback_section)

        layer_actions = QGridLayout()
        layer_actions.setContentsMargins(0, 0, 0, 0)
        layer_actions.setHorizontalSpacing(6)
        layer_actions.setVerticalSpacing(6)
        self._route_audio_btn = QPushButton("Route Audio", self._layer_controls)
        self._set_button_appearance(self._route_audio_btn, "subtle")
        self._gain_spin = QDoubleSpinBox(self._layer_controls)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply Gain", self._layer_controls)
        self._set_button_appearance(self._gain_apply_btn, "primary")
        layer_actions.addWidget(self._route_audio_btn, 0, 0, 1, 2)
        layer_actions.addWidget(self._gain_spin, 1, 0)
        layer_actions.addWidget(self._gain_apply_btn, 1, 1)
        layer_controls_layout.addLayout(layer_actions)
        details_layout.addWidget(self._layer_controls)

        self._route_audio_btn.clicked.connect(self._emit_route_audio)
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._action_buttons: dict[str, QPushButton] = {}
        self._settings_buttons: dict[str, QPushButton] = {}
        self._pipeline_action_plans: dict[str, ObjectActionSettingsPlan] = {}
        self._pipeline_action_rows: dict[str, QWidget] = {}
        self._set_controls_enabled(has_layer=False)
        self._event_preview_card.setVisible(False)
        self._layer_controls.setVisible(False)
        self._set_button_active(self._route_audio_btn, False)
        self._content_splitter.setStretchFactor(0, 0)
        self._content_splitter.setStretchFactor(1, 1)
        self._content_splitter.setSizes(
            [TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX, 320]
        )

    def _set_controls_enabled(self, *, has_layer: bool) -> None:
        self._route_audio_btn.setEnabled(has_layer)
        self._gain_spin.setEnabled(has_layer)
        self._gain_apply_btn.setEnabled(has_layer)

    @staticmethod
    def _set_button_active(button: QPushButton, active: bool) -> None:
        button.setProperty("active", active)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

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
        self._sync_event_preview(None)
        self._layer_controls.setVisible(False)
        self._set_button_active(self._route_audio_btn, False)

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

        has_layer = self._find_contract_action("set_active_playback_target") is not None
        self._set_controls_enabled(has_layer=has_layer)
        self._layer_controls.setVisible(has_layer)

        route_action = self._find_contract_action("set_active_playback_target")
        self._route_audio_btn.setText(
            route_action.label if route_action is not None else "Route Audio"
        )
        route_layer_id = route_action.params.get("layer_id") if route_action is not None else None
        self._set_button_active(
            self._route_audio_btn,
            route_layer_id is not None and route_layer_id == presentation.active_playback_layer_id,
        )
        if route_action is not None:
            self._route_audio_btn.setEnabled(route_action.enabled)
        self._sync_gain_controls(route_action)

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
