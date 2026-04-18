"""Object info sidebar for the timeline shell.
Exists to render inspector contract text and expose object-scoped actions.
Connects timeline selection state to operator-visible controls without duplicating app logic.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDoubleSpinBox, QFrame, QGridLayout, QLabel, QPushButton, QVBoxLayout

from echozero.application.presentation.inspector_contract import InspectorAction, InspectorContract, render_inspector_contract_text
from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.style.qt.qss import build_object_info_panel_qss


class ObjectInfoPanel(QFrame):
    """Sidebar panel that renders object facts and emits inspector actions."""

    action_requested = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        style = TIMELINE_STYLE.object_palette
        self.setObjectName(style.frame_object_name)
        self.setStyleSheet(build_object_info_panel_qss())
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

        title = QLabel("Object Palette", self)
        title.setObjectName(style.title_object_name)
        layout.addWidget(title)

        selection_section = QLabel("SELECTION", self)
        selection_section.setObjectName("timeline_object_info_section")
        layout.addWidget(selection_section)

        self._kind = QLabel("None", self)
        self._kind.setObjectName("timeline_object_info_kind")
        layout.addWidget(self._kind, 0, Qt.AlignmentFlag.AlignLeft)

        self._body = QLabel(self)
        self._body.setObjectName(style.body_object_name)
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setMinimumHeight(72)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._body)
        self._contract = InspectorContract(title="No timeline object selected.")

        event_section = QLabel("EVENT ACTIONS", self)
        event_section.setObjectName("timeline_object_info_section")
        layout.addWidget(event_section)

        event_actions = QGridLayout()
        event_actions.setHorizontalSpacing(6)
        event_actions.setVerticalSpacing(6)
        self._seek_btn = QPushButton("Seek", self)
        self._nudge_left_btn = QPushButton("Nudge -", self)
        self._nudge_right_btn = QPushButton("Nudge +", self)
        self._duplicate_btn = QPushButton("Duplicate", self)
        event_actions.addWidget(self._seek_btn, 0, 0)
        event_actions.addWidget(self._nudge_left_btn, 0, 1)
        event_actions.addWidget(self._nudge_right_btn, 1, 0)
        event_actions.addWidget(self._duplicate_btn, 1, 1)
        layout.addLayout(event_actions)

        layer_section = QLabel("LAYER ACTIONS", self)
        layer_section.setObjectName("timeline_object_info_section")
        layout.addWidget(layer_section)

        layer_actions = QGridLayout()
        layer_actions.setHorizontalSpacing(6)
        layer_actions.setVerticalSpacing(6)
        self._push_to_ma3_btn = QPushButton("Push to MA3", self)
        self._pull_from_ma3_btn = QPushButton("Pull from MA3", self)
        self._route_audio_btn = QPushButton("Route Audio", self)
        self._gain_spin = QDoubleSpinBox(self)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply Gain", self)
        layer_actions.addWidget(self._push_to_ma3_btn, 0, 0)
        layer_actions.addWidget(self._pull_from_ma3_btn, 0, 1)
        layer_actions.addWidget(self._route_audio_btn, 1, 0, 1, 2)
        layer_actions.addWidget(self._gain_spin, 2, 0)
        layer_actions.addWidget(self._gain_apply_btn, 2, 1)
        layout.addLayout(layer_actions)
        layout.addStretch(1)

        self._seek_btn.clicked.connect(self._emit_seek_selected_event)
        self._nudge_left_btn.clicked.connect(lambda: self._emit_contract_action("nudge_left"))
        self._nudge_right_btn.clicked.connect(lambda: self._emit_contract_action("nudge_right"))
        self._duplicate_btn.clicked.connect(lambda: self._emit_contract_action("duplicate"))
        self._push_to_ma3_btn.clicked.connect(lambda: self._emit_contract_action("push_to_ma3"))
        self._pull_from_ma3_btn.clicked.connect(lambda: self._emit_contract_action("pull_from_ma3"))
        self._route_audio_btn.clicked.connect(self._emit_route_audio)
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._set_controls_enabled(has_layer=False, has_event=False, has_transfer=False)

    def _set_controls_enabled(self, *, has_layer: bool, has_event: bool, has_transfer: bool) -> None:
        self._seek_btn.setEnabled(has_event)
        self._nudge_left_btn.setEnabled(has_event)
        self._nudge_right_btn.setEnabled(has_event)
        self._duplicate_btn.setEnabled(has_event)
        self._push_to_ma3_btn.setEnabled(has_transfer)
        self._pull_from_ma3_btn.setEnabled(has_transfer)
        self._route_audio_btn.setEnabled(has_layer)
        self._gain_spin.setEnabled(has_layer)
        self._gain_apply_btn.setEnabled(has_layer)

    def set_context(self, presentation: TimelinePresentation, text: str) -> None:
        """Set raw sidebar text for legacy callers during transition to full contracts."""

        self._body.setText(text)

    def _iter_contract_actions(self):
        for section in self._contract.context_sections:
            for action in section.actions:
                yield action

    def _find_contract_action(self, action_id: str) -> InspectorAction | None:
        return next((action for action in self._iter_contract_actions() if action.action_id == action_id), None)

    def _emit_contract_action(self, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_seek_selected_event(self) -> None:
        self._emit_contract_action("seek_here")

    def _emit_route_audio(self) -> None:
        self._emit_contract_action("set_active_playback_target")

    def _emit_apply_gain(self) -> None:
        layer_action = self._find_contract_action("set_active_playback_target")
        layer_id = layer_action.params.get("layer_id") if layer_action is not None else None
        if layer_id is None:
            return
        self.action_requested.emit(
            InspectorAction(
                action_id="set_gain_custom",
                label="Set Gain",
                group="gain",
                params={"layer_id": layer_id, "gain_db": float(self._gain_spin.value())},
            )
        )

    def set_contract(self, contract: InspectorContract) -> None:
        """Render a new inspector contract and refresh the enabled controls."""

        self._contract = contract
        self._body.setText(render_inspector_contract_text(contract))

        object_type = contract.identity.object_type if contract.identity is not None else "none"
        self._kind.setText(object_type.capitalize())

        has_event = self._find_contract_action("seek_here") is not None
        has_layer = self._find_contract_action("set_active_playback_target") is not None
        push_action = self._find_contract_action("push_to_ma3")
        pull_action = self._find_contract_action("pull_from_ma3")
        has_transfer = push_action is not None and pull_action is not None
        self._set_controls_enabled(has_layer=has_layer, has_event=has_event, has_transfer=has_transfer)

        route_action = self._find_contract_action("set_active_playback_target")
        self._route_audio_btn.setText(route_action.label if route_action is not None else "Route Audio")
        self._push_to_ma3_btn.setText(push_action.label if push_action is not None else "Push to MA3")
        self._pull_from_ma3_btn.setText(pull_action.label if pull_action is not None else "Pull from MA3")

    def contract(self) -> InspectorContract:
        """Return the currently rendered inspector contract."""

        return self._contract

    def text(self) -> str:
        """Return the currently rendered sidebar body text."""

        return self._body.text()
