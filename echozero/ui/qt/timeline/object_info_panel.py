"""Object info sidebar for the timeline shell.
Exists to render inspector contract text and expose object-scoped actions.
Connects timeline selection state to operator-visible controls without duplicating app logic.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract import InspectorAction, InspectorContract
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.timeline.object_actions import ObjectActionSettingsPlan
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.style.qt.qss import build_object_info_panel_qss


class ObjectInfoPanel(QFrame):
    """Sidebar panel that renders inspector facts and emits object actions."""

    action_requested = pyqtSignal(object)
    settings_requested = pyqtSignal(object)

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

        title = QLabel("Inspector", self)
        title.setObjectName(style.title_object_name)
        layout.addWidget(title)

        selection_section = QLabel("SELECTION", self)
        selection_section.setObjectName("timeline_object_info_section")
        layout.addWidget(selection_section)

        self._kind = QLabel("None", self)
        self._kind.setObjectName("timeline_object_info_kind")
        layout.addWidget(self._kind, 0, Qt.AlignmentFlag.AlignLeft)

        self._selection_title = QLabel("No timeline object selected.", self)
        self._selection_title.setObjectName("selectionPrimaryLabel")
        self._selection_title.setWordWrap(True)
        layout.addWidget(self._selection_title)

        self._body = QLabel(self)
        self._body.setObjectName(style.body_object_name)
        self._body.setWordWrap(True)
        self._body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._body.setMinimumHeight(56)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._body)

        self._contract = InspectorContract(title="No timeline object selected.")
        self._actions_scroll = QScrollArea(self)
        self._actions_scroll.setObjectName("timeline_object_info_scroll")
        self._actions_scroll.setWidgetResizable(True)
        self._actions_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._actions_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._actions_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self._actions_scroll, 1)

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

        self._layer_controls = QWidget(self)
        layer_actions = QGridLayout(self._layer_controls)
        layer_actions.setContentsMargins(0, 0, 0, 0)
        layer_actions.setHorizontalSpacing(6)
        layer_actions.setVerticalSpacing(6)
        self._route_audio_btn = QPushButton("Route Audio", self._layer_controls)
        self._gain_spin = QDoubleSpinBox(self)
        self._gain_spin.setRange(-60.0, 12.0)
        self._gain_spin.setSingleStep(0.5)
        self._gain_spin.setSuffix(" dB")
        self._gain_apply_btn = QPushButton("Apply Gain", self._layer_controls)
        layer_actions.addWidget(self._route_audio_btn, 0, 0, 1, 2)
        layer_actions.addWidget(self._gain_spin, 1, 0)
        layer_actions.addWidget(self._gain_apply_btn, 1, 1)
        layout.addWidget(self._layer_controls)

        self._route_audio_btn.clicked.connect(self._emit_route_audio)
        self._gain_apply_btn.clicked.connect(self._emit_apply_gain)

        self._action_buttons: dict[str, QPushButton] = {}
        self._settings_buttons: dict[str, QPushButton] = {}
        self._pipeline_action_plans: dict[str, ObjectActionSettingsPlan] = {}
        self._pipeline_action_rows: dict[str, QWidget] = {}
        self._set_controls_enabled(has_layer=False)
        self._set_button_active(self._route_audio_btn, False)

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

    def set_context(self, presentation: TimelinePresentation, text: str) -> None:
        """Set raw sidebar text for legacy callers during transition to full contracts."""

        del presentation
        self._contract = InspectorContract(title=text, empty_state=text)
        self._kind.setText("None")
        self._selection_title.setText("Selection")
        self._body.setText(text)
        self._clear_action_sections()
        self._pipeline_action_plans = {}
        self._layer_controls.setVisible(False)
        self._set_button_active(self._route_audio_btn, False)

    def _iter_contract_actions(self):
        for section in self._contract.context_sections:
            for action in section.actions:
                yield action

    def _find_contract_action(self, action_id: str) -> InspectorAction | None:
        return next(
            (action for action in self._iter_contract_actions() if action.action_id == action_id),
            None,
        )

    def _emit_contract_action(self, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

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

    def set_contract(
        self, presentation: TimelinePresentation, contract: InspectorContract
    ) -> None:
        """Render a new inspector contract and refresh the enabled controls."""

        self._contract = contract
        self._selection_title.setText(contract.title)
        self._body.setText(_contract_detail_text(contract))
        self._rebuild_action_sections()

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

    def contract(self) -> InspectorContract:
        """Return the currently rendered inspector contract."""

        return self._contract

    def text(self) -> str:
        """Return the currently rendered sidebar body text."""

        return _rendered_contract_text(self._contract, fallback=self._body.text())

    def _rebuild_action_sections(self) -> None:
        self._clear_action_sections()
        for section in self._contract.context_sections:
            actions = tuple(
                action
                for action in section.actions
                if action.action_id != "set_active_playback_target"
            )
            if not actions:
                continue
            section_label = QLabel(section.label.upper(), self._action_sections)
            section_label.setObjectName("timeline_object_info_section")
            self._action_sections_layout.addWidget(section_label)

            for action in actions:
                plan = self._pipeline_action_plans.get(action.action_id)
                if plan is not None:
                    row = self._build_pipeline_action_row(action, plan)
                    self._action_sections_layout.addWidget(row)
                    continue
                button = QPushButton(action.label, self._action_sections)
                button.setEnabled(action.enabled)
                button.clicked.connect(
                    lambda _checked=False, action_id=action.action_id: self._emit_contract_action(
                        action_id
                    )
                )
                self._action_sections_layout.addWidget(button)
                self._action_buttons[action.action_id] = button

    def _clear_action_sections(self) -> None:
        self._action_buttons.clear()
        self._settings_buttons.clear()
        self._pipeline_action_rows = {}
        while self._action_sections_layout.count():
            item = self._action_sections_layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
                continue
            if child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    child_widget = child_item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def _build_pipeline_action_row(
        self, action: InspectorAction, plan: ObjectActionSettingsPlan
    ) -> QWidget:
        row = QFrame(self._action_sections)
        row.setProperty("section", True)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel(plan.title, row)
        title.setObjectName("selectionPrimaryLabel")
        title.setWordWrap(True)
        layout.addWidget(title)

        summary_text = plan.summary or action.label
        if summary_text:
            summary = QLabel(summary_text, row)
            summary.setObjectName("selectionSecondaryLabel")
            summary.setWordWrap(True)
            layout.addWidget(summary)

        detail_text = _plan_detail_text(plan)
        if detail_text:
            details = QLabel(detail_text, row)
            details.setObjectName("selectionMetaLabel")
            details.setWordWrap(True)
            layout.addWidget(details)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        settings_button = QPushButton(plan.settings_label, row)
        settings_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_settings_request(
                action_id
            )
        )
        actions_row.addWidget(settings_button)
        run_button = QPushButton(plan.run_label, row)
        run_button.setEnabled(action.enabled)
        run_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_pipeline_action(
                action_id
            )
        )
        actions_row.addWidget(run_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self._action_buttons[action.action_id] = run_button
        self._settings_buttons[action.action_id] = settings_button
        self._pipeline_action_rows[action.action_id] = row
        return row

    def _emit_pipeline_action(self, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_settings_request(self, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None:
            return
        self.settings_requested.emit(replace(action, params=dict(action.params)))

    def _sync_gain_controls(self, route_action: InspectorAction | None) -> None:
        gain_actions = [
            action
            for action in self._iter_contract_actions()
            if action.group == "gain" and action.action_id != "set_gain_custom"
        ]
        has_gain_controls = bool(gain_actions)
        self._gain_spin.setEnabled(
            has_gain_controls and route_action is not None and route_action.enabled
        )
        self._gain_apply_btn.setEnabled(
            has_gain_controls and route_action is not None and route_action.enabled
        )


def _contract_kind_label(contract: InspectorContract) -> str:
    if contract.identity is not None:
        return contract.identity.object_type.capitalize()
    if contract.sections:
        return "Timeline"
    return "None"


def _contract_detail_text(contract: InspectorContract) -> str:
    if not contract.sections:
        return contract.empty_state
    lines: list[str] = []
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)


def _plan_detail_text(plan: ObjectActionSettingsPlan) -> str:
    parts: list[str] = []
    overrides = _plan_override_preview(plan)
    if overrides:
        parts.append(f"Saved: {overrides}")
    elif plan.editable_fields or plan.advanced_fields:
        parts.append("Saved: defaults")
    if plan.locked_bindings:
        locked = ", ".join(f"{key}: {value}" for key, value in plan.locked_bindings)
        parts.append(f"Locked: {locked}")
    elif plan.rerun_hint:
        parts.append(plan.rerun_hint)
    return "\n".join(parts)


def _plan_override_preview(plan: ObjectActionSettingsPlan) -> str:
    highlighted: list[str] = []
    for field in (*plan.editable_fields, *plan.advanced_fields):
        if field.value == field.default_value:
            continue
        highlighted.append(f"{field.label} {field.value}")
        if len(highlighted) == 2:
            break
    return ", ".join(highlighted)


def _rendered_contract_text(contract: InspectorContract, *, fallback: str) -> str:
    if contract.identity is None and not contract.sections:
        return contract.empty_state or fallback
    lines: list[str] = [contract.title]
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)
