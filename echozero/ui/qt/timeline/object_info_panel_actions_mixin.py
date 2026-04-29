"""Object info panel action-section helpers.
Exists to keep contract-action lookup and action-row rendering out of the panel shell.
Connects inspector context actions and settings plans to the panel's action area.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.ui.qt.timeline.object_info_panel_text import plan_detail_text

_ACTION_ROW_CONTENT_MARGIN_PX = 8
_OUTPUT_BUS_ACTION_PREFIX = "set_layer_output_bus_"


class _ObjectInfoPanelActionsMixin:
    _OBJECT_INFO_MA3_ACTION_IDS = frozenset({"route_layer_to_ma3_track"})

    def _iter_contract_actions(self: Any):
        for section in self._contract.context_sections:
            for action in section.actions:
                yield action

    def _find_contract_action(self: Any, action_id: str) -> InspectorAction | None:
        return next(
            (action for action in self._iter_contract_actions() if action.action_id == action_id),
            None,
        )

    def _emit_contract_action(self: Any, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_route_audio(self: Any) -> None:
        self._emit_contract_action("set_active_playback_target")

    def _emit_apply_gain(self: Any) -> None:
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

    def _emit_apply_output_bus(self: Any) -> None:
        if not self._output_bus_actions:
            return
        index = int(self._output_bus_combo.currentIndex())
        if index < 0 or index >= len(self._output_bus_actions):
            return
        action = self._output_bus_actions[index]
        if not action.enabled:
            return
        self.action_requested.emit(action)

    def _rebuild_action_sections(self: Any) -> None:
        self._clear_action_sections()
        for section in self._contract.context_sections:
            actions = tuple(
                action
                for action in section.actions
                if action.action_id not in {"set_active_playback_target", "preview_event_clip"}
                and not action.action_id.startswith(_OUTPUT_BUS_ACTION_PREFIX)
                and (
                    (action.group or "").strip().lower() != "transfer"
                    or action.action_id in self._OBJECT_INFO_MA3_ACTION_IDS
                )
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
                self._set_button_appearance(button, "subtle")
                button.setEnabled(action.enabled)
                button.clicked.connect(
                    lambda _checked=False, action_id=action.action_id: self._emit_contract_action(
                        action_id
                    )
                )
                self._action_sections_layout.addWidget(button)
                self._action_buttons[action.action_id] = button

    def _clear_action_sections(self: Any) -> None:
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

    def _build_pipeline_action_row(self: Any, action: InspectorAction, plan) -> QWidget:
        row = QFrame(self._action_sections)
        row.setObjectName("timeline_object_info_action_row")
        row.setProperty("section", True)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
        )
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

        detail_text = plan_detail_text(plan)
        if detail_text:
            details = QLabel(detail_text, row)
            details.setObjectName("selectionMetaLabel")
            details.setWordWrap(True)
            layout.addWidget(details)

        actions_container = QWidget(row)
        actions_container.setObjectName("timeline_object_info_action_buttons")
        actions_row = QHBoxLayout(actions_container)
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        settings_button = QPushButton(plan.settings_label, row)
        self._set_button_appearance(settings_button, "subtle")
        settings_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_settings_request(
                action_id
            )
        )
        actions_row.addWidget(settings_button)
        run_button = QPushButton(plan.run_label, row)
        self._set_button_appearance(run_button, "primary")
        run_button.setEnabled(action.enabled and not plan.is_running)
        run_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_pipeline_action(
                action_id
            )
        )
        actions_row.addWidget(run_button)
        actions_row.addStretch(1)
        layout.addWidget(actions_container)

        self._action_buttons[action.action_id] = run_button
        self._settings_buttons[action.action_id] = settings_button
        self._pipeline_action_rows[action.action_id] = row
        return row

    def _emit_pipeline_action(self: Any, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_settings_request(self: Any, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None:
            return
        self.settings_requested.emit(replace(action, params=dict(action.params)))

    def _sync_gain_controls(self: Any, route_action: InspectorAction | None) -> None:
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

    def _sync_output_bus_controls(
        self: Any,
        *,
        route_action: InspectorAction | None,
        selected_output_bus: str | None,
    ) -> None:
        output_bus_actions = tuple(
            action
            for action in self._iter_contract_actions()
            if action.action_id.startswith(_OUTPUT_BUS_ACTION_PREFIX)
        )
        self._output_bus_actions = output_bus_actions
        self._output_bus_combo.blockSignals(True)
        self._output_bus_combo.clear()
        selected_bus = self._normalize_output_bus(selected_output_bus)
        selected_index = 0
        for index, action in enumerate(output_bus_actions):
            output_bus = self._normalize_output_bus(action.params.get("output_bus"))
            self._output_bus_combo.addItem(self._output_bus_option_label(output_bus))
            if output_bus == selected_bus:
                selected_index = index
        if self._output_bus_combo.count() > 0:
            self._output_bus_combo.setCurrentIndex(selected_index)
        self._output_bus_combo.blockSignals(False)
        has_controls = bool(output_bus_actions)
        enabled = has_controls and route_action is not None and route_action.enabled
        self._output_bus_combo.setVisible(has_controls)
        self._output_bus_apply_btn.setVisible(has_controls)
        self._output_bus_combo.setEnabled(enabled)
        self._output_bus_apply_btn.setEnabled(enabled)

    @staticmethod
    def _normalize_output_bus(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        return normalized or None

    @staticmethod
    def _output_bus_option_label(output_bus: str | None) -> str:
        if output_bus is None:
            return "Default Output (1/2)"
        parts = output_bus.strip().lower().split("_")
        if len(parts) == 3 and parts[0] == "outputs" and parts[1].isdigit() and parts[2].isdigit():
            start_channel = int(parts[1])
            end_channel = int(parts[2])
            if start_channel == end_channel:
                return f"Output {start_channel}"
            return f"Outputs {start_channel}/{end_channel}"
        return output_bus
