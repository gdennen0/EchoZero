"""Object info panel action-section helpers.
Exists to keep contract-action lookup and action-row rendering out of the panel shell.
Connects inspector context actions and settings plans to the panel's action area.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.inspector_contract import InspectorAction
from echozero.ui.qt.timeline.object_info_panel_text import plan_detail_text

_ACTION_ROW_CONTENT_MARGIN_PX = 6
_ACTION_ROW_SPACING_PX = 4
_ACTION_ROW_BUTTON_SPACING_PX = 4
_ACTION_SETTINGS_BUTTON_WIDTH_PX = 24
_ACTION_RUN_BUTTON_MIN_WIDTH_PX = 48
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

    def _emit_apply_gain(self: Any) -> None:
        layer_id = self._layer_id_for_controls()
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

    def _emit_toggle_mute_from_panel(self: Any) -> None:
        action = (
            self._find_contract_action("set_layer_mute_off")
            or self._find_contract_action("set_layer_mute_on")
        )
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_toggle_solo_from_panel(self: Any) -> None:
        action = (
            self._find_contract_action("set_layer_solo_off")
            or self._find_contract_action("set_layer_solo_on")
        )
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _emit_gain_preset(self: Any, action_id: str) -> None:
        action = self._find_contract_action(action_id)
        if action is None or not action.enabled:
            return
        self.action_requested.emit(action)

    def _layer_id_for_controls(self: Any) -> object | None:
        for action_id in (
            "set_layer_mute_on",
            "set_layer_mute_off",
            "set_layer_solo_on",
            "set_layer_solo_off",
            "gain_unity",
            "gain_down",
            "gain_up",
            "delete_layer",
        ):
            action = self._find_contract_action(action_id)
            if action is None:
                continue
            layer_id = action.params.get("layer_id")
            if layer_id is not None:
                return layer_id
        for action in self._iter_contract_actions():
            layer_id = action.params.get("layer_id")
            if layer_id is not None:
                return layer_id
        return None

    def _rebuild_action_sections(self: Any) -> None:
        self._clear_action_sections()
        visible_index = 0
        for section in self._contract.context_sections:
            actions = tuple(
                action
                for action in section.actions
                if action.action_id != "preview_event_clip"
                and not action.action_id.startswith(_OUTPUT_BUS_ACTION_PREFIX)
                and (
                    (action.group or "").strip().lower() != "transfer"
                    or action.action_id in self._OBJECT_INFO_MA3_ACTION_IDS
                )
            )
            if not actions:
                continue
            section_id = section.section_id
            default_expanded = visible_index == 0
            expanded = bool(self._action_section_expanded.get(section_id, default_expanded))
            section_toggle = QToolButton(self._action_sections)
            section_toggle.setObjectName("timeline_object_info_section_toggle")
            section_toggle.setCheckable(True)
            section_toggle.setChecked(expanded)
            section_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            section_toggle.setArrowType(
                Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
            )
            section_toggle.setText(f"{section.label.upper()} ({len(actions)})")
            section_toggle.setProperty("compact", True)
            self._action_sections_layout.addWidget(section_toggle)

            section_body = QWidget(self._action_sections)
            section_body.setObjectName("timeline_object_info_section_body")
            section_body_layout = QVBoxLayout(section_body)
            section_body_layout.setContentsMargins(0, 0, 0, 0)
            section_body_layout.setSpacing(_ACTION_ROW_SPACING_PX)
            section_body.setVisible(expanded)
            self._action_sections_layout.addWidget(section_body)
            section_toggle.clicked.connect(
                lambda checked, section_key=section_id, button=section_toggle, body=section_body: self._toggle_action_section(
                    section_key, checked, button, body
                )
            )
            for action in actions:
                plan = self._pipeline_action_plans.get(action.action_id)
                if plan is not None:
                    row = self._build_pipeline_action_row(action, plan, section_body)
                    section_body_layout.addWidget(row)
                    continue
                row = self._build_simple_action_row(action, section_body)
                section_body_layout.addWidget(row)
            visible_index += 1

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

    def _toggle_action_section(
        self: Any,
        section_id: str,
        expanded: bool,
        button: QToolButton,
        body: QWidget,
    ) -> None:
        self._action_section_expanded[section_id] = expanded
        body.setVisible(expanded)
        button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

    def _build_pipeline_action_row(self: Any, action: InspectorAction, plan, parent: QWidget) -> QWidget:
        row = QFrame(parent)
        row.setObjectName("timeline_object_info_action_row")
        layout = QVBoxLayout(row)
        layout.setContentsMargins(
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
        )
        layout.setSpacing(_ACTION_ROW_SPACING_PX)

        actions_container = QWidget(row)
        actions_container.setObjectName("timeline_object_info_action_buttons")
        actions_row = QHBoxLayout(actions_container)
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(_ACTION_ROW_BUTTON_SPACING_PX)

        title = QLabel(plan.title, actions_container)
        title.setObjectName("timeline_object_info_action_label")
        title.setWordWrap(False)
        title.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        actions_row.addWidget(title, 1)

        settings_button = QPushButton(plan.settings_label, actions_container)
        settings_button.setText("...")
        settings_button.setToolTip(plan.settings_label)
        self._set_button_appearance(settings_button, "subtle")
        settings_button.setProperty("compact", True)
        settings_button.setFixedWidth(_ACTION_SETTINGS_BUTTON_WIDTH_PX)
        settings_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_settings_request(
                action_id
            )
        )
        actions_row.addWidget(settings_button)
        run_button = QPushButton(plan.run_label, actions_container)
        run_button.setText("Run")
        run_button.setToolTip(plan.run_label)
        self._set_button_appearance(run_button, "primary")
        run_button.setProperty("compact", True)
        run_button.setMinimumWidth(_ACTION_RUN_BUTTON_MIN_WIDTH_PX)
        run_button.setEnabled(action.enabled and not plan.is_running)
        run_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_pipeline_action(
                action_id
            )
        )
        actions_row.addWidget(run_button)
        layout.addWidget(actions_container)

        summary_text = plan.summary or action.label
        tooltip_lines: list[str] = []
        if summary_text and summary_text.strip().casefold() != plan.title.strip().casefold():
            tooltip_lines.append(summary_text.strip())
        detail_text = plan_detail_text(plan)
        if detail_text:
            tooltip_lines.append(detail_text)
        if tooltip_lines:
            row.setToolTip("\n".join(tooltip_lines))
            title.setToolTip(row.toolTip())

        self._action_buttons[action.action_id] = run_button
        self._settings_buttons[action.action_id] = settings_button
        self._pipeline_action_rows[action.action_id] = row
        return row

    def _build_simple_action_row(self: Any, action: InspectorAction, parent: QWidget) -> QWidget:
        row = QFrame(parent)
        row.setObjectName("timeline_object_info_action_row")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
            _ACTION_ROW_CONTENT_MARGIN_PX,
        )
        layout.setSpacing(_ACTION_ROW_BUTTON_SPACING_PX)

        label = QLabel(action.label, row)
        label.setObjectName("timeline_object_info_action_label")
        label.setWordWrap(False)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(label, 1)

        run_button = QPushButton("Run", row)
        run_button.setToolTip(action.label)
        self._set_button_appearance(run_button, "subtle")
        run_button.setProperty("compact", True)
        run_button.setMinimumWidth(_ACTION_RUN_BUTTON_MIN_WIDTH_PX)
        run_button.setEnabled(action.enabled)
        run_button.clicked.connect(
            lambda _checked=False, action_id=action.action_id: self._emit_contract_action(action_id)
        )
        layout.addWidget(run_button)

        self._action_buttons[action.action_id] = run_button
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

    def _sync_mute_solo_controls(self: Any, *, selected_layer: object | None) -> None:
        muted = bool(getattr(selected_layer, "muted", False))
        soloed = bool(getattr(selected_layer, "soloed", False))
        mute_action = (
            self._find_contract_action("set_layer_mute_off")
            or self._find_contract_action("set_layer_mute_on")
        )
        solo_action = (
            self._find_contract_action("set_layer_solo_off")
            or self._find_contract_action("set_layer_solo_on")
        )

        self._panel_mute_btn.setText("Unmute" if muted else "Mute")
        self._panel_solo_btn.setText("Unsolo" if soloed else "Solo")
        self._set_button_active(self._panel_mute_btn, muted)
        self._set_button_active(self._panel_solo_btn, soloed)
        self._panel_mute_btn.setEnabled(mute_action is not None and mute_action.enabled)
        self._panel_solo_btn.setEnabled(solo_action is not None and solo_action.enabled)

    def _sync_gain_controls(
        self: Any,
        layer_action: InspectorAction | None,
        *,
        selected_layer: object | None,
    ) -> None:
        del layer_action
        gain_down_action = self._find_contract_action("gain_down")
        gain_unity_action = self._find_contract_action("gain_unity")
        gain_up_action = self._find_contract_action("gain_up")
        gain_actions = [
            action
            for action in (gain_down_action, gain_unity_action, gain_up_action)
            if action is not None
        ]
        has_gain_controls = bool(gain_actions)
        gains_enabled = any(action.enabled for action in gain_actions)

        gain_db = float(getattr(selected_layer, "gain_db", 0.0))
        self._gain_spin.blockSignals(True)
        self._gain_spin.setValue(gain_db)
        self._gain_spin.blockSignals(False)

        self._gain_down_btn.setEnabled(gain_down_action is not None and gain_down_action.enabled)
        self._gain_unity_btn.setEnabled(gain_unity_action is not None and gain_unity_action.enabled)
        self._gain_up_btn.setEnabled(gain_up_action is not None and gain_up_action.enabled)
        self._set_button_active(self._gain_down_btn, abs(gain_db + 6.0) < 0.01)
        self._set_button_active(self._gain_unity_btn, abs(gain_db) < 0.01)
        self._set_button_active(self._gain_up_btn, abs(gain_db - 6.0) < 0.01)
        self._gain_spin.setEnabled(has_gain_controls and gains_enabled)
        self._gain_apply_btn.setEnabled(has_gain_controls and gains_enabled)

    def _sync_output_bus_controls(
        self: Any,
        *,
        layer_action: InspectorAction | None,
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
        enabled = (
            has_controls
            and layer_action is not None
            and layer_action.enabled
            and any(action.enabled for action in output_bus_actions)
        )
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

    @staticmethod
    def _set_button_active(button: QPushButton, is_active: bool) -> None:
        button.setProperty("active", bool(is_active))
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()
