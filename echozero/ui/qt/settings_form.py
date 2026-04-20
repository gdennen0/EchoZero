"""Reusable settings form for action and pipeline configuration surfaces.
Exists so the same settings plan can render inside bounded settings editors.
Connects neutral settings contracts to Qt widgets without duplicating host-specific UI.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.application.timeline.object_actions import ObjectActionSettingField, ObjectActionSettingsPlan


class ActionSettingsForm(QWidget):
    """Embeddable renderer/editor for one object action settings plan."""

    field_value_changed = pyqtSignal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(self._scroll)

        self._content = QWidget(self._scroll)
        self._scroll.setWidget(self._content)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(10)

        self._inputs: dict[str, QWidget] = {}
        self._plan: ObjectActionSettingsPlan | None = None
        self._suspend_field_events = False
        self._show_advanced = False

    def set_plan(self, plan: ObjectActionSettingsPlan) -> None:
        """Render one settings plan into the form."""
        self._suspend_field_events = True
        try:
            self._plan = plan
            self._clear_sections()
            self._inputs = {}
            if not plan.editable_fields and not plan.advanced_fields:
                empty_state = QLabel("No editable settings for this action.", self._content)
                empty_state.setWordWrap(True)
                self._content_layout.addWidget(empty_state)
                return
            self._add_section("Stage Settings", plan.editable_fields)
            if plan.advanced_fields:
                if any(field.is_dirty for field in plan.advanced_fields):
                    self._show_advanced = True
                self._add_advanced_section(plan.advanced_fields)
        finally:
            self._suspend_field_events = False

    def values(self) -> dict[str, object]:
        """Return the current edited values keyed by settings field."""

        return {key: self._read_widget_value(widget) for key, widget in self._inputs.items()}

    def set_values(self, values: dict[str, object]) -> None:
        """Apply one value set onto the currently rendered widgets."""
        self._suspend_field_events = True
        try:
            for key, value in values.items():
                widget = self._inputs.get(key)
                if widget is None:
                    continue
                self._write_widget_value(widget, value)
        finally:
            self._suspend_field_events = False

    def _add_section(self, label: str, fields: tuple[ObjectActionSettingField, ...]) -> None:
        if not fields:
            return
        title = QLabel(label, self._content)
        self._content_layout.addWidget(title)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        for field in fields:
            widget = self._build_field_widget(field)
            widget.setEnabled(field.enabled)
            self._inputs[field.key] = widget
            form.addRow(field.label, widget)
        self._content_layout.addLayout(form)

    def _add_advanced_section(self, fields: tuple[ObjectActionSettingField, ...]) -> None:
        toggle = QCheckBox("Show advanced settings", self._content)
        toggle.setChecked(self._show_advanced)
        self._content_layout.addWidget(toggle)

        container = QWidget(self._content)
        container.setVisible(self._show_advanced)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)

        title = QLabel("Advanced", container)
        container_layout.addWidget(title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        for field in fields:
            widget = self._build_field_widget(field)
            widget.setEnabled(field.enabled)
            self._inputs[field.key] = widget
            form.addRow(field.label, widget)
        container_layout.addLayout(form)
        self._content_layout.addWidget(container)

        toggle.toggled.connect(lambda checked: self._on_advanced_toggled(container, checked))

    def _clear_sections(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
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
        self._show_advanced = False

    def _build_field_widget(self, field: ObjectActionSettingField) -> QWidget:
        if field.widget == "dropdown":
            combo = QComboBox(self._content)
            for option in field.options:
                combo.addItem(option.label, option.value)
            index = combo.findData(field.value)
            if index >= 0:
                combo.setCurrentIndex(index)
            combo.currentIndexChanged.connect(
                lambda _index, key=field.key, widget=combo: self._emit_field_value_changed(key, widget)
            )
            return combo
        if field.widget == "toggle":
            checkbox = QCheckBox(self._content)
            checkbox.setChecked(bool(field.value))
            checkbox.toggled.connect(
                lambda _checked, key=field.key, widget=checkbox: self._emit_field_value_changed(key, widget)
            )
            return checkbox
        if field.widget == "number":
            if isinstance(field.value, int) and not isinstance(field.value, bool):
                spin = QSpinBox(self._content)
                spin.setMinimum(int(field.min_value) if field.min_value is not None else -999999)
                spin.setMaximum(int(field.max_value) if field.max_value is not None else 999999)
                spin.setSingleStep(int(field.step or 1))
                spin.setValue(int(field.value))
                if field.units:
                    spin.setSuffix(f" {field.units}")
                spin.valueChanged.connect(
                    lambda _value, key=field.key, widget=spin: self._emit_field_value_changed(key, widget)
                )
                return spin
            spin = QDoubleSpinBox(self._content)
            spin.setDecimals(3)
            spin.setRange(
                field.min_value if field.min_value is not None else -999999.0,
                field.max_value if field.max_value is not None else 999999.0,
            )
            spin.setSingleStep(field.step or 0.1)
            spin.setValue(float(field.value if field.value is not None else 0.0))
            if field.units:
                spin.setSuffix(f" {field.units}")
            spin.valueChanged.connect(
                lambda _value, key=field.key, widget=spin: self._emit_field_value_changed(key, widget)
            )
            return spin
        line_edit = QLineEdit(self._content)
        line_edit.setText("" if field.value is None else str(field.value))
        if field.placeholder:
            line_edit.setPlaceholderText(field.placeholder)
        if field.description:
            line_edit.setToolTip(field.description)
        line_edit.textChanged.connect(
            lambda _text, key=field.key, widget=line_edit: self._emit_field_value_changed(key, widget)
        )
        return line_edit

    def _emit_field_value_changed(self, key: str, widget: QWidget) -> None:
        if self._suspend_field_events:
            return
        self.field_value_changed.emit(key, self._read_widget_value(widget))

    def _on_advanced_toggled(self, container: QWidget, checked: bool) -> None:
        self._show_advanced = checked
        container.setVisible(checked)

    @staticmethod
    def _read_widget_value(widget: QWidget) -> object:
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    @staticmethod
    def _write_widget_value(widget: QWidget, value: object) -> None:
        if isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
            return
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
            return
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)) and value is not None:
            widget.setValue(value)
            return
        if isinstance(widget, QLineEdit):
            widget.setText("" if value is None else str(value))
