"""Reusable neutral settings-page form for EchoZero Qt surfaces.
Exists so app and feature settings can share one renderer instead of widget-local forms.
Connects typed settings pages to editable Qt widgets with consistent advanced-field handling.
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
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

from echozero.application.settings import (
    SettingsField,
    SettingsFieldSurface,
    SettingsFieldWidget,
    SettingsPage,
    SettingsSection,
)


class SettingsPageForm(QWidget):
    """Embeddable renderer/editor for one neutral settings page."""

    field_value_changed = pyqtSignal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(self._scroll)

        self._content = QWidget(self._scroll)
        self._scroll.setWidget(self._content)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(6)

        self._inputs: dict[str, QWidget] = {}
        self._page: SettingsPage | None = None
        self._suspend_field_events = False

    def set_page(self, page: SettingsPage, *, empty_message: str = "No settings available.") -> None:
        """Render one reusable settings page into the form."""

        self._suspend_field_events = True
        try:
            self._page = page
            self._clear_sections()
            self._inputs = {}
            rendered_any_field = False
            for section in page.sections:
                if not section.fields:
                    continue
                rendered_any_field = True
                self._add_section(section)
            if not rendered_any_field:
                empty_state = QLabel(empty_message, self._content)
                empty_state.setWordWrap(True)
                self._content_layout.addWidget(empty_state)
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

    def _add_section(self, section: SettingsSection) -> None:
        title = QLabel(section.title, self._content)
        self._content_layout.addWidget(title)
        if section.description:
            description = QLabel(section.description, self._content)
            description.setWordWrap(True)
            self._content_layout.addWidget(description)

        primary_fields = tuple(
            field for field in section.fields if field.surface is SettingsFieldSurface.PRIMARY
        )
        advanced_fields = tuple(
            field for field in section.fields if field.surface is SettingsFieldSurface.ADVANCED
        )

        self._add_form_fields(primary_fields)
        if advanced_fields:
            self._add_advanced_section(advanced_fields)

    def _add_form_fields(self, fields: tuple[SettingsField, ...]) -> None:
        if not fields:
            return
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        for field in fields:
            self._add_form_field_row(form, field)
        self._content_layout.addLayout(form)

    def _add_advanced_section(self, fields: tuple[SettingsField, ...]) -> None:
        show_advanced = any(field.is_dirty for field in fields)
        toggle = QCheckBox("Show advanced settings", self._content)
        toggle.setChecked(show_advanced)
        self._content_layout.addWidget(toggle)

        container = QWidget(self._content)
        container.setVisible(show_advanced)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(6)

        title = QLabel("Advanced", container)
        container_layout.addWidget(title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(4)
        for field in fields:
            self._add_form_field_row(form, field)
        container_layout.addLayout(form)
        self._content_layout.addWidget(container)

        toggle.toggled.connect(container.setVisible)

    def _clear_sections(self) -> None:
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            self._dispose_layout_item(item)

    def _dispose_layout_item(self, item) -> None:
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
            return

        child_layout = item.layout()
        if child_layout is None:
            return
        while child_layout.count():
            child_item = child_layout.takeAt(0)
            self._dispose_layout_item(child_item)
        child_layout.deleteLater()

    def _build_field_widget(self, field: SettingsField) -> QWidget:
        if field.widget is SettingsFieldWidget.DROPDOWN:
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

        if field.widget is SettingsFieldWidget.TOGGLE:
            checkbox = QCheckBox(self._content)
            checkbox.setChecked(bool(field.value))
            checkbox.toggled.connect(
                lambda _checked, key=field.key, widget=checkbox: self._emit_field_value_changed(key, widget)
            )
            return checkbox

        if field.widget is SettingsFieldWidget.NUMBER:
            if isinstance(field.value, int) and not isinstance(field.value, bool):
                spin = QSpinBox(self._content)
                spin.setMinimum(int(field.min_value) if field.min_value is not None else -999999)
                spin.setMaximum(int(field.max_value) if field.max_value is not None else 999999)
                spin.setSingleStep(int(field.step or 1))
                spin.setValue(int(field.value))
                if field.units:
                    spin.setSuffix(f" {field.units}")
                spin.setKeyboardTracking(False)
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
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(
                lambda _value, key=field.key, widget=spin: self._emit_field_value_changed(key, widget)
            )
            return spin

        line_edit = QLineEdit(self._content)
        line_edit.setText("" if field.value is None else str(field.value))
        if field.placeholder:
            line_edit.setPlaceholderText(field.placeholder)
        line_edit.textChanged.connect(
            lambda _text, key=field.key, widget=line_edit: self._emit_field_value_changed(key, widget)
        )
        return line_edit

    def _add_form_field_row(self, form: QFormLayout, field: SettingsField) -> None:
        widget = self._build_field_widget(field)
        widget.setEnabled(field.enabled)
        tooltip = self._field_tooltip_text(field)
        label = QLabel(field.label, self._content)
        if tooltip:
            widget.setToolTip(tooltip)
            widget.setStatusTip(tooltip)
            label.setToolTip(tooltip)
            label.setStatusTip(tooltip)
        self._inputs[field.key] = widget
        form.addRow(label, widget)

    @staticmethod
    def _field_tooltip_text(field: SettingsField) -> str:
        lines: list[str] = []
        description = str(field.description or "").strip()
        if description:
            lines.append(description)
        else:
            threshold_hint = SettingsPageForm._threshold_tooltip_hint(field)
            if threshold_hint:
                lines.append(threshold_hint)

        range_line = SettingsPageForm._field_range_line(field)
        if range_line:
            lines.append(range_line)
        default_line = SettingsPageForm._field_default_line(field)
        if default_line:
            lines.append(default_line)
        return "\n".join(lines)

    @staticmethod
    def _threshold_tooltip_hint(field: SettingsField) -> str:
        key_and_label = f"{field.key} {field.label}".lower()
        if "threshold" not in key_and_label:
            return ""
        if "positive_threshold" in key_and_label or "classification" in key_and_label:
            return (
                "Classification confidence cutoff. Lower values accept more candidates;"
                " higher values accept fewer."
            )
        if "onset_threshold" in key_and_label or "detection" in key_and_label:
            return (
                "Onset detection sensitivity. Lower values detect more candidate events;"
                " higher values detect fewer."
            )
        return "Threshold strictness. Lower values detect more events; higher values detect fewer."

    @staticmethod
    def _field_range_line(field: SettingsField) -> str:
        if field.min_value is None and field.max_value is None:
            return ""
        if field.min_value is None:
            value = SettingsPageForm._format_tooltip_scalar(field.max_value)
            return f"Max: {value}{SettingsPageForm._tooltip_unit_suffix(field.units)}"
        if field.max_value is None:
            value = SettingsPageForm._format_tooltip_scalar(field.min_value)
            return f"Min: {value}{SettingsPageForm._tooltip_unit_suffix(field.units)}"
        min_value = SettingsPageForm._format_tooltip_scalar(field.min_value)
        max_value = SettingsPageForm._format_tooltip_scalar(field.max_value)
        return f"Range: {min_value} to {max_value}{SettingsPageForm._tooltip_unit_suffix(field.units)}"

    @staticmethod
    def _field_default_line(field: SettingsField) -> str:
        if field.default_value is None:
            return ""
        formatted = SettingsPageForm._format_tooltip_value(field.default_value, options=field.options)
        if not formatted:
            return ""
        return f"Default: {formatted}"

    @staticmethod
    def _tooltip_unit_suffix(units: str) -> str:
        unit_text = str(units or "").strip()
        return f" {unit_text}" if unit_text else ""

    @staticmethod
    def _format_tooltip_scalar(value: object) -> str:
        if isinstance(value, float):
            return format(value, "g")
        return str(value)

    @staticmethod
    def _format_tooltip_value(value: object, *, options: Sequence[object]) -> str:
        for option in options:
            option_value = getattr(option, "value", None)
            if option_value == value:
                label = str(getattr(option, "label", "")).strip()
                return label or str(value)
        if isinstance(value, bool):
            return "On" if value else "Off"
        if isinstance(value, float):
            return format(value, "g")
        return str(value)

    def _emit_field_value_changed(self, key: str, widget: QWidget) -> None:
        if self._suspend_field_events:
            return
        self.field_value_changed.emit(key, self._read_widget_value(widget))

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
