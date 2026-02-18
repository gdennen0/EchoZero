"""
Settings binding helpers for block panel widgets.

These helpers enforce commit-on-complete behavior so we do not save partial
user input while a field is still being edited.
"""

from __future__ import annotations

from typing import Callable, Any, Optional

from PyQt6.QtWidgets import QLineEdit, QAbstractSpinBox, QComboBox, QCheckBox


class SettingsBinding:
    """Small helper to wire widgets to settings writes consistently."""

    @staticmethod
    def bind_line_edit(
        widget: QLineEdit,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        transform: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """Commit a line edit value when editing is finished."""

        def _commit() -> None:
            text = widget.text()
            value = transform(text) if transform else text
            if value != getter():
                setter(value)

        widget.editingFinished.connect(_commit)

    @staticmethod
    def bind_spinbox(
        widget: QAbstractSpinBox,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
    ) -> None:
        """Commit a spinbox value when editing is finished."""

        def _commit() -> None:
            value = widget.value()  # type: ignore[attr-defined]
            if value != getter():
                setter(value)

        widget.editingFinished.connect(_commit)

    @staticmethod
    def bind_combo(
        widget: QComboBox,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
        *,
        use_data: bool = True,
    ) -> None:
        """Commit combo changes when user explicitly selects a value."""

        def _commit(index: int) -> None:
            value = widget.itemData(index) if use_data else widget.currentText()
            if value != getter():
                setter(value)

        widget.currentIndexChanged.connect(_commit)

    @staticmethod
    def bind_checkbox(
        widget: QCheckBox,
        getter: Callable[[], Any],
        setter: Callable[[Any], None],
    ) -> None:
        """Commit checkbox changes on toggle."""

        def _commit(_state: int) -> None:
            value = widget.isChecked()
            if value != getter():
                setter(value)

        widget.stateChanged.connect(_commit)
