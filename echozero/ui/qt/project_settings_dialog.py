"""Project settings dialog for project-local EchoZero controls.
Exists because project-scoped values should be edited from one explicit modal surface.
Connects launcher project settings actions to runtime-backed project fields.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ProjectSettingsDialog(QDialog):
    """Modal editor for project-local settings."""

    def __init__(
        self,
        *,
        current_offset_seconds: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("projectSettingsDialog")
        self.setWindowTitle("Project Settings")
        self.resize(560, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)

        summary = QLabel(
            (
                "Adjust project-wide MA3 push timing.\n"
                "Negative values push events earlier (left on the clock)."
            ),
            self,
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        self._ma3_push_offset = QDoubleSpinBox(self)
        self._ma3_push_offset.setDecimals(3)
        self._ma3_push_offset.setRange(-3600.0, 3600.0)
        self._ma3_push_offset.setSingleStep(0.05)
        self._ma3_push_offset.setSuffix(" s")
        self._ma3_push_offset.setValue(_coerce_offset_seconds(current_offset_seconds))
        form.addRow("Global MA3 push offset:", self._ma3_push_offset)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def ma3_push_offset_seconds(self) -> float:
        return float(self._ma3_push_offset.value())


def _coerce_offset_seconds(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0

