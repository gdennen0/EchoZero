"""Find-similar event dialog helpers."""

from __future__ import annotations

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget


class EventComparisonDialog(QDialog):
    def __init__(self, *_args: object, parent: QWidget | None = None, **_kwargs: object) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find Similar Events")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Find similar events", self))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_payload(self) -> dict[str, object]:
        return {}


__all__ = ["EventComparisonDialog"]
