"""Minimal improve-model confirmation dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
)


@dataclass(frozen=True, slots=True)
class ImproveModelDialogPayload:
    request: ImproveModelTrainingRequest


class ImproveModelDialog(QDialog):
    def __init__(self, summary: object, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary = summary
        self.setWindowTitle("Improve Model From Selection")
        layout = QVBoxLayout(self)
        count = getattr(summary, "selected_event_count", 0)
        layout.addWidget(QLabel(f"Train a candidate model from {count} selected events?", self))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_payload(self) -> ImproveModelDialogPayload:
        return ImproveModelDialogPayload(
            ImproveModelTrainingRequest(
                target_label=str(getattr(self._summary, "target_label", "selection") or "selection"),
                base_model_path=getattr(self._summary, "base_model_path", None),
            )
        )


__all__ = ["ImproveModelDialog", "ImproveModelDialogPayload"]
