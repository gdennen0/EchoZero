"""Polished improve-model dialog for model evolution candidate runs.
Exists so the app can expose retraining as a small, clear operator action.
Connects selected fixed Events to Foundry model-evolution run requests.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
)


@dataclass(frozen=True, slots=True)
class ImproveModelDialogPayload:
    request: ImproveModelTrainingRequest

from echozero.ui.style.qt import ensure_qt_theme_installed


class ImproveModelDialog(QDialog):
    def __init__(self, summary: object, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        ensure_qt_theme_installed()
        self._summary = summary
        self._label_checks: dict[str, QCheckBox] = {}
        self.setWindowTitle("Improve Models")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        count = getattr(summary, "selected_event_count", 0)
        title = QLabel("Improve models from selected Events", self)
        title.setObjectName("DialogTitle")
        layout.addWidget(title)

        detail = QLabel(
            f"{count} selected Events will become event-span training examples.",
            self,
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)

        panel = QFrame(self)
        panel.setObjectName("ImproveModelPanel")
        grid = QGridLayout(panel)
        grid.setColumnStretch(1, 1)

        self._identity = QLineEdit(str(getattr(summary, "target_identity", "EchoZero Core")), panel)
        grid.addWidget(QLabel("Model name", panel), 0, 0)
        grid.addWidget(self._identity, 0, 1)

        self._profile = QComboBox(panel)
        self._profile.addItem("Beefy", "beefy")
        self._profile.addItem("Quick Check", "quick_check")
        self._profile.addItem("Release Candidate", "release_candidate")
        grid.addWidget(QLabel("Training", panel), 1, 0)
        grid.addWidget(self._profile, 1, 1)

        label_row = QHBoxLayout()
        label_counts = dict(getattr(summary, "label_counts", {}) or {})
        candidate_labels = tuple(getattr(summary, "labels", ()) or ("kick", "snare"))
        for label in candidate_labels:
            checkbox = QCheckBox(_label_title(label), panel)
            checkbox.setChecked(True)
            checkbox.setToolTip(f"{label_counts.get(label, 0)} selected examples")
            self._label_checks[label] = checkbox
            label_row.addWidget(checkbox)
        label_row.addStretch(1)
        grid.addWidget(QLabel("Targets", panel), 2, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addLayout(label_row, 2, 1)

        counts = QLabel(_counts_text(summary), panel)
        counts.setWordWrap(True)
        grid.addWidget(QLabel("Examples", panel), 3, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(counts, 3, 1)
        layout.addWidget(panel)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create Candidate Runs")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_payload(self) -> ImproveModelDialogPayload:
        return ImproveModelDialogPayload(
            ImproveModelTrainingRequest(
                target_label=str(
                    getattr(self._summary, "target_label", "selection") or "selection"
                ),
                target_identity=self._identity.text().strip() or "EchoZero Core",
                labels=tuple(
                    label for label, checkbox in self._label_checks.items() if checkbox.isChecked()
                ),
                profile_name=str(self._profile.currentData() or "beefy"),
                base_model_path=getattr(self._summary, "base_model_path", None),
                truths=tuple(getattr(self._summary, "truths", ()) or ()),
            )
        )


def _label_title(label: object) -> str:
    return str(label).strip().replace("_", " ").title()


def _counts_text(summary: object) -> str:
    label_counts = dict(getattr(summary, "label_counts", {}) or {})
    if not label_counts:
        return "No eligible fixed Events were found in the selection."
    parts = [f"{_label_title(label)} {count}" for label, count in sorted(label_counts.items())]
    source_count = int(getattr(summary, "source_audio_count", 0) or 0)
    return f"{'  |  '.join(parts)}\n{source_count} source audio file(s)"


__all__ = ["ImproveModelDialog", "ImproveModelDialogPayload"]
