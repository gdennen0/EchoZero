"""
ImproveModelDialog collects one clean EZ-first candidate-training request.
Exists because operators need a simple local-model improvement prompt, not raw Foundry controls.
Connects reviewed-selection summaries to one larger single-panel dialog without hidden workflow steps.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry.services.selection_model_improvement_service import (
    ImproveModelTrainingRequest,
)
from echozero.ui.qt.app_shell_selection_model_improvement import ImproveModelSelectionSummary


@dataclass(frozen=True, slots=True)
class ImproveModelDialogResult:
    """Accepted operator choices for one improve-model candidate run."""

    request: ImproveModelTrainingRequest


class ImproveModelDialog(QDialog):
    """Single-panel dialog for turning one reviewed selection into a candidate-training request."""

    _strength_options: tuple[tuple[str, str], ...] = (
        ("Light", "light"),
        ("Balanced", "balanced"),
        ("Strong", "strong"),
    )
    _scope_options: tuple[tuple[str, str], ...] = (
        ("Selection Only", "selected_events"),
        ("This Song/Layer", "song_layer"),
        ("This Song", "song"),
        ("Whole Project", "project"),
    )

    def __init__(
        self,
        summary: ImproveModelSelectionSummary,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self.setWindowTitle("Improve Model From Selection")
        self.setModal(True)
        self.resize(560, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QLabel(
            (
                "Train a new local candidate model from the reviewed events you selected. "
                "The selection stays the anchor, and the scope decides how much nearby reviewed "
                "project context EZ can use."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary_group = QGroupBox("Selection", self)
        summary_form = QFormLayout(summary_group)
        summary_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        summary_form.addRow("Target label", QLabel(summary.target_label, summary_group))
        summary_form.addRow(
            "Reviewed events",
            QLabel(str(summary.reviewed_event_count), summary_group),
        )
        summary_form.addRow(
            "Signal mix",
            QLabel(
                f"{summary.positive_signal_count} positive, {summary.negative_signal_count} negative",
                summary_group,
            ),
        )
        summary_form.addRow(
            "Anchor behavior",
            QLabel("Always starts from the reviewed events you selected", summary_group),
        )
        layout.addWidget(summary_group)

        config_group = QGroupBox("Training", self)
        config_form = QFormLayout(config_group)
        config_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._candidate_name = QLineEdit(
            f"{summary.target_label.title()} Selection Candidate",
            config_group,
        )
        config_form.addRow("Candidate name", self._candidate_name)

        self._strength = QComboBox(config_group)
        for label, value in self._strength_options:
            self._strength.addItem(label, userData=value)
        self._strength.setCurrentIndex(1)
        config_form.addRow("Training strength", self._strength)

        self._scope_mode = QComboBox(config_group)
        for label, value in self._scope_options:
            self._scope_mode.addItem(label, userData=value)
        default_scope_index = next(
            (
                index
                for index, (_label, value) in enumerate(self._scope_options)
                if value == summary.default_scope_mode
            ),
            1,
        )
        self._scope_mode.setCurrentIndex(default_scope_index)
        self._scope_mode.currentIndexChanged.connect(self._sync_scope_controls)
        config_form.addRow("Training scope", self._scope_mode)

        self._include_related = QCheckBox("Expand with similar reviewed examples in chosen scope", self)
        self._include_related.setChecked(True)
        config_form.addRow("Context expansion", self._include_related)

        self._base_model = QComboBox(config_group)
        self._base_model.addItem("No base comparison", userData=None)
        for option in summary.base_model_options:
            self._base_model.addItem(option.label, userData=option.option_id)
        config_form.addRow("Base model", self._base_model)

        layout.addWidget(config_group)

        guide_group = QGroupBox("What EZ Will Do", self)
        guide_layout = QVBoxLayout(guide_group)
        guide_layout.setContentsMargins(12, 12, 12, 12)
        guide_layout.setSpacing(6)
        for line in (
            "Use your selected reviewed events as the anchor examples.",
            "Optionally pull in related reviewed examples from the chosen scope.",
            "Train a local candidate model instead of silently replacing the active one.",
            "Use the chosen base model as the comparison reference for this V1 run.",
        ):
            label = QLabel(f"- {line}", guide_group)
            label.setWordWrap(True)
            guide_layout.addWidget(label)
        layout.addWidget(guide_group)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._train_button = self._require_button(QDialogButtonBox.StandardButton.Ok)
        self._train_button.setText("Train Candidate Model")
        layout.addWidget(self._buttons)
        self._sync_scope_controls()

    def result_payload(self) -> ImproveModelDialogResult:
        """Return the normalized training request after acceptance."""
        candidate_name = self._candidate_name.text().strip() or (
            f"{self._summary.target_label.title()} Selection Candidate"
        )
        strength = str(self._strength.currentData()).strip().lower() or "balanced"
        scope_mode = str(self._scope_mode.currentData()).strip().lower() or "song_layer"
        base_model_option_id = self._base_model.currentData()
        normalized_option_id = (
            None
            if base_model_option_id is None or not str(base_model_option_id).strip()
            else str(base_model_option_id).strip()
        )
        return ImproveModelDialogResult(
            request=ImproveModelTrainingRequest(
                target_label=self._summary.target_label,
                selected_signal_ids=self._summary.selected_signal_ids,
                candidate_name=candidate_name,
                scope_mode=scope_mode,
                strength=strength,
                include_related_examples=(
                    self._include_related.isChecked() and scope_mode != "selected_events"
                ),
                base_model_option_id=normalized_option_id,
            )
        )

    def _sync_scope_controls(self) -> None:
        scope_mode = str(self._scope_mode.currentData()).strip().lower()
        uses_related_context = scope_mode != "selected_events"
        self._include_related.setEnabled(uses_related_context)
        if not uses_related_context:
            self._include_related.setChecked(False)
        elif not self._include_related.isChecked():
            self._include_related.setChecked(True)

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Dialog button is missing: {standard_button!r}")
        return button
