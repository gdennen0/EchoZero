"""Run-tab widget builders for the Foundry window.
Exists to keep training and artifact widget construction out of the run orchestration mixin.
Connects Foundry run/artifact actions to the tab surfaces used by the main window shell.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.ui.style import SHELL_TOKENS


class _FoundryWindowRunBuildMixin:
    epochs: QSpinBox
    batch_size: QSpinBox
    learning_rate: QDoubleSpinBox
    class_names: QLineEdit
    run_summary: QPlainTextEdit
    artifact_summary: QPlainTextEdit
    create_run_btn: QPushButton
    start_run_btn: QPushButton
    create_start_run_btn: QPushButton
    _run_action_buttons: list[QPushButton]

    def _build_training_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        form = QGroupBox("Training Run")
        grid = QGridLayout(form)
        grid.setHorizontalSpacing(SHELL_TOKENS.scales.layout_gap)
        grid.setVerticalSpacing(SHELL_TOKENS.scales.inline_gap)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(4)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(4)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.0005)
        self.learning_rate.setValue(0.01)

        self.create_run_btn = QPushButton("Create Run")
        self.create_run_btn.clicked.connect(self._create_run)
        self.start_run_btn = QPushButton("Start Run")
        self.start_run_btn.clicked.connect(self._start_run)
        self.create_start_run_btn = QPushButton("Create + Start")
        self.create_start_run_btn.clicked.connect(self._create_and_start_run)
        checkpoint_btn = QPushButton("Save Checkpoint")
        checkpoint_btn.clicked.connect(self._checkpoint_run)
        complete_btn = QPushButton("Mark Complete")
        complete_btn.clicked.connect(self._complete_run)
        fail_btn = QPushButton("Mark Failed")
        fail_btn.clicked.connect(self._fail_run)
        open_exports_btn = QPushButton("Open Exports Dir")
        open_exports_btn.clicked.connect(self._open_exports_dir)
        open_metrics_btn = QPushButton("Open metrics.json")
        open_metrics_btn.clicked.connect(self._open_metrics_json)
        open_run_summary_btn = QPushButton("Open run_summary.json")
        open_run_summary_btn.clicked.connect(self._open_run_summary_json)
        self._run_action_buttons = [
            self.create_run_btn,
            self.start_run_btn,
            self.create_start_run_btn,
            checkpoint_btn,
            complete_btn,
            fail_btn,
        ]

        grid.addWidget(QLabel("Epochs"), 0, 0)
        grid.addWidget(self.epochs, 0, 1)
        grid.addWidget(QLabel("Batch"), 0, 2)
        grid.addWidget(self.batch_size, 0, 3)
        grid.addWidget(QLabel("LR"), 0, 4)
        grid.addWidget(self.learning_rate, 0, 5)
        grid.addWidget(self.create_run_btn, 1, 0)
        grid.addWidget(self.start_run_btn, 1, 1)
        grid.addWidget(self.create_start_run_btn, 1, 2)
        grid.addWidget(checkpoint_btn, 1, 3)
        grid.addWidget(complete_btn, 1, 4)
        grid.addWidget(fail_btn, 1, 5)
        grid.addWidget(open_exports_btn, 2, 0, 1, 2)
        grid.addWidget(open_metrics_btn, 2, 2, 1, 2)
        grid.addWidget(open_run_summary_btn, 2, 4, 1, 2)

        self.run_summary = QPlainTextEdit()
        self.run_summary.setReadOnly(True)
        self.run_summary.setPlaceholderText("Current run details will appear here.")

        layout.addWidget(form)
        layout.addWidget(QLabel("Current Run"))
        layout.addWidget(self.run_summary, stretch=1)
        return box

    def _build_artifact_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        form = QGroupBox("Artifact Actions")
        grid = QGridLayout(form)
        grid.setHorizontalSpacing(SHELL_TOKENS.scales.layout_gap)
        grid.setVerticalSpacing(SHELL_TOKENS.scales.inline_gap)

        self.class_names = QLineEdit("kick,snare")
        finalize_btn = QPushButton("Finalize Artifact")
        finalize_btn.clicked.connect(self._finalize_artifact)
        validate_btn = QPushButton("Validate Selected Artifact")
        validate_btn.clicked.connect(self._validate_artifact)
        open_manifest_btn = QPushButton("Open Artifact Manifest")
        open_manifest_btn.clicked.connect(self._open_artifact_manifest)
        open_latest_package_btn = QPushButton("Open Latest Artifact Package")
        open_latest_package_btn.clicked.connect(self._open_latest_artifact_package)

        grid.addWidget(QLabel("Classes (comma-separated)"), 0, 0)
        grid.addWidget(self.class_names, 0, 1, 1, 3)
        grid.addWidget(finalize_btn, 1, 2)
        grid.addWidget(validate_btn, 1, 3)
        grid.addWidget(open_latest_package_btn, 2, 2)
        grid.addWidget(open_manifest_btn, 2, 3)

        self.artifact_summary = QPlainTextEdit()
        self.artifact_summary.setReadOnly(True)
        self.artifact_summary.setPlaceholderText(
            "Selected artifact and validation details will appear here."
        )

        layout.addWidget(form)
        layout.addWidget(QLabel("Selected Artifact"))
        layout.addWidget(self.artifact_summary, stretch=1)
        return box
