from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp


class FoundryWindow(QMainWindow):
    """Official EchoZero Foundry v1 window (clean/simple mode)."""

    def __init__(self, root: Path):
        super().__init__()
        self._root = Path(root)
        self._app = FoundryApp(self._root)
        self._app.activity.set_listener(self._on_activity)

        self.setWindowTitle("EchoZero Foundry v1")
        self.resize(920, 700)

        self._dataset_id: str | None = None
        self._version_id: str | None = None
        self._run_id: str | None = None
        self._artifact_id: str | None = None

        container = QWidget()
        self.setCentralWidget(container)
        root_layout = QVBoxLayout(container)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_dataset_box())
        root_layout.addWidget(self._build_training_box())
        root_layout.addWidget(self._build_artifact_box())
        root_layout.addWidget(self._build_activity_box(), stretch=1)

        self._set_status(f"Workspace: {self._root}")

    # ------------------------------------------------------------------
    # UI Sections
    # ------------------------------------------------------------------

    def _build_dataset_box(self) -> QWidget:
        box = QGroupBox("1) Dataset")
        grid = QGridLayout(box)

        self.dataset_name = QLineEdit("Drums")
        self.dataset_folder = QLineEdit("")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._pick_dataset_folder)
        ingest_btn = QPushButton("Create + Ingest")
        ingest_btn.clicked.connect(self._create_and_ingest_dataset)

        self.val_split = QDoubleSpinBox()
        self.val_split.setRange(0.0, 0.9)
        self.val_split.setSingleStep(0.05)
        self.val_split.setValue(0.15)

        self.test_split = QDoubleSpinBox()
        self.test_split.setRange(0.0, 0.9)
        self.test_split.setSingleStep(0.05)
        self.test_split.setValue(0.10)

        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)

        self.balance = QLineEdit("none")
        plan_btn = QPushButton("Plan Split/Balance")
        plan_btn.clicked.connect(self._plan_version)

        grid.addWidget(QLabel("Name"), 0, 0)
        grid.addWidget(self.dataset_name, 0, 1, 1, 3)

        grid.addWidget(QLabel("Folder"), 1, 0)
        grid.addWidget(self.dataset_folder, 1, 1, 1, 2)
        grid.addWidget(browse_btn, 1, 3)

        grid.addWidget(ingest_btn, 2, 3)

        grid.addWidget(QLabel("Val Split"), 3, 0)
        grid.addWidget(self.val_split, 3, 1)
        grid.addWidget(QLabel("Test Split"), 3, 2)
        grid.addWidget(self.test_split, 3, 3)

        grid.addWidget(QLabel("Seed"), 4, 0)
        grid.addWidget(self.seed, 4, 1)
        grid.addWidget(QLabel("Balance"), 4, 2)
        grid.addWidget(self.balance, 4, 3)

        grid.addWidget(plan_btn, 5, 3)
        return box

    def _build_training_box(self) -> QWidget:
        box = QGroupBox("2) Training Run")
        layout = QGridLayout(box)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(10)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 1024)
        self.batch_size.setValue(16)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.000001, 1.0)
        self.learning_rate.setDecimals(6)
        self.learning_rate.setSingleStep(0.0001)
        self.learning_rate.setValue(0.001)

        create_btn = QPushButton("Create Run")
        start_btn = QPushButton("Start")
        checkpoint_btn = QPushButton("Checkpoint e1")
        complete_btn = QPushButton("Complete")
        fail_btn = QPushButton("Fail")

        create_btn.clicked.connect(self._create_run)
        start_btn.clicked.connect(self._start_run)
        checkpoint_btn.clicked.connect(self._checkpoint_run)
        complete_btn.clicked.connect(self._complete_run)
        fail_btn.clicked.connect(self._fail_run)

        layout.addWidget(QLabel("Epochs"), 0, 0)
        layout.addWidget(self.epochs, 0, 1)
        layout.addWidget(QLabel("Batch"), 0, 2)
        layout.addWidget(self.batch_size, 0, 3)
        layout.addWidget(QLabel("LR"), 0, 4)
        layout.addWidget(self.learning_rate, 0, 5)

        layout.addWidget(create_btn, 1, 0)
        layout.addWidget(start_btn, 1, 1)
        layout.addWidget(checkpoint_btn, 1, 2)
        layout.addWidget(complete_btn, 1, 3)
        layout.addWidget(fail_btn, 1, 4)
        return box

    def _build_artifact_box(self) -> QWidget:
        box = QGroupBox("3) Artifact")
        layout = QGridLayout(box)

        self.class_names = QLineEdit("kick,snare")
        finalize_btn = QPushButton("Finalize Artifact")
        validate_btn = QPushButton("Validate")

        finalize_btn.clicked.connect(self._finalize_artifact)
        validate_btn.clicked.connect(self._validate_artifact)

        layout.addWidget(QLabel("Classes (comma-separated)"), 0, 0)
        layout.addWidget(self.class_names, 0, 1, 1, 3)
        layout.addWidget(finalize_btn, 1, 2)
        layout.addWidget(validate_btn, 1, 3)
        return box

    def _build_activity_box(self) -> QWidget:
        box = QGroupBox("Activity")
        layout = QVBoxLayout(box)
        self.status_line = QLabel("Ready")
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.status_line)
        layout.addWidget(self.activity, stretch=1)
        return box

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pick_dataset_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select dataset folder", str(self._root))
        if path:
            self.dataset_folder.setText(path)

    def _create_and_ingest_dataset(self) -> None:
        try:
            name = self.dataset_name.text().strip() or "Dataset"
            folder = Path(self.dataset_folder.text().strip())
            dataset = self._app.datasets.create_dataset(name)
            version = self._app.datasets.ingest_from_folder(dataset.id, folder)
            self._dataset_id = dataset.id
            self._version_id = version.id
            self._set_status(f"Dataset {dataset.id} / Version {version.id} / Samples {len(version.samples)}")
        except Exception as exc:
            self._error(exc)

    def _plan_version(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("Ingest dataset first")
            planned = self._app.plan_version(
                self._version_id,
                validation_split=float(self.val_split.value()),
                test_split=float(self.test_split.value()),
                seed=int(self.seed.value()),
                balance_strategy=self.balance.text().strip() or "none",
            )
            self._set_status(
                f"Planned {planned['version_id']} (train={len(planned['split_plan']['train_ids'])}, "
                f"val={len(planned['split_plan']['val_ids'])}, test={len(planned['split_plan']['test_ids'])})"
            )
        except Exception as exc:
            self._error(exc)

    def _create_run(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("No dataset version")
            run = self._app.create_run(self._version_id, self._build_run_spec())
            self._run_id = run.id
            self._set_status(f"Run created: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _start_run(self) -> None:
        try:
            run = self._app.start_run(self._require_run_id())
            self._set_status(f"Run started: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _checkpoint_run(self) -> None:
        try:
            path = self._app.runs.save_checkpoint(self._require_run_id(), epoch=1, metric_snapshot={"loss": 0.123})
            self._set_status(f"Checkpoint: {path.name}")
        except Exception as exc:
            self._error(exc)

    def _complete_run(self) -> None:
        try:
            run = self._app.runs.complete_run(self._require_run_id(), metrics={"f1": 0.91})
            self._set_status(f"Run completed: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _fail_run(self) -> None:
        try:
            run = self._app.runs.fail_run(self._require_run_id(), error="manual-failure")
            self._set_status(f"Run failed: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _finalize_artifact(self) -> None:
        try:
            manifest = self._build_artifact_manifest()
            artifact = self._app.finalize_artifact(self._require_run_id(), manifest)
            self._artifact_id = artifact.id
            self._set_status(f"Artifact finalized: {artifact.id}")
        except Exception as exc:
            self._error(exc)

    def _validate_artifact(self) -> None:
        try:
            if not self._artifact_id:
                raise ValueError("No artifact yet")
            report = self._app.validate_artifact(self._artifact_id)
            self._set_status(f"Validation: ok={report.ok}")
        except Exception as exc:
            self._error(exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_run_spec(self) -> dict:
        if not self._version_id:
            raise ValueError("No dataset version")
        return {
            "schema": "foundry.train_run_spec.v1",
            "classificationMode": "multiclass",
            "data": {
                "datasetVersionId": self._version_id,
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
            "training": {
                "epochs": int(self.epochs.value()),
                "batchSize": int(self.batch_size.value()),
                "learningRate": float(self.learning_rate.value()),
            },
        }

    def _build_artifact_manifest(self) -> dict:
        classes = [c.strip() for c in self.class_names.text().split(",") if c.strip()]
        if not classes:
            raise ValueError("At least one class required")
        return {
            "weightsPath": "exports/model.pth",
            "classes": classes,
            "classificationMode": "multiclass",
            "inferencePreprocessing": {
                "sampleRate": 22050,
                "maxLength": 22050,
                "nFft": 2048,
                "hopLength": 512,
                "nMels": 128,
                "fmax": 8000,
            },
        }

    def _require_run_id(self) -> str:
        if not self._run_id:
            raise ValueError("No run yet")
        return self._run_id

    def _on_activity(self, item) -> None:
        self.activity.appendPlainText(f"[{item.kind}] {item.message}")

    def _set_status(self, text: str) -> None:
        self.status_line.setText(text)

    def _error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        QMessageBox.critical(self, "Foundry Error", str(exc))


def run_foundry_ui(root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(root or Path.cwd())
    window.show()
    return app.exec()
