from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
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
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp


_DEFAULT_RUN_SPEC = {
    "schema": "foundry.train_run_spec.v1",
    "classificationMode": "multiclass",
    "data": {
        "datasetVersionId": "",
        "sampleRate": 22050,
        "maxLength": 22050,
        "nFft": 2048,
        "hopLength": 512,
        "nMels": 128,
        "fmax": 8000,
    },
    "training": {
        "epochs": 10,
        "batchSize": 16,
        "learningRate": 0.001,
    },
}

_DEFAULT_ARTIFACT = {
    "weightsPath": "exports/model.pth",
    "classes": ["kick", "snare"],
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


class FoundryWindow(QMainWindow):
    def __init__(self, root: Path):
        super().__init__()
        self._root = Path(root)
        self._app = FoundryApp(self._root)
        self._app.activity.set_listener(self._on_activity)

        self.setWindowTitle("EchoZero Foundry v1")
        self.resize(980, 760)

        self._dataset_id: str | None = None
        self._version_id: str | None = None
        self._run_id: str | None = None
        self._artifact_id: str | None = None

        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)

        layout.addWidget(self._build_dataset_box())
        layout.addWidget(self._build_plan_box())
        layout.addWidget(self._build_run_box())
        layout.addWidget(self._build_artifact_box())
        layout.addWidget(self._build_activity_box(), stretch=1)

        self._set_status(f"Root: {self._root}")

    def _build_dataset_box(self) -> QWidget:
        box = QGroupBox("Dataset")
        grid = QGridLayout(box)

        self.dataset_name = QLineEdit("Drums")
        self.dataset_folder = QLineEdit("")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dataset_folder)
        ingest = QPushButton("Create + Ingest")
        ingest.clicked.connect(self._create_and_ingest_dataset)

        grid.addWidget(QLabel("Name"), 0, 0)
        grid.addWidget(self.dataset_name, 0, 1, 1, 3)
        grid.addWidget(QLabel("Folder"), 1, 0)
        grid.addWidget(self.dataset_folder, 1, 1, 1, 2)
        grid.addWidget(browse, 1, 3)
        grid.addWidget(ingest, 2, 3)

        return box

    def _build_plan_box(self) -> QWidget:
        box = QGroupBox("Split/Balance Plan")
        row = QHBoxLayout(box)

        self.val_split = QLineEdit("0.15")
        self.test_split = QLineEdit("0.10")
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        self.balance = QLineEdit("none")

        apply_btn = QPushButton("Plan Version")
        apply_btn.clicked.connect(self._plan_version)

        row.addWidget(QLabel("Val"))
        row.addWidget(self.val_split)
        row.addWidget(QLabel("Test"))
        row.addWidget(self.test_split)
        row.addWidget(QLabel("Seed"))
        row.addWidget(self.seed)
        row.addWidget(QLabel("Balance"))
        row.addWidget(self.balance)
        row.addWidget(apply_btn)

        return box

    def _build_run_box(self) -> QWidget:
        box = QGroupBox("Run")
        layout = QVBoxLayout(box)

        self.run_spec = QPlainTextEdit(json.dumps(_DEFAULT_RUN_SPEC, indent=2))
        self.run_spec.setMinimumHeight(160)

        actions = QHBoxLayout()
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

        for btn in [create_btn, start_btn, checkpoint_btn, complete_btn, fail_btn]:
            actions.addWidget(btn)

        layout.addWidget(self.run_spec)
        layout.addLayout(actions)
        return box

    def _build_artifact_box(self) -> QWidget:
        box = QGroupBox("Artifact")
        layout = QVBoxLayout(box)

        self.artifact_manifest = QPlainTextEdit(json.dumps(_DEFAULT_ARTIFACT, indent=2))
        self.artifact_manifest.setMinimumHeight(140)

        row = QHBoxLayout()
        finalize_btn = QPushButton("Finalize Artifact")
        validate_btn = QPushButton("Validate Artifact")
        finalize_btn.clicked.connect(self._finalize_artifact)
        validate_btn.clicked.connect(self._validate_artifact)
        row.addWidget(finalize_btn)
        row.addWidget(validate_btn)

        layout.addWidget(self.artifact_manifest)
        layout.addLayout(row)
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
            spec = self._safe_json(self.run_spec.toPlainText(), _DEFAULT_RUN_SPEC)
            spec.setdefault("data", {})["datasetVersionId"] = version.id
            self.run_spec.setPlainText(json.dumps(spec, indent=2))
            self._set_status(f"Dataset {dataset.id}, version {version.id}, samples={len(version.samples)}")
        except Exception as exc:
            self._error(exc)

    def _plan_version(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("No dataset version yet. Ingest dataset first.")
            planned = self._app.plan_version(
                self._version_id,
                validation_split=float(self.val_split.text()),
                test_split=float(self.test_split.text()),
                seed=int(self.seed.value()),
                balance_strategy=self.balance.text().strip() or "none",
            )
            self._set_status(f"Planned version {planned['version_id']}")
        except Exception as exc:
            self._error(exc)

    def _create_run(self) -> None:
        try:
            spec = self._safe_json(self.run_spec.toPlainText(), _DEFAULT_RUN_SPEC)
            dataset_version_id = spec.get("data", {}).get("datasetVersionId") or self._version_id
            if not dataset_version_id:
                raise ValueError("datasetVersionId missing in run spec")
            run = self._app.create_run(dataset_version_id, spec)
            self._run_id = run.id
            self._set_status(f"Run created: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _start_run(self) -> None:
        try:
            run_id = self._require_run_id()
            run = self._app.start_run(run_id)
            self._set_status(f"Run started: {run.id} ({run.status.value})")
        except Exception as exc:
            self._error(exc)

    def _checkpoint_run(self) -> None:
        try:
            run_id = self._require_run_id()
            path = self._app.runs.save_checkpoint(run_id, epoch=1, metric_snapshot={"loss": 0.123})
            self._set_status(f"Checkpoint saved: {path.name}")
        except Exception as exc:
            self._error(exc)

    def _complete_run(self) -> None:
        try:
            run_id = self._require_run_id()
            run = self._app.runs.complete_run(run_id, metrics={"f1": 0.91})
            self._set_status(f"Run completed: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _fail_run(self) -> None:
        try:
            run_id = self._require_run_id()
            run = self._app.runs.fail_run(run_id, error="manual-failure")
            self._set_status(f"Run failed: {run.id}")
        except Exception as exc:
            self._error(exc)

    def _finalize_artifact(self) -> None:
        try:
            run_id = self._require_run_id()
            manifest = self._safe_json(self.artifact_manifest.toPlainText(), _DEFAULT_ARTIFACT)
            artifact = self._app.finalize_artifact(run_id, manifest)
            self._artifact_id = artifact.id
            self._set_status(f"Artifact finalized: {artifact.id}")
        except Exception as exc:
            self._error(exc)

    def _validate_artifact(self) -> None:
        try:
            if not self._artifact_id:
                raise ValueError("No artifact yet. Finalize first.")
            report = self._app.validate_artifact(self._artifact_id)
            self._set_status(f"Artifact validation ok={report.ok}")
        except Exception as exc:
            self._error(exc)

    def _require_run_id(self) -> str:
        if not self._run_id:
            raise ValueError("No run yet. Create run first.")
        return self._run_id

    def _on_activity(self, item) -> None:
        self.activity.appendPlainText(f"[{item.kind}] {item.message}")

    def _set_status(self, text: str) -> None:
        self.status_line.setText(text)

    def _error(self, exc: Exception) -> None:
        self._set_status(f"Error: {exc}")
        QMessageBox.critical(self, "Foundry Error", str(exc))

    @staticmethod
    def _safe_json(text: str, default: dict) -> dict:
        payload = json.loads(text.strip() or "{}")
        merged = json.loads(json.dumps(default))
        if isinstance(payload, dict):
            for k, v in payload.items():
                merged[k] = v
        return merged


def run_foundry_ui(root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(root or Path.cwd())
    window.show()
    return app.exec()
