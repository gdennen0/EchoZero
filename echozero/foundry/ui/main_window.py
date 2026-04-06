from __future__ import annotations

import json
import os
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
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QDoubleSpinBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from echozero.foundry import FoundryApp
from echozero.foundry.persistence import DatasetRepository, DatasetVersionRepository, EvalReportRepository, ModelArtifactRepository


class FoundryWindow(QMainWindow):
    """Official EchoZero Foundry v1 window for local desktop workflows."""

    def __init__(self, root: Path):
        super().__init__()
        self._root = Path(root)
        self._app = FoundryApp(self._root)
        self._app.activity.set_listener(self._on_activity)
        self._datasets = DatasetRepository(self._root)
        self._versions = DatasetVersionRepository(self._root)
        self._artifacts = ModelArtifactRepository(self._root)
        self._evals = EvalReportRepository(self._root)
        self._show_error_dialogs = os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"

        self.setWindowTitle("EchoZero Foundry v1")
        self.resize(1280, 860)

        self._dataset_id: str | None = None
        self._version_id: str | None = None
        self._run_id: str | None = None
        self._artifact_id: str | None = None
        self._selected_artifact_id: str | None = None

        container = QWidget()
        self.setCentralWidget(container)
        root_layout = QVBoxLayout(container)
        root_layout.setSpacing(10)

        root_layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_workflow_tabs())
        splitter.addWidget(self._build_workspace_panel())
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        root_layout.addWidget(splitter, stretch=1)

        self._load_defaults()
        self._refresh_workspace_state()
        self._set_status(f"Workspace ready: {self._root}")

    # ------------------------------------------------------------------
    # UI Sections
    # ------------------------------------------------------------------

    def _build_header(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.workspace_path = QLineEdit(str(self._root))
        self.workspace_path.setReadOnly(True)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_workspace_state)

        layout.addWidget(QLabel("Workspace"))
        layout.addWidget(self.workspace_path, stretch=1)
        layout.addWidget(refresh_btn)
        return widget

    def _build_workflow_tabs(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._build_dataset_box(), "Dataset")
        tabs.addTab(self._build_training_box(), "Run")
        tabs.addTab(self._build_artifact_box(), "Artifacts")
        tabs.addTab(self._build_activity_box(), "Activity")
        return tabs

    def _build_dataset_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)

        form = QGroupBox("Dataset Import")
        grid = QGridLayout(form)

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
        plan_btn = QPushButton("Plan Split / Balance")
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

        self.dataset_summary = QPlainTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setPlaceholderText("Dataset and version details will appear here.")

        layout.addWidget(form)
        layout.addWidget(QLabel("Current Dataset"))
        layout.addWidget(self.dataset_summary, stretch=1)
        return box

    def _build_training_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)

        form = QGroupBox("Training Run")
        grid = QGridLayout(form)

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

        create_btn = QPushButton("Create Run")
        create_btn.clicked.connect(self._create_run)
        start_btn = QPushButton("Start Run")
        start_btn.clicked.connect(self._start_run)
        create_start_btn = QPushButton("Create + Start")
        create_start_btn.clicked.connect(self._create_and_start_run)
        checkpoint_btn = QPushButton("Save Checkpoint")
        checkpoint_btn.clicked.connect(self._checkpoint_run)
        complete_btn = QPushButton("Mark Complete")
        complete_btn.clicked.connect(self._complete_run)
        fail_btn = QPushButton("Mark Failed")
        fail_btn.clicked.connect(self._fail_run)

        grid.addWidget(QLabel("Epochs"), 0, 0)
        grid.addWidget(self.epochs, 0, 1)
        grid.addWidget(QLabel("Batch"), 0, 2)
        grid.addWidget(self.batch_size, 0, 3)
        grid.addWidget(QLabel("LR"), 0, 4)
        grid.addWidget(self.learning_rate, 0, 5)
        grid.addWidget(create_btn, 1, 0)
        grid.addWidget(start_btn, 1, 1)
        grid.addWidget(create_start_btn, 1, 2)
        grid.addWidget(checkpoint_btn, 1, 3)
        grid.addWidget(complete_btn, 1, 4)
        grid.addWidget(fail_btn, 1, 5)

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

        form = QGroupBox("Artifact Actions")
        grid = QGridLayout(form)

        self.class_names = QLineEdit("kick,snare")
        finalize_btn = QPushButton("Finalize Artifact")
        finalize_btn.clicked.connect(self._finalize_artifact)
        validate_btn = QPushButton("Validate Selected Artifact")
        validate_btn.clicked.connect(self._validate_artifact)

        grid.addWidget(QLabel("Classes (comma-separated)"), 0, 0)
        grid.addWidget(self.class_names, 0, 1, 1, 3)
        grid.addWidget(finalize_btn, 1, 2)
        grid.addWidget(validate_btn, 1, 3)

        self.artifact_summary = QPlainTextEdit()
        self.artifact_summary.setReadOnly(True)
        self.artifact_summary.setPlaceholderText("Selected artifact and validation details will appear here.")

        layout.addWidget(form)
        layout.addWidget(QLabel("Selected Artifact"))
        layout.addWidget(self.artifact_summary, stretch=1)
        return box

    def _build_activity_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        self.status_line = QLabel("Ready")
        self.activity = QPlainTextEdit()
        self.activity.setReadOnly(True)
        self.activity.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.status_line)
        layout.addWidget(self.activity, stretch=1)
        return box

    def _build_workspace_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        overview_box = QGroupBox("Workspace Overview")
        overview_layout = QVBoxLayout(overview_box)
        self.workspace_summary = QPlainTextEdit()
        self.workspace_summary.setReadOnly(True)
        overview_layout.addWidget(self.workspace_summary)

        runs_box = QGroupBox("Runs")
        runs_layout = QVBoxLayout(runs_box)
        self.run_list = QListWidget()
        self.run_list.currentTextChanged.connect(self._select_run_from_list)
        self.run_list.setMinimumHeight(140)
        runs_layout.addWidget(self.run_list)

        artifacts_box = QGroupBox("Artifacts")
        artifacts_layout = QVBoxLayout(artifacts_box)
        self.artifact_list = QListWidget()
        self.artifact_list.currentTextChanged.connect(self._select_artifact_from_list)
        self.artifact_list.setMinimumHeight(110)
        artifacts_layout.addWidget(self.artifact_list)

        evals_box = QGroupBox("Eval Reports")
        evals_layout = QVBoxLayout(evals_box)
        self.eval_list = QListWidget()
        self.eval_list.setMinimumHeight(110)
        evals_layout.addWidget(self.eval_list)
        self.eval_summary = QPlainTextEdit()
        self.eval_summary.setReadOnly(True)
        evals_layout.addWidget(self.eval_summary, stretch=1)

        layout.addWidget(overview_box)
        layout.addWidget(runs_box)
        layout.addWidget(artifacts_box)
        layout.addWidget(evals_box, stretch=1)
        return widget

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
            dataset = self._app.datasets.create_dataset(name, source_ref=str(folder.resolve()))
            version = self._app.datasets.ingest_from_folder(dataset.id, folder)
            self._dataset_id = dataset.id
            self._version_id = version.id
            self.class_names.setText(",".join(version.class_map))
            self._set_status(f"Dataset ready: {dataset.id} -> {version.id} ({len(version.samples)} samples)")
            self._refresh_workspace_state()
        except Exception as exc:
            self._error(exc)

    def _plan_version(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("Ingest a dataset before planning")
            planned = self._app.plan_version(
                self._version_id,
                validation_split=float(self.val_split.value()),
                test_split=float(self.test_split.value()),
                seed=int(self.seed.value()),
                balance_strategy=self.balance.text().strip() or "none",
            )
            self._set_status(
                f"Plan saved for {planned['version_id']} "
                f"(train={len(planned['split_plan']['train_ids'])}, "
                f"val={len(planned['split_plan']['val_ids'])}, "
                f"test={len(planned['split_plan']['test_ids'])})"
            )
            self._refresh_workspace_state()
        except Exception as exc:
            self._error(exc)

    def _create_run(self) -> None:
        try:
            if not self._version_id:
                raise ValueError("Create and plan a dataset before creating a run")
            run = self._app.create_run(self._version_id, self._build_run_spec())
            self._run_id = run.id
            self._set_status(f"Run created: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _create_and_start_run(self) -> None:
        self._create_run()
        if self._run_id:
            self._start_run()

    def _start_run(self) -> None:
        try:
            run = self._app.start_run(self._require_run_id())
            artifacts = self._artifacts.list_for_run(run.id)
            evals = self._evals.list_for_run(run.id)
            self._artifact_id = artifacts[-1].id if artifacts else None
            self._set_status(
                f"Run {run.id} finished with status {run.status.value} "
                f"({len(evals)} eval, {len(artifacts)} artifact)"
            )
            self._refresh_workspace_state(select_run_id=run.id, select_artifact_id=self._artifact_id)
        except Exception as exc:
            self._error(exc)

    def _checkpoint_run(self) -> None:
        try:
            path = self._app.runs.save_checkpoint(self._require_run_id(), epoch=1, metric_snapshot={"loss": 0.123})
            self._set_status(f"Checkpoint saved: {path.name}")
            self._refresh_workspace_state(select_run_id=self._run_id)
        except Exception as exc:
            self._error(exc)

    def _complete_run(self) -> None:
        try:
            run = self._app.runs.complete_run(self._require_run_id(), metrics={"f1": 0.91})
            self._set_status(f"Run marked completed: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _fail_run(self) -> None:
        try:
            run = self._app.runs.fail_run(self._require_run_id(), error="manual-failure")
            self._set_status(f"Run marked failed: {run.id}")
            self._refresh_workspace_state(select_run_id=run.id)
        except Exception as exc:
            self._error(exc)

    def _finalize_artifact(self) -> None:
        try:
            artifact = self._app.finalize_artifact(self._require_run_id(), self._build_artifact_manifest())
            self._artifact_id = artifact.id
            self._set_status(f"Artifact finalized: {artifact.id}")
            self._refresh_workspace_state(select_run_id=self._run_id, select_artifact_id=artifact.id)
        except Exception as exc:
            self._error(exc)

    def _validate_artifact(self) -> None:
        try:
            artifact_id = self._selected_artifact_id or self._artifact_id
            if not artifact_id:
                raise ValueError("Select or create an artifact first")
            report = self._app.validate_artifact(artifact_id)
            self._set_status(
                f"Validation for {artifact_id}: ok={report.ok}, "
                f"errors={len(report.errors)}, warnings={len(report.warnings)}"
            )
            self._refresh_workspace_state(select_run_id=self._run_id, select_artifact_id=artifact_id)
        except Exception as exc:
            self._error(exc)

    # ------------------------------------------------------------------
    # Refresh + Selection
    # ------------------------------------------------------------------

    def _load_defaults(self) -> None:
        candidate = self._root / "data" / "drum-oneshots"
        if candidate.exists() and candidate.is_dir():
            self.dataset_folder.setText(str(candidate))
            self.dataset_name.setText(candidate.name.replace("-", " ").title())

    def _refresh_workspace_state(
        self,
        *,
        select_run_id: str | None = None,
        select_artifact_id: str | None = None,
    ) -> None:
        datasets = self._datasets.list()
        runs = sorted(self._app.runs.list_runs(), key=lambda item: item.created_at)

        latest_dataset = datasets[-1] if datasets else None
        if self._dataset_id is None and latest_dataset is not None:
            self._dataset_id = latest_dataset.id

        if self._dataset_id:
            versions = self._app.datasets.list_versions(self._dataset_id)
            if versions:
                latest_version = versions[-1]
                if self._version_id is None or latest_version.id == self._version_id:
                    self._version_id = latest_version.id

        self.workspace_summary.setPlainText(self._format_workspace_summary(datasets, runs))
        self.dataset_summary.setPlainText(self._format_dataset_summary())
        self._populate_run_list(runs, select_run_id=select_run_id)
        self._update_selection_details(select_run_id=select_run_id, select_artifact_id=select_artifact_id)

    def _populate_run_list(self, runs: list[object], *, select_run_id: str | None = None) -> None:
        current_run_id = select_run_id or self._run_id
        if not current_run_id and runs:
            current_run_id = runs[-1].id

        self.run_list.blockSignals(True)
        self.run_list.clear()
        selected_row = -1
        for index, run in enumerate(reversed(runs)):
            label = f"{run.id} [{run.status.value}] {run.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            self.run_list.addItem(label)
            self.run_list.item(index).setData(Qt.ItemDataRole.UserRole, run.id)
            if run.id == current_run_id:
                selected_row = index
        self.run_list.blockSignals(False)

        if selected_row >= 0:
            self.run_list.setCurrentRow(selected_row)
        else:
            self.run_summary.setPlainText("No runs yet.")
            self.artifact_list.clear()
            self.artifact_summary.clear()
            self.eval_list.clear()
            self.eval_summary.clear()

    def _select_run_from_list(self, _: str) -> None:
        item = self.run_list.currentItem()
        if item is None:
            return
        run_id = item.data(Qt.ItemDataRole.UserRole)
        self._run_id = str(run_id) if run_id else None
        self._update_selection_details(select_run_id=self._run_id)

    def _select_artifact_from_list(self, _: str) -> None:
        item = self.artifact_list.currentItem()
        if item is None:
            return
        artifact_id = item.data(Qt.ItemDataRole.UserRole)
        self._selected_artifact_id = str(artifact_id) if artifact_id else None
        self._artifact_id = self._selected_artifact_id
        self.artifact_summary.setPlainText(self._format_artifact_summary(self._selected_artifact_id))

    def _update_selection_details(
        self,
        *,
        select_run_id: str | None = None,
        select_artifact_id: str | None = None,
    ) -> None:
        run_id = select_run_id or self._run_id
        if not run_id:
            return

        run = self._app.runs.get_run(run_id)
        if run is None:
            self.run_summary.setPlainText(f"Run not found: {run_id}")
            return

        self._run_id = run.id
        self._version_id = run.dataset_version_id
        version = self._app.datasets.get_version(run.dataset_version_id)
        if version is not None:
            self._dataset_id = version.dataset_id
            self.class_names.setText(",".join(version.class_map))

        self.run_summary.setPlainText(self._format_run_summary(run))

        artifacts = sorted(self._artifacts.list_for_run(run.id), key=lambda item: item.created_at)
        evals = sorted(self._evals.list_for_run(run.id), key=lambda item: item.created_at)
        self._populate_artifact_list(artifacts, select_artifact_id=select_artifact_id)
        self._populate_eval_list(evals)

    def _populate_artifact_list(self, artifacts: list[object], *, select_artifact_id: str | None = None) -> None:
        artifact_id = select_artifact_id or self._artifact_id
        if not artifact_id and artifacts:
            artifact_id = artifacts[-1].id

        self.artifact_list.blockSignals(True)
        self.artifact_list.clear()
        selected_row = -1
        for index, artifact in enumerate(reversed(artifacts)):
            label = f"{artifact.id} [{artifact.consumer_hints.get('consumer', 'consumer?')}]"
            self.artifact_list.addItem(label)
            self.artifact_list.item(index).setData(Qt.ItemDataRole.UserRole, artifact.id)
            if artifact.id == artifact_id:
                selected_row = index
        self.artifact_list.blockSignals(False)

        if selected_row >= 0:
            self.artifact_list.setCurrentRow(selected_row)
            self._selected_artifact_id = artifact_id
            self._artifact_id = artifact_id
            self.artifact_summary.setPlainText(self._format_artifact_summary(artifact_id))
        else:
            self._selected_artifact_id = None
            self.artifact_summary.setPlainText("No artifacts yet.")

    def _populate_eval_list(self, evals: list[object]) -> None:
        self.eval_list.clear()
        if not evals:
            self.eval_summary.setPlainText("No eval reports yet.")
            return
        for report in reversed(evals):
            macro_f1 = report.metrics.get("macro_f1", report.aggregate_metrics.get("macro_f1", "n/a"))
            accuracy = report.metrics.get("accuracy", report.aggregate_metrics.get("accuracy", "n/a"))
            self.eval_list.addItem(f"{report.id} [{report.split_name}] macro_f1={macro_f1} accuracy={accuracy}")
        self.eval_summary.setPlainText(self._format_eval_summary(evals[-1]))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_run_spec(self) -> dict:
        if not self._version_id:
            raise ValueError("No dataset version selected")
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
                "seed": int(self.seed.value()),
            },
        }

    def _build_artifact_manifest(self) -> dict:
        run = self._app.runs.get_run(self._require_run_id())
        if run is None:
            raise ValueError("Run not found")

        version = self._app.datasets.get_version(run.dataset_version_id)
        classes = [c.strip() for c in self.class_names.text().split(",") if c.strip()]
        if version is not None and not classes:
            classes = list(version.class_map)
        if not classes:
            raise ValueError("At least one class is required")

        return {
            "weightsPath": "model.pth",
            "classes": classes,
            "classificationMode": "multiclass",
            "inferencePreprocessing": dict(run.spec.get("data", {})),
        }

    def _require_run_id(self) -> str:
        if not self._run_id:
            raise ValueError("No run selected")
        return self._run_id

    def _format_workspace_summary(self, datasets: list[object], runs: list[object]) -> str:
        latest_run = runs[-1] if runs else None
        latest_run_line = "none"
        if latest_run is not None:
            latest_run_line = f"{latest_run.id} ({latest_run.status.value})"
        return "\n".join(
            [
                f"Root: {self._root}",
                f"Datasets: {len(datasets)}",
                f"Runs: {len(runs)}",
                f"Latest run: {latest_run_line}",
                f"State dir: {self._root / 'foundry' / 'state'}",
            ]
        )

    def _format_dataset_summary(self) -> str:
        if not self._dataset_id:
            return "No dataset loaded yet."
        dataset = self._app.datasets.get_dataset(self._dataset_id)
        versions = self._app.datasets.list_versions(self._dataset_id)
        version = self._app.datasets.get_version(self._version_id) if self._version_id else (versions[-1] if versions else None)
        if dataset is None:
            return f"Dataset not found: {self._dataset_id}"

        lines = [
            f"Dataset: {dataset.name}",
            f"Dataset ID: {dataset.id}",
            f"Source: {dataset.source_ref or '(not set)'}",
            f"Versions: {len(versions)}",
        ]
        if version is None:
            return "\n".join(lines + ["Current version: none"])

        split_plan = version.split_plan or {}
        lines.extend(
            [
                f"Current version: {version.id}",
                f"Classes: {', '.join(version.class_map) or '(none)'}",
                f"Samples: {version.sample_count}",
                f"Planned train/val/test: "
                f"{len(split_plan.get('train_ids', []))}/"
                f"{len(split_plan.get('val_ids', []))}/"
                f"{len(split_plan.get('test_ids', []))}",
            ]
        )
        return "\n".join(lines)

    def _format_run_summary(self, run: object) -> str:
        checkpoints = sorted(run.checkpoints_dir(self._root).glob("epoch_*.json"))
        exports_dir = run.exports_dir(self._root)
        metrics_path = exports_dir / "metrics.json"
        run_summary_path = exports_dir / "run_summary.json"

        lines = [
            f"Run ID: {run.id}",
            f"Status: {run.status.value}",
            f"Dataset version: {run.dataset_version_id}",
            f"Backend / Device: {run.backend} / {run.device}",
            f"Epochs: {run.spec.get('training', {}).get('epochs')}",
            f"Batch size: {run.spec.get('training', {}).get('batchSize')}",
            f"Learning rate: {run.spec.get('training', {}).get('learningRate')}",
            f"Checkpoints: {len(checkpoints)}",
            f"Exports dir: {exports_dir}",
            f"metrics.json: {'yes' if metrics_path.exists() else 'no'}",
            f"run_summary.json: {'yes' if run_summary_path.exists() else 'no'}",
        ]
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            final_metrics = (metrics.get("finalEval") or {}).get("metrics", {})
            if final_metrics:
                lines.append(
                    "Final eval: "
                    f"macro_f1={final_metrics.get('macro_f1', 'n/a')} "
                    f"accuracy={final_metrics.get('accuracy', 'n/a')}"
                )
        return "\n".join(lines)

    def _format_artifact_summary(self, artifact_id: str | None) -> str:
        if not artifact_id:
            return "No artifact selected."
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            return f"Artifact not found: {artifact_id}"

        validation_note = "Run Validate Selected Artifact to check compatibility."
        report = None
        try:
            report = self._app.artifacts.validate_compatibility(artifact.id)
        except Exception:
            report = None
        if report is not None:
            validation_note = f"Validation: ok={report.ok}, errors={len(report.errors)}, warnings={len(report.warnings)}"

        return "\n".join(
            [
                f"Artifact ID: {artifact.id}",
                f"Run ID: {artifact.run_id}",
                f"Manifest: {artifact.path}",
                f"Weights path: {artifact.manifest.get('weightsPath', 'n/a')}",
                f"Classes: {', '.join(artifact.manifest.get('classes', [])) or '(none)'}",
                f"Consumer: {artifact.consumer_hints.get('consumer', 'n/a')}",
                validation_note,
            ]
        )

    def _format_eval_summary(self, report: object) -> str:
        metrics = report.metrics or report.aggregate_metrics or {}
        return "\n".join(
            [
                f"Eval ID: {report.id}",
                f"Run ID: {report.run_id}",
                f"Split: {report.split_name}",
                f"macro_f1: {metrics.get('macro_f1', 'n/a')}",
                f"accuracy: {metrics.get('accuracy', 'n/a')}",
                f"Summary keys: {', '.join(sorted(report.summary.keys())) or '(none)'}",
            ]
        )

    def _on_activity(self, item) -> None:
        self.activity.appendPlainText(f"[{item.kind}] {item.message}")

    def _set_status(self, text: str) -> None:
        self.status_line.setText(text)
        self.activity.appendPlainText(f"[status] {text}")

    def _error(self, exc: Exception) -> None:
        message = str(exc)
        self._set_status(f"Error: {message}")
        self.activity.appendPlainText(f"[error] {message}")
        if self._show_error_dialogs:
            QMessageBox.critical(self, "Foundry Error", message)


def run_foundry_ui(root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = FoundryWindow(root or Path.cwd())
    window.show()
    return app.exec()
