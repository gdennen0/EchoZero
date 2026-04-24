"""Dataset tab and selection helpers for the Foundry window.
Exists to keep dataset ingest and version-planning concerns out of the main window shell.
Connects dataset services to the dataset tab widgets and summary surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from echozero.foundry import FoundryApp
from echozero.foundry.domain import Dataset
from echozero.foundry.ui.main_window_types import (
    DatasetSelectorRow,
    DatasetVersionSelectorRow,
)
from echozero.ui.style import SHELL_TOKENS


class FoundryWindowDatasetMixin:
    _root: Path
    _app: FoundryApp
    _dataset_id: str | None
    _version_id: str | None
    _run_id: str | None

    dataset_name: QLineEdit
    dataset_folder: QLineEdit
    dataset_selector: QComboBox
    version_selector: QComboBox
    val_split: QDoubleSpinBox
    test_split: QDoubleSpinBox
    seed: QSpinBox
    balance: QLineEdit
    class_names: QLineEdit
    dataset_summary: QPlainTextEdit

    _set_status: Callable[[str], None]
    _refresh_workspace_state: Callable[..., None]
    _error: Callable[[Exception], None]

    def _build_dataset_box(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setSpacing(SHELL_TOKENS.scales.layout_gap)

        form = QGroupBox("Dataset Import")
        grid = QGridLayout(form)
        grid.setHorizontalSpacing(SHELL_TOKENS.scales.layout_gap)
        grid.setVerticalSpacing(SHELL_TOKENS.scales.inline_gap)

        self.dataset_name = QLineEdit("Drums")
        self.dataset_folder = QLineEdit("")
        self.dataset_selector = QComboBox()
        self.dataset_selector.currentIndexChanged.connect(self._on_dataset_selected)
        self.version_selector = QComboBox()
        self.version_selector.currentIndexChanged.connect(self._on_version_selected)
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

        grid.addWidget(QLabel("Dataset"), 0, 0)
        grid.addWidget(self.dataset_selector, 0, 1, 1, 3)
        grid.addWidget(QLabel("Version"), 1, 0)
        grid.addWidget(self.version_selector, 1, 1, 1, 3)
        grid.addWidget(QLabel("Name"), 2, 0)
        grid.addWidget(self.dataset_name, 2, 1, 1, 3)
        grid.addWidget(QLabel("Folder"), 3, 0)
        grid.addWidget(self.dataset_folder, 3, 1, 1, 2)
        grid.addWidget(browse_btn, 3, 3)
        grid.addWidget(ingest_btn, 4, 3)
        grid.addWidget(QLabel("Val Split"), 5, 0)
        grid.addWidget(self.val_split, 5, 1)
        grid.addWidget(QLabel("Test Split"), 5, 2)
        grid.addWidget(self.test_split, 5, 3)
        grid.addWidget(QLabel("Seed"), 6, 0)
        grid.addWidget(self.seed, 6, 1)
        grid.addWidget(QLabel("Balance"), 6, 2)
        grid.addWidget(self.balance, 6, 3)
        grid.addWidget(plan_btn, 7, 3)

        self.dataset_summary = QPlainTextEdit()
        self.dataset_summary.setReadOnly(True)
        self.dataset_summary.setPlaceholderText("Dataset and version details will appear here.")

        layout.addWidget(form)
        layout.addWidget(QLabel("Current Dataset"))
        layout.addWidget(self.dataset_summary, stretch=1)
        return box

    def _pick_dataset_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            cast(QWidget, self),
            "Select dataset folder",
            str(self._root),
        )
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
            self._set_status(
                f"Dataset ready: {dataset.id} -> {version.id} ({len(version.samples)} samples)"
            )
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

    def _load_defaults(self) -> None:
        candidate = self._root / "data" / "drum-oneshots"
        if candidate.exists() and candidate.is_dir():
            self.dataset_folder.setText(str(candidate))
            self.dataset_name.setText(candidate.name.replace("-", " ").title())

    def _populate_dataset_selectors(self, datasets: list[Dataset]) -> None:
        selected_dataset_id = self._dataset_id
        if selected_dataset_id and all(dataset.id != selected_dataset_id for dataset in datasets):
            selected_dataset_id = None
        if selected_dataset_id is None and datasets:
            selected_dataset_id = datasets[-1].id
        self._dataset_id = selected_dataset_id

        dataset_rows = [
            DatasetSelectorRow(dataset=dataset, label=f"{dataset.name} ({dataset.id})")
            for dataset in datasets
        ]

        self.dataset_selector.blockSignals(True)
        self.dataset_selector.clear()
        for row in dataset_rows:
            self.dataset_selector.addItem(row.label, row.dataset.id)
        if selected_dataset_id:
            index = self.dataset_selector.findData(selected_dataset_id)
            if index >= 0:
                self.dataset_selector.setCurrentIndex(index)
        self.dataset_selector.blockSignals(False)

        versions = (
            self._app.datasets.list_versions(selected_dataset_id) if selected_dataset_id else []
        )
        if self._version_id and all(version.id != self._version_id for version in versions):
            self._version_id = None
        if self._version_id is None and versions:
            self._version_id = versions[-1].id

        version_rows: list[DatasetVersionSelectorRow] = [
            DatasetVersionSelectorRow(
                version=version,
                label=(
                    f"v{version.version} [{version.id}] "
                    f"samples={version.sample_count} "
                    f"train/val/test={len((version.split_plan or {}).get('train_ids', []))}/"
                    f"{len((version.split_plan or {}).get('val_ids', []))}/"
                    f"{len((version.split_plan or {}).get('test_ids', []))}"
                ),
            )
            for version in versions
        ]

        self.version_selector.blockSignals(True)
        self.version_selector.clear()
        for version_row in version_rows:
            self.version_selector.addItem(version_row.label, version_row.version.id)
        if self._version_id:
            index = self.version_selector.findData(self._version_id)
            if index >= 0:
                self.version_selector.setCurrentIndex(index)
        self.version_selector.blockSignals(False)

    def _on_dataset_selected(self, index: int) -> None:
        dataset_id = self.dataset_selector.itemData(index)
        self._dataset_id = str(dataset_id) if dataset_id else None
        self._version_id = None
        self._refresh_workspace_state()
        dataset = self._app.datasets.get_dataset(self._dataset_id) if self._dataset_id else None
        if dataset is not None:
            self.dataset_name.setText(dataset.name)
            self.dataset_folder.setText(dataset.source_ref or "")

    def _on_version_selected(self, index: int) -> None:
        version_id = self.version_selector.itemData(index)
        self._version_id = str(version_id) if version_id else None
        version = self._app.datasets.get_version(self._version_id) if self._version_id else None
        if version is not None:
            self.class_names.setText(",".join(version.class_map))
        self.dataset_summary.setPlainText(self._format_dataset_summary())

    def _format_dataset_summary(self) -> str:
        if not self._dataset_id:
            return "No dataset loaded yet."
        dataset = self._app.datasets.get_dataset(self._dataset_id)
        versions = self._app.datasets.list_versions(self._dataset_id)
        version = (
            self._app.datasets.get_version(self._version_id)
            if self._version_id
            else (versions[-1] if versions else None)
        )
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
