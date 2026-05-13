"""Qt model manager for app-installed EchoZero runtime models.
Exists so operators can inspect, download, import, and validate models outside the app bundle.
Connects central model registry installs to the local ~/.echozero/models store.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.models.distribution import (
    default_registry_manifest_source,
    discover_registry_models,
    import_local_model_bundle,
    install_model_from_registry,
    list_installed_models,
    save_registry_manifest_source,
    validate_installed_model,
)
from echozero.models.paths import ensure_installed_models_dir
from echozero.ui.style.qt import ensure_qt_theme_installed


class ModelManagerDialog(QDialog):
    """Modal manager for mutable app-installed model assets."""

    def __init__(
        self,
        *,
        models_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        ensure_qt_theme_installed()
        self.setObjectName("modelManagerDialog")
        self.setWindowTitle("Model Manager")
        self.resize(820, 520)
        self._models_dir = models_dir or ensure_installed_models_dir()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        self._root_label = QLabel(f"Models: {self._models_dir}", self)
        self._root_label.setWordWrap(True)
        layout.addWidget(self._root_label)

        source_layout = QHBoxLayout()
        self._registry_source = QLineEdit(self)
        self._registry_source.setPlaceholderText("Registry manifest URL or local path")
        self._registry_source.setText(default_registry_manifest_source(self._models_dir) or "")
        source_layout.addWidget(self._registry_source, 1)
        self._save_source_button = QPushButton("Save Registry", self)
        self._save_source_button.clicked.connect(self._save_registry_source)
        source_layout.addWidget(self._save_source_button)
        layout.addLayout(source_layout)

        self._table = QTableWidget(0, 7, self)
        self._table.setHorizontalHeaderLabels(
            ["Model", "Version", "Label", "Type", "Runtime", "Status", "Source"]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._table, 1)

        actions = QHBoxLayout()
        self._discover_button = QPushButton("Discover", self)
        self._discover_button.clicked.connect(self._render_models)
        actions.addWidget(self._discover_button)
        self._install_button = QPushButton("Install Selected", self)
        self._install_button.clicked.connect(self._install_selected)
        actions.addWidget(self._install_button)
        self._import_button = QPushButton("Import Local Bundle", self)
        self._import_button.clicked.connect(self._import_local_bundle)
        actions.addWidget(self._import_button)
        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.clicked.connect(self._render_models)
        actions.addWidget(self._refresh_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._render_models()

    def _render_models(self) -> None:
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        source = self._registry_source.text().strip()
        if source:
            try:
                listings = discover_registry_models(
                    manifest_source=source,
                    models_dir=self._models_dir,
                )
            except Exception as exc:
                QMessageBox.warning(self, "Model Registry", str(exc))
                listings = ()
            for listing in listings:
                entry = listing.entry
                rows.append(
                    (
                        entry.model_id,
                        entry.version,
                        entry.label,
                        entry.model_type,
                        entry.runtime_consumer or "",
                        listing.state.value,
                        "registry",
                    )
                )
        installed_ids = {row[0] for row in rows}
        for record in list_installed_models(self._models_dir):
            if record.model_id in installed_ids:
                continue
            status = (
                "Ready"
                if validate_installed_model(record, models_dir=self._models_dir)
                else "Invalid"
            )
            rows.append(
                (
                    record.model_id,
                    record.version,
                    record.label,
                    record.model_type,
                    record.runtime_consumer or "",
                    status.lower(),
                    "local",
                )
            )
        self._table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self._table.setItem(row, column, QTableWidgetItem(value))
        self._table.resizeColumnsToContents()

    def _save_registry_source(self) -> None:
        source = self._registry_source.text().strip()
        if not source:
            return
        try:
            save_registry_manifest_source(source, models_dir=self._models_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Model Registry", str(exc))
            return
        self._render_models()

    def _install_selected(self) -> None:
        row = self._table.currentRow()
        source = self._registry_source.text().strip()
        if row < 0 or not source:
            model_id, ok = QInputDialog.getText(self, "Install Model", "Model id")
            if not ok or not model_id.strip():
                return
            model_id_text = model_id.strip()
        else:
            item = self._table.item(row, 0)
            if item is None or not item.text().strip():
                return
            model_id_text = item.text().strip()
        try:
            install_model_from_registry(
                model_id=model_id_text,
                manifest_source=source,
                models_dir=self._models_dir,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Model Install Failed", str(exc))
            return
        self._render_models()

    def _import_local_bundle(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Import Local Model Bundle",
            str(Path.home()),
        )
        if not selected:
            return
        model_id, ok = QInputDialog.getText(self, "Import Model", "Model id")
        if not ok or not model_id.strip():
            return
        model_type, ok = QInputDialog.getText(self, "Import Model", "Model type")
        if not ok or not model_type.strip():
            return
        label, ok = QInputDialog.getText(self, "Import Model", "Label")
        if not ok or not label.strip():
            return
        version, ok = QInputDialog.getText(self, "Import Model", "Version")
        if not ok or not version.strip():
            return
        classes_text, ok = QInputDialog.getText(
            self,
            "Import Model",
            "Classes, comma-separated",
        )
        if not ok:
            return
        runtime_consumer, ok = QInputDialog.getText(
            self,
            "Import Model",
            "Runtime consumer",
            text="BinaryDrumClassify",
        )
        if not ok:
            return
        try:
            import_local_model_bundle(
                bundle_path=Path(selected),
                model_id=model_id.strip(),
                model_type=model_type.strip(),
                label=label.strip(),
                version=version.strip(),
                classes=tuple(
                    item.strip().lower()
                    for item in classes_text.split(",")
                    if item.strip()
                ),
                runtime_consumer=runtime_consumer.strip() or None,
                models_dir=self._models_dir,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Model Import Failed", str(exc))
            return
        self._render_models()
