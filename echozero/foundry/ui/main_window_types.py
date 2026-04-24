"""Typed Foundry UI row and selection helpers.
Exists to keep ad hoc `object`-typed table/list payloads out of the Foundry window.
Connects domain entities to stable Qt-facing view models and nullable item helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHeaderView, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem

from echozero.foundry.domain import Dataset, DatasetVersion, EvalReport, ModelArtifact, TrainRun


@dataclass(slots=True)
class DatasetSelectorRow:
    dataset: Dataset
    label: str


@dataclass(slots=True)
class DatasetVersionSelectorRow:
    version: DatasetVersion
    label: str


@dataclass(slots=True)
class RunTableRow:
    run: TrainRun
    model_type: str
    epochs_label: str
    learning_rate_label: str
    updated_label: str
    artifact_summary: str

    @property
    def cells(self) -> tuple[str, ...]:
        return (
            self.run.id,
            str(self.run.status.value),
            str(self.run.dataset_version_id),
            self.model_type,
            self.epochs_label,
            self.learning_rate_label,
            self.updated_label,
            self.artifact_summary,
        )


@dataclass(slots=True)
class QueueRunRow:
    run: TrainRun
    label: str


@dataclass(slots=True)
class ArtifactListRow:
    artifact: ModelArtifact
    label: str


@dataclass(slots=True)
class EvalListRow:
    report: EvalReport
    label: str


def require_horizontal_header(table: QTableWidget) -> QHeaderView:
    header = table.horizontalHeader()
    if header is None:
        raise RuntimeError("Foundry run table is missing a horizontal header")
    return header


def table_item_or_none(
    table: QTableWidget,
    row: int,
    column: int,
) -> QTableWidgetItem | None:
    return table.item(row, column)


def selected_table_row(table: QTableWidget) -> int | None:
    selection_model = table.selectionModel()
    if selection_model is None:
        return None
    selected_rows = selection_model.selectedRows()
    if not selected_rows:
        return None
    return selected_rows[0].row()


def selected_list_run_id(list_widget: QListWidget) -> str | None:
    return list_item_user_role(list_widget.currentItem())


def list_item_user_role(item: QListWidgetItem | None) -> str | None:
    if item is None:
        return None
    value = item.data(Qt.ItemDataRole.UserRole)
    return str(value) if value else None


__all__ = [
    "ArtifactListRow",
    "DatasetSelectorRow",
    "DatasetVersionSelectorRow",
    "EvalListRow",
    "QueueRunRow",
    "RunTableRow",
    "list_item_user_role",
    "require_horizontal_header",
    "selected_list_run_id",
    "selected_table_row",
    "table_item_or_none",
]
