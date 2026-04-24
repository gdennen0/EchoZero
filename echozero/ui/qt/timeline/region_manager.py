"""Timeline region manager dialog for Stage Zero editing workflows.
Exists to provide one operator-facing surface for create, update, and delete region spans.
Connects timeline presentation regions to canonical region intents through widget dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import RegionId


@dataclass(slots=True)
class RegionDraft:
    region_id: RegionId | None
    start: float
    end: float
    label: str
    color: str | None = None
    kind: str = "custom"


class RegionManagerDialog(QDialog):
    """Edit timeline regions as a flat list of labeled time spans."""

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Region Manager")
        self.resize(760, 420)

        self._rows: list[RegionDraft] = [
            RegionDraft(
                region_id=region.region_id,
                start=float(region.start),
                end=float(region.end),
                label=region.label,
                color=region.color,
                kind=region.kind,
            )
            for region in presentation.regions
        ]
        self._loading_editors = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Manage timeline regions", self)
        title.setProperty("timelineToolbarLabel", True)
        root.addWidget(title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        root.addLayout(row, 1)

        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Label", "Start", "End", "Color", "Kind"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_row_selection_changed)
        row.addWidget(self._table, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        row.addLayout(side)

        self._add_button = QPushButton("Add Region", self)
        self._add_button.clicked.connect(self._on_add_region)
        side.addWidget(self._add_button)

        self._delete_button = QPushButton("Delete Region", self)
        self._delete_button.clicked.connect(self._on_delete_region)
        side.addWidget(self._delete_button)

        side.addSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        side.addLayout(form)

        self._label_input = QLineEdit(self)
        self._label_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Label", self._label_input)

        self._start_input = QDoubleSpinBox(self)
        self._start_input.setDecimals(3)
        self._start_input.setRange(0.0, 24 * 60 * 60)
        self._start_input.valueChanged.connect(self._apply_editor_values)
        form.addRow("Start (s)", self._start_input)

        self._end_input = QDoubleSpinBox(self)
        self._end_input.setDecimals(3)
        self._end_input.setRange(0.001, 24 * 60 * 60)
        self._end_input.valueChanged.connect(self._apply_editor_values)
        form.addRow("End (s)", self._end_input)

        self._color_input = QLineEdit(self)
        self._color_input.setPlaceholderText("#RRGGBB (optional)")
        self._color_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Color", self._color_input)

        self._kind_input = QLineEdit(self)
        self._kind_input.setPlaceholderText("custom")
        self._kind_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Kind", self._kind_input)

        side.addStretch(1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._refresh_table(select_row=0 if self._rows else None)

    def region_drafts(self) -> list[RegionDraft]:
        """Return normalized region edits in timeline display order."""

        normalized = [self._normalized_row(row) for row in self._rows]
        return sorted(
            normalized,
            key=lambda row: (
                float(row.start),
                float(row.end),
                row.label.lower(),
                str(row.region_id or ""),
            ),
        )

    def _on_add_region(self) -> None:
        start, end = self._next_region_range()
        self._rows.append(
            RegionDraft(
                region_id=None,
                start=start,
                end=end,
                label=f"Region {len(self._rows) + 1}",
            )
        )
        self._refresh_table(select_row=len(self._rows) - 1)

    def _on_delete_region(self) -> None:
        row_index = self._selected_row_index()
        if row_index is None:
            return
        del self._rows[row_index]
        next_index = min(row_index, len(self._rows) - 1) if self._rows else None
        self._refresh_table(select_row=next_index)

    def _on_row_selection_changed(self) -> None:
        row_index = self._selected_row_index()
        if row_index is None:
            self._set_editors_enabled(False)
            return
        self._set_editors_enabled(True)
        self._load_row_into_editors(self._rows[row_index])

    def _apply_editor_values(self) -> None:
        if self._loading_editors:
            return
        row_index = self._selected_row_index()
        if row_index is None:
            return
        row = self._rows[row_index]
        row.label = self._label_input.text().strip() or "Region"
        row.start = float(max(0.0, self._start_input.value()))
        row.end = float(max(row.start + 0.001, self._end_input.value()))
        color = self._color_input.text().strip()
        row.color = color or None
        row.kind = self._kind_input.text().strip().lower() or "custom"
        self._refresh_table(select_row=row_index)

    def _refresh_table(self, *, select_row: int | None) -> None:
        self._table.setRowCount(len(self._rows))
        for row_index, row in enumerate(self._rows):
            self._set_table_item(row_index, 0, row.label)
            self._set_table_item(row_index, 1, f"{row.start:.3f}")
            self._set_table_item(row_index, 2, f"{row.end:.3f}")
            self._set_table_item(row_index, 3, row.color or "auto")
            self._set_table_item(row_index, 4, row.kind)
        self._table.resizeColumnsToContents()
        if select_row is None or not self._rows:
            self._table.clearSelection()
            self._set_editors_enabled(False)
            return
        safe_index = max(0, min(select_row, len(self._rows) - 1))
        self._table.selectRow(safe_index)
        self._set_editors_enabled(True)
        self._load_row_into_editors(self._rows[safe_index])

    def _load_row_into_editors(self, row: RegionDraft) -> None:
        self._loading_editors = True
        try:
            self._label_input.setText(row.label)
            self._start_input.setValue(float(row.start))
            self._end_input.setValue(float(row.end))
            self._color_input.setText(row.color or "")
            self._kind_input.setText(row.kind)
        finally:
            self._loading_editors = False

    def _set_editors_enabled(self, enabled: bool) -> None:
        self._delete_button.setEnabled(enabled)
        self._label_input.setEnabled(enabled)
        self._start_input.setEnabled(enabled)
        self._end_input.setEnabled(enabled)
        self._color_input.setEnabled(enabled)
        self._kind_input.setEnabled(enabled)

    def _set_table_item(self, row: int, column: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, column, item)

    def _selected_row_index(self) -> int | None:
        indexes = self._table.selectionModel().selectedRows()
        if not indexes:
            return None
        return int(indexes[0].row())

    def _next_region_range(self) -> tuple[float, float]:
        if not self._rows:
            return 0.0, 1.0
        latest = max(self._rows, key=lambda row: (float(row.end), float(row.start)))
        start = float(max(0.0, latest.end))
        return start, start + 1.0

    @staticmethod
    def _normalized_row(row: RegionDraft) -> RegionDraft:
        label = row.label.strip() or "Region"
        start = max(0.0, float(row.start))
        end = max(start + 0.001, float(row.end))
        color = row.color.strip() if isinstance(row.color, str) else ""
        kind = row.kind.strip().lower() or "custom"
        return RegionDraft(
            region_id=row.region_id,
            start=start,
            end=end,
            label=label,
            color=color or None,
            kind=kind,
        )
__all__ = ["RegionDraft", "RegionManagerDialog"]
