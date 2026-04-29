"""Timeline sections manager dialog.
Exists to provide one operator-facing surface for cue-owned section editing.
Connects timeline presentation section cues to canonical section-edit intents.
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
from echozero.application.shared.ids import SectionCueId


@dataclass(slots=True)
class SectionCueDraft:
    cue_id: SectionCueId | None
    start: float
    cue_ref: str
    name: str
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


class SectionManagerDialog(QDialog):
    """Edit canonical section cues as a flat ordered list."""

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sections")
        self.resize(820, 440)

        self._rows: list[SectionCueDraft] = [
            SectionCueDraft(
                cue_id=cue.cue_id,
                start=float(cue.start),
                cue_ref=cue.cue_ref,
                name=cue.name,
                color=cue.color,
                notes=cue.notes,
                payload_ref=cue.payload_ref,
            )
            for cue in presentation.section_cues
        ]
        self._loading_editors = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel("Manage section cues", self)
        title.setProperty("timelineToolbarLabel", True)
        root.addWidget(title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        root.addLayout(row, 1)

        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Cue Ref", "Name", "Start", "Color", "Notes"])
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

        self._add_button = QPushButton("Add Section", self)
        self._add_button.clicked.connect(self._on_add_section)
        side.addWidget(self._add_button)

        self._delete_button = QPushButton("Delete Section", self)
        self._delete_button.clicked.connect(self._on_delete_section)
        side.addWidget(self._delete_button)

        side.addSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        side.addLayout(form)

        self._cue_ref_input = QLineEdit(self)
        self._cue_ref_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Cue Ref", self._cue_ref_input)

        self._name_input = QLineEdit(self)
        self._name_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Name", self._name_input)

        self._start_input = QDoubleSpinBox(self)
        self._start_input.setDecimals(3)
        self._start_input.setRange(0.0, 24 * 60 * 60)
        self._start_input.valueChanged.connect(self._apply_editor_values)
        form.addRow("Start (s)", self._start_input)

        self._color_input = QLineEdit(self)
        self._color_input.setPlaceholderText("#RRGGBB (optional)")
        self._color_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Color", self._color_input)

        self._notes_input = QLineEdit(self)
        self._notes_input.setPlaceholderText("Optional notes")
        self._notes_input.editingFinished.connect(self._apply_editor_values)
        form.addRow("Notes", self._notes_input)

        side.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_table(select_row=0 if self._rows else None)

    def section_cue_drafts(self) -> list[SectionCueDraft]:
        normalized = [self._normalized_row(row) for row in self._rows]
        return sorted(
            normalized,
            key=lambda row: (
                float(row.start),
                row.cue_ref.casefold(),
                str(row.cue_id or ""),
            ),
        )

    def _on_add_section(self) -> None:
        start = self._next_section_start()
        cue_number = len(self._rows) + 1
        self._rows.append(
            SectionCueDraft(
                cue_id=None,
                start=start,
                cue_ref=f"Q{cue_number}",
                name=f"Section {cue_number}",
            )
        )
        self._refresh_table(select_row=len(self._rows) - 1)

    def _on_delete_section(self) -> None:
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
            self._loading_editors = True
            try:
                self._cue_ref_input.clear()
                self._name_input.clear()
                self._start_input.setValue(0.0)
                self._color_input.clear()
                self._notes_input.clear()
            finally:
                self._loading_editors = False
            return
        row = self._rows[row_index]
        self._set_editors_enabled(True)
        self._loading_editors = True
        try:
            self._cue_ref_input.setText(row.cue_ref)
            self._name_input.setText(row.name)
            self._start_input.setValue(float(row.start))
            self._color_input.setText(row.color or "")
            self._notes_input.setText(row.notes or "")
        finally:
            self._loading_editors = False

    def _apply_editor_values(self) -> None:
        if self._loading_editors:
            return
        row_index = self._selected_row_index()
        if row_index is None:
            return
        current = self._rows[row_index]
        cue_ref = self._cue_ref_input.text().strip() or current.cue_ref or f"Q{row_index + 1}"
        name = self._name_input.text().strip() or cue_ref
        self._rows[row_index] = SectionCueDraft(
            cue_id=current.cue_id,
            start=max(0.0, float(self._start_input.value())),
            cue_ref=cue_ref,
            name=name,
            color=self._color_input.text().strip() or None,
            notes=self._notes_input.text().strip() or None,
            payload_ref=current.payload_ref,
        )
        self._refresh_table(select_row=row_index)

    def _refresh_table(self, *, select_row: int | None) -> None:
        self._loading_editors = True
        try:
            self._table.setRowCount(len(self._rows))
            for row_index, row in enumerate(self._rows):
                values = (
                    row.cue_ref,
                    row.name,
                    f"{float(row.start):.3f}",
                    row.color or "",
                    row.notes or "",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column == 2:
                        item.setTextAlignment(
                            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        )
                    self._table.setItem(row_index, column, item)
            if select_row is None:
                self._table.clearSelection()
            else:
                self._table.selectRow(max(0, min(select_row, len(self._rows) - 1)))
        finally:
            self._loading_editors = False
        self._on_row_selection_changed()

    def _set_editors_enabled(self, enabled: bool) -> None:
        self._cue_ref_input.setEnabled(enabled)
        self._name_input.setEnabled(enabled)
        self._start_input.setEnabled(enabled)
        self._color_input.setEnabled(enabled)
        self._notes_input.setEnabled(enabled)
        self._delete_button.setEnabled(enabled)

    def _selected_row_index(self) -> int | None:
        selected_rows = self._table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        return int(selected_rows[0].row())

    def _next_section_start(self) -> float:
        if not self._rows:
            return 0.0
        return max(0.0, float(max(row.start for row in self._rows)) + 8.0)

    @staticmethod
    def _normalized_row(row: SectionCueDraft) -> SectionCueDraft:
        cue_ref = str(row.cue_ref or "").strip() or "Q1"
        name = str(row.name or "").strip() or cue_ref
        color = str(row.color or "").strip() or None
        notes = str(row.notes or "").strip() or None
        payload_ref = str(row.payload_ref or "").strip() or None
        return SectionCueDraft(
            cue_id=row.cue_id,
            start=max(0.0, float(row.start)),
            cue_ref=cue_ref,
            name=name,
            color=color,
            notes=notes,
            payload_ref=payload_ref,
        )
