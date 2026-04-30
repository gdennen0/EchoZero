"""Timeline sections manager dialog.
Exists to provide one operator-facing surface for cue-owned section editing.
Connects timeline presentation section cues to canonical section-edit intents.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QItemSelectionModel, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.cue_numbers import (
    CueNumber,
    cue_number_from_ref_text,
    cue_number_text,
    parse_positive_cue_number,
)
from echozero.application.shared.ids import SectionCueId


@dataclass(slots=True)
class SectionCueDraft:
    cue_id: SectionCueId | None
    start: float
    cue_ref: str
    name: str
    cue_number: CueNumber | None = None
    color: str | None = None
    notes: str | None = None
    payload_ref: str | None = None


class SectionManagerDialog(QDialog):
    """Edit canonical section cues as a flat ordered list."""

    _QUICK_LABEL_OPTIONS: tuple[str, ...] = (
        "Intro",
        "Verse",
        "Chorus",
        "Bridge",
        "Pause",
        "Break",
        "Vocal",
        "End",
        "Outro",
        "FTB",
    )

    def __init__(
        self,
        presentation: TimelinePresentation,
        parent: QWidget | None = None,
        *,
        cues: list[SectionCueDraft] | None = None,
        worksheet_title: str | None = None,
        selected_cue_id: SectionCueId | None = None,
    ) -> None:
        super().__init__(parent)
        resolved_title = str(worksheet_title or "").strip() or "Section Cue Stack"
        self.setWindowTitle(resolved_title)
        self.resize(820, 440)

        self._rows: list[SectionCueDraft] = (
            [self._normalized_row(row) for row in list(cues or [])]
            if cues is not None
            else [
                SectionCueDraft(
                    cue_id=cue.cue_id,
                    start=float(cue.start),
                    cue_ref=cue.cue_ref,
                    name=cue.name,
                    cue_number=cue_number_from_ref_text(cue.cue_ref),
                    color=cue.color,
                    notes=cue.notes,
                    payload_ref=cue.payload_ref,
                )
                for cue in presentation.section_cues
            ]
        )
        self._loading_editors = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        title = QLabel(resolved_title, self)
        title.setProperty("timelineToolbarLabel", True)
        root.addWidget(title)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        root.addLayout(row, 1)

        self._table = QTableWidget(self)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Cue No", "Cue Ref", "Name", "Start", "Color", "Notes"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.SelectedClicked
            | QTableWidget.EditTrigger.EditKeyPressed
        )
        self._table.setAlternatingRowColors(True)
        self._table.itemSelectionChanged.connect(self._on_row_selection_changed)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._table.horizontalHeader().setStretchLastSection(True)
        row.addWidget(self._table, 1)

        side = QVBoxLayout()
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(8)
        row.addLayout(side)

        self._add_button = QPushButton("Add Section", self)
        self._add_button.clicked.connect(self._on_add_section)
        side.addWidget(self._add_button)

        self._add_before_button = QPushButton("Add Before", self)
        self._add_before_button.clicked.connect(lambda: self._insert_section_relative(before=True))
        side.addWidget(self._add_before_button)

        self._add_after_button = QPushButton("Add After", self)
        self._add_after_button.clicked.connect(lambda: self._insert_section_relative(before=False))
        side.addWidget(self._add_after_button)

        self._delete_button = QPushButton("Delete Section", self)
        self._delete_button.clicked.connect(self._on_delete_section)
        side.addWidget(self._delete_button)

        self._renumber_button = QPushButton("Renumber Cues", self)
        self._renumber_button.clicked.connect(self._on_renumber_cues)
        side.addWidget(self._renumber_button)

        self._renumber_from_button = QPushButton("Renumber From…", self)
        self._renumber_from_button.clicked.connect(self._on_renumber_cues_from_prompt)
        side.addWidget(self._renumber_from_button)

        side.addSpacing(10)

        quick_label_title = QLabel("Quick Labels", self)
        quick_label_title.setProperty("timelineToolbarLabel", True)
        side.addWidget(quick_label_title)

        quick_label_box = QWidget(self)
        quick_label_layout = QGridLayout(quick_label_box)
        quick_label_layout.setContentsMargins(0, 0, 0, 0)
        quick_label_layout.setHorizontalSpacing(6)
        quick_label_layout.setVerticalSpacing(6)
        for index, label in enumerate(self._QUICK_LABEL_OPTIONS):
            button = QPushButton(label, quick_label_box)
            button.clicked.connect(lambda _checked=False, text=label: self._apply_quick_label(text))
            quick_label_layout.addWidget(button, index // 2, index % 2)
        side.addWidget(quick_label_box)

        side.addSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        side.addLayout(form)

        self._cue_ref_input = QLineEdit(self)
        self._cue_ref_input.editingFinished.connect(lambda: self._apply_editor_field("cue_ref"))
        form.addRow("Cue Ref", self._cue_ref_input)

        self._cue_number_input = QLineEdit(self)
        self._cue_number_input.setPlaceholderText("e.g. 7 or 7.5")
        self._cue_number_input.editingFinished.connect(lambda: self._apply_editor_field("cue_number"))
        form.addRow("Cue No", self._cue_number_input)

        self._name_input = QLineEdit(self)
        self._name_input.editingFinished.connect(lambda: self._apply_editor_field("name"))
        form.addRow("Name", self._name_input)

        self._start_input = QDoubleSpinBox(self)
        self._start_input.setDecimals(3)
        self._start_input.setRange(0.0, 24 * 60 * 60)
        self._start_input.valueChanged.connect(lambda _value: self._apply_editor_field("start"))
        form.addRow("Start (s)", self._start_input)

        self._color_input = QLineEdit(self)
        self._color_input.setPlaceholderText("#RRGGBB (optional)")
        self._color_input.editingFinished.connect(lambda: self._apply_editor_field("color"))
        form.addRow("Color", self._color_input)

        self._notes_input = QLineEdit(self)
        self._notes_input.setPlaceholderText("Optional notes")
        self._notes_input.editingFinished.connect(lambda: self._apply_editor_field("notes"))
        form.addRow("Notes", self._notes_input)

        side.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        initial_row = 0 if self._rows else None
        if selected_cue_id is not None and self._rows:
            initial_row = next(
                (
                    index
                    for index, row in enumerate(self._rows)
                    if row.cue_id == selected_cue_id
                ),
                initial_row,
            )
        self._refresh_table(select_row=initial_row)

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
                cue_ref=self._cue_ref_from_number(cue_number),
                name=f"Section {cue_number}",
                cue_number=cue_number,
            )
        )
        self._refresh_table(select_row=len(self._rows) - 1)

    def _insert_section_relative(self, *, before: bool) -> None:
        selected_row = self._selected_row_index()
        if selected_row is None:
            self._on_add_section()
            return
        insert_at = selected_row if before else selected_row + 1
        start = self._suggest_insert_start(insert_at=insert_at)
        cue_number = self._suggest_insert_cue_number(insert_at=insert_at)
        cue_ref = self._cue_ref_from_value(cue_number)
        self._rows.insert(
            insert_at,
            SectionCueDraft(
                cue_id=None,
                start=start,
                cue_ref=cue_ref,
                name=f"Section {cue_number_text(cue_number) or insert_at + 1}",
                cue_number=cue_number,
            ),
        )
        self._refresh_table(select_row=insert_at)

    def _apply_quick_label(self, label: str) -> None:
        selected_rows = self._selected_row_indexes()
        if not selected_rows:
            return
        resolved = str(label or "").strip()
        if not resolved:
            return
        for row_index in selected_rows:
            current = self._rows[row_index]
            self._rows[row_index] = SectionCueDraft(
                cue_id=current.cue_id,
                start=float(current.start),
                cue_ref=current.cue_ref,
                name=resolved,
                cue_number=current.cue_number,
                color=current.color,
                notes=current.notes,
                payload_ref=current.payload_ref,
            )
        self._refresh_table(select_rows=selected_rows)

    def _on_renumber_cues(self) -> None:
        self._renumber_cues_from_start(1)

    def _on_renumber_cues_from_prompt(self) -> None:
        default_start = "1"
        selected_row = self._selected_row_index()
        if selected_row is not None and 0 <= selected_row < len(self._rows):
            default_start = cue_number_text(self._rows[selected_row].cue_number) or "1"
        raw_start, accepted = QInputDialog.getText(
            self,
            "Renumber Cues",
            "Start cue number",
            text=default_start,
        )
        if not accepted:
            return
        start_at = parse_positive_cue_number(raw_start)
        if start_at is None:
            QMessageBox.warning(
                self,
                "Invalid Cue Number",
                "Start cue number must be positive numeric data (for example: 7 or 7.5).",
            )
            return
        self._renumber_cues_from_start(start_at)

    def _renumber_cues_from_start(self, start_at: CueNumber) -> None:
        if not self._rows:
            return
        selected_row = self._selected_row_index()
        selected_cue_id = (
            self._rows[selected_row].cue_id
            if selected_row is not None and 0 <= selected_row < len(self._rows)
            else None
        )
        ordered_rows = sorted(
            self._rows,
            key=lambda row: (
                float(row.start),
                str(row.cue_id or ""),
                row.cue_ref.casefold(),
            ),
        )
        renumbered: list[SectionCueDraft] = []
        start_value = float(start_at)
        for index, row in enumerate(ordered_rows):
            cue_number_value = parse_positive_cue_number(start_value + float(index))
            if cue_number_value is None:
                cue_number_value = 1
            renumbered.append(
                SectionCueDraft(
                    cue_id=row.cue_id,
                    start=float(row.start),
                    cue_ref=self._cue_ref_from_value(cue_number_value),
                    name=row.name,
                    cue_number=cue_number_value,
                    color=row.color,
                    notes=row.notes,
                    payload_ref=row.payload_ref,
                )
            )
        self._rows = renumbered
        next_selected_index = 0 if self._rows else None
        if selected_cue_id is not None:
            next_selected_index = next(
                (
                    index
                    for index, row in enumerate(self._rows)
                    if row.cue_id == selected_cue_id
                ),
                next_selected_index,
            )
        self._refresh_table(select_row=next_selected_index)

    def _on_delete_section(self) -> None:
        selected_rows = self._selected_row_indexes()
        if not selected_rows:
            return
        for row_index in sorted(selected_rows, reverse=True):
            del self._rows[row_index]
        if not self._rows:
            self._refresh_table(select_row=None)
            return
        anchor_index = min(selected_rows[0], len(self._rows) - 1)
        next_index = max(0, anchor_index)
        self._refresh_table(select_row=next_index)

    def _on_row_selection_changed(self) -> None:
        selected_rows = self._selected_row_indexes()
        if not selected_rows:
            self._set_editors_enabled(False)
            self._loading_editors = True
            try:
                self._cue_ref_input.clear()
                self._cue_number_input.clear()
                self._name_input.clear()
                self._start_input.setValue(0.0)
                self._color_input.clear()
                self._notes_input.clear()
            finally:
                self._loading_editors = False
            return
        row_values = [self._rows[index] for index in selected_rows]
        row = row_values[0]

        def _common_text(values: list[str]) -> str:
            if not values:
                return ""
            first = values[0]
            return first if all(value == first for value in values[1:]) else ""

        self._set_editors_enabled(True)
        self._loading_editors = True
        try:
            self._cue_ref_input.setText(_common_text([candidate.cue_ref for candidate in row_values]))
            self._cue_number_input.setText(
                _common_text([cue_number_text(candidate.cue_number) or "" for candidate in row_values])
            )
            self._name_input.setText(_common_text([candidate.name for candidate in row_values]))
            start_values = [float(candidate.start) for candidate in row_values]
            self._start_input.setValue(start_values[0])
            self._color_input.setText(_common_text([(candidate.color or "") for candidate in row_values]))
            self._notes_input.setText(_common_text([(candidate.notes or "") for candidate in row_values]))
        finally:
            self._loading_editors = False

    def _apply_editor_field(self, field: str) -> None:
        if self._loading_editors:
            return
        selected_rows = self._selected_row_indexes()
        if not selected_rows:
            return
        multi_select = len(selected_rows) > 1
        for row_index in selected_rows:
            current = self._rows[row_index]
            cue_ref = current.cue_ref
            cue_number = current.cue_number
            name = current.name
            start = float(current.start)
            color = current.color
            notes = current.notes

            if field == "cue_ref":
                raw = self._cue_ref_input.text().strip()
                if raw:
                    cue_ref = raw
                elif not multi_select:
                    cue_ref = current.cue_ref or self._cue_ref_from_number(row_index + 1)
            elif field == "cue_number":
                raw = self._cue_number_input.text().strip()
                parsed = parse_positive_cue_number(raw)
                if raw and parsed is not None:
                    cue_number = parsed
                elif raw and parsed is None:
                    cue_number = current.cue_number
                elif not multi_select:
                    cue_number = None
            elif field == "name":
                raw = self._name_input.text().strip()
                if raw:
                    name = raw
                elif not multi_select:
                    name = cue_ref
            elif field == "start":
                start = max(0.0, float(self._start_input.value()))
            elif field == "color":
                raw = self._color_input.text().strip()
                if raw:
                    color = raw
                elif not multi_select:
                    color = None
            elif field == "notes":
                raw = self._notes_input.text().strip()
                if raw:
                    notes = raw
                elif not multi_select:
                    notes = None
            else:
                return

            self._rows[row_index] = SectionCueDraft(
                cue_id=current.cue_id,
                start=start,
                cue_ref=cue_ref,
                name=name,
                cue_number=cue_number,
                color=color,
                notes=notes,
                payload_ref=current.payload_ref,
            )
        self._refresh_table(select_rows=selected_rows)

    def _on_table_item_changed(self, item: QTableWidgetItem | None) -> None:
        if self._loading_editors or item is None:
            return
        row_index = int(item.row())
        if row_index < 0 or row_index >= len(self._rows):
            return
        column = int(item.column())
        current = self._rows[row_index]
        raw_value = str(item.text() or "").strip()

        cue_number = current.cue_number
        cue_ref = current.cue_ref
        name = current.name
        start = float(current.start)
        color = current.color
        notes = current.notes

        if column == 0:
            parsed = parse_positive_cue_number(raw_value)
            cue_number = parsed if parsed is not None else cue_number
        elif column == 1:
            cue_ref = raw_value or cue_ref
        elif column == 2:
            name = raw_value or cue_ref
        elif column == 3:
            try:
                start = max(0.0, float(raw_value))
            except ValueError:
                start = float(current.start)
        elif column == 4:
            color = raw_value or None
        elif column == 5:
            notes = raw_value or None
        else:
            return

        self._rows[row_index] = SectionCueDraft(
            cue_id=current.cue_id,
            start=start,
            cue_ref=cue_ref,
            name=name,
            cue_number=cue_number,
            color=color,
            notes=notes,
            payload_ref=current.payload_ref,
        )
        self._refresh_table(select_row=row_index)

    def _refresh_table(
        self,
        *,
        select_row: int | None = None,
        select_rows: list[int] | None = None,
    ) -> None:
        self._loading_editors = True
        try:
            self._table.setRowCount(len(self._rows))
            for row_index, row in enumerate(self._rows):
                values = (
                    cue_number_text(row.cue_number) or "",
                    row.cue_ref,
                    row.name,
                    f"{float(row.start):.3f}",
                    row.color or "",
                    row.notes or "",
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column in {0, 3}:
                        item.setTextAlignment(
                            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        )
                    self._table.setItem(row_index, column, item)
            if select_rows is not None:
                self._table.clearSelection()
                selection_model = self._table.selectionModel()
                if selection_model is not None:
                    for row in sorted(set(select_rows)):
                        if row < 0 or row >= len(self._rows):
                            continue
                        index = self._table.model().index(row, 0)
                        selection_model.select(
                            index,
                            QItemSelectionModel.SelectionFlag.Select
                            | QItemSelectionModel.SelectionFlag.Rows,
                        )
            elif select_row is None:
                self._table.clearSelection()
            else:
                self._table.selectRow(max(0, min(select_row, len(self._rows) - 1)))
        finally:
            self._loading_editors = False
        self._on_row_selection_changed()

    def _set_editors_enabled(self, enabled: bool) -> None:
        self._cue_ref_input.setEnabled(enabled)
        self._cue_number_input.setEnabled(enabled)
        self._name_input.setEnabled(enabled)
        self._start_input.setEnabled(enabled)
        self._color_input.setEnabled(enabled)
        self._notes_input.setEnabled(enabled)
        self._delete_button.setEnabled(enabled)
        self._add_before_button.setEnabled(enabled)
        self._add_after_button.setEnabled(enabled)
        self._renumber_button.setEnabled(bool(self._rows))
        self._renumber_from_button.setEnabled(bool(self._rows))

    def _selected_row_index(self) -> int | None:
        selected_rows = self._selected_row_indexes()
        if not selected_rows:
            return None
        return int(selected_rows[0])

    def _selected_row_indexes(self) -> list[int]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return []
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            return []
        return sorted({int(index.row()) for index in selected_rows})

    def _next_section_start(self) -> float:
        if not self._rows:
            return 0.0
        return max(0.0, float(max(row.start for row in self._rows)) + 8.0)

    def _suggest_insert_start(self, *, insert_at: int) -> float:
        previous_start = (
            float(self._rows[insert_at - 1].start)
            if insert_at - 1 >= 0 and insert_at - 1 < len(self._rows)
            else None
        )
        next_start = (
            float(self._rows[insert_at].start)
            if insert_at >= 0 and insert_at < len(self._rows)
            else None
        )
        if previous_start is not None and next_start is not None:
            midpoint = (previous_start + next_start) * 0.5
            if midpoint <= previous_start:
                return round(previous_start + 0.01, 3)
            return round(midpoint, 3)
        if previous_start is not None:
            return round(previous_start + 8.0, 3)
        if next_start is not None:
            return round(max(0.0, next_start - 8.0), 3)
        return 0.0

    def _suggest_insert_cue_number(self, *, insert_at: int) -> CueNumber:
        previous = (
            parse_positive_cue_number(self._rows[insert_at - 1].cue_number)
            if insert_at - 1 >= 0 and insert_at - 1 < len(self._rows)
            else None
        )
        next_value = (
            parse_positive_cue_number(self._rows[insert_at].cue_number)
            if insert_at >= 0 and insert_at < len(self._rows)
            else None
        )
        candidate: CueNumber
        if previous is not None and next_value is not None:
            prev_float = float(previous)
            next_float = float(next_value)
            if next_float > prev_float:
                candidate = (prev_float + next_float) * 0.5
            else:
                candidate = prev_float + 1.0
        elif previous is not None:
            candidate = float(previous) + 1.0
        elif next_value is not None:
            next_float = float(next_value)
            candidate = next_float - 1.0 if next_float > 1.0 else 1.0
        else:
            candidate = float(len(self._rows) + 1)
        if float(candidate) < 1.0:
            candidate = 1.0
        normalized = parse_positive_cue_number(candidate)
        if normalized is None:
            return 1
        return normalized

    @staticmethod
    def _normalized_row(row: SectionCueDraft) -> SectionCueDraft:
        parsed_cue_number = parse_positive_cue_number(row.cue_number)
        if parsed_cue_number is None:
            parsed_cue_number = cue_number_from_ref_text(row.cue_ref)
        cue_ref = str(row.cue_ref or "").strip()
        if not cue_ref:
            cue_ref = SectionManagerDialog._cue_ref_from_number(
                int(parsed_cue_number) if parsed_cue_number is not None else 1
            )
        name = str(row.name or "").strip() or cue_ref
        color = str(row.color or "").strip() or None
        notes = str(row.notes or "").strip() or None
        payload_ref = str(row.payload_ref or "").strip() or None
        return SectionCueDraft(
            cue_id=row.cue_id,
            start=max(0.0, float(row.start)),
            cue_ref=cue_ref,
            name=name,
            cue_number=parsed_cue_number,
            color=color,
            notes=notes,
            payload_ref=payload_ref,
        )

    @staticmethod
    def _cue_ref_from_number(number: int) -> str:
        return f"Cue {max(1, int(number))}"

    @staticmethod
    def _cue_ref_from_value(number: CueNumber | None) -> str:
        text = cue_number_text(number)
        if text is None:
            return SectionManagerDialog._cue_ref_from_number(1)
        return f"Cue {text}"

    @staticmethod
    def cue_number_from_ref_text(cue_ref: str | None) -> CueNumber | None:
        return cue_number_from_ref_text(cue_ref)
