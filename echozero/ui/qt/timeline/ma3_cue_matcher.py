"""Fast worksheet dialog for mapping EZ timeline events to MA3 cue numbers.
Exists to keep MA3 cue-matching UI concerns out of transfer action routing.
Connects operator-selected layer events to deterministic cue-number mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from echozero.application.shared.cue_numbers import (
    CueNumber,
    cue_number_from_ref_text,
    cue_number_text,
    parse_positive_cue_number,
)


@dataclass(frozen=True, slots=True)
class MA3CueOption:
    cue_number: CueNumber
    name: str


@dataclass(frozen=True, slots=True)
class EventCueMatchRow:
    event_id: str
    start: float
    label: str
    current_cue_number: CueNumber | None
    current_cue_ref: str | None


@dataclass(frozen=True, slots=True)
class EventCueMatchResult:
    event_id: str
    cue_number: CueNumber
    cue_ref: str
    cue_name: str | None = None

from echozero.ui.style.qt import ensure_qt_theme_installed


class MA3CueMatcherDialog(QDialog):
    _COL_START = 0
    _COL_EVENT = 1
    _COL_CURRENT = 2
    _COL_TARGET = 3
    _COL_TARGET_NAME = 4

    def __init__(
        self,
        *,
        title: str,
        subtitle: str,
        rows: list[EventCueMatchRow],
        cue_options: list[MA3CueOption],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        ensure_qt_theme_installed()
        self.setWindowTitle(title)
        self.resize(980, 620)
        self._rows = list(rows)
        self._cue_options = sorted(
            list(cue_options),
            key=lambda item: float(item.cue_number),
        )
        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            [
                "Start",
                "EZ Event",
                "Current Cue",
                "MA3 Cue",
                "MA3 Cue Name",
            ]
        )
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.verticalHeader().setVisible(False)

        subtitle_label = QLabel(subtitle, self)
        subtitle_label.setWordWrap(True)

        actions_row = QHBoxLayout()
        auto_button = QPushButton("Auto Match", self)
        auto_button.clicked.connect(self._auto_match_rows)
        fill_down_button = QPushButton("Fill Down", self)
        fill_down_button.clicked.connect(self._fill_down_selection)
        clear_button = QPushButton("Clear Selected", self)
        clear_button.clicked.connect(self._clear_selected)
        actions_row.addWidget(auto_button)
        actions_row.addWidget(fill_down_button)
        actions_row.addWidget(clear_button)
        actions_row.addStretch(1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(subtitle_label)
        layout.addLayout(actions_row)
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(button_box)

        self._populate_table()
        self._auto_match_rows()

    def selected_mappings(self) -> list[EventCueMatchResult]:
        results: list[EventCueMatchResult] = []
        for row_idx, row in enumerate(self._rows):
            combo = self._target_combo(row_idx)
            cue_number = self._cue_number_from_combo(combo)
            if cue_number is None:
                continue
            cue_name = self._cue_name_for_number(cue_number)
            cue_ref = cue_number_text(cue_number) or str(cue_number)
            results.append(
                EventCueMatchResult(
                    event_id=row.event_id,
                    cue_number=cue_number,
                    cue_ref=cue_ref,
                    cue_name=cue_name,
                )
            )
        return results

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row_idx, row in enumerate(self._rows):
            start_item = QTableWidgetItem(f"{float(row.start):.3f}")
            start_item.setFlags(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, self._COL_START, start_item)

            label_item = QTableWidgetItem(str(row.label or "").strip() or "Event")
            label_item.setFlags(label_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, self._COL_EVENT, label_item)

            current_text = cue_number_text(row.current_cue_number) or str(
                row.current_cue_ref or ""
            )
            current_item = QTableWidgetItem(current_text)
            current_item.setFlags(current_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, self._COL_CURRENT, current_item)

            combo = QComboBox(self._table)
            combo.setEditable(True)
            combo.addItem("")
            for cue in self._cue_options:
                number_text = cue_number_text(cue.cue_number) or str(cue.cue_number)
                cue_label = cue.name.strip()
                combo.addItem(f"{number_text} - {cue_label}" if cue_label else number_text)
            combo.currentTextChanged.connect(
                lambda _text, idx=row_idx: self._sync_target_name_cell(idx)
            )
            self._table.setCellWidget(row_idx, self._COL_TARGET, combo)

            target_name_item = QTableWidgetItem("")
            target_name_item.setFlags(target_name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row_idx, self._COL_TARGET_NAME, target_name_item)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(self._COL_START, 92)
        header.resizeSection(self._COL_EVENT, 220)
        header.resizeSection(self._COL_CURRENT, 130)
        header.resizeSection(self._COL_TARGET, 220)
        header.resizeSection(self._COL_TARGET_NAME, 250)

    def _auto_match_rows(self) -> None:
        cue_by_number: dict[str, MA3CueOption] = {}
        cue_by_name: dict[str, MA3CueOption] = {}
        for cue in self._cue_options:
            cue_no_text = cue_number_text(cue.cue_number)
            if cue_no_text:
                cue_by_number[cue_no_text.casefold()] = cue
            cue_name = str(cue.name or "").strip()
            if cue_name:
                cue_by_name[cue_name.casefold()] = cue

        for row_idx, row in enumerate(self._rows):
            matched: MA3CueOption | None = None
            current_no_text = cue_number_text(row.current_cue_number)
            if current_no_text:
                matched = cue_by_number.get(current_no_text.casefold())
            if matched is None and row.current_cue_ref:
                cue_ref_text = str(row.current_cue_ref).strip()
                matched = cue_by_number.get(cue_ref_text.casefold())
                if matched is None:
                    cue_ref_number = cue_number_from_ref_text(cue_ref_text)
                    cue_ref_number_text = cue_number_text(cue_ref_number)
                    if cue_ref_number_text:
                        matched = cue_by_number.get(cue_ref_number_text.casefold())
            if matched is None:
                matched = cue_by_name.get(str(row.label or "").strip().casefold())
            if matched is None:
                continue
            self._set_row_target_cue(row_idx, matched.cue_number)

    def _fill_down_selection(self) -> None:
        selection = sorted({index.row() for index in self._table.selectedIndexes()})
        if not selection:
            return
        source_row = selection[0]
        source_cue = self._cue_number_from_combo(self._target_combo(source_row))
        if source_cue is None:
            return
        source_option_index = self._cue_option_index(source_cue)
        for row_idx in selection[1:]:
            if source_option_index is None:
                self._set_row_target_cue(row_idx, source_cue)
                continue
            option_index = source_option_index + (row_idx - source_row)
            if option_index < 0 or option_index >= len(self._cue_options):
                continue
            self._set_row_target_cue(row_idx, self._cue_options[option_index].cue_number)

    def _clear_selected(self) -> None:
        selection = {index.row() for index in self._table.selectedIndexes()}
        for row_idx in selection:
            combo = self._target_combo(row_idx)
            combo.setCurrentText("")
            self._sync_target_name_cell(row_idx)

    def _set_row_target_cue(self, row_idx: int, cue_number: CueNumber) -> None:
        combo = self._target_combo(row_idx)
        cue_no_text = cue_number_text(cue_number)
        if cue_no_text is None:
            combo.setCurrentText("")
            self._sync_target_name_cell(row_idx)
            return
        for index in range(combo.count()):
            raw_text = str(combo.itemText(index) or "").strip()
            parsed = self._parse_combo_cue_text(raw_text)
            if parsed is not None and cue_number_text(parsed) == cue_no_text:
                combo.setCurrentIndex(index)
                self._sync_target_name_cell(row_idx)
                return
        combo.setCurrentText(cue_no_text)
        self._sync_target_name_cell(row_idx)

    def _sync_target_name_cell(self, row_idx: int) -> None:
        cue_number = self._cue_number_from_combo(self._target_combo(row_idx))
        item = self._table.item(row_idx, self._COL_TARGET_NAME)
        if item is None:
            return
        if cue_number is None:
            item.setText("")
            return
        cue_name = self._cue_name_for_number(cue_number)
        item.setText(cue_name or "")

    def _cue_name_for_number(self, cue_number: CueNumber) -> str | None:
        cue_text = cue_number_text(cue_number)
        if cue_text is None:
            return None
        for cue in self._cue_options:
            option_text = cue_number_text(cue.cue_number)
            if option_text == cue_text:
                name = str(cue.name or "").strip()
                return name or None
        return None

    def _cue_number_from_combo(self, combo: QComboBox) -> CueNumber | None:
        return self._parse_combo_cue_text(str(combo.currentText() or ""))

    def _cue_option_index(self, cue_number: CueNumber) -> int | None:
        cue_text = cue_number_text(cue_number)
        if cue_text is None:
            return None
        for index, cue in enumerate(self._cue_options):
            if cue_number_text(cue.cue_number) == cue_text:
                return index
        return None

    @staticmethod
    def _parse_combo_cue_text(value: str) -> CueNumber | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        number_token = raw.split("-", 1)[0].strip()
        parsed = parse_positive_cue_number(number_token)
        if parsed is not None:
            return parsed
        parsed = parse_positive_cue_number(raw)
        if parsed is not None:
            return parsed
        return cue_number_from_ref_text(raw)

    def _target_combo(self, row_idx: int) -> QComboBox:
        widget = self._table.cellWidget(row_idx, self._COL_TARGET)
        if isinstance(widget, QComboBox):
            return widget
        raise RuntimeError(f"Missing target cue selector for row {row_idx}")
