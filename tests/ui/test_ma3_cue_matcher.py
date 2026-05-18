"""MA3 cue matcher dialog contract tests.
Exists because section sends rely on the dialog to write cue mappings before MA3 push.
Connects operator worksheet actions to serialized event cue mapping results.
"""

from PyQt6.QtWidgets import QApplication, QTableWidgetSelectionRange

from echozero.ui.qt.timeline.ma3_cue_matcher import (
    EventCueMatchRow,
    MA3CueMatcherDialog,
    MA3CueOption,
)


def test_ma3_cue_matcher_auto_matches_section_cue_refs() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = MA3CueMatcherDialog(
        title="Match",
        subtitle="Sections",
        rows=[
            EventCueMatchRow(
                event_id="section_1",
                start=0.0,
                label="Intro",
                current_cue_number=None,
                current_cue_ref="Cue 1",
            ),
            EventCueMatchRow(
                event_id="section_2",
                start=8.0,
                label="Verse",
                current_cue_number=None,
                current_cue_ref="Cue 2",
            ),
        ],
        cue_options=[
            MA3CueOption(cue_number=1, name="Intro"),
            MA3CueOption(cue_number=2, name="Verse"),
        ],
    )
    try:
        mappings = dialog.selected_mappings()
        assert [mapping.cue_number for mapping in mappings] == [1, 2]
        assert [mapping.cue_ref for mapping in mappings] == ["1", "2"]
    finally:
        dialog.close()
        app.processEvents()


def test_ma3_cue_matcher_fill_down_advances_through_cue_options() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = MA3CueMatcherDialog(
        title="Match",
        subtitle="Sections",
        rows=[
            EventCueMatchRow(
                event_id="section_1",
                start=0.0,
                label="Intro",
                current_cue_number=None,
                current_cue_ref=None,
            ),
            EventCueMatchRow(
                event_id="section_2",
                start=8.0,
                label="Verse",
                current_cue_number=None,
                current_cue_ref=None,
            ),
            EventCueMatchRow(
                event_id="section_3",
                start=16.0,
                label="Chorus",
                current_cue_number=None,
                current_cue_ref=None,
            ),
        ],
        cue_options=[
            MA3CueOption(cue_number=10, name="Intro"),
            MA3CueOption(cue_number=11, name="Verse"),
            MA3CueOption(cue_number=12, name="Chorus"),
        ],
    )
    try:
        dialog._set_row_target_cue(0, 10)
        dialog._table.setRangeSelected(
            QTableWidgetSelectionRange(0, 0, 2, dialog._COL_TARGET_NAME),
            True,
        )
        dialog._fill_down_selection()

        mappings = dialog.selected_mappings()
        assert [mapping.cue_number for mapping in mappings] == [10, 11, 12]
        assert [mapping.cue_name for mapping in mappings] == ["Intro", "Verse", "Chorus"]
    finally:
        dialog.close()
        app.processEvents()


def test_ma3_cue_matcher_creates_missing_cues_from_empty_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = MA3CueMatcherDialog(
        title="Match",
        subtitle="Sections",
        rows=[
            EventCueMatchRow(
                event_id="section_1",
                start=0.0,
                label="Part 1",
                current_cue_number=1,
                current_cue_ref="Cue 1",
            ),
            EventCueMatchRow(
                event_id="section_2",
                start=8.0,
                label="Part 2",
                current_cue_number=2,
                current_cue_ref="Cue 2",
            ),
            EventCueMatchRow(
                event_id="section_3",
                start=16.0,
                label="Bridge",
                current_cue_number=None,
                current_cue_ref="Cue 5",
            ),
        ],
        cue_options=[MA3CueOption(cue_number=1000, name="Part 1")],
    )
    try:
        assert [mapping.cue_number for mapping in dialog.selected_mappings()] == [1000]

        dialog._create_missing_cues_from_ez_rows()

        mappings = dialog.selected_mappings()
        assert [mapping.cue_number for mapping in mappings] == [1000, 2, 5]
        assert [mapping.cue_name for mapping in mappings] == [
            "Part 1",
            "Create: Part 2",
            "Create: Bridge",
        ]
    finally:
        dialog.close()
        app.processEvents()


def test_ma3_cue_matcher_parses_typed_section_cue_refs() -> None:
    assert MA3CueMatcherDialog._parse_combo_cue_text("Cue 10") == 10
    assert MA3CueMatcherDialog._parse_combo_cue_text("Cue 10 - Bridge") == 10
