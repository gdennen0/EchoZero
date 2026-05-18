from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import TimelineId
from echozero.ui.qt.timeline.section_manager import SectionCueDraft, SectionManagerDialog


def _empty_presentation() -> TimelinePresentation:
    return TimelinePresentation(
        timeline_id=TimelineId("timeline_section_manager_dialog"),
        title="Sections",
        end_time_label="00:10.00",
    )


def test_section_manager_add_section_creates_clean_ez_marker() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(_empty_presentation())
    try:
        dialog._on_add_section()
        drafts = dialog.section_cue_drafts()
        assert len(drafts) == 1
        assert drafts[0].name == "Section"
        assert drafts[0].cue_ref is None
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_orders_sections_by_start_without_renumbering() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_b", start=30.0, cue_ref="Q44", name="Bridge"),
            SectionCueDraft(cue_id="section_a", start=12.0, cue_ref="Q7", name="Verse"),
            SectionCueDraft(cue_id="section_c", start=42.0, cue_ref="Q9A", name="Chorus"),
        ],
    )
    try:
        drafts = dialog.section_cue_drafts()
        assert [(row.name, row.start) for row in drafts] == [
            ("Verse", 12.0),
            ("Bridge", 30.0),
            ("Chorus", 42.0),
        ]
        assert [row.cue_ref for row in drafts] == ["Q7", "Q44", "Q9A"]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_add_after_selected_inserts_without_cue_number_churn() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_a", start=10.0, name="Verse"),
            SectionCueDraft(cue_id="section_b", start=20.0, name="Chorus"),
            SectionCueDraft(cue_id="section_c", start=30.0, name="Bridge"),
        ],
    )
    try:
        dialog._refresh_table(select_row=0)
        dialog._insert_section_relative(before=False)
        inserted = dialog._rows[1]
        assert inserted.name == "Section"
        assert inserted.cue_ref is None
        assert inserted.start == 15.0
        assert [row.name for row in dialog._rows] == [
            "Verse",
            "Section",
            "Chorus",
            "Bridge",
        ]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_acceptance_preserves_section_names_by_start_time() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_1", start=10.0, name="Intro"),
            SectionCueDraft(cue_id="section_2", start=20.0, name="Verse"),
            SectionCueDraft(cue_id="section_3", start=30.0, name="Chorus"),
            SectionCueDraft(cue_id=None, start=25.0, name="Section"),
        ],
    )
    try:
        drafts = dialog.section_cue_drafts()
        assert [draft.start for draft in drafts] == [10.0, 20.0, 25.0, 30.0]
        assert [draft.name for draft in drafts] == ["Intro", "Verse", "Section", "Chorus"]
        assert all(draft.cue_ref is None for draft in drafts)
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_table_edit_updates_selected_row_values() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[SectionCueDraft(cue_id="section_a", start=10.0, name="Verse")],
    )
    try:
        dialog._refresh_table(select_row=0)
        dialog._table.item(0, 0).setText("Bridge")
        dialog._table.item(0, 1).setText("12.25")
        row = dialog._rows[0]
        assert row.name == "Bridge"
        assert row.start == 12.25
        assert row.cue_ref is None
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_quick_label_applies_to_multiple_selected_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_a", start=10.0, name="Verse"),
            SectionCueDraft(cue_id="section_b", start=20.0, name="Bridge"),
            SectionCueDraft(cue_id="section_c", start=30.0, name="Outro"),
        ],
    )
    try:
        dialog._refresh_table(select_rows=[0, 2])
        dialog._apply_quick_label("Chorus")
        assert [row.name for row in dialog._rows] == ["Chorus", "Bridge", "Chorus"]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_multi_edit_name_updates_all_selected_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_a", start=10.0, name="Verse"),
            SectionCueDraft(cue_id="section_b", start=20.0, name="Bridge"),
        ],
    )
    try:
        dialog._refresh_table(select_rows=[0, 1])
        dialog._name_input.setText("Vocal")
        dialog._apply_editor_field("name")
        assert [row.name for row in dialog._rows] == ["Vocal", "Vocal"]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_delete_removes_multiple_selected_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="section_a", start=10.0, name="Verse"),
            SectionCueDraft(cue_id="section_b", start=20.0, name="Bridge"),
            SectionCueDraft(cue_id="section_c", start=30.0, name="Outro"),
        ],
    )
    try:
        dialog._refresh_table(select_rows=[0, 2])
        dialog._on_delete_section()
        assert len(dialog._rows) == 1
        assert dialog._rows[0].cue_id == "section_b"
    finally:
        dialog.close()
        app.processEvents()
