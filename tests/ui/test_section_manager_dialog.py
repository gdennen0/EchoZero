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


def test_section_manager_add_section_uses_cue_prefix() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(_empty_presentation())
    try:
        dialog._on_add_section()
        drafts = dialog.section_cue_drafts()
        assert len(drafts) == 1
        assert drafts[0].cue_ref == "Cue 1"
        assert drafts[0].cue_number == 1
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_renumber_cues_assigns_sequential_cue_labels() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_b", start=30.0, cue_ref="Q44", name="Bridge"),
            SectionCueDraft(cue_id="cue_a", start=12.0, cue_ref="Q7", name="Verse"),
            SectionCueDraft(cue_id="cue_c", start=42.0, cue_ref="Q9A", name="Chorus"),
        ],
    )
    try:
        dialog._on_renumber_cues()
        assert [row.cue_ref for row in dialog._rows] == ["Cue 1", "Cue 2", "Cue 3"]
        assert [row.cue_number for row in dialog._rows] == [1, 2, 3]
        assert [row.name for row in dialog._rows] == ["Verse", "Bridge", "Chorus"]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_renumber_cues_from_specific_start_number() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_b", start=30.0, cue_ref="Q44", name="Bridge"),
            SectionCueDraft(cue_id="cue_a", start=12.0, cue_ref="Q7", name="Verse"),
            SectionCueDraft(cue_id="cue_c", start=42.0, cue_ref="Q9A", name="Chorus"),
        ],
    )
    try:
        dialog._renumber_cues_from_start(10)
        assert [row.cue_ref for row in dialog._rows] == ["Cue 10", "Cue 11", "Cue 12"]
        assert [row.cue_number for row in dialog._rows] == [10, 11, 12]
        assert [row.name for row in dialog._rows] == ["Verse", "Bridge", "Chorus"]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_renumber_cues_from_prompt_accepts_float_start(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_b", start=30.0, cue_ref="Q44", name="Bridge"),
            SectionCueDraft(cue_id="cue_a", start=12.0, cue_ref="Q7", name="Verse"),
        ],
    )
    try:
        monkeypatch.setattr(
            "echozero.ui.qt.timeline.section_manager.QInputDialog.getText",
            lambda *args, **kwargs: ("7.5", True),
        )
        dialog._on_renumber_cues_from_prompt()
        assert [row.cue_ref for row in dialog._rows] == ["Cue 7.5", "Cue 8.5"]
        assert [row.cue_number for row in dialog._rows] == [7.5, 8.5]
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_section_cue_drafts_preserve_explicit_cue_numbers() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 99", name="Verse", cue_number=7.5),
        ],
    )
    try:
        drafts = dialog.section_cue_drafts()
        assert len(drafts) == 1
        assert drafts[0].cue_number == 7.5
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_add_after_selected_auto_numbers_between_neighbors() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 1", name="Verse", cue_number=1),
            SectionCueDraft(cue_id="cue_b", start=20.0, cue_ref="Cue 2", name="Chorus", cue_number=2),
        ],
    )
    try:
        dialog._refresh_table(select_row=0)
        dialog._insert_section_relative(before=False)
        inserted = dialog._rows[1]
        assert inserted.cue_number == 1.5
        assert inserted.cue_ref == "Cue 1.5"
        assert inserted.start == 15.0
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_add_before_selected_auto_numbers_before_first() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 3", name="Verse", cue_number=3),
        ],
    )
    try:
        dialog._refresh_table(select_row=0)
        dialog._insert_section_relative(before=True)
        inserted = dialog._rows[0]
        assert inserted.cue_number == 2
        assert inserted.cue_ref == "Cue 2"
        assert inserted.start == 2.0
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_table_edit_updates_selected_row_values() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 1", name="Verse", cue_number=1),
        ],
    )
    try:
        dialog._refresh_table(select_row=0)
        dialog._table.item(0, 0).setText("7.5")
        dialog._table.item(0, 2).setText("Bridge")
        dialog._table.item(0, 3).setText("12.25")
        row = dialog._rows[0]
        assert row.cue_number == 7.5
        assert row.name == "Bridge"
        assert row.start == 12.25
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_quick_label_applies_to_multiple_selected_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 1", name="Verse", cue_number=1),
            SectionCueDraft(cue_id="cue_b", start=20.0, cue_ref="Cue 2", name="Bridge", cue_number=2),
            SectionCueDraft(cue_id="cue_c", start=30.0, cue_ref="Cue 3", name="Outro", cue_number=3),
        ],
    )
    try:
        dialog._refresh_table(select_rows=[0, 2])
        dialog._apply_quick_label("Chorus")
        assert dialog._rows[0].name == "Chorus"
        assert dialog._rows[1].name == "Bridge"
        assert dialog._rows[2].name == "Chorus"
    finally:
        dialog.close()
        app.processEvents()


def test_section_manager_multi_edit_name_updates_all_selected_rows() -> None:
    app = QApplication.instance() or QApplication([])
    dialog = SectionManagerDialog(
        _empty_presentation(),
        cues=[
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 1", name="Verse", cue_number=1),
            SectionCueDraft(cue_id="cue_b", start=20.0, cue_ref="Cue 2", name="Bridge", cue_number=2),
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
            SectionCueDraft(cue_id="cue_a", start=10.0, cue_ref="Cue 1", name="Verse", cue_number=1),
            SectionCueDraft(cue_id="cue_b", start=20.0, cue_ref="Cue 2", name="Bridge", cue_number=2),
            SectionCueDraft(cue_id="cue_c", start=30.0, cue_ref="Cue 3", name="Outro", cue_number=3),
        ],
    )
    try:
        dialog._refresh_table(select_rows=[0, 2])
        dialog._on_delete_section()
        assert len(dialog._rows) == 1
        assert dialog._rows[0].cue_id == "cue_b"
    finally:
        dialog.close()
        app.processEvents()
