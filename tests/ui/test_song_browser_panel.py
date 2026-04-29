from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication, QAbstractItemView

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.song_browser_drop import SongBrowserAudioDrop
from echozero.ui.qt.song_browser_panel import SongBrowserPanel
from tests.ui.timeline_shell_shared_support import _song_switching_presentation


def test_song_browser_panel_renders_song_and_version_lists():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        assert panel._title.text() == "Setlist"
        assert panel._songs_tree.topLevelItemCount() == 2
        assert panel._songs_tree.columnCount() == 2
        assert panel._songs_tree.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
        assert panel._songs_tree.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
        assert panel._songs_tree.topLevelItem(0).text(0) == "1"
        assert panel._songs_tree.topLevelItem(1).text(0) == "2"
        assert panel._songs_tree.topLevelItem(0).text(1) == "Alpha Song"
        assert panel._songs_tree.topLevelItem(1).text(1) == "Beta Song"

        assert panel._version_list.count() == 2
        assert panel._version_list.item(0).text() == "V1: Original"
        assert panel._version_list.item(1).text() == "V2: Festival Edit [Active]"
        assert panel._active_song_title.text() == "Alpha Song"
        assert "Festival Edit" in panel._active_song_version.text()
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_collapses_to_compact_rail():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        assert panel.is_collapsed is False
        assert panel.width() == 300
        assert panel.property("collapsed") is False
        assert panel._add_button.isHidden() is False
        assert panel._collapse_button.text() == "<"

        panel.toggle_collapsed()
        app.processEvents()

        assert panel.is_collapsed is True
        assert panel.width() == 56
        assert panel.property("collapsed") is True
        assert panel._add_button.isHidden() is True
        assert panel._title.isHidden() is True
        assert panel._collapse_button.text() == ">"
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_empty_state_uses_clean_first_song_prompt():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        panel.set_presentation(
            TimelinePresentation(
                timeline_id=panel._presentation.timeline_id,
                title="Empty",
            )
        )

        assert "No songs in this project yet." in panel._empty_state.text()
        assert "Use + above or drop audio here." in panel._empty_state.text()
        assert panel._content_stack.currentIndex() == 1
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_song_click_emits_song_selected():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    selected: list[str] = []
    panel.song_selected.connect(selected.append)
    try:
        song_item = panel._songs_tree.topLevelItem(1)
        panel._handle_song_item_clicked(song_item, 0)

        assert selected == ["song_beta"]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_version_click_emits_song_version_selected():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    selected: list[str] = []
    panel.song_version_selected.connect(selected.append)
    try:
        version_item = panel._version_list.item(0)
        panel._handle_version_clicked(version_item)

        assert selected == ["song_version_original"]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_add_version_button_uses_active_song():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    requested: list[str] = []
    panel.add_song_version_requested.connect(requested.append)
    try:
        panel._emit_add_version_for_active_song()

        assert requested == ["song_alpha"]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_multi_select_enables_batch_actions_and_emits_batch_delete():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    captured: list[tuple[str, ...]] = []
    panel.batch_delete_songs_requested.connect(captured.append)
    try:
        alpha = panel._songs_tree.topLevelItem(0)
        beta = panel._songs_tree.topLevelItem(1)
        alpha.setSelected(True)
        beta.setSelected(True)
        panel._handle_song_selection_changed()

        assert panel._batch_top_button.isEnabled() is True
        assert panel._batch_bottom_button.isEnabled() is True
        assert panel._batch_delete_button.isEnabled() is True

        panel._emit_batch_delete_songs()

        assert captured == [("song_alpha", "song_beta")]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_drag_reorder_updates_numbers_and_emits_song_order():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    captured: list[tuple[str, ...]] = []
    panel.songs_reordered_requested.connect(captured.append)
    try:
        moved_item = panel._songs_tree.takeTopLevelItem(1)
        assert moved_item is not None
        panel._songs_tree.insertTopLevelItem(0, moved_item)

        panel._handle_song_rows_moved()
        app.processEvents()

        assert panel._songs_tree.topLevelItem(0).text(0) == "1"
        assert panel._songs_tree.topLevelItem(0).text(1) == "Beta Song"
        assert panel._songs_tree.topLevelItem(1).text(0) == "2"
        assert panel._songs_tree.topLevelItem(1).text(1) == "Alpha Song"
        assert captured == [("song_beta", "song_alpha")]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_drag_reorder_coalesces_reorder_emits():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    captured: list[tuple[str, ...]] = []
    panel.songs_reordered_requested.connect(captured.append)
    try:
        moved_item = panel._songs_tree.takeTopLevelItem(1)
        assert moved_item is not None
        panel._songs_tree.insertTopLevelItem(0, moved_item)

        panel._handle_song_rows_moved()
        panel._handle_song_rows_moved()
        app.processEvents()

        assert captured == [("song_beta", "song_alpha")]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_drop_on_song_row_targets_that_song(tmp_path):
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    audio_path = tmp_path / "drop-target.wav"
    audio_path.write_bytes(b"RIFF")
    captured: list[SongBrowserAudioDrop] = []
    panel.audio_paths_dropped.connect(captured.append)
    try:
        panel.show()
        app.processEvents()
        song_item = panel._songs_tree.topLevelItem(1)
        drop_rect = panel._songs_tree.visualItemRect(song_item)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(audio_path))])

        panel._songs_tree.dropEvent(
            QDropEvent(
                QPointF(drop_rect.center()),
                Qt.DropAction.CopyAction,
                mime_data,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        assert captured == [
            SongBrowserAudioDrop(
                audio_paths=(str(audio_path),),
                target_song_id="song_beta",
                target_song_title="Beta Song",
            )
        ]
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_drop_folder_expands_audio_files_in_natural_order(tmp_path):
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    song_dir = tmp_path / "setlist-folder"
    song_dir.mkdir()
    for file_name in ("Song 10.wav", "Song 2.wav", "Song 1.wav", "README.txt"):
        (song_dir / file_name).write_bytes(b"RIFF")
    captured: list[SongBrowserAudioDrop] = []
    panel.audio_paths_dropped.connect(captured.append)
    try:
        panel.show()
        app.processEvents()
        song_item = panel._songs_tree.topLevelItem(1)
        drop_rect = panel._songs_tree.visualItemRect(song_item)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(song_dir))])

        panel._songs_tree.dropEvent(
            QDropEvent(
                QPointF(drop_rect.center()),
                Qt.DropAction.CopyAction,
                mime_data,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )

        assert captured == [
            SongBrowserAudioDrop(
                audio_paths=(
                    str(song_dir / "Song 1.wav"),
                    str(song_dir / "Song 2.wav"),
                    str(song_dir / "Song 10.wav"),
                ),
                target_song_id="song_beta",
                target_song_title="Beta Song",
            )
        ]
    finally:
        panel.close()
        app.processEvents()
