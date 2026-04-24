from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import TimelinePresentation
from echozero.ui.qt.song_browser_drop import SongBrowserAudioDrop
from echozero.ui.qt.song_browser_panel import SongBrowserPanel
from tests.ui.timeline_shell_shared_support import _song_switching_presentation


def test_song_browser_panel_renders_songs_and_versions():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        assert panel._tree.topLevelItemCount() == 2
        assert panel._title.text() == "Setlist"
        assert panel._tree.topLevelItem(0).text(0) == "▶ Alpha Song"
        assert panel._tree.topLevelItem(0).childCount() == 2
        assert panel._tree.topLevelItem(0).child(0).text(0) == "Version 1 · Original"
        assert (
            panel._tree.topLevelItem(0).child(1).text(0)
            == "Version 2 · Festival Edit · Active"
        )
        assert panel._tree.topLevelItem(0).isExpanded() is False
        assert panel._tree.currentItem().text(0) == "▶ Alpha Song"
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_song_rows_show_disclosure_state():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        song_item = panel._tree.topLevelItem(0)
        assert song_item.text(0) == "▶ Alpha Song"

        panel._tree.expandItem(song_item)
        app.processEvents()

        assert song_item.text(0) == "▼ Alpha Song"

        panel._tree.collapseItem(song_item)
        app.processEvents()

        assert song_item.text(0) == "▶ Alpha Song"
    finally:
        panel.close()
        app.processEvents()


def test_song_browser_panel_collapses_to_compact_rail():
    app = QApplication.instance() or QApplication([])
    panel = SongBrowserPanel(_song_switching_presentation())
    try:
        assert panel.is_collapsed is False
        assert panel.width() == 280
        assert panel.property("collapsed") is False
        assert panel._add_button.isHidden() is False
        assert panel._collapse_button.text() == "◂"

        panel.toggle_collapsed()
        app.processEvents()

        assert panel.is_collapsed is True
        assert panel.width() == 64
        assert panel.property("collapsed") is True
        assert panel._add_button.isHidden() is True
        assert panel._title.isHidden() is True
        assert panel._collapse_button.text() == "▸"
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
        song_item = panel._tree.topLevelItem(1)
        drop_rect = panel._tree.visualItemRect(song_item)
        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(audio_path))])

        panel._tree.dropEvent(
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
