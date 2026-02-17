from typing import TYPE_CHECKING, Optional, List, Any

from src.application.commands.base_command import EchoZeroCommand
from src.application.settings.show_manager_settings import normalize_ma3_coord
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SyncLayerPairCommand(EchoZeroCommand):
    """
    Unified sync command for creating a synced MA3<->Editor pair.
    """

    COMMAND_TYPE = "layer_sync.sync_pair"

    def __init__(
        self,
        facade: "ApplicationFacade",
        show_manager_block_id: str,
        source: str,
        editor_layer_id: Optional[str] = None,
        editor_block_id: Optional[str] = None,
        ma3_coord: Optional[str] = None,
        ma3_timecode_no: Optional[int] = None,
        ma3_track_group: Optional[int] = None,
        ma3_track: Optional[int] = None,
        ma3_track_name: Optional[str] = None,
        ma3_events: Optional[List[Any]] = None,
        sequence_no: Optional[int] = None,
    ):
        title = f"Sync Layer Pair ({source})"
        super().__init__(facade, title)
        self._show_manager_block_id = show_manager_block_id
        self._source = source
        self._editor_layer_id = editor_layer_id
        self._editor_block_id = editor_block_id
        self._ma3_coord = normalize_ma3_coord(ma3_coord) if ma3_coord else None
        self._ma3_timecode_no = ma3_timecode_no
        self._ma3_track_group = ma3_track_group
        self._ma3_track = ma3_track
        self._ma3_track_name = ma3_track_name
        self._ma3_events = ma3_events or []
        self._sequence_no = sequence_no
        self._inner_command = None

    def redo(self):
        if self._source == "ma3":
            self._redo_from_ma3()
        elif self._source == "editor":
            self._redo_from_editor()
        else:
            Log.warning(f"SyncLayerPairCommand: Unknown source '{self._source}'")

    def undo(self):
        if self._inner_command and hasattr(self._inner_command, "undo"):
            self._inner_command.undo()

    def _redo_from_ma3(self):
        from src.features.show_manager.application.commands.add_synced_ma3_track_command import (
            AddSyncedMA3TrackCommand,
        )

        if not self._ma3_coord:
            Log.warning("SyncLayerPairCommand: Missing MA3 coord")
            return
        if not self._ma3_timecode_no or not self._ma3_track_group or not self._ma3_track:
            Log.warning("SyncLayerPairCommand: Missing MA3 track details")
            return

        self._inner_command = AddSyncedMA3TrackCommand(
            facade=self._facade,
            show_manager_block_id=self._show_manager_block_id,
            coord=self._ma3_coord,
            timecode_no=self._ma3_timecode_no,
            track_group=self._ma3_track_group,
            track=self._ma3_track,
            name=self._ma3_track_name or "",
            ma3_events=self._ma3_events,
            sequence_no=self._sequence_no,
        )
        self._inner_command.redo()

    def _redo_from_editor(self):
        from src.features.show_manager.application.commands.add_synced_editor_layer_command import (
            AddSyncedEditorLayerCommand,
        )

        if not self._editor_layer_id or not self._editor_block_id:
            Log.warning("SyncLayerPairCommand: Missing Editor layer details")
            return

        self._inner_command = AddSyncedEditorLayerCommand(
            facade=self._facade,
            show_manager_block_id=self._show_manager_block_id,
            editor_layer_id=self._editor_layer_id,
            editor_block_id=self._editor_block_id,
            ma3_events=self._ma3_events,
        )
        self._inner_command.redo()
