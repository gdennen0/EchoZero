"""
Sync Layer Polling Commands

Commands for re-hooking and polling MA3 synced tracks.
"""

from typing import TYPE_CHECKING, Dict, Any, Iterable, List, Optional, Tuple

from src.application.commands.base_command import EchoZeroCommand
from src.application.settings.show_manager_settings import ShowManagerSettingsManager
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


def _parse_coord(coord: str) -> Optional[Tuple[int, int, int]]:
    """
    Parse MA3 coord to (tc, tg, track).

    Supports:
    - "tc101_tg1_tr2"
    - "101.1.2"
    """
    if not coord:
        return None
    try:
        if "." in coord:
            parts = coord.split(".")
            if len(parts) >= 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
        cleaned = coord.replace("tc", "").replace("_tg", ".").replace("_tr", ".").replace("_", ".")
        parts = [p for p in cleaned.split(".") if p]
        if len(parts) >= 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, TypeError):
        return None
    return None


def _iter_ma3_tracks(synced_layers: Iterable[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    """Return list of (tc, tg, track) from synced_layers entries."""
    tracks: List[Tuple[int, int, int]] = []
    for entity in synced_layers:
        if not isinstance(entity, dict) or "coord" not in entity:
            continue
        tc = entity.get("timecode_no")
        tg = entity.get("track_group")
        tr = entity.get("track")
        if isinstance(tc, int) and isinstance(tg, int) and isinstance(tr, int):
            tracks.append((tc, tg, tr))
            continue
        parsed = _parse_coord(entity.get("coord", ""))
        if parsed:
            tracks.append(parsed)
    return tracks


def _send_lua_command(ip: str, port: int, lua_code: str) -> bool:
    """Send a Lua command to MA3 via OSC /cmd."""
    import socket

    def build_osc_string(value: str) -> bytes:
        encoded = value.encode("utf-8") + b"\x00"
        padding = (4 - len(encoded) % 4) % 4
        return encoded + b"\x00" * padding

    try:
        cmd = f'Lua "{lua_code}"'
        msg = build_osc_string("/cmd")
        msg += build_osc_string(",s")
        msg += build_osc_string(cmd)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(msg, (ip, port))
        sock.close()
        return True
    except Exception as exc:
        Log.error(f"Failed to send MA3 command '{lua_code}': {exc}")
        return False


class RehookSyncedMA3TracksCommand(EchoZeroCommand):
    """Re-hook all synced MA3 tracks for real-time change notifications."""

    COMMAND_TYPE = "layer_sync.rehook_synced_ma3_tracks"

    def __init__(self, facade: "ApplicationFacade", show_manager_block_id: str):
        super().__init__(facade, "Rehook Synced MA3 Tracks")
        self._show_manager_block_id = show_manager_block_id
        self.hooked_count: int = 0
        self.requested_count: int = 0

    def redo(self):
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        ma3_ip = settings_manager.ma3_ip
        ma3_port = settings_manager.ma3_port
        if not ma3_ip or not ma3_port:
            Log.warning("RehookSyncedMA3TracksCommand: MA3 IP/port not configured")
            return

        tracks = _iter_ma3_tracks(settings_manager.synced_layers)
        self.requested_count = len(tracks)
        
        # Request fresh track list first (triggers auto_reconnect_layers
        # in on_tracks_received, which may update coords before hooking)
        requested_tctg = set()
        for tc, tg, tr in tracks:
            tctg_key = (tc, tg)
            if tctg_key not in requested_tctg:
                _send_lua_command(ma3_ip, ma3_port, f"EZ.GetTracks({tc}, {tg})")
                requested_tctg.add(tctg_key)
        
        # Then hook all tracks for real-time change notifications
        for tc, tg, tr in tracks:
            if _send_lua_command(ma3_ip, ma3_port, f"EZ.HookCmdSubTrack({tc}, {tg}, {tr})"):
                self.hooked_count += 1

        Log.info(
            f"RehookSyncedMA3TracksCommand: sent {self.hooked_count}/{self.requested_count} hook requests "
            f"(requested track list for {len(requested_tctg)} TC/TG pairs)"
        )

    def undo(self):
        """No-op. Rehook is not undone automatically."""
        return


class PollSyncedMA3TracksCommand(EchoZeroCommand):
    """Poll all synced MA3 tracks by requesting their event lists."""

    COMMAND_TYPE = "layer_sync.poll_synced_ma3_tracks"

    def __init__(self, facade: "ApplicationFacade", show_manager_block_id: str):
        super().__init__(facade, "Poll Synced MA3 Tracks")
        self._show_manager_block_id = show_manager_block_id
        self.polled_count: int = 0
        self.requested_count: int = 0

    def redo(self):
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        ma3_ip = settings_manager.ma3_ip
        ma3_port = settings_manager.ma3_port
        if not ma3_ip or not ma3_port:
            Log.warning("PollSyncedMA3TracksCommand: MA3 IP/port not configured")
            return

        tracks = _iter_ma3_tracks(settings_manager.synced_layers)
        self.requested_count = len(tracks)
        for tc, tg, tr in tracks:
            if _send_lua_command(ma3_ip, ma3_port, f"EZ.GetEvents({tc}, {tg}, {tr})"):
                self.polled_count += 1

        Log.info(
            f"PollSyncedMA3TracksCommand: sent {self.polled_count}/{self.requested_count} poll requests"
        )

    def undo(self):
        """No-op. Poll is not undone."""
        return


class TestSyncedMA3TrackCommand(EchoZeroCommand):
    """Rehook and poll a single MA3 track."""

    COMMAND_TYPE = "layer_sync.test_synced_ma3_track"

    def __init__(self, facade: "ApplicationFacade", show_manager_block_id: str, coord: str):
        super().__init__(facade, f"Test MA3 Track: {coord}")
        self._show_manager_block_id = show_manager_block_id
        self._coord = coord
        self.hooked: bool = False
        self.polled: bool = False

    def redo(self):
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        ma3_ip = settings_manager.ma3_ip
        ma3_port = settings_manager.ma3_port
        if not ma3_ip or not ma3_port:
            Log.warning("TestSyncedMA3TrackCommand: MA3 IP/port not configured")
            return

        parsed = _parse_coord(self._coord)
        if not parsed:
            Log.warning(f"TestSyncedMA3TrackCommand: Invalid coord '{self._coord}'")
            return

        tc, tg, tr = parsed
        self.hooked = _send_lua_command(ma3_ip, ma3_port, f"EZ.HookCmdSubTrack({tc}, {tg}, {tr})")
        self.polled = _send_lua_command(ma3_ip, ma3_port, f"EZ.GetEvents({tc}, {tg}, {tr})")
        Log.info(
            f"TestSyncedMA3TrackCommand: hook={'ok' if self.hooked else 'fail'} "
            f"poll={'ok' if self.polled else 'fail'} for {self._coord}"
        )

    def undo(self):
        """No-op. Test is not undone."""
        return


class SetApplyUpdatesEnabledCommand(EchoZeroCommand):
    """Enable/disable applying MA3 updates to Editor."""

    COMMAND_TYPE = "layer_sync.set_apply_updates_enabled"

    def __init__(self, facade: "ApplicationFacade", show_manager_block_id: str, enabled: bool):
        super().__init__(facade, "Set Apply Updates")
        self._show_manager_block_id = show_manager_block_id
        self._enabled = bool(enabled)

    def redo(self):
        settings_manager = ShowManagerSettingsManager(self._facade, self._show_manager_block_id)
        settings_manager.apply_updates_enabled = self._enabled
        Log.info(
            f"SetApplyUpdatesEnabledCommand: apply_updates_enabled={self._enabled} "
            f"for {self._show_manager_block_id}"
        )

    def undo(self):
        """No-op. Settings change is not undone."""
        return
