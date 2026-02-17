"""
Setlist Snapshot Service

Handles song switching via snapshot save/restore.
Extracted from SetlistService for single-responsibility.
"""
from typing import Optional, Tuple
import threading

from src.features.setlists.domain import SetlistRepository, SetlistSongRepository
from src.shared.domain.entities import DataStateSnapshot
from src.features.projects.application import SnapshotService, ProjectService
from src.application.api.application_facade import ApplicationFacade
from src.utils.message import Log


class SetlistSnapshotService:
    """
    Handles song switching by saving and restoring data state snapshots.
    
    Thread-safe: uses a lock to prevent concurrent switches.
    Simple: project file is the backup, no extra complexity.
    """
    
    def __init__(
        self,
        setlist_repo: SetlistRepository,
        setlist_song_repo: SetlistSongRepository,
        snapshot_service: SnapshotService,
        project_service: ProjectService,
        facade: ApplicationFacade
    ):
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._snapshot_service = snapshot_service
        self._project_service = project_service
        self._facade = facade
        self._switching_lock = threading.Lock()
        self._is_switching = False
        Log.info("SetlistSnapshotService: Initialized")
    
    @property
    def is_switching(self) -> bool:
        """Whether a song switch is currently in progress."""
        return self._is_switching
    
    def switch_active_song(
        self,
        setlist_id: str,
        song_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Switch active song by restoring its snapshot.
        
        Uses current project (setlist.project_id). Snapshots are loaded from project file.
        
        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier to switch to
            
        Returns:
            (success, error_message)
        """
        if not self._switching_lock.acquire(blocking=False):
            return False, "Another song switch is already in progress. Please wait."
        
        try:
            self._is_switching = True
            
            # Validate inputs
            setlist = self._setlist_repo.get(setlist_id)
            if not setlist:
                return False, f"Setlist {setlist_id} not found"
            
            song = self._setlist_song_repo.get(song_id)
            if not song:
                return False, f"Song {song_id} not found"
            
            if song.setlist_id != setlist_id:
                return False, f"Song {song_id} does not belong to setlist {setlist_id}"
            
            if song.status != "completed":
                return False, f"Song has not been processed yet (status: {song.status}). Please process the song first."
            
            # Verify current project
            if not self._facade.current_project_id:
                return False, "No current project. Please open the project used to create this setlist."
            
            if self._facade.current_project_id != setlist.project_id:
                return False, (
                    f"Current project ({self._facade.current_project_id}) does not match "
                    f"setlist's project ({setlist.project_id}). Please open the correct project."
                )
            
            project = self._project_service.load_project(setlist.project_id)
            if not project:
                return False, f"Project not found: {setlist.project_id}"
            
            # Get snapshot from project file
            snapshot_data = self._project_service.get_snapshot(song_id, project)
            if not snapshot_data:
                return False, "Song has not been processed yet. Please process the song first."
            
            # Parse snapshot
            try:
                snapshot = DataStateSnapshot.from_dict(snapshot_data)
            except Exception as e:
                return False, f"Failed to parse snapshot: {e}"
            
            # Restore snapshot
            event_bus = getattr(self._facade, 'event_bus', None)
            if not event_bus:
                Log.warning("SetlistSnapshotService: No event_bus available, UI may not refresh correctly")
            
            from src.utils.paths import get_project_workspace_dir
            project_dir = get_project_workspace_dir(project.id) if project.id else None
            
            try:
                Log.info(
                    f"SetlistSnapshotService: Restoring snapshot for song {snapshot.song_id} "
                    f"({len(snapshot.data_items)} data items, {len(snapshot.block_local_state)} block states)"
                )
                self._snapshot_service.restore_snapshot(
                    project_id=project.id,
                    snapshot=snapshot,
                    project_dir=project_dir,
                    event_bus=event_bus,
                    progress_callback=None
                )
                Log.info(f"SetlistSnapshotService: Successfully restored snapshot for song {snapshot.song_id}")
            except Exception as e:
                error_msg = str(e)
                Log.error(f"SetlistSnapshotService: Failed to restore snapshot: {error_msg}")
                import traceback
                Log.error(traceback.format_exc())
                return False, f"Failed to restore snapshot: {error_msg}"
            
            Log.info(f"SetlistSnapshotService: Successfully switched to song {song.audio_path}")
            return True, None
            
        except Exception as e:
            error_msg = f"Unexpected error during switch: {e}"
            Log.error(f"SetlistSnapshotService: {error_msg}")
            import traceback
            Log.error(traceback.format_exc())
            return False, error_msg
            
        finally:
            self._is_switching = False
            self._switching_lock.release()
