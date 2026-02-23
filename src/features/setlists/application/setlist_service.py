"""
Setlist Service

Manages setlist CRUD and song switching. Delegates to:
- SetlistSnapshotService: Song switching via snapshot save/restore

Processing is handled directly by SetlistProcessingService
(called from the facade without going through this service).

Responsibilities:
- Creating setlists from audio folders
- Discovering available block actions
- Managing songs (add/remove)
- Song switching (via snapshot service)
"""
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
from pathlib import Path
import glob

from src.features.setlists.domain import Setlist
from src.features.setlists.domain import SetlistSong
from src.features.setlists.domain import SetlistRepository
from src.features.setlists.domain import SetlistSongRepository
from src.features.blocks.domain import BlockRepository
from src.features.projects.application import SnapshotService
from src.features.projects.application import ProjectService
from src.application.blocks.quick_actions import get_quick_actions
from src.features.setlists.application.setlist_snapshot_service import SetlistSnapshotService
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SetlistService:
    """
    Coordinator service for managing setlists.

    Handles:
    - Creating setlists from audio folders (uses current project)
    - Discovering available block actions
    - Managing songs (add/remove)

    Delegates to:
    - SetlistProcessingService for song/setlist processing
    - SetlistSnapshotService for song switching
    """

    def __init__(
        self,
        setlist_repo: SetlistRepository,
        setlist_song_repo: SetlistSongRepository,
        block_repo: BlockRepository,
        snapshot_service: SnapshotService,
        project_service: ProjectService,
    ):
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._block_repo = block_repo
        self._project_service = project_service

        self._snapshot_switch_service = SetlistSnapshotService(
            setlist_repo=setlist_repo,
            setlist_song_repo=setlist_song_repo,
            snapshot_service=snapshot_service,
            project_service=project_service,
        )

        Log.info("SetlistService: Initialized")

    def create_setlist_from_folder(
        self,
        audio_folder_path: str,
        facade: "ApplicationFacade",
        default_actions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Setlist:
        """
        Create a new setlist from audio folder - uses current project.

        Args:
            audio_folder_path: Path to folder containing audio files
            facade: ApplicationFacade for current project context
            default_actions: Optional default actions for all songs

        Returns:
            Created Setlist entity

        Raises:
            ValueError: If no current project or folder doesn't exist
        """
        from datetime import datetime

        if not facade.current_project_id:
            raise ValueError("No current project. Please open or create a project first.")

        project = self._project_service.load_project(facade.current_project_id)
        if not project:
            raise ValueError(f"Current project not found: {facade.current_project_id}")

        folder_path = Path(audio_folder_path).expanduser()
        if not folder_path.is_dir():
            raise ValueError(f"Audio folder does not exist: {audio_folder_path}")

        existing_setlist = self._setlist_repo.get_by_project(project.id)

        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aif', '.aiff']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(str(folder_path / f"*{ext}")))
            audio_files.extend(glob.glob(str(folder_path / f"*{ext.upper()}")))

        audio_files = sorted(audio_files)

        if not audio_files:
            raise ValueError(f"No audio files found in folder: {audio_folder_path}")

        if existing_setlist:
            existing_setlist.audio_folder_path = str(folder_path)
            if default_actions:
                existing_setlist.default_actions = default_actions
            existing_setlist.update_modified()
            setlist = self._setlist_repo.update(existing_setlist)
            setlist = existing_setlist
            self._setlist_song_repo.delete_by_setlist(setlist.id)
            Log.info(f"SetlistService: Updated setlist for project, clearing old songs")
        else:
            setlist = Setlist(
                id="",
                audio_folder_path=str(folder_path),
                project_id=project.id,
                default_actions=default_actions or {},
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow()
            )
            setlist = self._setlist_repo.create(setlist)

        for index, audio_path in enumerate(audio_files):
            song = SetlistSong(
                id="",
                setlist_id=setlist.id,
                audio_path=audio_path,
                order_index=index
            )
            self._setlist_song_repo.create(song)

        Log.info(f"SetlistService: {'Updated' if existing_setlist else 'Created'} setlist with {len(audio_files)} song(s) from folder {audio_folder_path}")
        return setlist

    def discover_available_actions(self, project_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Discover all available actions for blocks and project-level operations.

        Args:
            project_id: Project identifier

        Returns:
            Dict mapping block_id or "project" -> action info dict
        """
        blocks = self._block_repo.list_by_project(project_id)
        actions_by_block = {}

        for block in blocks:
            quick_actions = get_quick_actions(block.type)

            action_list = []
            for action in quick_actions:
                action_dict = {
                    "name": action.name,
                    "description": action.description,
                    "category": action.category.value if hasattr(action.category, 'value') else str(action.category),
                    "icon": action.icon,
                    "primary": action.primary,
                    "dangerous": action.dangerous,
                    "keyboard_shortcut": action.keyboard_shortcut
                }
                action_list.append(action_dict)

            actions_by_block[block.id] = {
                "block_name": block.name,
                "block_type": block.type,
                "actions": action_list
            }

        project_actions = [
            {
                "category": "execute",
                "icon": "play",
                "primary": True,
                "dangerous": False,
                "keyboard_shortcut": None
            },
            {
                "name": "validate_project",
                "description": "Validate project graph (check connections and data types)",
                "category": "configure",
                "icon": "check",
                "primary": False,
                "dangerous": False,
                "keyboard_shortcut": None
            },
            {
                "name": "save_project",
                "description": "Save project changes to disk",
                "category": "file",
                "icon": "save",
                "primary": False,
                "dangerous": False,
                "keyboard_shortcut": None
            },
            {
                "name": "save_as",
                "description": "Save project to new location (supports {song_name}, {song_audio_path}, etc.)",
                "category": "file",
                "icon": "save_as",
                "primary": False,
                "dangerous": False,
                "keyboard_shortcut": None
            }
        ]

        actions_by_block["project"] = {
            "block_name": "Project",
            "block_type": "Project",
            "actions": project_actions
        }

        Log.debug(f"SetlistService: Discovered actions for {len(actions_by_block)} block(s) + project actions")
        return actions_by_block

    def switch_active_song(
        self,
        setlist_id: str,
        song_id: str,
        facade: "ApplicationFacade"
    ) -> Tuple[bool, Optional[str]]:
        """
        Switch active song by restoring its snapshot.
        Delegates to SetlistSnapshotService.
        """
        return self._snapshot_switch_service.switch_active_song(setlist_id, song_id, facade)

    def add_song_to_setlist(self, project_id: str, audio_path: str) -> SetlistSong:
        """
        Add a song to the project's setlist.

        Args:
            project_id: Project identifier
            audio_path: Path to audio file

        Returns:
            Created SetlistSong entity

        Raises:
            ValueError: If setlist doesn't exist or audio file doesn't exist
        """
        setlist = self._setlist_repo.get_by_project(project_id)
        if not setlist:
            raise ValueError(f"Project {project_id} has no setlist. Setlists are auto-created on project initialization.")

        audio_file = Path(audio_path).expanduser()
        if not audio_file.is_file():
            raise ValueError(f"Audio file does not exist: {audio_path}")

        existing_songs = self._setlist_song_repo.list_by_setlist(setlist.id)
        max_order = max([s.order_index for s in existing_songs], default=-1)

        song = SetlistSong(
            id="",
            setlist_id=setlist.id,
            audio_path=str(audio_file),
            order_index=max_order + 1
        )
        song = self._setlist_song_repo.create(song)

        setlist.update_modified()
        self._setlist_repo.update(setlist)

        Log.info(f"SetlistService: Added song {audio_path} to setlist")
        return song

    def remove_song_from_setlist(self, project_id: str, song_id: str) -> None:
        """
        Remove a song from the project's setlist.

        Args:
            project_id: Project identifier
            song_id: Song identifier to remove

        Raises:
            ValueError: If project has no setlist or song not found
        """
        setlist = self._setlist_repo.get_by_project(project_id)
        if not setlist:
            raise ValueError(f"Project {project_id} has no setlist.")

        song = self._setlist_song_repo.get(song_id)
        if not song:
            raise ValueError(f"Song {song_id} not found")

        if song.setlist_id != setlist.id:
            raise ValueError(f"Song {song_id} does not belong to project's setlist")

        self._setlist_song_repo.delete(song_id)

        remaining_songs = self._setlist_song_repo.list_by_setlist(setlist.id)
        for index, remaining_song in enumerate(sorted(remaining_songs, key=lambda s: s.order_index)):
            if remaining_song.order_index != index:
                remaining_song.order_index = index
                self._setlist_song_repo.update(remaining_song)

        setlist.update_modified()
        self._setlist_repo.update(setlist)

        Log.info(f"SetlistService: Removed song {song_id} from setlist")
