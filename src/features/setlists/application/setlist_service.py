"""
Setlist Service

Coordinator for setlist-related use cases. Delegates to:
- SetlistProcessingService: Song processing, action execution, pre/post hooks
- SetlistSnapshotService: Song switching via snapshot save/restore

Direct responsibilities:
- Creating setlists from audio folders
- Discovering available block actions
- Managing songs (add/remove)

Follows EchoZero standards:
- Lean: Coordinator delegates to focused services
- Explicit: Every step visible, no magic
- Simple: Clear structure, easy to understand
"""
from typing import Optional, Dict, Any, List, Callable, Tuple
from pathlib import Path
import glob

from src.features.setlists.domain import Setlist
from src.features.setlists.domain import SetlistSong
from src.features.setlists.domain import SetlistRepository
from src.features.setlists.domain import SetlistSongRepository
from src.features.blocks.domain import BlockRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.features.projects.application import SnapshotService
from src.features.projects.application import ProjectService
from src.features.execution.application import BlockExecutionEngine
from src.application.api.application_facade import ApplicationFacade
from src.application.blocks.quick_actions import get_quick_actions
from src.features.setlists.application.setlist_processing_service import SetlistProcessingService
from src.features.setlists.application.setlist_snapshot_service import SetlistSnapshotService
from src.utils.message import Log


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
        data_item_repo: DataItemRepository,
        block_local_state_repo: BlockLocalStateRepository,
        snapshot_service: SnapshotService,
        project_service: ProjectService,
        execution_engine: BlockExecutionEngine,
        facade: ApplicationFacade
    ):
        """
        Initialize setlist service.
        
        Args:
            setlist_repo: Repository for setlists
            setlist_song_repo: Repository for setlist songs
            block_repo: Repository for blocks
            data_item_repo: Repository for data items
            block_local_state_repo: Repository for block local state
            snapshot_service: Service for saving/restoring snapshots
            project_service: Service for loading projects
            execution_engine: Execution engine for running blocks
            facade: Application facade for executing projects
        """
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._block_repo = block_repo
        self._data_item_repo = data_item_repo
        self._block_local_state_repo = block_local_state_repo
        self._snapshot_service = snapshot_service
        self._project_service = project_service
        self._execution_engine = execution_engine
        self._facade = facade
        
        # Delegate services
        self._processing_service = SetlistProcessingService(
            setlist_repo=setlist_repo,
            setlist_song_repo=setlist_song_repo,
            block_repo=block_repo,
            data_item_repo=data_item_repo,
            block_local_state_repo=block_local_state_repo,
            snapshot_service=snapshot_service,
            project_service=project_service,
            execution_engine=execution_engine,
            facade=facade
        )
        self._snapshot_switch_service = SetlistSnapshotService(
            setlist_repo=setlist_repo,
            setlist_song_repo=setlist_song_repo,
            snapshot_service=snapshot_service,
            project_service=project_service,
            facade=facade
        )
        
        Log.info("SetlistService: Initialized")
    
    def create_setlist_from_folder(
        self,
        audio_folder_path: str,
        default_actions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Setlist:
        """
        Create a new setlist from audio folder - uses current project.
        
        Args:
            audio_folder_path: Path to folder containing audio files
            default_actions: Optional default actions for all songs
            
        Returns:
            Created Setlist entity
            
        Raises:
            ValueError: If no current project or folder doesn't exist
        """
        from datetime import datetime
        
        # Get current project
        if not self._facade.current_project_id:
            raise ValueError("No current project. Please open or create a project first.")
        
        project = self._project_service.load_project(self._facade.current_project_id)
        if not project:
            raise ValueError(f"Current project not found: {self._facade.current_project_id}")
        
        # Validate folder exists
        folder_path = Path(audio_folder_path).expanduser()
        if not folder_path.is_dir():
            raise ValueError(f"Audio folder does not exist: {audio_folder_path}")
        
        # Get or create setlist for this project (one setlist per project)
        existing_setlist = self._setlist_repo.get_by_project(project.id)
        
        # Scan folder for audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aif', '.aiff']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(str(folder_path / f"*{ext}")))
            audio_files.extend(glob.glob(str(folder_path / f"*{ext.upper()}")))
        
        # Sort for consistent ordering
        audio_files = sorted(audio_files)
        
        if not audio_files:
            raise ValueError(f"No audio files found in folder: {audio_folder_path}")
        
        if existing_setlist:
            # Update existing setlist (one per project)
            existing_setlist.audio_folder_path = str(folder_path)
            if default_actions:
                existing_setlist.default_actions = default_actions
            existing_setlist.update_modified()
            setlist = self._setlist_repo.update(existing_setlist)
            setlist = existing_setlist  # update() doesn't return, use existing
            
            # Clear existing songs and add new ones from folder
            self._setlist_song_repo.delete_by_setlist(setlist.id)
            Log.info(f"SetlistService: Updated setlist for project, clearing old songs")
        else:
            # Create new setlist
            setlist = Setlist(
                id="",
                audio_folder_path=str(folder_path),
                project_id=project.id,
                default_actions=default_actions or {},
                created_at=datetime.utcnow(),
                modified_at=datetime.utcnow()
            )
            setlist = self._setlist_repo.create(setlist)
        
        # Add songs for each audio file
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
        
        Uses existing quick_actions system for blocks, adds project-level actions.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Dict mapping block_id or "project" -> {
                "block_name": str,
                "block_type": str,
                "actions": [action1, action2, ...]
            }
            Each action dict contains: name, description, category, icon, etc.
        """
        blocks = self._block_repo.list_by_project(project_id)
        actions_by_block = {}
        
        # Discover block-level actions
        for block in blocks:
            # Get quick actions for this block type
            quick_actions = get_quick_actions(block.type)
            
            # Convert to dict format for UI
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
            
            # Include ALL blocks, even if they have no actions
            # This allows users to see all blocks in the dropdown
            actions_by_block[block.id] = {
                "block_name": block.name,
                "block_type": block.type,
                "actions": action_list  # Can be empty list
            }
        
        # Add project-level actions
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
    
    def process_song(
        self,
        setlist_id: str,
        song_id: str,
        progress_callback: Optional[Any] = None,
        action_progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> bool:
        """
        Process a single song through the project action pipeline.
        Delegates to SetlistProcessingService.
        """
        return self._processing_service.process_song(
            setlist_id=setlist_id,
            song_id=song_id,
            progress_callback=progress_callback,
            action_progress_callback=action_progress_callback
        )
    
    def switch_active_song(
        self,
        setlist_id: str,
        song_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Switch active song by restoring its snapshot.
        Delegates to SetlistSnapshotService.
        """
        return self._snapshot_switch_service.switch_active_song(setlist_id, song_id)
    
    def process_setlist(
        self,
        setlist_id: str,
        error_callback: Optional[Callable[[str, str], None]] = None,
        action_progress_callback: Optional[Callable[[str, int, int, str, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, bool]:
        """
        Process all songs in a setlist with error recovery.
        Delegates to SetlistProcessingService.
        """
        return self._processing_service.process_setlist(
            setlist_id=setlist_id,
            error_callback=error_callback,
            action_progress_callback=action_progress_callback,
            cancel_check=cancel_check
        )
    
    def validate_action_items(self, project_id: str) -> List[str]:
        """
        Validate action items before processing begins.
        Delegates to SetlistProcessingService.
        """
        return self._processing_service.validate_action_items(project_id)
    
    def add_song_to_setlist(self, project_id: str, audio_path: str) -> SetlistSong:
        """
        Add a song to the project's setlist.
        
        One setlist per project - setlist should already exist (created on project init).
        
        Args:
            project_id: Project identifier
            audio_path: Path to audio file
            
        Returns:
            Created SetlistSong entity
            
        Raises:
            ValueError: If setlist doesn't exist or audio file doesn't exist
        """
        # Get existing setlist (one per project - should always exist)
        setlist = self._setlist_repo.get_by_project(project_id)
        if not setlist:
            raise ValueError(f"Project {project_id} has no setlist. Setlists are auto-created on project initialization.")
        
        # Validate audio file exists
        audio_file = Path(audio_path).expanduser()
        if not audio_file.is_file():
            raise ValueError(f"Audio file does not exist: {audio_path}")
        
        # Get current max order_index
        existing_songs = self._setlist_song_repo.list_by_setlist(setlist.id)
        max_order = max([s.order_index for s in existing_songs], default=-1)
        
        # Create new song
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
        # Get project's setlist (one per project)
        setlist = self._setlist_repo.get_by_project(project_id)
        if not setlist:
            raise ValueError(f"Project {project_id} has no setlist.")
        
        # Verify song belongs to this setlist
        song = self._setlist_song_repo.get(song_id)
        if not song:
            raise ValueError(f"Song {song_id} not found")
        
        if song.setlist_id != setlist.id:
            raise ValueError(f"Song {song_id} does not belong to project's setlist")
        
        # Delete song
        self._setlist_song_repo.delete(song_id)
        
        # Reorder remaining songs
        remaining_songs = self._setlist_song_repo.list_by_setlist(setlist.id)
        for index, remaining_song in enumerate(sorted(remaining_songs, key=lambda s: s.order_index)):
            if remaining_song.order_index != index:
                remaining_song.order_index = index
                self._setlist_song_repo.update(remaining_song)
        
        setlist.update_modified()
        self._setlist_repo.update(setlist)
        
        Log.info(f"SetlistService: Removed song {song_id} from setlist")

