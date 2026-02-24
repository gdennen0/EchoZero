"""
Project Service

Orchestrates project-related use cases.
Handles business rules and emits domain events.
"""
import json
import os
import tempfile
import shutil
import zipfile
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

from src.features.blocks.domain import Block
from src.features.connections.domain import Connection
from src.features.projects.domain.project import Project
from src.shared.domain.entities import DataItem
from src.shared.domain.entities import AudioDataItem as ExportedAudioDataItem
from src.shared.domain.entities import EventDataItem as ExportedEventDataItem
from src.shared.domain.entities.layer_order import LayerOrder, LayerKey
from src.features.setlists.domain import Setlist
from src.features.setlists.domain import SetlistSong
from src.features.projects.domain.project_repository import ProjectRepository
from src.application.events.event_bus import EventBus
from src.application.events import ProjectCreated, ProjectUpdated, ProjectDeleted
from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log
from src.utils.recent_projects import RecentProjectsStore


class ProjectService:
    """
    Service for managing projects.
    
    Orchestrates project lifecycle operations:
    - Creating projects
    - Loading projects
    - Updating projects
    - Deleting projects
    
    Emits domain events for UI synchronization.
    """
    
    def __init__(
        self, 
        project_repo: ProjectRepository, 
        event_bus: EventBus,
        block_repo=None,
        connection_repo=None,
        data_item_repo=None,
        database: Optional[Database] = None,
        recent_store: Optional[RecentProjectsStore] = None,
        ui_state_repo=None,
        block_local_state_repo=None,
        layer_order_repo=None,
        setlist_repo=None,
        setlist_song_repo=None,
        action_set_repo=None,
        action_item_repo=None
    ):
        """
        Initialize project service.
        
        Args:
            project_repo: Repository for project persistence
            event_bus: Event bus for publishing domain events
            block_repo: Optional block repository (for exporting project data)
            connection_repo: Optional connection repository (for exporting project data)
            data_item_repo: Optional data item repository (for exporting project data)
            ui_state_repo: Optional UI state repository (for block positions)
            block_local_state_repo: Optional block local state repository (for input/output references)
            layer_order_repo: Optional layer order repository (for editor layer ordering)
            setlist_repo: Optional setlist repository (for exporting project data)
            setlist_song_repo: Optional setlist song repository (for exporting project data)
            action_set_repo: Optional action set repository (for exporting project data)
            action_item_repo: Optional action item repository (for exporting project data)
        """
        self._project_repo = project_repo
        self._event_bus = event_bus
        self._block_repo = block_repo
        self._connection_repo = connection_repo
        self._data_item_repo = data_item_repo
        self._database = database
        self._recent_store = recent_store
        self._ui_state_repo = ui_state_repo
        self._block_local_state_repo = block_local_state_repo
        self._layer_order_repo = layer_order_repo
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._action_set_repo = action_set_repo
        self._action_item_repo = action_item_repo
        # In-memory snapshot store: song_id -> snapshot_dict
        self._loaded_snapshots: Dict[str, dict] = {}
        # Single-entry cache for fast repeated access to the same snapshot
        self._snapshot_cache: Optional[Tuple[str, dict]] = None  # (song_id, snapshot_dict)
        # Track snapshots that exist only in memory and need to be flushed to disk
        self._dirty_snapshot_ids: set = set()
        Log.info("ProjectService: Initialized")
    
    def set_snapshot(self, song_id: str, snapshot_dict: dict, project: Optional[Project] = None) -> None:
        """
        Save snapshot directly to project file (memory-efficient).
        
        Reads project file, updates snapshots dict, writes back to file immediately.
        Also updates cache for fast subsequent access.
        
        Args:
            song_id: Song identifier
            snapshot_dict: Snapshot dictionary (from DataStateSnapshot.to_dict())
            project: Optional Project entity (if not provided, will try to find from current project)
        """
        # Get project if not provided
        if not project:
            # Try to get current project from repository
            # This is a fallback - callers should provide project for efficiency
            projects = self._project_repo.list_all()
            if projects:
                project = projects[0]  # Use first project as fallback
            else:
                Log.warning(f"ProjectService: Cannot save snapshot - no project found")
                return
        
        project_file = self._get_export_file_path(project)
        if not project_file or not project_file.exists():
            Log.warning(f"ProjectService: Cannot save snapshot - project file not found: {project_file}")
            return
        
        try:
            project_dir = project_file.parent
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.ez.tmp',
                prefix='.snapshot_save_',
                dir=project_dir
            )
            
            try:
                # Check if file is a ZIP archive
                if zipfile.is_zipfile(project_file):
                    # New ZIP format - read project.json, update, write back to ZIP
                    with zipfile.ZipFile(project_file, 'r') as zip_read:
                        # Read project.json
                        if "project.json" not in zip_read.namelist():
                            raise ValueError("project.json not found in ZIP archive")
                        project_json_str = zip_read.read("project.json").decode('utf-8')
                        data = json.loads(project_json_str)
                        
                        # Update snapshots dict
                        if "snapshots" not in data:
                            data["snapshots"] = {}
                        data["snapshots"][song_id] = snapshot_dict
                        
                        # Write updated ZIP
                        with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zip_write:
                            # Write updated project.json
                            project_json_str = json.dumps(data, indent=2, ensure_ascii=False)
                            zip_write.writestr("project.json", project_json_str.encode('utf-8'))
                            
                            # Copy all other files from original ZIP
                            for item in zip_read.namelist():
                                if item != "project.json":
                                    zip_write.writestr(item, zip_read.read(item))
                else:
                    # Legacy JSON format
                    with open(project_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Update snapshots dict
                    if "snapshots" not in data:
                        data["snapshots"] = {}
                    data["snapshots"][song_id] = snapshot_dict
                    
                    # Write back to file
                    with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        temp_fd = None  # Prevent closing in finally
                
                # Atomic rename
                if temp_fd is not None:
                    os.close(temp_fd)
                shutil.move(temp_path, str(project_file))
            except Exception as e:
                # Clean up temp file on error
                if temp_fd is not None:
                    try:
                        os.close(temp_fd)
                    except:
                        pass
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                raise e
            
            # Keep in-memory stores in sync
            self._loaded_snapshots[song_id] = snapshot_dict
            self._snapshot_cache = (song_id, snapshot_dict)
            self._dirty_snapshot_ids.discard(song_id)
            
            Log.debug(f"ProjectService: Saved snapshot for song {song_id} to project file")
        except Exception as e:
            Log.error(f"ProjectService: Failed to save snapshot to project file: {e}")
            raise

    def cache_snapshot(self, song_id: str, snapshot_dict: dict) -> None:
        """
        Update in-memory snapshot cache without writing to the project file.

        Marks the snapshot as dirty so it will be flushed to disk on the
        next project save or explicit flush_dirty_snapshots() call.
        """
        self._loaded_snapshots[song_id] = snapshot_dict
        self._snapshot_cache = (song_id, snapshot_dict)
        self._dirty_snapshot_ids.add(song_id)

    def flush_dirty_snapshots(self, project: Optional["Project"] = None) -> None:
        """
        Write any cached-only snapshots to the project file on disk.

        Called during project save and application shutdown to ensure
        no snapshot data is lost.
        """
        if not self._dirty_snapshot_ids:
            return

        if not project:
            projects = self._project_repo.list_all()
            project = projects[0] if projects else None
        if not project:
            Log.warning("ProjectService: Cannot flush dirty snapshots - no project")
            return

        dirty_ids = list(self._dirty_snapshot_ids)
        Log.info(f"ProjectService: Flushing {len(dirty_ids)} dirty snapshot(s) to disk")
        for sid in dirty_ids:
            snap = self._loaded_snapshots.get(sid)
            if snap:
                try:
                    self.set_snapshot(sid, snap, project)
                except Exception as e:
                    Log.error(f"ProjectService: Failed to flush snapshot {sid}: {e}")

    def get_snapshot(self, song_id: str, project: Optional[Project] = None) -> Optional[dict]:
        """
        Get snapshot dictionary for a song.
        
        Checks single-entry cache, then in-memory store, then falls back to
        reading the project file for legacy compatibility.
        
        Args:
            song_id: Song identifier
            project: Optional Project entity (if not provided, will try to find from current project)
            
        Returns:
            Snapshot dictionary or None if not found
        """
        if self._snapshot_cache and self._snapshot_cache[0] == song_id:
            return self._snapshot_cache[1]
        
        # Check in-memory store (populated on project load/import)
        snapshot = self._loaded_snapshots.get(song_id)
        if snapshot is not None:
            self._snapshot_cache = (song_id, snapshot)
            return snapshot
        
        # Fallback: read from .ez file for cases where _loaded_snapshots
        # wasn't populated (e.g. snapshot written by external tool)
        if not project:
            projects = self._project_repo.list_all()
            if projects:
                project = projects[0]
            else:
                Log.warning(f"ProjectService: Cannot load snapshot - no project found")
                return None
        
        project_file = self._get_export_file_path(project)
        if not project_file or not project_file.exists():
            return None
        
        try:
            if zipfile.is_zipfile(project_file):
                with zipfile.ZipFile(project_file, 'r') as zip_file:
                    if "project.json" not in zip_file.namelist():
                        return None
                    project_json_str = zip_file.read("project.json").decode('utf-8')
                    data = json.loads(project_json_str)
            else:
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            snapshots = data.get("snapshots", {})
            snapshot = snapshots.get(song_id)
            
            if snapshot:
                self._loaded_snapshots[song_id] = snapshot
                self._snapshot_cache = (song_id, snapshot)
            
            return snapshot
        except Exception as e:
            Log.warning(f"ProjectService: Failed to load snapshot from project file: {e}")
            return None
    
    def create_project(self, name: str = "Untitled", save_directory: Optional[str] = None, version: str = "0.0.1") -> Project:
        """
        Create a new project.
        
        Projects start as "untitled" (unsaved) until first save.
        Use save_project_as() to set the save location.
        
        Args:
            name: Project name (default: "Untitled")
            save_directory: Directory where project will be saved (None for untitled/unsaved)
            version: Project version (default: "0.0.1")
            
        Returns:
            Created Project entity
            
        Raises:
            ValueError: If name validation fails
        """
        # Validate inputs
        if not name or not name.strip():
            name = "Untitled"
        
        # Create project entity (untitled if no save_directory)
        project = Project(
            id="",  # Will be generated
            name=name.strip(),
            version=version,
            save_directory=save_directory,  # Can be None for untitled
            created_at=datetime.now(timezone.utc),
            modified_at=datetime.now(timezone.utc)
        )
        
        # Save to repository
        created_project = self._project_repo.create(project)
        
        # Fresh project has no snapshots
        self._loaded_snapshots = {}
        self._snapshot_cache = None
        
        # Emit event
        self._event_bus.publish(ProjectCreated(
            project_id=created_project.id,
            data={"name": created_project.name, "version": created_project.version, "untitled": created_project.is_untitled()}
        ))
        
        status = "untitled" if created_project.is_untitled() else "saved"
        Log.info(f"ProjectService: Created {status} project '{created_project.name}' (id: {created_project.id})")
        return created_project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """
        Get a project by ID without reloading from disk.
        
        Lightweight lookup from the in-memory repository. Use this when you
        already have the project loaded and just need the entity (e.g. during
        song switching).
        """
        return self._project_repo.get(project_id)

    def load_project(self, project_id: str) -> Optional[Project]:
        """
        Load a project by ID.
        
        If project file exists, also reload snapshots from file.
        Use get_project() instead when the project is already loaded and
        you don't need to re-read the project file.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project entity or None if not found
        """
        project = self._project_repo.get(project_id)
        
        if project:
            project_file = self._get_export_file_path(project)
            self._record_recent_project(project, project_file)
            
            # Populate in-memory snapshot store from .ez file so saves preserve them
            self._snapshot_cache = None
            snapshots = self._read_snapshots_from_file(project)
            self._loaded_snapshots = dict(snapshots) if snapshots else {}
            if self._loaded_snapshots:
                Log.debug(f"ProjectService: Loaded {len(self._loaded_snapshots)} snapshot(s) into memory")
            
            Log.info(f"ProjectService: Loaded project '{project.name}' (id: {project_id})")
        else:
            Log.warning(f"ProjectService: Project not found: {project_id}")
        
        return project
    
    def save_project(self, project: Project) -> None:
        """
        Save project changes.
        
        If project is untitled, raises ValueError. Use save_project_as() instead.
        
        Args:
            project: Project entity to save
            
        Raises:
            ValueError: If project is untitled or not found
        """
        if project.is_untitled():
            raise ValueError("Cannot save untitled project. Use save_project_as() to set save location first.")
        
        # Update modified timestamp
        project.update_modified()
        
        # Save to repository
        self._project_repo.update(project)
        
        # Write project file to disk
        project_file_path = self._write_project_file(project)
        self._record_recent_project(project, project_file_path)
        
        # Emit event
        self._event_bus.publish(ProjectUpdated(
            project_id=project.id,
            data={"name": project.name, "version": project.version}
        ))
        
        Log.info(f"ProjectService: Saved project '{project.name}' (id: {project.id})")
    
    def save_project_as(self, project_id: str, save_directory: str, name: Optional[str] = None) -> Project:
        """
        Save project with a new location (Save As).
        
        If project is untitled, this sets the save location.
        If project already has a location, this creates a copy at the new location.
        
        Args:
            project_id: Project identifier
            save_directory: Directory where project will be saved
            name: Optional new name for the project
            
        Returns:
            Updated Project entity
            
        Raises:
            ValueError: If project not found or validation fails
        """
        project = self.load_project(project_id)
        if not project:
            raise ValueError(f"Project with id '{project_id}' not found")
        
        if not save_directory or not save_directory.strip():
            raise ValueError("Save directory cannot be empty")
        
        # Update name if provided
        if name and name.strip():
            project.rename(name.strip())
        
        # Set save directory
        project.set_save_directory(save_directory.strip())
        
        # Save to repository
        self._project_repo.update(project)
        
        # Write project file to disk
        project_file = self._write_project_file(project)
        self._record_recent_project(project, project_file)
        
        # Emit event
        self._event_bus.publish(ProjectUpdated(
            project_id=project.id,
            data={"name": project.name, "version": project.version, "save_directory": project.save_directory}
        ))
        
        Log.info(f"ProjectService: Saved project '{project.name}' to '{project.save_directory}' (id: {project.id})")
        return project

    def import_project_from_file(self, file_path: str) -> Project:
        """
        Import a project from a saved .ez ZIP archive into the runtime database.
        
        ZIP projects are extracted to a workspace directory in the application
        cache (not next to the .ez file). This keeps the user's save directory
        clean and makes .ez files fully portable.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"Project file not found: {file_path}")

        workspace_dir = None
        
        try:
            if not zipfile.is_zipfile(file_path):
                raise ValueError(f"Project file is not a valid .ez ZIP archive: {file_path}")
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                try:
                    project_json_bytes = zip_file.read("project.json")
                except KeyError:
                    raise ValueError(f"project.json not found in ZIP archive: {file_path}")
                
                data = json.loads(project_json_bytes.decode('utf-8'))
                
                from src.utils.paths import get_project_workspace_dir, cleanup_project_workspace
                project_id = data.get("project_id", "")
                if not project_id:
                    raise ValueError(f"project_id missing from project.json in: {file_path}")
                
                cleanup_project_workspace(project_id)
                workspace_dir = get_project_workspace_dir(project_id)
                
                zip_file.extractall(workspace_dir)
                Log.debug(f"ProjectService: Extracted ZIP project to workspace {workspace_dir}")
            
            Log.info(f"ProjectService: Loaded project from ZIP archive: {file_path}")
        except zipfile.BadZipFile:
            raise ValueError(f"Project file is not a valid .ez ZIP archive: {file_path}")
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in project file '{file_path}': {e.msg} (line {e.lineno}, col {e.colno})"
            Log.error(f"ProjectService: {error_msg}")
            raise ValueError(error_msg) from e

        project = Project.from_dict(data)
        
        # Update save_directory to the local machine path (the .ez file's parent).
        # The project may have been saved on a different machine with a different path.
        local_save_dir = str(Path(file_path).parent)
        if project.save_directory != local_save_dir:
            Log.info(
                f"ProjectService: Updating save_directory from '{project.save_directory}' "
                f"to local path '{local_save_dir}'"
            )
            project.set_save_directory(local_save_dir)
        
        self._clear_runtime_tables()
        self._loaded_snapshots = {}  # Clear snapshots when loading new project
        created_project = self._project_repo.create(project)

        if self._block_repo:
            # Import here to avoid circular dependency
            from src.application.block_registry import get_block_registry
            from src.features.blocks.domain import PortDirection
            
            registry = get_block_registry()
            
            for block_data in data.get("blocks", []):
                block_data["project_id"] = created_project.id
                block = Block.from_dict(block_data)
                self._block_repo.create(block)
                
                # Ensure block has all default ports according to registry
                # This ensures blocks loaded from old project files get new ports
                block_metadata = registry.get(block.type)
                if block_metadata:
                    ports_added = False
                    
                    # Check and add input ports
                    for port_name, port_type in block_metadata.inputs.items():
                        if not block.get_port(port_name, PortDirection.INPUT):
                            try:
                                block.add_port(port_name, port_type, PortDirection.INPUT)
                                ports_added = True
                            except ValueError:
                                pass
                    
                    # Check and add output ports
                    for port_name, port_type in block_metadata.outputs.items():
                        if not block.get_port(port_name, PortDirection.OUTPUT):
                            try:
                                block.add_port(port_name, port_type, PortDirection.OUTPUT)
                                ports_added = True
                            except ValueError:
                                pass
                    
                    # Check and add bidirectional ports
                    if hasattr(block_metadata, 'bidirectional') and block_metadata.bidirectional:
                        for port_name, port_type in block_metadata.bidirectional.items():
                            if not block.get_port(port_name, PortDirection.BIDIRECTIONAL):
                                try:
                                    block.add_port(port_name, port_type, PortDirection.BIDIRECTIONAL)
                                    ports_added = True
                                except ValueError:
                                    pass
                    
                    # Update block in repository if ports were added
                    if ports_added:
                        self._block_repo.update(block)

        if self._connection_repo:
            for conn_data in data.get("connections", []):
                connection = Connection.from_dict(conn_data)
                self._connection_repo.create(connection)

        # Restore data items with file validation
        missing_files = []
        resolve_dir = workspace_dir if workspace_dir else Path(file_path).parent
        
        raw_data_items = data.get("data_items", [])
        audio_count = sum(1 for d in raw_data_items if (d.get("type") or "").lower() == "audio")
        event_count = sum(1 for d in raw_data_items if (d.get("type") or "").lower() == "event")
        Log.info(f"ProjectService: Restoring {len(raw_data_items)} data item(s) ({audio_count} audio, {event_count} event) from project file")
        
        if self._data_item_repo:
            for item_data in raw_data_items:
                data_item = self._build_data_item_from_dict(item_data)
                
                # Clean stale metadata keys that should not persist across machines
                for stale_key in ("file_path", "sample_rate", "channels", "duration_samples", "_file_valid", "original_path"):
                    data_item.metadata.pop(stale_key, None)
                
                # Resolve file path (all .ez paths are relative)
                if data_item.file_path:
                    resolved_path = resolve_dir / data_item.file_path
                    data_item.file_path = str(resolved_path)
                    
                    if not os.path.exists(resolved_path):
                        missing_files.append({
                            'data_item': data_item.name,
                            'block_id': data_item.block_id,
                            'path': str(resolved_path),
                            'original_relative_path': item_data.get('file_path')
                        })
                        Log.error(f"ProjectService: File not found for DataItem '{data_item.name}': {resolved_path}")
                
                # Restore waveform (AudioDataItems only)
                if isinstance(data_item, ExportedAudioDataItem):
                    waveform_info = data_item.metadata.get("waveform", {})
                    waveform_rel_path = waveform_info.get("file_path")
                    if waveform_rel_path:
                        try:
                            resolved_waveform = resolve_dir / waveform_rel_path
                            
                            if resolved_waveform.exists():
                                from src.shared.application.services.waveform_service import get_waveform_service
                                import numpy as np
                                waveform_service = get_waveform_service()
                                resolution = waveform_info.get("resolution") or waveform_service.DEFAULT_RESOLUTION
                                
                                # Copy to deterministic cache path
                                cache_path = waveform_service.get_waveform_path(data_item.id, resolution)
                                cache_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(resolved_waveform, cache_path)
                                
                                # Validate: must be 1D (strict -- no conversion)
                                arr = np.load(str(cache_path))
                                if arr.ndim != 1:
                                    Log.error(
                                        f"ProjectService: Waveform for {data_item.name} is not 1D "
                                        f"(shape: {arr.shape}). Rejecting. Delete and re-run pipeline."
                                    )
                                    cache_path.unlink(missing_ok=True)
                                    data_item.metadata.pop("waveform", None)
                                else:
                                    # Minimal metadata -- no file_path, no resolutions dict
                                    data_item.metadata['waveform'] = {
                                        'resolution': resolution,
                                        'points_count': len(arr),
                                    }
                                    Log.info(f"ProjectService: Restored waveform for {data_item.name} ({len(arr)} pts, r{resolution})")
                            else:
                                data_item.metadata.pop("waveform", None)
                                Log.warning(f"ProjectService: Waveform not found in project for {data_item.name}: {resolved_waveform}")
                        except Exception as e:
                            Log.error(f"ProjectService: Failed to restore waveform for {data_item.name}: {e}")
                            data_item.metadata.pop("waveform", None)
                
                self._data_item_repo.create(data_item)
            
            # Verify data items were persisted correctly
            if self._block_repo:
                total_persisted = 0
                for blk in self._block_repo.list_by_project(created_project.id):
                    items = self._data_item_repo.list_by_block(blk.id)
                    for item in items:
                        total_persisted += 1
                Log.info(f"ProjectService: Verified {total_persisted} data item(s) in database after import (expected {len(raw_data_items)})")

        # Restore UI state
        ui_state_serialized = data.get("ui_state", {})
        if ui_state_serialized:
            self._ui_state_repo.deserialize_from_project(created_project.id, ui_state_serialized)

        # Restore block local state
        block_local_state_serialized = data.get("block_local_state", {})
        for block_id, local_state in block_local_state_serialized.items():
            self._block_local_state_repo.set_inputs(block_id, local_state)
        if block_local_state_serialized:
            Log.info(f"ProjectService: Restored {len(block_local_state_serialized)} block local state entry(ies)")

        # Restore layer orders
        layer_orders_serialized = data.get("layer_orders", {})
        for block_id, order_list in layer_orders_serialized.items():
            order = [
                LayerKey.from_dict(entry)
                for entry in (order_list or [])
                if isinstance(entry, dict)
            ]
            self._layer_order_repo.set_order(LayerOrder(block_id=block_id, order=order))
        if layer_orders_serialized:
            Log.info(f"ProjectService: Restored {len(layer_orders_serialized)} layer order entry(ies)")

        # Restore setlists and songs
        for setlist_data in data.get("setlists", []):
            setlist_data["project_id"] = created_project.id
            setlist = Setlist.from_dict(setlist_data)
            songs_data = setlist_data.get("songs", [])

            setlist = self._setlist_repo.create(setlist)
            self._setlist_song_repo.delete_by_setlist(setlist.id)

            missing_paths = []
            for song_data in songs_data:
                song_data["setlist_id"] = setlist.id
                song = SetlistSong.from_dict(song_data)
                self._setlist_song_repo.create(song)
                if song.audio_path and not os.path.exists(song.audio_path):
                    missing_paths.append(song.audio_path)

            if missing_paths:
                Log.warning(
                    f"ProjectService: {len(missing_paths)} song audio path(s) not found on this machine "
                    f"(first: {missing_paths[0]}). Setlist audio folder may need to be re-linked."
                )
            if setlist.audio_folder_path and not os.path.isdir(setlist.audio_folder_path):
                Log.warning(f"ProjectService: Setlist audio folder not found: {setlist.audio_folder_path}")

            Log.info(f"ProjectService: Restored setlist with {len(songs_data)} song(s)")

        # Restore action sets (before action items so the table can be rebuilt)
        from src.features.projects.domain.action_set import ActionSet, ActionItem
        action_sets_data = data.get("action_sets", [])
        for action_set_data in action_sets_data:
            action_set_data["project_id"] = created_project.id
            action_set = ActionSet.from_dict(action_set_data)
            self._action_set_repo.create(action_set)
        if action_sets_data:
            Log.info(f"ProjectService: Restored {len(action_sets_data)} action set(s)")

        # Rebuild action_items table from action sets (authoritative) + standalone items
        action_ids_from_sets = set()
        for action_set_data in action_sets_data:
            for action_dict in action_set_data.get("actions", []):
                action_dict["project_id"] = created_project.id
                item = ActionItem.from_dict(action_dict)
                self._action_item_repo.create(item)
                action_ids_from_sets.add(item.id)

        standalone_count = 0
        for action_item_data in data.get("action_items", []):
            if action_item_data.get("id", "") in action_ids_from_sets:
                continue
            action_item_data["project_id"] = created_project.id
            item = ActionItem.from_dict(action_item_data)
            self._action_item_repo.create(item)
            standalone_count += 1

        total_items = len(action_ids_from_sets) + standalone_count
        if total_items > 0:
            Log.info(f"ProjectService: Restored {total_items} action item(s) ({len(action_ids_from_sets)} from sets, {standalone_count} standalone)")

        # Load snapshots
        snapshots_serialized = data.get("snapshots", {})
        self._loaded_snapshots = dict(snapshots_serialized)
        if snapshots_serialized:
            Log.info(f"ProjectService: Loaded {len(snapshots_serialized)} snapshot(s)")

        self._record_recent_project(created_project, file_path)
        
        # Report load status
        load_data = {
            "name": created_project.name,
            "version": created_project.version,
            "imported": True,
            "blocks_loaded": len(data.get("blocks", [])),
            "connections_loaded": len(data.get("connections", [])),
            "data_items_loaded": len(data.get("data_items", [])),
            "missing_files": len(missing_files)
        }
        
        self._event_bus.publish(ProjectCreated(
            project_id=created_project.id,
            data=load_data
        ))
        
        if missing_files:
            Log.warning(
                f"ProjectService: Imported project '{created_project.name}' with {len(missing_files)} missing file(s). "
                f"Use validate command to see details."
            )
        else:
            Log.info(
                f"ProjectService: Imported project '{created_project.name}' from {file_path} "
                f"({len(data.get('blocks', []))} blocks, {len(data.get('data_items', []))} data items)"
            )
        return created_project

    def reset_session(self) -> None:
        """Clear any runtime data so a new project can be loaded cleanly."""
        self._clear_runtime_tables()

    def _clear_runtime_tables(self) -> None:
        if self._database:
            self._database.clear_runtime_tables()

    def _record_recent_project(self, project: Project, project_file: Optional[str]) -> None:
        if not self._recent_store:
            return

        try:
            self._recent_store.update(project, project_file)
        except Exception as e:
            Log.warning(f"ProjectService: Failed to record recent project: {e}")

    def _build_data_item_from_dict(self, data: dict) -> DataItem:
        item_type = (data.get("type") or "").lower()
        if item_type == "audio":
            return ExportedAudioDataItem.from_dict(data)
        if item_type == "event":
            return ExportedEventDataItem.from_dict(data)

        created_at_str = data.get("created_at")
        created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)
        return DataItem(
            id=data.get("id", ""),
            block_id=data.get("block_id", ""),
            name=data.get("name", "DataItem"),
            type=data.get("type", "Data"),
            created_at=created_at,
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {})
        )
    
    def _read_snapshots_from_file(self, project: Project) -> Dict[str, dict]:
        """
        Read snapshots from project file (memory-efficient).
        
        Supports both ZIP archive format (new) and JSON format (legacy).
        
        Args:
            project: Project entity
            
        Returns:
            Dict mapping song_id to snapshot_dict
        """
        project_file = self._get_export_file_path(project)
        if not project_file or not project_file.exists():
            return {}
        
        try:
            # Check if file is a ZIP archive
            if zipfile.is_zipfile(project_file):
                # New ZIP format - read project.json from ZIP
                with zipfile.ZipFile(project_file, 'r') as zip_file:
                    if "project.json" not in zip_file.namelist():
                        return {}
                    project_json_str = zip_file.read("project.json").decode('utf-8')
                    data = json.loads(project_json_str)
            else:
                # Legacy JSON format - read directly
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            return data.get("snapshots", {})
        except Exception as e:
            Log.warning(f"ProjectService: Failed to read snapshots from project file: {e}")
            return {}
    
    def _write_project_file(self, project: Project) -> str:
        """
        Write project file to disk in the save directory.
        
        Creates a ZIP archive (.ez) containing:
        - project.json: All project data (metadata, blocks, connections, etc.)
        - data/: All data files bundled in the archive
        
        Raises on any failure -- callers see the real error, not a silent None.
        
        Args:
            project: Project entity to save
            
        Returns:
            Path to the written project file
            
        Raises:
            ValueError: If project has no save directory
        """
        project_file = self._get_export_file_path(project)
        if project_file is None:
            raise ValueError("Cannot determine project file path: no save_directory set")
        
        save_dir = project_file.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Gather blocks
        block_entities = self._block_repo.list_by_project(project.id)
        blocks = [block.to_dict() for block in block_entities]
        
        # Gather data items and collect files for ZIP packaging
        files_to_zip = {}
        data_items = []
        for block_entity in block_entities:
            block_data_items = self._data_item_repo.list_by_block(block_entity.id)
            for data_item in block_data_items:
                item_dict = data_item.to_dict()
                
                original_path = item_dict.get("file_path")
                if original_path and os.path.exists(original_path):
                    safe_id = item_dict["id"].replace("-", "_")
                    file_ext = os.path.splitext(os.path.basename(original_path))[1] or ".dat"
                    zip_path = f"data/{safe_id}{file_ext}"
                    files_to_zip[zip_path] = original_path
                    item_dict["file_path"] = zip_path
                
                item_metadata = item_dict.get("metadata", {})
                for stale_key in ("file_path", "sample_rate", "channels", "duration_samples", "_file_valid", "original_path"):
                    item_metadata.pop(stale_key, None)
                
                if item_dict.get("type") == "Audio":
                    from src.shared.application.services.waveform_service import get_waveform_service
                    import numpy as np
                    waveform_service = get_waveform_service()
                    waveform_info = item_metadata.get("waveform", {})
                    resolution = waveform_info.get("resolution") or waveform_service.DEFAULT_RESOLUTION
                    
                    det_path = waveform_service.get_waveform_path(item_dict["id"], resolution)
                    if det_path.exists():
                        arr = np.load(str(det_path))
                        if arr.ndim != 1:
                            raise ValueError(
                                f"Waveform for {item_dict.get('name', item_dict['id'])} is not 1D "
                                f"(shape: {arr.shape}). Re-run pipeline to regenerate."
                            )
                        safe_id = item_dict["id"].replace("-", "_")
                        waveform_zip_path = f"data/{safe_id}_waveform.npy"
                        files_to_zip[waveform_zip_path] = str(det_path)
                        item_metadata["waveform"] = {
                            "file_path": waveform_zip_path,
                            "resolution": resolution,
                            "points_count": len(arr),
                        }
                    else:
                        item_metadata.pop("waveform", None)
                
                item_dict.pop("original_path", None)
                data_items.append(item_dict)
        
        # Gather connections
        connections = [conn.to_dict() for conn in self._connection_repo.list_by_project(project.id)]
        
        # Serialize UI state
        ui_state_serialized = self._ui_state_repo.serialize_for_project(project.id)
        
        # Serialize block local state
        block_local_state_serialized = {}
        for block_entity in block_entities:
            local_state = self._block_local_state_repo.get_inputs(block_entity.id)
            if local_state:
                block_local_state_serialized[block_entity.id] = local_state
        
        # Serialize layer orders
        layer_orders_serialized = {}
        for block_entity in block_entities:
            layer_order = self._layer_order_repo.get_order(block_entity.id)
            if layer_order and layer_order.order:
                layer_orders_serialized[block_entity.id] = [
                    key.to_dict() for key in layer_order.order
                ]
        
        # Serialize setlists and songs
        setlists_serialized = []
        for setlist in self._setlist_repo.list_by_project(project.id):
            setlist_dict = setlist.to_dict()
            songs = self._setlist_song_repo.list_by_setlist(setlist.id)
            setlist_dict["songs"] = [song.to_dict() for song in songs]
            setlists_serialized.append(setlist_dict)
        
        # Serialize action sets
        action_sets_serialized = []
        for action_set in self._action_set_repo.list_by_project(project.id):
            live_items = self._action_item_repo.list_by_action_set(action_set.id)
            if live_items:
                action_set.actions = live_items
            action_sets_serialized.append(action_set.to_dict())
        
        # Derive action items from action sets (authoritative) + standalone items
        action_items_serialized = []
        action_ids_from_sets = set()
        for as_dict in action_sets_serialized:
            for action_dict in as_dict.get("actions", []):
                action_items_serialized.append(action_dict)
                if action_dict.get("id"):
                    action_ids_from_sets.add(action_dict["id"])
        for item in self._action_item_repo.list_by_project(project.id):
            if item.id not in action_ids_from_sets:
                action_items_serialized.append(item.to_dict())
        
        # Build project data
        project_data = {
            "format_version": "1.0",
            "project_id": project.id,
            "name": project.name,
            "version": project.version,
            "created_at": project.created_at.isoformat(),
            "modified_at": project.modified_at.isoformat(),
            "metadata": project.metadata,
            "save_directory": project.save_directory,
            "blocks": blocks,
            "connections": connections,
            "data_items": data_items,
            "ui_state": ui_state_serialized,
            "block_local_state": block_local_state_serialized,
            "layer_orders": layer_orders_serialized,
            "setlists": setlists_serialized,
            "action_sets": action_sets_serialized,
            "action_items": action_items_serialized,
            "snapshots": dict(self._loaded_snapshots),
        }
        
        # Write ZIP archive atomically (temp file + rename)
        project_dir = project_file.parent
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.ez.tmp',
            prefix='.ez_save_',
            dir=str(project_dir)
        )
        os.close(temp_fd)
        try:
            with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                project_json_str = json.dumps(project_data, indent=2, ensure_ascii=False)
                zip_file.writestr("project.json", project_json_str.encode('utf-8'))
                
                for zip_path, source_path in files_to_zip.items():
                    zip_file.write(source_path, zip_path)
            
            shutil.move(temp_path, str(project_file))
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
        
        self._dirty_snapshot_ids.clear()
        Log.info(
            f"ProjectService: Created project file: {project_file} "
            f"({len(blocks)} blocks, {len(connections)} connections, {len(data_items)} data items, "
            f"{len(setlists_serialized)} setlist(s), {len(action_sets_serialized)} action set(s), "
            f"{len(action_items_serialized)} action item(s), {len(self._loaded_snapshots)} snapshot(s))"
        )
        return str(project_file)

    def _get_export_file_path(self, project: Project) -> Optional[Path]:
        """Get the file path for exported project files."""
        if not project.save_directory:
            return None
        safe_name = self._sanitize_project_name(project.name)
        return Path(project.save_directory) / f"{safe_name}.ez"

    @staticmethod
    def _sanitize_project_name(name: str) -> str:
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        return safe_name or "project"
    
    def delete_project(self, project_id: str) -> None:
        """
        Delete a project.
        
        Args:
            project_id: Project identifier
            
        Raises:
            ValueError: If project not found
        """
        # Get project before deletion (for event)
        project = self._project_repo.get(project_id)
        if not project:
            raise ValueError(f"Project with id '{project_id}' not found")
        
        project_name = project.name
        
        # Delete from repository (cascade will delete blocks, connections, etc.)
        self._project_repo.delete(project_id)
        
        # Emit event
        self._event_bus.publish(ProjectDeleted(
            project_id=project_id,
            data={"name": project_name}
        ))
        
        Log.info(f"ProjectService: Deleted project '{project_name}' (id: {project_id})")
    
    def rename_project(self, project_id: str, new_name: str) -> Project:
        """
        Rename a project.
        
        Args:
            project_id: Project identifier
            new_name: New project name
            
        Returns:
            Updated Project entity
            
        Raises:
            ValueError: If project not found or name validation fails
        """
        project = self.load_project(project_id)
        if not project:
            raise ValueError(f"Project with id '{project_id}' not found")
        
        # Rename
        project.rename(new_name)
        self.save_project(project)
        
        return project
    
    def list_recent_projects(self, limit: int = 10):
        """
        List recently accessed projects.
        """
        if not self._recent_store:
            return []

        recent_entries = self._recent_store.list_recent(limit)
        projects = []
        for entry in recent_entries:
            last_accessed = entry.get("last_accessed")
            try:
                timestamp = datetime.fromisoformat(last_accessed)
            except Exception:
                timestamp = datetime.now(timezone.utc)

            projects.append(Project(
                id=entry.get("project_id"),
                name=entry.get("name", "Untitled"),
                version=entry.get("version") or "0.0.1",
                save_directory=entry.get("save_directory"),
                created_at=timestamp,
                modified_at=timestamp,
                metadata={}
            ))

        return projects

