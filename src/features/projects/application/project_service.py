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
        # Small cache for last accessed snapshot (memory-efficient)
        self._snapshot_cache: Optional[Tuple[str, dict]] = None  # (song_id, snapshot_dict)
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
            
            # Update cache
            self._snapshot_cache = (song_id, snapshot_dict)
            
            Log.debug(f"ProjectService: Saved snapshot for song {song_id} to project file")
        except Exception as e:
            Log.error(f"ProjectService: Failed to save snapshot to project file: {e}")
            raise
    
    def get_snapshot(self, song_id: str, project: Optional[Project] = None) -> Optional[dict]:
        """
        Get snapshot dictionary from project file (memory-efficient).
        
        Checks cache first, then reads from project file if not cached.
        
        Args:
            song_id: Song identifier
            project: Optional Project entity (if not provided, will try to find from current project)
            
        Returns:
            Snapshot dictionary or None if not found
        """
        # Check cache first
        if self._snapshot_cache and self._snapshot_cache[0] == song_id:
            snapshot = self._snapshot_cache[1]
            return snapshot
        
        # Get project if not provided
        if not project:
            # Try to get current project from repository
            projects = self._project_repo.list_all()
            if projects:
                project = projects[0]  # Use first project as fallback
            else:
                Log.warning(f"ProjectService: Cannot load snapshot - no project found")
                return None
        
        project_file = self._get_export_file_path(project)
        if not project_file or not project_file.exists():
            Log.debug(f"ProjectService: Project file not found: {project_file}")
            return None
        
        try:
            # Read project file - handle both ZIP and JSON formats
            if zipfile.is_zipfile(project_file):
                # New ZIP format - read project.json from ZIP
                with zipfile.ZipFile(project_file, 'r') as zip_file:
                    if "project.json" not in zip_file.namelist():
                        return None
                    project_json_str = zip_file.read("project.json").decode('utf-8')
                    data = json.loads(project_json_str)
            else:
                # Legacy JSON format - read directly
                with open(project_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            # Get snapshot from file
            snapshots = data.get("snapshots", {})
            snapshot = snapshots.get(song_id)
            
            # Update cache
            if snapshot:
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
        
        # Emit event
        self._event_bus.publish(ProjectCreated(
            project_id=created_project.id,
            data={"name": created_project.name, "version": created_project.version, "untitled": created_project.is_untitled()}
        ))
        
        status = "untitled" if created_project.is_untitled() else "saved"
        Log.info(f"ProjectService: Created {status} project '{created_project.name}' (id: {created_project.id})")
        return created_project
    
    def load_project(self, project_id: str) -> Optional[Project]:
        """
        Load a project by ID.
        
        If project file exists, also reload snapshots from file.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Project entity or None if not found
        """
        project = self._project_repo.get(project_id)
        
        if project:
            project_file = self._get_export_file_path(project)
            self._record_recent_project(project, project_file)
            
            # Clear snapshot cache when loading project (snapshots are read from file on demand)
            self._snapshot_cache = None
            
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

        # Restore UI state from project file (built-in, automatic)
        if self._ui_state_repo:
            try:
                ui_state_serialized = data.get("ui_state", {})
                if ui_state_serialized:
                    self._ui_state_repo.deserialize_from_project(created_project.id, ui_state_serialized)
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore UI state from project file: {e}")
                # Don't fail project load - UI state is non-critical

        # Restore block local state from project file
        if self._block_local_state_repo:
            try:
                block_local_state_serialized = data.get("block_local_state", {})
                if block_local_state_serialized:
                    for block_id, local_state in block_local_state_serialized.items():
                        try:
                            self._block_local_state_repo.set_inputs(block_id, local_state)
                        except Exception as e:
                            Log.warning(f"ProjectService: Failed to restore block local state for {block_id}: {e}")
                    Log.info(f"ProjectService: Restored {len(block_local_state_serialized)} block local state entry(ies) from project file")
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore block local state from project file: {e}")
                # Don't fail project load - block local state is non-critical

        # Restore layer order from project file
        if self._layer_order_repo:
            try:
                layer_orders_serialized = data.get("layer_orders", {})
                if layer_orders_serialized:
                    for block_id, order_list in layer_orders_serialized.items():
                        try:
                            order = [
                                LayerKey.from_dict(entry)
                                for entry in (order_list or [])
                                if isinstance(entry, dict)
                            ]
                            self._layer_order_repo.set_order(LayerOrder(block_id=block_id, order=order))
                        except Exception as e:
                            Log.warning(f"ProjectService: Failed to restore layer order for {block_id}: {e}")
                    Log.info(f"ProjectService: Restored {len(layer_orders_serialized)} layer order entry(ies) from project file")
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore layer order from project file: {e}")
                # Don't fail project load - layer order is non-critical

        # Restore setlists and songs from project file (follows blocks pattern - tables cleared by _clear_runtime_tables)
        if self._setlist_repo and self._setlist_song_repo:
            try:
                setlists_serialized = data.get("setlists", [])
                for setlist_data in setlists_serialized:
                    try:
                        # Update project_id to match the loaded project
                        setlist_data["project_id"] = created_project.id
                        setlist = Setlist.from_dict(setlist_data)
                        songs_data = setlist_data.get("songs", [])
                        
                        # Create setlist (tables already cleared by _clear_runtime_tables)
                        setlist = self._setlist_repo.create(setlist)
                        
                        # Create songs for this setlist
                        for song_data in songs_data:
                            song_data["setlist_id"] = setlist.id
                            song = SetlistSong.from_dict(song_data)
                            self._setlist_song_repo.create(song)
                        
                        Log.info(f"ProjectService: Restored setlist with {len(songs_data)} song(s) from project file")
                    except Exception as e:
                        Log.warning(f"ProjectService: Failed to restore setlist: {e}")
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore setlists from project file: {e}")
                # Don't fail project load - setlists are non-critical

        # Restore action sets from project file (follows blocks pattern - tables cleared by _clear_runtime_tables)
        # IMPORTANT: Action sets must be restored BEFORE action items, so action items can reference them
        if self._action_set_repo:
            try:
                from src.features.projects.domain.action_set import ActionSet
                action_sets_serialized = data.get("action_sets", [])
                Log.debug(f"ProjectService: Found {len(action_sets_serialized)} action set(s) in project file")
                restored_count = 0
                for action_set_data in action_sets_serialized:
                    try:
                        # Update project_id to match the loaded project
                        action_set_data["project_id"] = created_project.id
                        action_set = ActionSet.from_dict(action_set_data)
                        
                        # Check if action set already exists (by ID or name)
                        existing = None
                        if action_set.id:
                            try:
                                existing = self._action_set_repo.get(action_set.id)
                            except Exception:
                                pass
                        
                        if existing:
                            # Update existing action set
                            self._action_set_repo.update(action_set)
                            Log.debug(f"ProjectService: Updated existing action set '{action_set.name}' (id: {action_set.id})")
                        else:
                            # Create new action set
                            self._action_set_repo.create(action_set)
                            Log.debug(f"ProjectService: Created new action set '{action_set.name}' (id: {action_set.id})")
                        
                        restored_count += 1
                        Log.debug(f"ProjectService: Restored action set '{action_set.name}' with {len(action_set.actions)} action(s)")
                    except Exception as e:
                        Log.warning(f"ProjectService: Failed to restore action set '{action_set_data.get('name', 'unknown')}': {e}")
                
                if restored_count > 0:
                    Log.info(f"ProjectService: Restored {restored_count} action set(s) from project file")
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore action sets from project file: {e}")
                # Don't fail project load - action sets are non-critical
        else:
            Log.debug("ProjectService: action_set_repo not available, skipping action sets restoration")

        # Restore action items from project file (user-configured action events)
        if self._action_item_repo:
            try:
                from src.features.projects.domain.action_set import ActionItem
                action_items_serialized = data.get("action_items", [])
                Log.debug(f"ProjectService: Found {len(action_items_serialized)} action item(s) in project file")
                restored_count = 0
                for action_item_data in action_items_serialized:
                    try:
                        # Update project_id to match the loaded project
                        action_item_data["project_id"] = created_project.id
                        action_item = ActionItem.from_dict(action_item_data)
                        
                        # Check if action item already exists (by ID)
                        existing = None
                        if action_item.id:
                            try:
                                existing = self._action_item_repo.get(action_item.id)
                            except Exception:
                                pass
                        
                        if existing:
                            # Update existing action item
                            self._action_item_repo.update(action_item)
                            Log.debug(f"ProjectService: Updated existing action item '{action_item.action_name}' (id: {action_item.id})")
                        else:
                            # Create new action item
                            self._action_item_repo.create(action_item)
                            Log.debug(f"ProjectService: Created new action item '{action_item.action_name}' (id: {action_item.id})")
                        
                        restored_count += 1
                    except Exception as e:
                        Log.warning(f"ProjectService: Failed to restore action item '{action_item_data.get('action_name', 'unknown')}': {e}")
                
                if restored_count > 0:
                    Log.info(f"ProjectService: Restored {restored_count} action item(s) from project file")
            except Exception as e:
                Log.warning(f"ProjectService: Failed to restore action items from project file: {e}")
                # Don't fail project load - action items are non-critical
        else:
            Log.debug("ProjectService: action_item_repo not available, skipping action items restoration")

        # Load snapshots from project file
        try:
            snapshots_serialized = data.get("snapshots", {})
            self._loaded_snapshots = dict(snapshots_serialized)  # song_id -> snapshot_dict
            if snapshots_serialized:
                Log.info(f"ProjectService: Loaded {len(snapshots_serialized)} snapshot(s) from project file")
        except Exception as e:
            Log.warning(f"ProjectService: Failed to load snapshots from project file: {e}")
            self._loaded_snapshots = {}  # Clear on error
            # Don't fail project load - snapshots are non-critical

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
    
    def _write_project_file(self, project: Project) -> Optional[str]:
        """
        Write project file to disk in the save directory.
        
        Creates a ZIP archive (.ez) containing:
        - project.json: All project data (metadata, blocks, connections, etc.)
        - data/: All data files bundled in the archive
        
        This allows the project to be transported as a single file.
        
        Args:
            project: Project entity to save
        """
        project_file = self._get_export_file_path(project)
        if project_file is None:
            return None
        
        try:
            save_dir = project_file.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Gather all project data
            blocks = []
            connections = []
            data_items = []
            
            if self._block_repo:
                try:
                    block_entities = self._block_repo.list_by_project(project.id)
                    blocks = [block.to_dict() for block in block_entities]
                    
                    # Get data items for each block and collect files for ZIP packaging
                    files_to_zip = {}  # Maps ZIP path -> source file path
                    if self._data_item_repo:
                        try:
                            for block_entity in block_entities:
                                block_data_items = self._data_item_repo.list_by_block(block_entity.id)
                                for data_item in block_data_items:
                                    item_dict = data_item.to_dict()
                                    
                                    # Package audio file if it exists
                                    original_path = item_dict.get("file_path")
                                    if original_path and os.path.exists(original_path):
                                        try:
                                            safe_id = item_dict["id"].replace("-", "_")
                                            file_ext = os.path.splitext(os.path.basename(original_path))[1] or ".dat"
                                            zip_path = f"data/{safe_id}{file_ext}"
                                            
                                            files_to_zip[zip_path] = original_path
                                            item_dict["file_path"] = zip_path
                                            
                                            Log.debug(f"ProjectService: Will package file {original_path} -> {zip_path} in ZIP")
                                        except Exception as e:
                                            Log.warning(f"ProjectService: Failed to prepare file {original_path} for packaging: {e}")
                                    
                                    # Clean metadata before export: remove duplicates and runtime flags
                                    item_metadata = item_dict.get("metadata", {})
                                    for stale_key in ("file_path", "sample_rate", "channels", "duration_samples", "_file_valid", "original_path"):
                                        item_metadata.pop(stale_key, None)
                                    
                                    # Package waveform (AudioDataItems only)
                                    if item_dict.get("type") == "Audio":
                                        try:
                                            from src.shared.application.services.waveform_service import get_waveform_service
                                            import numpy as np
                                            waveform_service = get_waveform_service()
                                            waveform_info = item_metadata.get("waveform", {})
                                            resolution = waveform_info.get("resolution") or waveform_service.DEFAULT_RESOLUTION
                                            
                                            det_path = waveform_service.get_waveform_path(item_dict["id"], resolution)
                                            if det_path.exists():
                                                arr = np.load(str(det_path))
                                                if arr.ndim != 1:
                                                    Log.error(
                                                        f"ProjectService: Waveform for {item_dict.get('name', item_dict['id'])} is not 1D "
                                                        f"(shape: {arr.shape}). Skipping waveform. Re-run pipeline to regenerate."
                                                    )
                                                    item_metadata.pop("waveform", None)
                                                else:
                                                    safe_id = item_dict["id"].replace("-", "_")
                                                    waveform_zip_path = f"data/{safe_id}_waveform.npy"
                                                    files_to_zip[waveform_zip_path] = str(det_path)
                                                    item_metadata["waveform"] = {
                                                        "file_path": waveform_zip_path,
                                                        "resolution": resolution,
                                                        "points_count": len(arr),
                                                    }
                                                    Log.debug(f"ProjectService: Will package waveform -> {waveform_zip_path} (r{resolution}, {len(arr)} pts)")
                                            else:
                                                item_metadata.pop("waveform", None)
                                        except Exception as e:
                                            Log.error(f"ProjectService: Failed to prepare waveform for export ({item_dict.get('name', item_dict['id'])}): {e}")
                                    
                                    # Remove original_path -- do not leak machine paths
                                    item_dict.pop("original_path", None)
                                    
                                    data_items.append(item_dict)
                        except Exception as e:
                            Log.warning(f"ProjectService: Failed to get data items for block {block_entity.id}: {e}")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to get blocks for export: {e}")
            
            if self._connection_repo:
                try:
                    connections = [conn.to_dict() for conn in self._connection_repo.list_by_project(project.id)]
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to get connections for export: {e}")
            
            # Serialize UI state (built-in, automatic)
            ui_state_serialized = {}
            if self._ui_state_repo:
                try:
                    ui_state_serialized = self._ui_state_repo.serialize_for_project(project.id)
                    total_entries = sum(len(entries) for entries in ui_state_serialized.values())
                    if total_entries > 0:
                        Log.debug(f"ProjectService: Serialized {total_entries} UI state entry(ies) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize UI state: {e}")
            
            # Serialize block local state (input/output references)
            block_local_state_serialized = {}
            if self._block_local_state_repo:
                try:
                    for block_entity in block_entities:
                        local_state = self._block_local_state_repo.get_inputs(block_entity.id)
                        if local_state:
                            block_local_state_serialized[block_entity.id] = local_state
                    if block_local_state_serialized:
                        Log.debug(f"ProjectService: Serialized {len(block_local_state_serialized)} block local state entry(ies) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize block local state: {e}")

            # Serialize layer orders (Editor ordering)
            layer_orders_serialized = {}
            if self._layer_order_repo:
                try:
                    for block_entity in block_entities:
                        layer_order = self._layer_order_repo.get_order(block_entity.id)
                        if layer_order and layer_order.order:
                            layer_orders_serialized[block_entity.id] = [
                                key.to_dict() for key in layer_order.order
                            ]
                    if layer_orders_serialized:
                        Log.debug(f"ProjectService: Serialized {len(layer_orders_serialized)} layer order entry(ies) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize layer orders: {e}")
            
            # Serialize setlists and songs (follows blocks pattern)
            setlists_serialized = []
            if self._setlist_repo and self._setlist_song_repo:
                try:
                    setlist_entities = self._setlist_repo.list_by_project(project.id)
                    for setlist in setlist_entities:
                        setlist_dict = setlist.to_dict()
                        songs = self._setlist_song_repo.list_by_setlist(setlist.id)
                        setlist_dict["songs"] = [song.to_dict() for song in songs]
                        setlists_serialized.append(setlist_dict)
                    if setlists_serialized:
                        total_songs = sum(len(s.get("songs", [])) for s in setlists_serialized)
                        Log.debug(f"ProjectService: Serialized {len(setlists_serialized)} setlist(s) with {total_songs} song(s) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize setlists: {e}")
            
            # Serialize action sets (follows blocks pattern)
            action_sets_serialized = []
            if self._action_set_repo:
                try:
                    action_set_entities = self._action_set_repo.list_by_project(project.id)
                    Log.debug(f"ProjectService: Found {len(action_set_entities)} action set(s) in database for project {project.id}")
                    for action_set in action_set_entities:
                        action_set_dict = action_set.to_dict()
                        action_sets_serialized.append(action_set_dict)
                        Log.debug(f"ProjectService: Serializing action set '{action_set.name}' with {len(action_set.actions)} action(s)")
                    if action_sets_serialized:
                        Log.info(f"ProjectService: Serialized {len(action_sets_serialized)} action set(s) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize action sets: {e}")
            else:
                Log.debug("ProjectService: action_set_repo not available, skipping action sets serialization")
            
            # Serialize action items (user-configured action events)
            action_items_serialized = []
            if self._action_item_repo:
                try:
                    action_item_entities = self._action_item_repo.list_by_project(project.id)
                    Log.debug(f"ProjectService: Found {len(action_item_entities)} action item(s) in database for project {project.id}")
                    for action_item in action_item_entities:
                        action_item_dict = action_item.to_dict()
                        action_items_serialized.append(action_item_dict)
                    if action_items_serialized:
                        Log.info(f"ProjectService: Serialized {len(action_items_serialized)} action item(s) for project file")
                except Exception as e:
                    Log.warning(f"ProjectService: Failed to serialize action items: {e}")
            else:
                Log.debug("ProjectService: action_item_repo not available, skipping action items serialization")
            
            # Create complete project file data
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
                "ui_state": ui_state_serialized,  # Built-in UI state persistence
                "block_local_state": block_local_state_serialized,  # Block input/output references
                "layer_orders": layer_orders_serialized,  # Editor layer ordering
                "setlists": setlists_serialized,  # Setlists and songs
                "action_sets": action_sets_serialized,  # Action sets
                "action_items": action_items_serialized,  # User-configured action events
                "snapshots": self._read_snapshots_from_file(project)  # Snapshots for setlist songs (read from file)
            }
            
            # Write ZIP archive atomically using temp file + rename
            # This prevents data corruption if the write is interrupted
            project_dir = project_file.parent
            temp_fd, temp_path = tempfile.mkstemp(
                suffix='.ez.tmp',
                prefix='.ez_save_',
                dir=str(project_dir)
            )
            os.close(temp_fd)  # Close fd immediately; zipfile opens by path
            try:
                # Create ZIP archive
                with zipfile.ZipFile(temp_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Write project.json to ZIP root
                    project_json_str = json.dumps(project_data, indent=2, ensure_ascii=False)
                    zip_file.writestr("project.json", project_json_str.encode('utf-8'))
                    
                    # Add all data files to ZIP under data/ folder
                    for zip_path, source_path in files_to_zip.items():
                        try:
                            zip_file.write(source_path, zip_path)
                            Log.debug(f"ProjectService: Added {source_path} to ZIP as {zip_path}")
                        except Exception as e:
                            Log.warning(f"ProjectService: Failed to add {source_path} to ZIP: {e}")
                
                # Atomic rename (preserves original if this fails)
                shutil.move(temp_path, str(project_file))
            except Exception:
                # Clean up temp file if rename failed
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
            
            ui_state_count = sum(len(entries) for entries in ui_state_serialized.values()) if ui_state_serialized else 0
            block_local_state_count = len(block_local_state_serialized) if block_local_state_serialized else 0
            layer_order_count = len(layer_orders_serialized) if layer_orders_serialized else 0
            setlist_count = len(setlists_serialized)
            total_songs = sum(len(s.get("songs", [])) for s in setlists_serialized)
            action_set_count = len(action_sets_serialized)
            action_item_count = len(action_items_serialized)
            snapshot_count = len(self._loaded_snapshots)
            Log.info(f"ProjectService: Created project file: {project_file} ({len(blocks)} blocks, {len(connections)} connections, {len(data_items)} data items, {ui_state_count} UI state entries, {block_local_state_count} block local state entries, {layer_order_count} layer order entries, {setlist_count} setlist(s) with {total_songs} song(s), {action_set_count} action set(s), {action_item_count} action item(s), {snapshot_count} snapshot(s))")
            return str(project_file)
            
        except Exception as e:
            Log.error(f"ProjectService: Failed to write project file: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - database save succeeded, file write is secondary
        return None

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

