"""
Application Facade

Unified interface for all application operations.
Used by CLI, GUI, and external APIs.

All methods return CommandResult[T] where T is the type of data returned:
- CommandResult[Block] for single block operations
- CommandResult[List[Block]] for multiple blocks
- CommandResult[Connection] for connection operations
- etc.
"""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from collections import defaultdict

from .result_types import CommandResult, ResultStatus
from src.features.blocks.domain import Block
from src.features.connections.domain import Connection
from src.features.projects.domain import Project
from src.features.projects.domain import ActionSet
from src.shared.domain.data_state import DataState
from src.application.events import (
    UIStateChanged,
    ExecutionStarted,
    ExecutionProgress,
    ExecutionCompleted,
    ProjectLoaded,
    SubprocessProgress,
)
from src.features.execution.application.topological_sort import (
    topological_sort_blocks,
    CyclicDependencyError
)
from src.application.processing.block_processor import FilterError
from src.features.execution.application.progress_tracker import create_progress_tracker
from src.utils.message import Log

class ApplicationFacade:
    """
    Unified interface for all application operations.
    
    Provides a clean, typed API that all user interfaces can use.
    Returns structured CommandResult objects instead of just booleans.
    
    Used by:
    - CLI (through CommandParser adapter)
    - GUI (direct method calls - future)
    - External APIs (future)
    """
    
    def __init__(self, services):
        """
        Initialize application facade.
        
        Args:
            services: ServiceContainer with all application services
        """
        self.project_service = services.project_service
        self.block_service = services.block_service
        self.connection_service = services.connection_service
        self.connection_repo = getattr(services, "connection_repo", None) or getattr(services.connection_service, "_connection_repo", None)
        self.execution_engine = services.execution_engine
        self.event_bus = services.event_bus
        self.database = getattr(services, "database", None)
        self.data_item_repo = getattr(services, "data_item_repo", None)
        self.block_local_state_repo = getattr(services, "block_local_state_repo", None)
        self.recent_store = getattr(services, "recent_store", None)
        self.layer_order_service = getattr(services, "layer_order_service", None)
        self.layer_group_order_service = getattr(services, "layer_group_order_service", None)
        
        # UI state repositories (Phase A Foundation)
        self.ui_state_repo = getattr(services, "ui_state_repo", None)
        self.preferences_repo = getattr(services, "preferences_repo", None)
        self.session_state_repo = getattr(services, "session_state_repo", None)
        
        # Application settings manager (standardized settings system)
        self.app_settings = getattr(services, "app_settings", None)
        
        # Data state and filter services
        self.data_state_service = getattr(services, "data_state_service", None)
        self.block_status_service = getattr(services, "block_status_service", None)
        self.data_filter_manager = getattr(services, "data_filter_manager", None)
        self.expected_outputs_service = getattr(services, "expected_outputs_service", None)
        
        # Setlist service
        self.setlist_service = getattr(services, "setlist_service", None)
        
        # Action set repositories
        # Database repo for project-specific storage
        self.action_set_repo = getattr(services, "action_set_repo", None)
        # File-based repo for global/reusable action sets
        from src.infrastructure.persistence.file import get_action_set_file_repo
        self.action_set_file_repo = get_action_set_file_repo()
        
        # Action item repository (user-configured action events)
        self.action_item_repo = getattr(services, "action_item_repo", None)
        
        # ShowManager listener service (manages OSC listeners independently of panel lifecycle)
        self.show_manager_listener_service = getattr(services, "show_manager_listener_service", None)
        # ShowManager state service (manages connection state independently of panel lifecycle)
        self.show_manager_state_service = getattr(services, "show_manager_state_service", None)
        # MA3 communication service (send commands to MA3)
        self.ma3_communication_service = getattr(services, "ma3_communication_service", None)

        # EditorAPI registry (shared instances for sync signals)
        self.editor_api_registry: Dict[str, Any] = {}
        # ShowManager settings manager cache
        self._show_manager_settings_cache: Dict[str, "ShowManagerSettingsManager"] = {}
        # SyncSystemManager cache (one per ShowManager block, shared with panel)
        self._sync_system_manager_cache: Dict[str, Any] = {}

        # Sync port (bidirectional Editor <-> ShowManager entrypoint)
        self.sync_port = getattr(services, "sync_port", None)
        
        # Command bus (set after MainWindow creates it, since it requires QUndoStack)
        self.command_bus = None
        
        self.current_project_id: Optional[str] = None
        Log.info("ApplicationFacade: Initialized")

    def show_manager_settings_manager(self, block_id: str):
        """Get cached ShowManagerSettingsManager for a block."""
        if not block_id:
            return None
        if block_id in self._show_manager_settings_cache:
            return self._show_manager_settings_cache[block_id]
        try:
            from src.application.settings.show_manager_settings import ShowManagerSettingsManager
            mgr = ShowManagerSettingsManager(self, block_id)
            self._show_manager_settings_cache[block_id] = mgr
            return mgr
        except Exception:
            return None
    
    def sync_system_manager(self, block_id: str):
        """Get or create a cached SyncSystemManager for a ShowManager block.
        
        The SSM is shared between the facade (for project-load monitoring)
        and the panel (for UI-driven operations). Both access the same instance.
        """
        if not block_id:
            return None
        if block_id in self._sync_system_manager_cache:
            return self._sync_system_manager_cache[block_id]
        try:
            from src.features.show_manager.application.sync_system_manager import SyncSystemManager
            settings_mgr = self.show_manager_settings_manager(block_id)
            if not settings_mgr:
                return None
            ssm = SyncSystemManager(
                facade=self,
                show_manager_block_id=block_id,
                settings_manager=settings_mgr,
            )
            self._sync_system_manager_cache[block_id] = ssm
            Log.info(f"ApplicationFacade: Created SyncSystemManager for block {block_id}")
            return ssm
        except Exception as e:
            Log.warning(f"ApplicationFacade: Failed to create SyncSystemManager for {block_id}: {e}")
            return None

    def set_command_bus(self, command_bus) -> None:
        """
        Set the command bus instance.
        
        Called by MainWindow after creating CommandBus (which requires QUndoStack).
        This makes the dependency explicit rather than using global state.
        
        Args:
            command_bus: CommandBus instance
        """
        self.command_bus = command_bus
        Log.info("ApplicationFacade: CommandBus set")
    
    # ==================== Project Operations ====================
    
    def create_project(self, name: str = "Untitled", save_directory: Optional[str] = None) -> CommandResult:
        """
        Create a new project.
        
        Auto-creates an empty setlist for the project (one setlist per project).
        
        Args:
            name: Project name (default: "Untitled")
            save_directory: Optional save directory (None for untitled)
            
        Returns:
            CommandResult with project data
        """
        self.reset_session()
        try:
            project = self.project_service.create_project(name, save_directory)
            self.current_project_id = project.id
            
            # Auto-create empty setlist for new project
            if self.setlist_service:
                try:
                    from datetime import datetime
                    from src.features.setlists.domain import Setlist
                    
                    # Check if setlist already exists (shouldn't, but be safe)
                    existing = self.setlist_service._setlist_repo.get_by_project(project.id)
                    if not existing:
                        setlist = Setlist(
                            id="",
                            audio_folder_path="",  # Empty - no folder dependency
                            project_id=project.id,
                            default_actions={},
                            created_at=datetime.utcnow(),
                            modified_at=datetime.utcnow()
                        )
                        self.setlist_service._setlist_repo.create(setlist)
                        Log.info(f"ApplicationFacade: Auto-created empty setlist for project '{project.name}'")
                except Exception as e:
                    # Don't fail project creation if setlist creation fails
                    Log.warning(f"ApplicationFacade: Failed to auto-create setlist: {e}")
            
            if project.is_untitled():
                return CommandResult.success_result(
                    message=f"Created untitled project '{project.name}'",
                    data=project
                )
            else:
                return CommandResult.success_result(
                    message=f"Created project '{project.name}'",
                    data=project
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to create project: {e}")
            return CommandResult.error_result(
                message=f"Failed to create project: {e}",
                errors=[str(e)]
            )
    
    def load_project(self, identifier: str) -> CommandResult:
        """
        Load a project by ID or name.
        
        Args:
            identifier: Project ID or project name
            
        Returns:
            CommandResult with project data
        """
        try:
            project = None
            project_file = None

            if os.path.isfile(identifier) and identifier.lower().endswith(".ez"):
                project_file = identifier
            elif self.recent_store:
                recent_entry = self.recent_store.get(identifier)
                if recent_entry:
                    project_file = recent_entry.get("project_file")

            self.reset_session()
            if project_file:
                project = self.project_service.import_project_from_file(project_file)
            else:
                project = self.project_service.load_project(identifier)
            
            if project:
                self.current_project_id = project.id
                
                # Publish project.loaded event for UI components
                if self.event_bus:
                    self.event_bus.publish(ProjectLoaded(
                        project_id=project.id,
                        data={"name": project.name, "version": project.version}
                    ))
                    Log.debug(f"ApplicationFacade: Published project.loaded event for '{project.name}'")
                
                # Ensure setlist exists for loaded project (auto-create if missing)
                if self.setlist_service:
                    try:
                        from datetime import datetime
                        from src.features.setlists.domain import Setlist
                        
                        # Check if setlist exists
                        existing = self.setlist_service._setlist_repo.get_by_project(project.id)
                        if not existing:
                            setlist = Setlist(
                                id="",
                                audio_folder_path="",  # Empty - no folder dependency
                                project_id=project.id,
                                default_actions={},
                                created_at=datetime.utcnow(),
                                modified_at=datetime.utcnow()
                            )
                            self.setlist_service._setlist_repo.create(setlist)
                            Log.info(f"ApplicationFacade: Auto-created empty setlist for loaded project '{project.name}'")
                    except Exception as e:
                        # Don't fail project loading if setlist creation fails
                        Log.warning(f"ApplicationFacade: Failed to auto-create setlist for loaded project: {e}")

                # Restore ShowManager connections and listeners after project load
                try:
                    self._restore_show_manager_state()
                except Exception as e:
                    Log.warning(f"ApplicationFacade: Failed to restore ShowManager state: {e}")
                
                return CommandResult.success_result(
                    message=f"Loaded project '{project.name}'",
                    data=project
                )
            else:
                return CommandResult.error_result(
                    message=f"Project not found: {identifier}"
                )
        except ValueError as e:
            # ValueError from project service includes detailed JSON error info
            error_msg = str(e)
            Log.error(f"ApplicationFacade: Failed to load project: {error_msg}")
            return CommandResult.error_result(
                message=error_msg,
                errors=[error_msg]
            )
        except Exception as e:
            # Other exceptions - provide generic error with file info if available
            error_msg = f"Failed to load project: {e}"
            if project_file:
                error_msg = f"Failed to load project from '{project_file}': {e}"
            Log.error(f"ApplicationFacade: {error_msg}")
            return CommandResult.error_result(
                message=error_msg,
                errors=[str(e)]
            )

    def _restore_show_manager_state(self) -> None:
        """
        Restore ShowManager connections and OSC listeners after project load.

        Ensures:
        - Editor connections are re-established when possible
        - OSC listeners are restarted if they were active
        - Listener failures clear stale metadata to keep status accurate
        """
        if not self.current_project_id:
            return

        if not (self.show_manager_listener_service or self.show_manager_state_service):
            return

        blocks_result = self.list_blocks()
        if not blocks_result.success or not blocks_result.data:
            return

        for block_summary in blocks_result.data:
            if getattr(block_summary, "type", None) != "ShowManager":
                continue

            # list_blocks() returns BlockSummary (id, name, type only).
            # We need the full Block entity to access metadata.
            full_block_result = self.describe_block(block_summary.id)
            if not full_block_result.success or not full_block_result.data:
                Log.warning(
                    f"ApplicationFacade: Could not load full block for "
                    f"ShowManager {block_summary.id}, skipping state restore"
                )
                continue
            block = full_block_result.data

            metadata = block.metadata or {}
            settings_manager = None
            try:
                from src.application.settings.show_manager_settings import ShowManagerSettingsManager
                settings_manager = ShowManagerSettingsManager(self, block.id)
            except Exception:
                settings_manager = None

            # Restore Editor connection if possible
            if self.show_manager_state_service:
                try:
                    self.show_manager_state_service.attempt_auto_connect(block.id)
                except Exception as e:
                    Log.warning(
                        f"ApplicationFacade: Auto-connect failed for ShowManager {block.id}: {e}"
                    )

            # Restore OSC listener if it was active or sync requires it
            should_restore_listener = False
            restore_reason = None
            if metadata.get("osc_listener_active"):
                should_restore_listener = True
                restore_reason = "metadata_active"
            elif metadata.get("synced_layers"):
                should_restore_listener = True
                restore_reason = "synced_layers"
            elif metadata.get("auto_sync_enabled"):
                should_restore_listener = True
                restore_reason = "auto_sync_enabled"
            elif settings_manager:
                if settings_manager.synced_layers:
                    should_restore_listener = True
                    restore_reason = "settings_synced_layers"
                elif settings_manager.auto_sync_enabled:
                    should_restore_listener = True
                    restore_reason = "settings_auto_sync_enabled"

            if self.show_manager_listener_service and should_restore_listener:
                listen_port = metadata.get("listen_port", 9000)
                listen_address = metadata.get("listen_address", "127.0.0.1")
                try:
                    listen_port = int(listen_port)
                except (TypeError, ValueError):
                    listen_port = 9000
                if settings_manager and getattr(settings_manager, "listen_address", None):
                    listen_address = settings_manager.listen_address or listen_address

                success, error_message = self.show_manager_listener_service.start_listener(
                    block_id=block.id,
                    listen_port=listen_port,
                    listen_address=listen_address
                )

                if not success:
                    # If listener is already running, keep metadata as-is
                    if error_message == "Listener is already running":
                        success = True
                    else:
                        Log.warning(
                            f"ApplicationFacade: Failed to restore OSC listener for "
                            f"ShowManager {block.id}: {error_message}"
                        )
                        # Clear stale listener flag so status reflects actual state
                        updated_metadata = dict(metadata)
                        updated_metadata["osc_listener_active"] = False
                        block.metadata = updated_metadata
                        try:
                            self.block_service.update_block(
                                self.current_project_id,
                                block.id,
                                block
                            )
                        except Exception as e:
                            Log.warning(
                                f"ApplicationFacade: Failed to clear OSC listener flag "
                                f"for ShowManager {block.id}: {e}"
                            )
                        success = False

                if success and self.show_manager_listener_service.is_listening(block.id):
                    if not metadata.get("osc_listener_active"):
                        updated_metadata = dict(metadata)
                        updated_metadata["osc_listener_active"] = True
                        block.metadata = updated_metadata
                        try:
                            self.block_service.update_block(
                                self.current_project_id,
                                block.id,
                                block
                            )
                        except Exception as e:
                            Log.warning(
                                f"ApplicationFacade: Failed to set OSC listener flag "
                                f"for ShowManager {block.id}: {e}"
                            )
                    try:
                        self._rehook_show_manager_sync_layers(block.id)
                    except Exception as e:
                        Log.warning(
                            f"ApplicationFacade: Failed to rehook synced layers for ShowManager {block.id}: {e}"
                        )
                    # Start SSM connection monitoring so MA3 layers auto-populate
                    # even before the ShowManager panel is opened
                    try:
                        ssm = self.sync_system_manager(block.id)
                        if ssm:
                            ssm.start_connection_monitoring()
                            Log.info(
                                f"ApplicationFacade: Started SSM connection monitoring for "
                                f"ShowManager {block.id} (reason: {restore_reason})"
                            )
                    except Exception as e:
                        Log.warning(
                            f"ApplicationFacade: Failed to start SSM monitoring for "
                            f"ShowManager {block.id}: {e}"
                        )
            elif self.show_manager_listener_service and self.show_manager_listener_service.is_listening(block.id):
                try:
                    self._rehook_show_manager_sync_layers(block.id)
                except Exception as e:
                    Log.warning(
                        f"ApplicationFacade: Failed to rehook synced layers for ShowManager {block.id}: {e}"
                    )
                # Start SSM connection monitoring for already-running listener
                try:
                    ssm = self.sync_system_manager(block.id)
                    if ssm:
                        ssm.start_connection_monitoring()
                        Log.info(
                            f"ApplicationFacade: Started SSM connection monitoring for "
                            f"ShowManager {block.id} (listener already running)"
                        )
                except Exception as e:
                    Log.warning(
                        f"ApplicationFacade: Failed to start SSM monitoring for "
                        f"ShowManager {block.id}: {e}"
                    )

    def _rehook_show_manager_sync_layers(self, show_manager_block_id: str) -> None:
        """Rehook synced MA3 tracks for a ShowManager block."""
        from src.features.show_manager.application.commands import RehookSyncedMA3TracksCommand

        cmd = RehookSyncedMA3TracksCommand(self, show_manager_block_id)
        if self.command_bus:
            self.command_bus.execute(cmd)
        else:
            cmd.redo()
    
    def _flush_all_block_settings(self) -> None:
        """Force save all pending block settings before project save."""
        blocks_result = self.list_blocks()
        if not blocks_result.success:
            return
        
        for block in blocks_result.data:
            if block.type == "ShowManager":
                try:
                    from src.application.settings.show_manager_settings import ShowManagerSettingsManager
                    settings_manager = ShowManagerSettingsManager(self, block.id)
                    if settings_manager.has_pending_save():
                        settings_manager.force_save()
                except Exception as e:
                    Log.warning(f"ApplicationFacade: Failed to flush settings for ShowManager {block.id}: {e}")
    
    def save_project(self) -> CommandResult:
        """
        Save current project.
        
        Returns:
            CommandResult with project data
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            project = self.project_service.load_project(self.current_project_id)
            if not project:
                return CommandResult.error_result(
                    message=f"Project not found: {self.current_project_id}"
                )
            
            # If project is untitled, return warning
            if project.is_untitled():
                return CommandResult.warning_result(
                    message="Project is untitled. Use save_as to set location.",
                    warnings=["Use 'save_as <directory>' to set save location"]
                )
            
            # Before saving project, ensure all block settings are flushed
            # This ensures synced layers and other settings persist
            self._flush_all_block_settings()
            
            
            self.project_service.save_project(project)
            return CommandResult.success_result(
                message=f"Saved project '{project.name}'",
                data=project
            )
        except ValueError as e:
            # Project is untitled
            return CommandResult.warning_result(
                message=str(e),
                warnings=["Use 'save_as <directory>' to set save location"]
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to save project: {e}")
            return CommandResult.error_result(
                message=f"Failed to save project: {e}",
                errors=[str(e)]
            )
    
    def save_project_as(self, save_directory: str, name: Optional[str] = None) -> CommandResult:
        """
        Save project with new location and optional name.
        
        Args:
            save_directory: Directory to save project
            name: Optional new name for project
            
        Returns:
            CommandResult with project data
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        # Expand and validate path
        try:
            save_directory = os.path.expanduser(save_directory)
            save_directory = os.path.abspath(save_directory)
            
            # If no name provided, derive from directory
            if not name:
                directory_name = os.path.basename(os.path.normpath(save_directory))
                name = directory_name if directory_name else "Untitled"
            
            # Validate directory
            validation_error = self._validate_save_directory(save_directory)
            if validation_error:
                return CommandResult.error_result(
                    message=validation_error
                )
            
            # Create directory if needed
            os.makedirs(save_directory, exist_ok=True)
            
            # Verify writable
            if not os.access(save_directory, os.W_OK):
                return CommandResult.error_result(
                    message=f"Directory '{save_directory}' is not writable"
                )
            
            # Flush settings before save_as (same as save_project)
            self._flush_all_block_settings()
            
            
            # Save project
            project = self.project_service.save_project_as(
                self.current_project_id,
                save_directory,
                name=name
            )
            
            return CommandResult.success_result(
                message=f"Saved project '{project.name}' to {project.save_directory}",
                data=project
            )
        except ValueError as e:
            return CommandResult.error_result(
                message=f"Validation error: {e}",
                errors=[str(e)]
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to save project: {e}")
            return CommandResult.error_result(
                message=f"Failed to save project: {e}",
                errors=[str(e)]
            )
    
    def delete_project(self, project_id: Optional[str] = None) -> CommandResult:
        """
        Delete a project.
        
        Args:
            project_id: Project ID (uses current project if None)
            
        Returns:
            CommandResult
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            return CommandResult.error_result(
                message="No project specified"
            )
        
        try:
            self.project_service.delete_project(project_id)
            if project_id == self.current_project_id:
                self.current_project_id = None
            return CommandResult.success_result(
                message=f"Deleted project: {project_id}"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to delete project: {e}")
            return CommandResult.error_result(
                message=f"Failed to delete project: {e}",
                errors=[str(e)]
            )

    def reset_session(self) -> CommandResult:
        """
        Reset the runtime session so the database cache is clean.
        Also cleans up the previous project's workspace directory from cache.
        """
        # Clean up previous project's workspace before loading a new one
        if self.current_project_id:
            try:
                from src.utils.paths import cleanup_project_workspace
                cleanup_project_workspace(self.current_project_id)
            except Exception as e:
                from src.utils.message import Log
                Log.warning(f"ApplicationFacade: Failed to clean workspace for project {self.current_project_id}: {e}")
        
        if hasattr(self.project_service, "reset_session"):
            self.project_service.reset_session()
        self.current_project_id = None
        return CommandResult.success_result(
            message="Runtime session cleared"
            )
    
    # ==================== Setlist Operations ====================
    
    def create_setlist_from_folder(
        self,
        audio_folder_path: str,
        default_actions: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> CommandResult:
        """
        Create a new setlist from audio folder - uses current project.
        
        Args:
            audio_folder_path: Path to folder containing audio files
            default_actions: Optional default actions for all songs
            
        Returns:
            CommandResult with setlist data
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            setlist = self.setlist_service.create_setlist_from_folder(
                audio_folder_path=audio_folder_path,
                default_actions=default_actions
            )
            return CommandResult.success_result(
                message=f"Created setlist from folder {audio_folder_path}",
                data=setlist
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to create setlist: {e}")
            return CommandResult.error_result(
                message=f"Failed to create setlist: {e}",
                errors=[str(e)]
            )
    
    def discover_setlist_actions(self, project_id: Optional[str] = None) -> CommandResult:
        """
        Discover available actions for blocks in current project.
        
        Args:
            project_id: Optional project ID (uses current project if not provided)
            
        Returns:
            CommandResult with actions data
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            # Use current project if not provided
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            actions = self.setlist_service.discover_available_actions(project_id)
            return CommandResult.success_result(
                message=f"Discovered actions for {len(actions)} block(s)",
                data=actions
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to discover actions: {e}")
            return CommandResult.error_result(
                message=f"Failed to discover actions: {e}",
                errors=[str(e)]
            )
    
    def get_setlist(self, setlist_id: str) -> CommandResult:
        """
        Get setlist by ID.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            CommandResult with setlist data
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            setlist = self.setlist_service._setlist_repo.get(setlist_id)
            if not setlist:
                return CommandResult.error_result(
                    message="Setlist not found",
                    errors=[f"Setlist {setlist_id} does not exist"]
                )
            return CommandResult.success_result(
                message="Setlist retrieved",
                data=setlist
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get setlist: {e}")
            return CommandResult.error_result(
                message=f"Failed to get setlist: {e}",
                errors=[str(e)]
            )
    
    def list_setlists(self, project_id: Optional[str] = None) -> CommandResult:
        """
        List setlists for current project (or specified project).
        
        Args:
            project_id: Optional project ID (uses current project if not provided)
            
        Returns:
            CommandResult with list of setlists
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            # Use current project if not provided
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            # One setlist per project
            setlist = self.setlist_service._setlist_repo.get_by_project(project_id)
            setlists = [setlist] if setlist else []
            return CommandResult.success_result(
                message=f"Found {'1' if setlist else '0'} setlist(s) for project",
                data=setlists
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list setlists: {e}")
            return CommandResult.error_result(
                message=f"Failed to list setlists: {e}",
                errors=[str(e)]
            )
    
    def get_setlist_songs(self, setlist_id: str) -> CommandResult:
        """
        Get all songs for a setlist.
        
        Args:
            setlist_id: Setlist identifier
            
        Returns:
            CommandResult with list of songs
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            songs = self.setlist_service._setlist_song_repo.list_by_setlist(setlist_id)
            return CommandResult.success_result(
                message=f"Found {len(songs)} song(s)",
                data=songs
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get setlist songs: {e}")
            return CommandResult.error_result(
                message=f"Failed to get setlist songs: {e}",
                errors=[str(e)]
            )
    
    def process_song(
        self,
        setlist_id: str,
        song_id: str,
        progress_callback: Optional[Any] = None,
        action_progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> CommandResult:
        """
        Process a single song in a setlist.
        
        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier
            progress_callback: Optional progress callback (message, current, total)
            action_progress_callback: Optional callback for action-level progress (action_index, total_actions, action_name, status)
            
        Returns:
            CommandResult indicating success/failure
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            success = self.setlist_service.process_song(
                setlist_id=setlist_id,
                song_id=song_id,
                progress_callback=progress_callback,
                action_progress_callback=action_progress_callback
            )
            if success:
                return CommandResult.success_result(
                    message="Song processed successfully"
                )
            else:
                return CommandResult.error_result(
                    message="Song processing failed",
                    errors=["See logs for details"]
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to process song: {e}")
            return CommandResult.error_result(
                message=f"Failed to process song: {e}",
                errors=[str(e)]
            )
    
    def switch_active_song(
        self,
        setlist_id: str,
        song_id: str
    ) -> CommandResult:
        """
        Switch active song with bulletproof state management.
        
        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier to switch to
            
        Returns:
            CommandResult indicating success/failure with detailed error message
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            success, error_msg = self.setlist_service.switch_active_song(
                setlist_id=setlist_id,
                song_id=song_id
            )
            if success:
                return CommandResult.success_result(
                    message="Switched to song successfully"
                )
            else:
                return CommandResult.error_result(
                    message=error_msg or "Failed to switch to song",
                    errors=[error_msg] if error_msg else []
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Error switching song: {e}")
            return CommandResult.error_result(
                message=f"Error switching song: {e}",
                errors=[str(e)]
            )
    
    def process_setlist(
        self,
        setlist_id: str,
        error_callback: Optional[Callable[[str, str], None]] = None,
        action_progress_callback: Optional[Callable[[str, int, int, str, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> CommandResult:
        """
        Process all songs in a setlist with error recovery.
        
        Args:
            setlist_id: Setlist identifier
            error_callback: Optional error callback (song_path, error_message)
            action_progress_callback: Optional callback for action-level progress (song_id, action_index, total_actions, action_name, status)
            cancel_check: Optional callable that returns True if processing should be cancelled
            
        Returns:
            CommandResult with processing results
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            results = self.setlist_service.process_setlist(
                setlist_id=setlist_id,
                error_callback=error_callback,
                action_progress_callback=action_progress_callback,
                cancel_check=cancel_check
            )
            success_count = sum(1 for s in results.values() if s)
            total_count = len(results)
            error_count = total_count - success_count
            message = f"Processed {success_count}/{total_count} song(s) successfully"
            if error_count > 0:
                message += f" ({error_count} error(s))"
            return CommandResult.success_result(
                message=message,
                data=results
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to process setlist: {e}")
            return CommandResult.error_result(
                message=f"Failed to process setlist: {e}",
                errors=[str(e)]
            )
    
    def validate_setlist_actions(self, project_id: Optional[str] = None) -> CommandResult:
        """
        Validate action items before setlist processing begins.
        
        Checks that all referenced blocks and actions exist and are valid.
        Call this before process_setlist to catch configuration errors early.
        
        Args:
            project_id: Optional project ID (uses current project if not provided)
            
        Returns:
            CommandResult with list of validation error strings (empty list = valid)
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            errors = self.setlist_service.validate_action_items(project_id)
            if errors:
                return CommandResult.error_result(
                    message=f"Validation found {len(errors)} issue(s)",
                    errors=errors,
                    data=errors
                )
            return CommandResult.success_result(
                message="All action items are valid",
                data=[]
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to validate setlist actions: {e}")
            return CommandResult.error_result(
                message=f"Validation failed: {e}",
                errors=[str(e)]
            )
    
    # ==================== Action Set Operations ====================
    
    def save_action_set_from_project(self, name: str, description: str = "", project_id: Optional[str] = None) -> CommandResult:
        """
        Save the current project's ActionItems as an ActionSet (bundle/preset).
        
        This creates a new ActionSet from the project's current ActionItems,
        which can then be loaded into other projects.
        
        Args:
            name: Name for the ActionSet
            description: Optional description
            project_id: Project ID (uses current project if not provided)
            
        Returns:
            CommandResult with created ActionSet
        """
        from src.features.projects.domain import ActionSet, ActionItem
        
        try:
            if not project_id:
                project_id = self.current_project_id
            
            if not project_id:
                return CommandResult.error_result(
                    message="No project specified",
                    errors=["Please open or create a project first"]
                )
            
            # Get current project ActionItems
            items_result = self.list_action_items_by_project(project_id)
            if not items_result.success:
                return items_result
            
            action_items = items_result.data or []
            
            if not action_items:
                return CommandResult.error_result(
                    message="No action items in project",
                    errors=["Add action items to the project before saving as ActionSet"]
                )
            
            # Create ActionSet from ActionItems
            # Copy ActionItems but clear project_id and action_set_id (these are template items)
            template_items = []
            for item in action_items:
                template_item = ActionItem(
                    action_type=item.action_type,
                    action_name=item.action_name,
                    action_description=item.action_description,
                    block_id=item.block_id,
                    block_name=item.block_name,
                    action_args=item.action_args.copy() if item.action_args else {},
                    order_index=item.order_index,
                    metadata=item.metadata.copy() if item.metadata else {}
                )
                template_items.append(template_item)
            
            action_set = ActionSet(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                actions=template_items,
                project_id=None  # ActionSets are global templates
            )
            
            # Save to file storage (global, reusable)
            saved = self.action_set_file_repo.save(action_set)
            
            Log.info(f"ApplicationFacade: Saved {len(template_items)} action(s) from project as ActionSet '{name}'")
            return CommandResult.success_result(
                message=f"Saved {len(template_items)} action(s) as ActionSet '{name}'",
                data=saved
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to save action set from project: {e}")
            return CommandResult.error_result(
                message=f"Failed to save action set: {e}",
                errors=[str(e)]
            )
    
    def save_action_set(self, action_set: ActionSet) -> CommandResult:
        """
        Save an action set to file and database.
        
        Action sets are saved:
        - As JSON files in the user data directory (global/reusable)
        - To the database for runtime access and project association
        
        Args:
            action_set: ActionSet entity to save
            
        Returns:
            CommandResult with saved action set
        """
        try:
            # Ensure project_id is set if we have a current project
            if not action_set.project_id and self.current_project_id:
                action_set.project_id = self.current_project_id
                Log.debug(f"ApplicationFacade: Set project_id={self.current_project_id} on action set '{action_set.name}'")
            
            # Save to file-based repository (global storage)
            saved = self.action_set_file_repo.save(action_set)
            
            # Also save to database repository (for runtime access and project association)
            if self.action_set_repo:
                try:
                    # Check if exists in DB (by ID or by name+project_id)
                    existing = None
                    if action_set.id:
                        try:
                            existing = self.action_set_repo.get(action_set.id)
                        except Exception:
                            pass
                    
                    # If not found by ID, try to find by name and project_id
                    if not existing and action_set.project_id:
                        try:
                            all_sets = self.action_set_repo.list_by_project(action_set.project_id)
                            existing = next((s for s in all_sets if s.name == action_set.name), None)
                        except Exception:
                            pass
                    
                    if existing:
                        # Update existing - preserve ID but update all other fields
                        action_set.id = existing.id  # Preserve existing ID
                        self.action_set_repo.update(action_set)
                        Log.debug(f"ApplicationFacade: Updated action set '{action_set.name}' in database (id: {action_set.id})")
                    else:
                        self.action_set_repo.create(action_set)
                        Log.debug(f"ApplicationFacade: Created action set '{action_set.name}' in database (id: {action_set.id})")
                except Exception as db_error:
                    Log.warning(f"ApplicationFacade: Failed to save action set to database: {db_error}")
                    # Don't fail - file save succeeded
            
            Log.info(f"ApplicationFacade: Saved action set '{action_set.name}'")
            return CommandResult.success_result(
                message=f"Action set '{action_set.name}' saved",
                data=saved
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to save action set: {e}")
            return CommandResult.error_result(
                message=f"Failed to save action set: {e}",
                errors=[str(e)]
            )
    
    def load_action_set(self, name_or_id: str) -> CommandResult:
        """
        Load an action set by name or ID.
        
        Searches file-based storage first, then database if not found.
        
        Args:
            name_or_id: Action set name or identifier
            
        Returns:
            CommandResult with action set
        """
        try:
            # Try loading from file-based storage first (by name)
            action_set = self.action_set_file_repo.load(name_or_id)
            
            # If not found by name, try by ID in files
            if not action_set:
                action_set = self.action_set_file_repo.load_by_id(name_or_id)
            
            # If still not found, try database
            if not action_set and self.action_set_repo:
                try:
                    action_set = self.action_set_repo.get(name_or_id)
                except Exception:
                    pass
            
            if not action_set:
                return CommandResult.error_result(
                    message=f"Action set '{name_or_id}' not found",
                    errors=["Action set not found"]
                )
            
            return CommandResult.success_result(
                message=f"Loaded action set '{action_set.name}'",
                data=action_set
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to load action set: {e}")
            return CommandResult.error_result(
                message=f"Failed to load action set: {e}",
                errors=[str(e)]
            )
    
    def load_action_set_into_project(self, name_or_id: str, project_id: Optional[str] = None, replace: bool = False) -> CommandResult:
        """
        Load an ActionSet's actions into the project's ActionItems list.
        
        ActionSets are bundles/presets that can be loaded into the project.
        This copies the ActionSet's ActionItems into the project's ActionItems.
        
        Args:
            name_or_id: Action set name or identifier
            project_id: Project ID (uses current project if not provided)
            replace: If True, replace existing ActionItems. If False, append to existing.
            
        Returns:
            CommandResult with list of loaded ActionItems
        """
        from src.features.projects.domain import ActionItem
        
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            if not project_id:
                project_id = self.current_project_id
            
            if not project_id:
                return CommandResult.error_result(
                    message="No project specified",
                    errors=["Please open or create a project first"]
                )
            
            # Load the ActionSet
            action_set_result = self.load_action_set(name_or_id)
            if not action_set_result.success:
                return action_set_result
            
            action_set = action_set_result.data
            
            # If replacing, clear existing ActionItems for this project
            if replace:
                existing_items = self.action_item_repo.list_by_project(project_id)
                for item in existing_items:
                    self.action_item_repo.delete(item.id)
                Log.info(f"ApplicationFacade: Cleared {len(existing_items)} existing action item(s) from project")
            
            # Get current count for ordering
            existing_items = self.action_item_repo.list_by_project(project_id)
            start_index = len(existing_items)
            
            # Copy ActionItems from ActionSet to project
            loaded_items = []
            for idx, source_item in enumerate(action_set.actions):
                # Create new ActionItem for project (don't copy action_set_id)
                new_item = ActionItem(
                    action_type=source_item.action_type,
                    action_name=source_item.action_name,
                    action_description=source_item.action_description,
                    block_id=source_item.block_id,
                    block_name=source_item.block_name,
                    action_args=source_item.action_args.copy() if source_item.action_args else {},
                    project_id=project_id,
                    action_set_id="",  # Clear action_set_id - these are project-level ActionItems
                    order_index=start_index + idx,
                    metadata=source_item.metadata.copy() if source_item.metadata else {}
                )
                
                # Generate new ID
                new_item.id = str(uuid.uuid4())
                
                # Save to project
                created = self.action_item_repo.create(new_item)
                loaded_items.append(created)
            
            Log.info(f"ApplicationFacade: Loaded {len(loaded_items)} action(s) from ActionSet '{action_set.name}' into project")
            return CommandResult.success_result(
                message=f"Loaded {len(loaded_items)} action(s) from '{action_set.name}' into project",
                data=loaded_items
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to load action set into project: {e}")
            return CommandResult.error_result(
                message=f"Failed to load action set into project: {e}",
                errors=[str(e)]
            )
    
    def list_action_sets(self, project_id: Optional[str] = None) -> CommandResult:
        """
        List all available action sets.
        
        Combines action sets from file storage (global) and database (project-specific).
        File-based sets take priority if there are name conflicts.
        
        Args:
            project_id: Optional project ID to include project-specific sets
            
        Returns:
            CommandResult with list of action sets
        """
        try:
            # Start with file-based action sets (global)
            action_sets = self.action_set_file_repo.list_all()
            seen_names = {a.name for a in action_sets}
            
            # Add database action sets that aren't already in files
            if self.action_set_repo:
                try:
                    db_sets = self.action_set_repo.list_all() if not project_id else self.action_set_repo.list_by_project(project_id)
                    for db_set in db_sets:
                        if db_set.name not in seen_names:
                            action_sets.append(db_set)
                            seen_names.add(db_set.name)
                except Exception as db_error:
                    Log.debug(f"ApplicationFacade: Could not list database action sets: {db_error}")
            
            return CommandResult.success_result(
                message=f"Found {len(action_sets)} action set(s)",
                data=action_sets
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list action sets: {e}")
            return CommandResult.error_result(
                message=f"Failed to list action sets: {e}",
                errors=[str(e)]
            )
    
    def delete_action_set(self, name: str) -> CommandResult:
        """
        Delete an action set by name.
        
        Args:
            name: Action set name
            
        Returns:
            CommandResult indicating success or failure
        """
        try:
            if self.action_set_file_repo.delete(name):
                return CommandResult.success_result(
                    message=f"Action set '{name}' deleted"
                )
            else:
                return CommandResult.error_result(
                    message=f"Action set '{name}' not found",
                    errors=["Action set not found"]
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to delete action set: {e}")
            return CommandResult.error_result(
                message=f"Failed to delete action set: {e}",
                errors=[str(e)]
            )
    
    # ==================== Action Item Operations (User-Configured Actions) ====================
    
    def add_action_item(self, action_set_id: str, action_item) -> CommandResult:
        """
        Add an action item to an action set (auto-saves to DB).
        
        Args:
            action_set_id: ID of the action set to add to
            action_item: ActionItem entity to add
            
        Returns:
            CommandResult with created action item
        """
        from src.features.projects.domain import ActionItem
        
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            # Ensure action_set_id and project_id are set
            action_item.action_set_id = action_set_id
            if not action_item.project_id and self.current_project_id:
                action_item.project_id = self.current_project_id
            
            # Get existing items to determine if we should set order_index
            existing_items = self.action_item_repo.list_by_action_set(action_set_id)
            existing_count = len(existing_items)
            
            # Only auto-set order_index to end if order_index is 0 (default) AND there are existing items
            # This allows callers to explicitly set order_index for insertions at any position (including 0)
            # If order_index is already set to a non-zero value, it was explicitly set - preserve it
            # If order_index is 0 and there are no existing items, keep as 0 (first item)
            if action_item.order_index == 0 and existing_count > 0:
                # Default value (0) with existing items - append to end
                action_item.order_index = existing_count
            # Otherwise, preserve the order_index that was set (allows explicit insertions)
            
            # Save to database
            created = self.action_item_repo.create(action_item)
            
            Log.info(f"ApplicationFacade: Added action item '{action_item.action_name}' to set {action_set_id}")
            return CommandResult.success_result(
                message=f"Action item '{action_item.action_name}' added",
                data=created
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to add action item: {e}")
            return CommandResult.error_result(
                message=f"Failed to add action item: {e}",
                errors=[str(e)]
            )
    
    def update_action_item(self, action_item) -> CommandResult:
        """
        Update an existing action item (auto-saves to DB).
        
        Args:
            action_item: ActionItem entity to update
            
        Returns:
            CommandResult indicating success or failure
        """
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            action_item.update_modified()
            self.action_item_repo.update(action_item)
            
            Log.info(f"ApplicationFacade: Updated action item '{action_item.action_name}'")
            return CommandResult.success_result(
                message=f"Action item '{action_item.action_name}' updated"
            )
        except ValueError as e:
            return CommandResult.error_result(
                message=str(e),
                errors=[str(e)]
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to update action item: {e}")
            return CommandResult.error_result(
                message=f"Failed to update action item: {e}",
                errors=[str(e)]
            )
    
    def remove_action_item(self, action_item_id: str) -> CommandResult:
        """
        Remove an action item from its action set (auto-removes from DB).
        
        Args:
            action_item_id: ID of the action item to remove
            
        Returns:
            CommandResult indicating success or failure
        """
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            # Get item to find action_set_id for reordering
            item = self.action_item_repo.get(action_item_id)
            if not item:
                return CommandResult.error_result(
                    message=f"Action item '{action_item_id}' not found",
                    errors=["Action item not found"]
                )
            
            action_set_id = item.action_set_id
            
            self.action_item_repo.delete(action_item_id)
            
            # Reorder remaining items
            remaining_items = self.action_item_repo.list_by_action_set(action_set_id)
            if remaining_items:
                item_ids = [i.id for i in remaining_items]
                self.action_item_repo.reorder(action_set_id, item_ids)
            
            Log.info(f"ApplicationFacade: Removed action item {action_item_id}")
            return CommandResult.success_result(
                message="Action item removed"
            )
        except ValueError as e:
            return CommandResult.error_result(
                message=str(e),
                errors=[str(e)]
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to remove action item: {e}")
            return CommandResult.error_result(
                message=f"Failed to remove action item: {e}",
                errors=[str(e)]
            )
    
    def list_action_items(self, action_set_id: str) -> CommandResult:
        """
        List all action items for an action set.
        
        Args:
            action_set_id: ID of the action set
            
        Returns:
            CommandResult with list of action items
        """
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            items = self.action_item_repo.list_by_action_set(action_set_id)
            return CommandResult.success_result(
                message=f"Found {len(items)} action item(s)",
                data=items
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list action items: {e}")
            return CommandResult.error_result(
                message=f"Failed to list action items: {e}",
                errors=[str(e)]
            )
    
    def list_action_items_by_project(self, project_id: Optional[str] = None) -> CommandResult:
        """
        List all action items for a project.
        
        Args:
            project_id: Project ID (uses current project if not provided)
            
        Returns:
            CommandResult with list of action items
        """
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            if not project_id:
                project_id = self.current_project_id
            
            if not project_id:
                return CommandResult.error_result(
                    message="No project specified",
                    errors=["Please open or create a project first"]
                )
            
            items = self.action_item_repo.list_by_project(project_id)
            Log.debug(f"ApplicationFacade: Found {len(items)} action item(s) for project {project_id}")
            return CommandResult.success_result(
                message=f"Found {len(items)} action item(s)",
                data=items
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list action items by project: {e}")
            return CommandResult.error_result(
                message=f"Failed to list action items: {e}",
                errors=[str(e)]
            )
    
    def reorder_action_items(self, action_set_id: str, item_ids: List[str]) -> CommandResult:
        """
        Reorder action items within an action set.
        
        Args:
            action_set_id: ID of the action set
            item_ids: List of action item IDs in desired order
            
        Returns:
            CommandResult indicating success or failure
        """
        try:
            if not self.action_item_repo:
                return CommandResult.error_result(
                    message="Action item repository not available",
                    errors=["Action item repository not initialized"]
                )
            
            self.action_item_repo.reorder(action_set_id, item_ids)
            
            Log.info(f"ApplicationFacade: Reordered {len(item_ids)} action items in set {action_set_id}")
            return CommandResult.success_result(
                message=f"Reordered {len(item_ids)} action items"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to reorder action items: {e}")
            return CommandResult.error_result(
                message=f"Failed to reorder action items: {e}",
                errors=[str(e)]
            )
    
    def add_song_to_setlist(self, audio_path: str, project_id: Optional[str] = None) -> CommandResult:
        """
        Add a song to the project's setlist.
        
        One setlist per project - adds song to that setlist.
        
        Args:
            audio_path: Path to audio file
            project_id: Optional project ID (uses current project if not provided)
            
        Returns:
            CommandResult with created song
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            song = self.setlist_service.add_song_to_setlist(project_id, audio_path)
            return CommandResult.success_result(
                message=f"Added song to setlist: {Path(audio_path).name}",
                data=song
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to add song to setlist: {e}")
            return CommandResult.error_result(
                message=f"Failed to add song: {e}",
                errors=[str(e)]
            )
    
    def remove_song_from_setlist(self, song_id: str, project_id: Optional[str] = None) -> CommandResult:
        """
        Remove a song from the project's setlist.
        
        Args:
            song_id: Song identifier to remove
            project_id: Optional project ID (uses current project if not provided)
            
        Returns:
            CommandResult indicating success/failure
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            self.setlist_service.remove_song_from_setlist(project_id, song_id)
            return CommandResult.success_result(
                message="Removed song from setlist"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to remove song from setlist: {e}")
            return CommandResult.error_result(
                message=f"Failed to remove song: {e}",
                errors=[str(e)]
            )
    
    def update_setlist_song(
        self,
        project_id: Optional[str] = None,
        song_id: str = "",
        updates: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """
        Update a setlist song's fields (e.g., action_overrides, metadata).
        
        Args:
            project_id: Optional project ID (uses current project if not provided)
            song_id: Song identifier to update
            updates: Dict of field names to new values
            
        Returns:
            CommandResult indicating success/failure
        """
        if not self.setlist_service:
            return CommandResult.error_result(
                message="Setlist service not available",
                errors=["Setlist service not initialized"]
            )
        
        try:
            if not project_id:
                if not self.current_project_id:
                    return CommandResult.error_result(
                        message="No current project",
                        errors=["Please open or create a project first"]
                    )
                project_id = self.current_project_id
            
            # Get the song from the repo
            song = self.setlist_service._setlist_song_repo.get(song_id)
            if not song:
                return CommandResult.error_result(
                    message=f"Song {song_id} not found",
                    errors=[f"Song {song_id} not found"]
                )
            
            # Apply updates
            if updates:
                for key, value in updates.items():
                    if hasattr(song, key):
                        setattr(song, key, value)
            
            self.setlist_service._setlist_song_repo.update(song)
            return CommandResult.success_result(
                message="Song updated",
                data=song
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to update setlist song: {e}")
            return CommandResult.error_result(
                message=f"Failed to update song: {e}",
                errors=[str(e)]
            )
    
    # ==================== Block Operations ====================
    
    def add_block(self, block_type: str, name: Optional[str] = None) -> CommandResult[Block]:
        """
        Add a block to current project.
        
        Args:
            block_type: Type of block (e.g., "LoadAudio")
            name: Optional block name
            
        Returns:
            CommandResult[Block] containing the newly created Block entity
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        # Validate block type (case-insensitive)
        from src.application.block_registry import get_block_registry
        registry = get_block_registry()
        block_metadata = registry.get(block_type)
        
        if not block_metadata:
            available_types = [bt.type_id for bt in registry.list_all()]
            return CommandResult.error_result(
                message=f"Unknown block type: '{block_type}'",
                errors=[f"Available types: {', '.join(available_types)}"]
            )
        
        # Use the correct type_id from metadata (in case user typed it with wrong case)
        correct_type_id = block_metadata.type_id
        
        try:
            block = self.block_service.add_block(
                self.current_project_id,
                correct_type_id,
                name=name
            )
            
            
            
            result = CommandResult.success_result(
                message=f"Added block '{block.name}' (type: {block.type})",
                data=block
            )
            
            
            
            return result
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to add block: {e}")
            
            
            return CommandResult.error_result(
                message=f"Failed to add block: {e}",
                errors=[str(e)]
            )
    
    def delete_block(self, identifier: str) -> CommandResult:
        """
        Delete a block by ID or name.
        
        Args:
            identifier: Block ID or block name
            
        Returns:
            CommandResult
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            # Try to find block by name first
            block = self.block_service.find_by_name(self.current_project_id, identifier)
            if not block:
                # Try as block_id
                block = self.block_service.get_block(self.current_project_id, identifier)
            
            if block:
                # Clean up processor resources (PyTorch models, etc.) before deletion
                if self.execution_engine:
                    processor = self.execution_engine.get_processor(block)
                    if processor and hasattr(processor, 'cleanup'):
                        try:
                            processor.cleanup(block)
                            Log.debug(f"ApplicationFacade: Cleaned up processor for block '{block.name}'")
                        except Exception as cleanup_err:
                            Log.warning(f"ApplicationFacade: Processor cleanup failed for '{block.name}': {cleanup_err}")

                self.block_service.remove_block(self.current_project_id, block.id)
                return CommandResult.success_result(
                    message=f"Deleted block '{block.name}'"
                )
            else:
                return CommandResult.error_result(
                    message=f"Block not found: {identifier}"
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to delete block: {e}")
            return CommandResult.error_result(
                message=f"Failed to delete block: {e}",
                errors=[str(e)]
            )
    
    def list_blocks(self) -> CommandResult[List[Block]]:
        """
        List all blocks in current project.
        
        Returns:
            CommandResult[List[Block]] containing all blocks in the current project
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            blocks = self.block_service.list_blocks(self.current_project_id)
            return CommandResult.success_result(
                message=f"Found {len(blocks)} block(s)",
                data=blocks
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list blocks: {e}")
            return CommandResult.error_result(
                message=f"Failed to list blocks: {e}",
                errors=[str(e)]
            )
    
    def list_block_types(self) -> CommandResult:
        """
        List all available block types.
        
        Returns:
            CommandResult with flat list of block types (sorted by name)
        """
        try:
            from src.application.block_registry import get_block_registry
            registry = get_block_registry()
            block_types = registry.list_all()
            
            # Return flat list of block type info
            block_list = []
            for bt in block_types:
                block_list.append({
                    "type_id": bt.type_id,
                    "name": bt.name,
                    "description": bt.description,
                    "inputs": {name: pt.name for name, pt in bt.inputs.items()},
                    "outputs": {name: pt.name for name, pt in bt.outputs.items()},
                    "tags": bt.tags
                })
            
            # Sort by name for consistent display
            block_list.sort(key=lambda x: x["name"])
            
            return CommandResult.success_result(
                message=f"Found {len(block_types)} block type(s)",
                data=block_list
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list block types: {e}")
            return CommandResult.error_result(
                message=f"Failed to list block types: {e}",
                errors=[str(e)]
            )

    def list_recent_projects(self, limit: int = 10):
        """
        Helper for CLI when presenting recent projects (not exposed via CLI command).
        """
        return self.project_service.list_recent_projects(limit)
    
    def rename_block(self, identifier: str, new_name: str) -> CommandResult[Block]:
        """
        Rename a block.
        
        Args:
            identifier: Block ID or name
            new_name: New block name
            
        Returns:
            CommandResult[Block] containing the renamed Block entity
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            # Find block
            block = self.block_service.find_by_name(self.current_project_id, identifier)
            if not block:
                block = self.block_service.get_block(self.current_project_id, identifier)
            
            if not block:
                return CommandResult.error_result(
                    message=f"Block not found: {identifier}"
                )
            
            # Rename
            updated_block = self.block_service.rename_block(
                self.current_project_id,
                block.id,
                new_name
            )
            return CommandResult.success_result(
                message=f"Renamed block to '{updated_block.name}'",
                data=updated_block
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to rename block: {e}")
            return CommandResult.error_result(
                message=f"Failed to rename block: {e}",
                errors=[str(e)]
            )

    def get_panel_state(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Get panel state for a block.
        
        Status conditions can use this to access panel internal state
        (e.g., listener status, connection state) that isn't in block metadata.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Dictionary of panel state, or None if no panel is registered or service unavailable
        """
        if not self.block_status_service:
            return None
        return self.block_status_service.get_panel_state(block_id)
    
    def describe_block(self, identifier: str, skip_port_check: bool = False) -> CommandResult[Block]:
        """
        Get detailed information about a block.
        
        Returns the Block entity which contains all metadata, ports, and configuration.
        Use block.metadata for custom metadata, block.ports for port definitions.
        
        Args:
            identifier: Block ID or name
            skip_port_check: If True, skip port validation and DB updates (for performance during scene refresh)
            
        Returns:
            CommandResult[Block] containing the Block entity with all its data
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )

        block = self._get_block_by_identifier(identifier, skip_port_check=skip_port_check)
        if not block:
            return CommandResult.error_result(
                message=f"Block not found: {identifier}"
            )

        try:
            # Block entity already contains all information:
            # - block.id, block.name, block.type
            # - block.ports (unified port definitions)
            # - block.metadata (all custom data)
            # No need to nest in a complex dict structure

            return CommandResult.success_result(
                message=f"Block: {block.name} ({block.type})",
                data=block
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to describe block: {e}")
            return CommandResult.error_result(
                message=f"Failed to describe block: {e}",
                errors=[str(e)]
            )

    def execute_block_command(
        self,
        identifier: str,
        command_name: str,
        args: List[str],
        kwargs: Dict[str, str]
    ) -> CommandResult:
        """
        Execute a block-specific command.
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )

        block = self._get_block_by_identifier(identifier)
        if not block:
            return CommandResult.error_result(
                message=f"Block not found: {identifier}"
            )

        from src.application.block_registry import get_block_registry
        registry = get_block_registry()
        metadata = registry.get(block.type)
        if not metadata or not metadata.commands:
            return CommandResult.error_result(
                message=f"No commands defined for block type '{block.type}'"
            )

        command_meta = self._find_block_command(metadata.commands, command_name)
        if not command_meta:
            return CommandResult.error_result(
                message=f"Command '{command_name}' not available for block type '{block.type}'",
                errors=["Use 'block_help <block_id|name>' to list commands"]
            )

        validation_error = self._validate_block_command_input(command_meta, args, kwargs)
        if validation_error:
            return CommandResult.error_result(
                message=validation_error["message"],
                errors=validation_error["errors"]
            )

        try:
            updated_block = self.block_service.execute_block_command(
                self.current_project_id,
                block.id,
                command_meta.get("name", command_name),
                args,
                kwargs
            )

            return CommandResult.success_result(
                message=f"Executed '{command_meta.get('name', command_name)}' on block '{updated_block.name}'",
                data={
                    "block": updated_block,
                    "command": command_meta,
                    "args": args,
                    "kwargs": kwargs
                }
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to execute block command: {e}")
            return CommandResult.error_result(
                message=f"Failed to execute block command: {e}",
                errors=[str(e)]
            )

    def _validate_block_command_input(
        self,
        command_meta: Dict[str, Any],
        args: List[str],
        kwargs: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Ensure block command receives its required arguments."""
        argument_defs = command_meta.get("arguments", [])
        missing_errors = []

        positional_required = [
            arg for arg in argument_defs
            if arg.get("source", "positional") == "positional" and arg.get("required", False)
        ]

        if len(args) < len(positional_required):
            missing_names = ", ".join(arg.get("name") for arg in positional_required[len(args):])
            usage = command_meta.get("usage") or "Consult block_help"
            missing_errors.append(f"Missing required positional arguments: {missing_names}")
            missing_errors.append(f"Usage: {usage}")
            return {
                "message": f"Invalid input for command '{command_meta.get('name')}'",
                "errors": missing_errors
            }

        for arg_def in argument_defs:
            if arg_def.get("source") == "kw" and arg_def.get("required", False):
                name = arg_def.get("name")
                if name not in kwargs:
                    usage = command_meta.get("usage") or "Consult block_help"
                    missing_errors.append(f"Missing required keyword argument: {name}")
                    missing_errors.append(f"Usage: {usage}")
                    return {
                        "message": f"Invalid input for command '{command_meta.get('name')}'",
                        "errors": missing_errors
                    }

        return None

    def _find_block_command(self, commands: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
        """Find the command metadata for a given command name or alias."""
        target_lower = target.lower()
        for command in commands:
            name = command.get("name", "")
            if name.lower() == target_lower:
                return command
            for alias in command.get("aliases", []):
                if alias.lower() == target_lower:
                    return command
        return None

    def _get_block_by_identifier(self, identifier: str, skip_port_check: bool = False):
        """
        Resolve a block by name or ID within the current project.
        
        Args:
            identifier: Block ID or name
            skip_port_check: If True, skip port validation (for performance during scene refresh)
        """
        if not self.current_project_id:
            return None

        block = self.block_service.find_by_name(self.current_project_id, identifier)
        if block:
            if not skip_port_check:
                # Ensure ports for name-based lookup too
                self.block_service._ensure_default_ports(block, skip_db_update=True)
            return block

        return self.block_service.get_block(self.current_project_id, identifier, skip_port_check=skip_port_check)
    
    # ==================== UI-Focused Block Methods ====================
    
    def get_block_metadata(self, identifier: str) -> CommandResult[Dict[str, Any]]:
        """
        Get block metadata without full block details.
        Useful for UI state persistence.
        
        Args:
            identifier: Block ID or name
            
        Returns:
            CommandResult[Dict[str, Any]] containing the block's metadata dictionary
        """
        if not self.current_project_id:
            return CommandResult.error_result(message="No project loaded")
        
        try:
            block = self._get_block_by_identifier(identifier)
            if not block:
                return CommandResult.error_result(message=f"Block not found: {identifier}")
            
            return CommandResult.success_result(
                message=f"Retrieved metadata for block '{block.name}'",
                data=block.metadata or {}
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get block metadata: {e}")
            return CommandResult.error_result(
                message=f"Failed to get block metadata: {e}",
                errors=[str(e)]
            )
    
    def update_block_metadata(self, identifier: str, metadata_updates: Dict[str, Any]) -> CommandResult[Block]:
        """
        Update block metadata (merges with existing metadata).
        
        IMPORTANT: Block metadata should contain DOMAIN DATA ONLY.
        For UI-specific state (positions, zoom, etc.), use set_ui_state() instead.
        
        Args:
            identifier: Block ID or name
            metadata_updates: Dict of metadata to merge (domain data only)
            
        Returns:
            CommandResult[Block] containing the updated Block entity
        """
        if not self.current_project_id:
            return CommandResult.error_result(message="No project loaded")
        
        try:
            # Validate: Reject UI-specific keys (Phase B enforcement)
            RESERVED_UI_KEYS = {'ui_position', 'x', 'y', 'ui_zoom', 'ui_viewport', 'ui_panel_open'}
            
            for key in metadata_updates.keys():
                if key in RESERVED_UI_KEYS or key.startswith('ui_'):
                    return CommandResult.error_result(
                        message=f"Cannot store UI state '{key}' in block metadata. "
                                f"Block metadata is for domain data only. "
                                f"Use facade.set_ui_state() for UI-specific data.",
                        errors=[f"Invalid metadata key: {key}"]
                    )
            
            block = self._get_block_by_identifier(identifier)
            if not block:
                return CommandResult.error_result(message=f"Block not found: {identifier}")
            
            # Merge metadata
            if not block.metadata:
                block.metadata = {}
            block.metadata.update(metadata_updates)
            
            # Update in database
            updated_block = self.block_service.update_block(
                self.current_project_id,
                block.id,
                block
            )
            
            return CommandResult.success_result(
                message=f"Updated metadata for block '{block.name}'",
                data=updated_block
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to update block metadata: {e}")
            return CommandResult.error_result(
                message=f"Failed to update block metadata: {e}",
                errors=[str(e)]
            )
    
    def reset_block_state(self, identifier: str) -> CommandResult[Block]:
        """
        Reset a block's metadata to its initial state (empty dict).
        
        This clears all configuration and settings, returning the block
        to the state it had when first created. This operation is undoable.
        
        Args:
            identifier: Block ID or name
            
        Returns:
            CommandResult[Block] containing the updated Block entity
        """
        if not self.current_project_id:
            return CommandResult.error_result(message="No project loaded")
        
        try:
            block = self._get_block_by_identifier(identifier)
            if not block:
                return CommandResult.error_result(message=f"Block not found: {identifier}")
            
            # Use command for undo support
            if self.command_bus:
                from src.application.commands.block_commands import ResetBlockStateCommand
                cmd = ResetBlockStateCommand(self, block.id)
                result = self.command_bus.execute(cmd)
                if result:
                    # Get updated block
                    updated_result = self.describe_block(block.id)
                    if updated_result.success and updated_result.data:
                        return CommandResult.success_result(
                            message=f"Reset state for block '{block.name}'",
                            data=updated_result.data
                        )
                    return CommandResult.success_result(
                        message=f"Reset state for block '{block.name}'"
                    )
                else:
                    return CommandResult.error_result(
                        message="Failed to reset block state"
                    )
            else:
                # Fallback if command_bus not available (shouldn't happen in normal usage)
                updated_block = self.block_service.reset_block_state(
                    self.current_project_id,
                    block.id
                )
                return CommandResult.success_result(
                    message=f"Reset state for block '{block.name}'",
                    data=updated_block
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to reset block state: {e}")
            return CommandResult.error_result(
                message=f"Failed to reset block state: {e}",
                errors=[str(e)]
            )
    
    def set_block_metadata(self, identifier: str, metadata: Dict[str, Any]) -> CommandResult:
        """
        Replace block metadata entirely (does not merge).
        
        IMPORTANT: Block metadata should contain DOMAIN DATA ONLY.
        For UI-specific state (positions, zoom, etc.), use set_ui_state() instead.
        
        Args:
            identifier: Block ID or name
            metadata: Complete metadata dict to set (domain data only)
            
        Returns:
            CommandResult with updated block
        """
        if not self.current_project_id:
            return CommandResult.error_result(message="No project loaded")
        
        try:
            # Validate: Reject UI-specific keys (Phase B enforcement)
            RESERVED_UI_KEYS = {'ui_position', 'x', 'y', 'ui_zoom', 'ui_viewport', 'ui_panel_open'}
            
            for key in (metadata or {}).keys():
                if key in RESERVED_UI_KEYS or key.startswith('ui_'):
                    return CommandResult.error_result(
                        message=f"Cannot store UI state '{key}' in block metadata. "
                                f"Block metadata is for domain data only. "
                                f"Use facade.set_ui_state() for UI-specific data.",
                        errors=[f"Invalid metadata key: {key}"]
                    )
            
            block = self._get_block_by_identifier(identifier)
            if not block:
                return CommandResult.error_result(message=f"Block not found: {identifier}")
            
            # Replace metadata
            block.metadata = metadata or {}
            
            # Update in database
            updated_block = self.block_service.update_block(
                self.current_project_id,
                block.id,
                block
            )
            
            return CommandResult.success_result(
                message=f"Set metadata for block '{block.name}'",
                data=updated_block
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to set block metadata: {e}")
            return CommandResult.error_result(
                message=f"Failed to set block metadata: {e}",
                errors=[str(e)]
            )
    
    def get_blocks_batch(self, block_ids: List[str]) -> CommandResult:
        """
        Get multiple blocks in one call.
        More efficient than calling describe_block multiple times.
        
        Uses optimized batch loading: loads all blocks for project, then filters by IDs.
        This is more efficient than individual queries when loading many blocks.
        
        Args:
            block_ids: List of block IDs
            
        Returns:
            CommandResult with dict mapping block_id -> block
        """
        if not self.current_project_id:
            return CommandResult.error_result(message="No project loaded")
        
        if not block_ids:
            return CommandResult.success_result(
                message="No blocks requested",
                data={}
            )
        
        try:
            # Optimize: If requesting most/all blocks, load all at once and filter
            # This is more efficient than individual queries
            all_blocks = self.block_service._block_repo.list_by_project(self.current_project_id)
            block_ids_set = set(block_ids)
            
            # Filter to requested blocks
            blocks = {block.id: block for block in all_blocks if block.id in block_ids_set}
            
            # Ensure all blocks have their default ports (skip DB update for performance)
            for block in blocks.values():
                self.block_service._ensure_default_ports(block, skip_db_update=True)
            
            # Ensure all requested blocks are found (log missing ones)
            missing = block_ids_set - set(blocks.keys())
            if missing:
                Log.debug(f"ApplicationFacade: {len(missing)} requested block(s) not found: {missing}")
            
            return CommandResult.success_result(
                message=f"Retrieved {len(blocks)} blocks",
                data=blocks
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get blocks batch: {e}")
            return CommandResult.error_result(
                message=f"Failed to get blocks batch: {e}",
                errors=[str(e)]
            )
    
    # ==================== Connection Operations ====================
    
    def connect_blocks(
        self,
        source_block_id: str,
        source_output: str,
        target_block_id: str,
        target_input: str
    ) -> CommandResult:
        """
        Connect two blocks.
        
        Args:
            source_block_id: Source block ID
            source_output: Source output port name
            target_block_id: Target block ID
            target_input: Target input port name
            
        Returns:
            CommandResult with connection data
        """
        source_block = self._get_block_by_identifier(source_block_id)
        target_block = self._get_block_by_identifier(target_block_id)

        if not source_block:
            return CommandResult.error_result(
                message=f"Source block not found: {source_block_id}"
            )
        if not target_block:
            return CommandResult.error_result(
                message=f"Target block not found: {target_block_id}"
            )

        try:
            connection = self.connection_service.connect_blocks(
                source_block.id,
                source_output,
                target_block.id,
                target_input
            )
            
            # Recalculate expected outputs for target block (may depend on connections)
            # Also recalculate for source block if it outputs based on inputs
            self._recalculate_expected_outputs(target_block.id)
            self._recalculate_expected_outputs(source_block.id)
            
            return CommandResult.success_result(
                message=f"Connected {source_block_id}.{source_output} -> {target_block_id}.{target_input}",
                data=connection
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to connect blocks: {e}")
            return CommandResult.error_result(
                message=f"Failed to connect blocks: {e}",
                errors=[str(e)]
            )
    
    def disconnect_blocks(self, connection_id: str) -> CommandResult:
        """
        Disconnect blocks by connection ID.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            CommandResult
        """
        try:
            # Get connection before deleting to know which blocks are affected
            connection = self.connection_service.get_connection(connection_id)
            
            self.connection_service.disconnect_blocks(connection_id)
            
            # Recalculate expected outputs for affected blocks
            if connection:
                self._recalculate_expected_outputs(connection.target_block_id)
                self._recalculate_expected_outputs(connection.source_block_id)
            
            return CommandResult.success_result(
                message=f"Disconnected: {connection_id}"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to disconnect: {e}")
            return CommandResult.error_result(
                message=f"Failed to disconnect: {e}",
                errors=[str(e)]
            )
    
    def disconnect_by_port(self, block_identifier: str, port_name: str) -> CommandResult:
        """
        Disconnect a specific input port on a block.
        
        Args:
            block_identifier: Block name or ID
            port_name: Input port name to disconnect
            
        Returns:
            CommandResult
        """
        if not self.current_project_id:
            return CommandResult.error_result("No project loaded")
        
        try:
            # Resolve block identifier
            block = self._resolve_block_identifier(block_identifier)
            if not block:
                return CommandResult.error_result(f"Block '{block_identifier}' not found")
            
            # Get connections before disconnecting to know which blocks are affected
            connections = self.connection_service._connection_repo.list_by_target(block.id, port_name)
            source_block_ids = {conn.source_block_id for conn in connections}
            
            # Disconnect
            self.connection_service.disconnect_by_target(block.id, port_name)
            
            # Recalculate expected outputs for affected blocks
            self._recalculate_expected_outputs(block.id)
            for source_block_id in source_block_ids:
                self._recalculate_expected_outputs(source_block_id)
            
            return CommandResult.success_result(
                message=f"Disconnected {block.name}.{port_name}"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to disconnect port: {e}")
            return CommandResult.error_result(
                message=f"Failed to disconnect: {e}",
                errors=[str(e)]
            )
    
    def _recalculate_expected_outputs(self, block_id: str) -> None:
        """
        Recalculate and update expected outputs for a block.
        
        Called when connections change, as blocks may output based on their inputs.
        
        Args:
            block_id: Block ID to recalculate expected outputs for
        """
        if not self.current_project_id or not self.expected_outputs_service:
            return
        
        try:
            block = self.block_service.get_block(self.current_project_id, block_id)
            if not block:
                return
            
            processor = self.execution_engine.get_processor(block)
            if not processor:
                return
            
            expected_outputs = self.expected_outputs_service.calculate_expected_outputs(
                block,
                processor,
                facade=self
            )
            
            block.metadata['expected_outputs'] = expected_outputs
            self.block_service.update_block(self.current_project_id, block_id, block)
            
            Log.debug(f"ApplicationFacade: Recalculated expected_outputs for block '{block.name}': {expected_outputs}")
        except Exception as e:
            # Non-critical - don't fail connection operations if expected outputs update fails
            Log.debug(f"ApplicationFacade: Failed to recalculate expected_outputs for {block_id}: {e}")
    
    def disconnect_all_from_block(self, block_identifier: str) -> CommandResult:
        """
        Disconnect all connections to/from a block.
        
        Args:
            block_identifier: Block name or ID
            
        Returns:
            CommandResult
        """
        if not self.current_project_id:
            return CommandResult.error_result("No project loaded")
        
        try:
            # Resolve block identifier
            block = self._resolve_block_identifier(block_identifier)
            if not block:
                return CommandResult.error_result(f"Block '{block_identifier}' not found")
            
            # Get all connections for this block
            connections = self.connection_service.list_connections_by_block(block.id)
            
            if not connections:
                return CommandResult.success_result(
                    message=f"No connections found for block '{block.name}'"
                )
            
            # Disconnect all
            disconnected_count = 0
            for conn in connections:
                try:
                    self.connection_service.disconnect_blocks(conn.id)
                    disconnected_count += 1
                except Exception as e:
                    Log.warning(f"Failed to disconnect {conn.id}: {e}")
            
            return CommandResult.success_result(
                message=f"Disconnected {disconnected_count} connection(s) from block '{block.name}'"
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to disconnect all: {e}")
            return CommandResult.error_result(
                message=f"Failed to disconnect all: {e}",
                errors=[str(e)]
            )
    
    def list_connections(self) -> CommandResult[List[Connection]]:
        """
        List all connections in current project.
        
        Returns:
            CommandResult[List[Connection]] containing all connections in the current project
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            connections = self.connection_service.list_connections_by_project(
                self.current_project_id
            )
            return CommandResult.success_result(
                message=f"Found {len(connections)} connection(s)",
                data=connections
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to list connections: {e}")
            return CommandResult.error_result(
                message=f"Failed to list connections: {e}",
                errors=[str(e)]
            )
    
    # ==================== Execution Operations ====================
    
    def execute_block_by_name(self, block_identifier: str) -> CommandResult:
        """
        Execute a single block by ID or name.
        
        DEPRECATED: Use execute_block() instead.
        This method is kept for backwards compatibility.
        
        Args:
            block_identifier: Block ID or name
            
        Returns:
            CommandResult with block outputs
        """
        return self.execute_block(block_identifier)
    
    def execute_block(
        self,
        block_identifier: str,
        progress_tracker_override: Optional[Any] = None,
    ) -> CommandResult:
        """
        Execute a single block by ID or name.
        
        This is the API layer - validates requests and delegates to ExecutionEngine.
        The ExecutionEngine handles all execution logic including pulling data,
        gathering inputs, executing the processor, and saving outputs.
        
        Args:
            block_identifier: Block ID or name
            progress_tracker_override: Optional progress tracker (e.g. for subprocess
                runners that stream progress to stdout). If None, uses event_bus tracker.
            
        Returns:
            CommandResult with block outputs
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        if not self.execution_engine:
            return CommandResult.error_result(
                message="Execution engine not available"
            )
        
        try:
            # Find the block by ID first, then by name
            block = self.block_service.get_block(self.current_project_id, block_identifier)
            if not block:
                # Try finding by name
                block = self.block_service.find_by_name(self.current_project_id, block_identifier)
            
            if not block:
                return CommandResult.error_result(
                    message=f"Block not found: '{block_identifier}'",
                    errors=[
                        f"No block with ID or name '{block_identifier}' found in the current project",
                        "Use 'listblocks' to see available blocks"
                    ]
                )
            
            # Reload block from database to ensure we have latest metadata (settings may have changed)
            describe_result = self.describe_block(block.id)
            if describe_result.success and describe_result.data:
                block = describe_result.data
            
            # Unified execution: honor use_subprocess_runner for both UI Run and setlist
            if self._use_subprocess_runner():
                return self._execute_block_via_subprocess(block.id, block.name)
            
            Log.info(f"ApplicationFacade: Executing block '{block.name}' (ID: {block.id})")
            
            # Prepare metadata for execution - STANDARDIZED for all block types
            from src.application.block_registry import get_block_registry
            registry = get_block_registry()
            execution_mode = registry.get_execution_mode(block.type)
            block_repo = getattr(self.execution_engine, '_block_repo', None)
            
            # Create progress tracker for block-level progress reporting
            # Override used by subprocess runner (run_block CLI) to stream progress to stdout
            if progress_tracker_override is not None:
                progress_tracker = progress_tracker_override
            else:
                Log.debug(f"ApplicationFacade: Creating progress_tracker with event_bus id: {id(self.event_bus) if self.event_bus else 'None'}")
                progress_tracker = create_progress_tracker(
                    block=block,
                    project_id=self.current_project_id,
                    event_bus=self.event_bus
                )
            
            # Create UI state service adapter for processors that need UI state access
            ui_state_service = None
            if self.ui_state_repo:
                class UIStateServiceAdapter:
                    """Simple adapter to provide UI state access to processors."""
                    def __init__(self, facade):
                        self._facade = facade
                    
                    def get_state(self, state_type: str, entity_id: str):
                        result = self._facade.get_ui_state(state_type, entity_id)
                        return result.data if result.success else None
                    
                    def set_state(self, state_type: str, entity_id: str, data: dict):
                        self._facade.set_ui_state(state_type, entity_id, data)
                
                ui_state_service = UIStateServiceAdapter(self)
            
            metadata = {
                "project_id": self.current_project_id,
                "data_item_repo": self.data_item_repo,
                "ui_state_service": ui_state_service,
                "execution_mode": execution_mode,
                "progress_tracker": progress_tracker
            }
            if block_repo:
                metadata["block_repo"] = block_repo
            
            # Delegate to ExecutionEngine - it handles everything:
            # 1. Pull data from upstream
            # 2. Gather inputs from local state
            # 3. Execute processor
            # 4. Save outputs to database
            outputs = self.execution_engine.execute_block(
                block=block,
                inputs=None,  # Let engine gather from local state
                metadata=metadata,
                auto_pull=True,  # Pull upstream data
                auto_save=True   # Save outputs to database
            )
            
            Log.info(f"ApplicationFacade: Successfully executed block '{block.name}'")
            
            return CommandResult.success_result(
                message=f"Successfully executed block '{block.name}'",
                data={
                    'block': block,
                    'outputs': outputs
                }
            )
        except FilterError as e:
            Log.error(f"ApplicationFacade: Filter error in block: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            
            # Include filter error details for UI
            error_details = {
                "error_type": "FilterError",
                "block_id": e.block_id,
                "block_name": e.block_name,
                "port_name": e.port_name,
                "remedy_action": e.remedy_action,
                "available_items": e.available_items,
                "selected_ids": e.selected_ids
            }
            
            result = CommandResult.error_result(
                message=f"Filter error: {e.message}",
                errors=[str(e), error_traceback]
            )
            result.data = error_details
            
            return result
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to execute block: {e}")
            import traceback
            return CommandResult.error_result(
                message=f"Failed to execute block: {e}",
                errors=[str(e), traceback.format_exc()]
            )
    
    # Backwards compatibility alias
    def execute_single_block(self, block_identifier: str) -> CommandResult:
        """
        DEPRECATED: Use execute_block() instead.
        
        This method is kept for backwards compatibility and will be removed in a future version.
        """
        return self.execute_block(block_identifier)
    
    def _use_subprocess_runner(self) -> bool:
        """
        Check if block execution should run in a separate process.
        
        Returns False when we are already inside run_block_cli (parent sets
        ECHOZERO_INSIDE_RUN_BLOCK_CLI when spawning) to avoid recursive subprocess spawning.
        """
        if os.environ.get("ECHOZERO_INSIDE_RUN_BLOCK_CLI") == "1":
            return False
        if getattr(self, "app_settings", None) is not None:
            if getattr(self.app_settings, "use_subprocess_runner", False):
                return True
        return os.environ.get("ECHOZERO_USE_SUBPROCESS_RUNNER") == "1"
    
    def _execute_block_via_subprocess(
        self,
        block_id: str,
        block_name: str,
    ) -> CommandResult:
        """
        Execute block in a separate process (run_block_cli).
        
        Blocks until the subprocess completes. Streams progress from stdout
        and publishes SubprocessProgress events for UI updates.
        Used by both MainWindow (via RunBlockThread) and SetlistProcessingService.
        """
        from src.utils.paths import get_database_path
        
        project_id = self.current_project_id
        if not project_id:
            return CommandResult.error_result(message="No project loaded")
        
        db_path = str(get_database_path("ez"))
        exe = sys.executable
        if getattr(sys, "frozen", False):
            args = ["--run-block-cli", "--db", db_path, "--project", project_id, "--block", block_id]
        else:
            args = [
                "-m", "src.features.execution.run_block_cli",
                "--db", db_path, "--project", project_id, "--block", block_id
            ]
        
        Log.info(f"ApplicationFacade: Executing block '{block_name}' via subprocess")
        
        env = os.environ.copy()
        env["ECHOZERO_INSIDE_RUN_BLOCK_CLI"] = "1"
        
        try:
            proc = subprocess.Popen(
                [exe] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to start subprocess: {e}")
            return CommandResult.error_result(
                message=f"Failed to start execution subprocess: {e}",
                errors=[str(e)]
            )
        
        result_line = None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                typ = obj.get("type")
                if typ == "progress":
                    if self.event_bus:
                        msg = obj.get("message", "")
                        pct = obj.get("percentage", 0)
                        self.event_bus.publish(SubprocessProgress(
                            project_id=project_id,
                            data={
                                "block_name": block_name,
                                "message": msg,
                                "percentage": pct,
                            }
                        ))
                elif typ in ("result", "error"):
                    result_line = obj
            except (json.JSONDecodeError, TypeError):
                pass
        
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        
        if result_line and result_line.get("type") == "result":
            success = result_line.get("success", False)
            msg = result_line.get("message", "")
            errs = list(result_line.get("errors") or [])
            if stderr:
                errs.append(f"Process stderr:\n{stderr}")
            if success:
                return CommandResult.success_result(
                    message=msg or f"Successfully executed block '{block_name}'",
                    data={"block_id": block_id}
                )
            return CommandResult.error_result(
                message=msg or "Execution failed",
                errors=errs
            )
        
        if result_line and result_line.get("type") == "error":
            errs = [result_line.get("traceback", result_line.get("message", ""))]
            if stderr:
                errs.append(f"Process stderr:\n{stderr}")
            return CommandResult.error_result(
                message=result_line.get("message", "Execution error"),
                errors=errs
            )
        
        if proc.returncode != 0:
            return CommandResult.error_result(
                message=f"Subprocess exited with code {proc.returncode}",
                errors=[stderr] if stderr else []
            )
        
        return CommandResult.error_result(
            message="Subprocess did not return a result",
            errors=[stderr] if stderr else []
        )
    
    def _load_inputs_for_block(self, block: Block) -> Dict[str, Any]:
        """
        Load input data items for a block from previous execution data.
        
        Applies filter_selections if they exist in block.metadata.
        
        Args:
            block: Block to load inputs for
            
        Returns:
            Dictionary mapping input port names to DataItem instances
        """
        inputs: Dict[str, Any] = {}
        
        if not self.data_item_repo:
            return inputs
        
        # Get all connections to this block
        connections = self.connection_service.list_connections_by_block(block.id)
        
        # Filter for incoming connections
        incoming_connections = [
            conn for conn in connections 
            if conn.target_block_id == block.id
        ]
        
        Log.info(f"ApplicationFacade: Found {len(incoming_connections)} incoming connection(s)")
        
        # Get filter_selections from block metadata
        filter_selections = block.metadata.get("filter_selections", {})
        
        # Group connections by target_input_name to handle multiple connections per port
        connections_by_port: Dict[str, List[Connection]] = defaultdict(list)
        for conn in incoming_connections:
            connections_by_port[conn.target_input_name].append(conn)
        
        # Process each input port, collecting data from all connections
        for port_name, port_connections in connections_by_port.items():
            all_matching_items = []
            all_available_items = []  # Track all available items before filtering
            
            for conn in port_connections:
                # Load data items from source block
                source_data_items = self.data_item_repo.list_by_block(conn.source_block_id)
                
                # Filter for items that match the source output port
                matching_items = [
                    item for item in source_data_items
                    if item.metadata.get('output_port') == conn.source_output_name
                ]
                
                # Track available items before filtering
                all_available_items.extend(matching_items)
                
                # Apply filter using DataFilterManager (single application point)
                if self.data_filter_manager and matching_items:
                    try:
                        matching_items = self.data_filter_manager.apply_filter(
                            block, port_name, matching_items
                        )
                    except ValueError as e:
                        Log.error(f"ApplicationFacade: Invalid filter: {e}")
                    except Exception as e:
                        Log.warning(f"ApplicationFacade: Failed to apply filter: {e}")
                
                if matching_items:
                    # Add items from this connection to the collection
                    all_matching_items.extend(matching_items)
                    Log.debug(
                        f"ApplicationFacade: Found {len(matching_items)} data item(s) "
                        f"for input '{port_name}' from connection '{conn.source_output_name}' -> '{port_name}'"
                    )
                else:
                    Log.warning(
                        f"ApplicationFacade: No data items found for connection "
                        f"{conn.source_output_name} -> {port_name}"
                    )
            
            # Check if filter_selections filtered out all items
            port_filter = filter_selections.get(port_name)
            
            # If filter exists and all items are disabled, don't attempt reconciliation
            if (port_filter and 
                isinstance(port_filter, dict) and
                sum(1 for v in port_filter.values() if v) == 0):
                # All items disabled - this is intentional, don't reconcile
                pass
            elif (port_filter and 
                  len(all_available_items) > 0 and len(all_matching_items) == 0):
                # Filter was applied and filtered out all items - attempt reconciliation
                Log.info(
                    f"ApplicationFacade: Filter selections for {block.name}.{port_name} are stale "
                    f"(no items match filter). Attempting reconciliation..."
                )
                
                # Attempt reconciliation: try to match items by name/type from available items
                reconciled_ids = []
                try:
                    # Try to find original items by ID to get their names/types for matching
                    original_items = []
                    for selected_id in selected_ids:
                        try:
                            original_item = self.data_item_repo.get(selected_id)
                            if original_item:
                                original_items.append(original_item)
                        except Exception:
                            continue
                    
                    # Try to match original items to available items by name and type
                    for original_item in original_items:
                        matching = [
                            item for item in all_available_items
                            if item.name == original_item.name 
                            and item.type == original_item.type
                        ]
                        if matching:
                            # Found exact match - use it
                            reconciled_ids.append(matching[0].id)
                        elif len(all_available_items) > 0:
                            # No exact match, but we have available items - use first item of same type
                            same_type = [
                                item for item in all_available_items
                                if item.type == original_item.type
                            ]
                            if same_type and same_type[0].id not in reconciled_ids:
                                reconciled_ids.append(same_type[0].id)
                    
                    # If we couldn't match any, use all available items as fallback
                    if not reconciled_ids and all_available_items:
                        reconciled_ids = [item.id for item in all_available_items[:len(selected_ids)]]
                    
                except Exception as e:
                    Log.warning(
                        f"ApplicationFacade: Filter reconciliation failed for {block.name}.{port_name}: {e}"
                    )
                
                # If reconciliation found matches, update filter_selections and use reconciled items
                if reconciled_ids:
                    # Update filter_selections with reconciled IDs
                    if "filter_selections" not in block.metadata:
                        block.metadata["filter_selections"] = {}
                    block.metadata["filter_selections"][port_name] = reconciled_ids
                    
                    # Update block in repository to persist reconciled filters
                    try:
                        self.block_service.update_block(self.current_project_id, block.id, block)
                        Log.info(
                            f"ApplicationFacade: Reconciled filter selections for {block.name}.{port_name}: "
                            f"{len(selected_ids)} stale IDs -> {len(reconciled_ids)} reconciled IDs"
                        )
                    except Exception as e:
                        Log.warning(
                            f"ApplicationFacade: Failed to save reconciled filters for {block.name}: {e}"
                        )
                    
                    # Use reconciled items
                    reconciled_set = set(reconciled_ids)
                    all_matching_items = [
                        item for item in all_available_items
                        if item.id in reconciled_set
                    ]
                    Log.info(
                        f"ApplicationFacade: Using {len(all_matching_items)} reconciled item(s) "
                        f"for {block.name}.{port_name}"
                    )
                else:
                    # Reconciliation failed - no items found, raise FilterError
                    available_item_ids = [item.id for item in all_available_items]
                    raise FilterError(
                        message=(
                            f"Filter for '{port_name}' filtered out all available items and reconciliation failed. "
                            f"Available items: {len(all_available_items)}, Selected IDs: {len(selected_ids)}"
                        ),
                        block_id=block.id,
                        block_name=block.name,
                        port_name=port_name,
                        available_items=available_item_ids,
                        selected_ids=selected_ids
                    )
            
            # Store collected items for this port (single item or list)
            if all_matching_items:
                if len(all_matching_items) > 1:
                    inputs[port_name] = all_matching_items
                    Log.info(
                        f"ApplicationFacade: Loaded {len(all_matching_items)} data items "
                        f"for input '{port_name}' from {len(port_connections)} connection(s)"
                    )
                else:
                    inputs[port_name] = all_matching_items[0]
                    Log.info(
                        f"ApplicationFacade: Loaded data item '{all_matching_items[0].name}' "
                        f"for input '{port_name}' from {len(port_connections)} connection(s)"
                    )
        
        return inputs

    # ==================== Block Local Inputs (MVP) ====================

    def clear_block_local_inputs(self, block_id: str) -> CommandResult:
        """Clear persisted local input references for a block."""
        if not self.block_local_state_repo:
            return CommandResult.error_result(message="Block local state repository not available")
        try:
            self.block_local_state_repo.clear_inputs(block_id)
            return CommandResult.success_result(message="Cleared block local inputs", data=True)
        except Exception as e:
            return CommandResult.error_result(message=f"Failed to clear local inputs: {e}", errors=[str(e)])

    def get_block_local_inputs(self, block_id: str) -> CommandResult:
        """Get persisted local input references for a block (port -> id(s))."""
        if not self.block_local_state_repo:
            return CommandResult.error_result(message="Block local state repository not available")
        try:
            mapping = self.block_local_state_repo.get_inputs(block_id) or {}
            return CommandResult.success_result(message="Loaded block local inputs", data=mapping)
        except Exception as e:
            return CommandResult.error_result(message=f"Failed to load local inputs: {e}", errors=[str(e)])

    def touch_block_local_state(self, block_id: str, reason: str = "touch") -> CommandResult:
        """
        Update (touch) a block's local state entry.

        Primary use: UI edits that mutate underlying DataItems should also update local state
        so downstream pulls see the latest routing references (and updated_at advances).

        Current behavior (minimal, deterministic):
        - Preserve existing local mapping, if any
        - Ensure `events` points at this block's owned EventDataItems (metadata.output_port == 'events')
        - Ensure `audio` is present if there's an incoming audio connection
        """
        if not self.block_local_state_repo or not self.data_item_repo:
            return CommandResult.error_result(message="Required repositories not available")
        try:
            current = self.block_local_state_repo.get_inputs(block_id) or {}

            # Re-point events at current owned outputs for this block
            try:
                from src.shared.domain.entities import EventDataItem
                owned_items = self.data_item_repo.list_by_block(block_id)
                event_ids = [
                    item.id for item in owned_items
                    if isinstance(item, EventDataItem) and item.metadata.get("output_port") == "events"
                ]
                if event_ids:
                    current["events"] = event_ids if len(event_ids) > 1 else event_ids[0]
            except Exception:
                pass

            # Ensure audio is present if there's an incoming audio connection
            if "audio" not in current:
                try:
                    connections = self.connection_service.list_connections_by_block(block_id)
                    incoming_audio = [
                        c for c in connections
                        if c.target_block_id == block_id and c.target_input_name == "audio"
                    ]
                    if incoming_audio:
                        conn = incoming_audio[0]
                        # Prefer upstream local state
                        src_local = self.block_local_state_repo.get_inputs(conn.source_block_id) or {}
                        audio_ref = src_local.get(conn.source_output_name)
                        if not audio_ref:
                            # Fall back to persisted outputs
                            source_items = self.data_item_repo.list_by_block(conn.source_block_id)
                            matching = [
                                item for item in source_items
                                if item.metadata.get("output_port") == conn.source_output_name
                            ]
                            if len(matching) == 1:
                                audio_ref = matching[0].id
                            elif len(matching) > 1:
                                audio_ref = [i.id for i in matching]
                        if audio_ref:
                            current["audio"] = audio_ref
                except Exception:
                    pass

            self.block_local_state_repo.set_inputs(block_id, current)
            return CommandResult.success_result(message=f"Touched block local state ({reason})", data=current)
        except Exception as e:
            return CommandResult.error_result(message=f"Failed to touch local state: {e}", errors=[str(e)])

    def pull_block_inputs_overwrite(self, block_id: str) -> CommandResult:
        """
        Pull inputs from incoming connections into local state (overwrite).
        
        This is part of the execution flow:
        1. Clears existing local state (inputs)
        2. Pulls fresh data from upstream connections
        3. Stores references in local state: {input_port: data_item_id | [data_item_id]}
        
        Note: This method clears inputs before pulling to ensure fresh data.
        """
        if not self.block_local_state_repo or not self.data_item_repo:
            return CommandResult.error_result(message="Required repositories not available")
        
        try:
            # Get block for logging filter selections
            block = None
            if self.current_project_id:
                try:
                    block = self.block_service.get_block(self.current_project_id, block_id)
                except Exception:
                    pass
            
            if block:
                Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - starting for block '{block.name}' (ID: {block_id})")
                
                # Get filter selections for logging
                filter_selections = block.metadata.get("filter_selections", {})
                if filter_selections:
                    Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - filter_selections: {filter_selections}")
                    for port_name, port_filter in filter_selections.items():
                        if isinstance(port_filter, dict):
                            enabled = [k for k, v in port_filter.items() if v]
                            disabled = [k for k, v in port_filter.items() if not v]
                            Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - {port_name}: enabled={enabled}, disabled={disabled}")
                else:
                    Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - no filter_selections set")
            else:
                Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - starting for block ID: {block_id} (block details not available for logging)")

            # Wipe existing local state first
            self.block_local_state_repo.clear_inputs(block_id)

            # Resolve incoming connections
            connections = self.connection_service.list_connections_by_block(block_id)
            incoming = [c for c in connections if c.target_block_id == block_id]

            inputs_map: Dict[str, Any] = {}
            missing_upstream: List[Dict[str, str]] = []
            
            # Get block early to access filter_selections and port information
            block = None
            try:
                block = self.block_service.get_block(self.current_project_id, block_id) if self.current_project_id else None
            except Exception:
                block = None

            # Filter to ONLY process connections where target port is an INPUT port
            # Bidirectional ports (like manipulator) should not be processed here
            # as they don't follow the input/output data flow pattern
            input_connections = []
            if block:
                input_ports = block.get_inputs()
                for conn in incoming:
                    # Only include connections where target port is an actual input port
                    if conn.target_input_name in input_ports:
                        input_connections.append(conn)
                    else:
                        # Log that we're skipping bidirectional ports
                        Log.debug(
                            f"ApplicationFacade: Skipping connection to '{conn.target_input_name}' "
                            f"(not an input port, likely bidirectional)"
                        )
            else:
                # If we can't get the block, fall back to processing all connections
                # but log a warning
                Log.warning(f"ApplicationFacade: Could not get block {block_id}, processing all connections")
                input_connections = incoming

            # If this block has no incoming input connections, treat it as a source:
            # initialize its local data from any persisted DataItems by output_port.
            # (Ports are routing labels; local state is the single source of truth.)
            if not input_connections:

                if block and getattr(block, "outputs", None):
                    try:
                        by_port: Dict[str, list] = {}
                        for item in self.data_item_repo.list_by_block(block_id):
                            port = item.metadata.get("output_port")
                            if not port:
                                continue
                            by_port.setdefault(port, []).append(item.id)

                        for port_name in block.get_outputs().keys():
                            ids = by_port.get(port_name, [])
                            if not ids:
                                continue
                            inputs_map[port_name] = ids[0] if len(ids) == 1 else ids

                        self.block_local_state_repo.set_inputs(block_id, inputs_map)

                        return CommandResult.success_result(
                            message=f"Initialized local data for {len(inputs_map)} output port(s)",
                            data=inputs_map
                        )
                    except Exception as e:
                        return CommandResult.error_result(message=f"Failed to initialize source local data: {e}", errors=[str(e)])
            
            # Group connections by target_input_name to handle multiple connections per port
            connections_by_port: Dict[str, List[Connection]] = defaultdict(list)
            for conn in input_connections:
                connections_by_port[conn.target_input_name].append(conn)
            
            # Process each input port, collecting data from all connections
            for port_name, port_connections in connections_by_port.items():
                # OPTIMIZATION: Check if filter would exclude all items for this port
                # Skip processing entirely if all items would be filtered out
                if block and self.data_filter_manager:
                    if self.data_filter_manager.is_filter_all_disabled(block, port_name):
                        Log.info(
                            f"ApplicationFacade: pull_block_inputs_overwrite - SKIPPING port {block.name}.{port_name} "
                            f"(filter excludes all items, {len(port_connections)} connection(s) not processed)"
                        )
                        continue
                
                all_item_ids = []
                
                for conn in port_connections:
                    # Gather diagnostic information for this connection
                    connection_diagnostics = {
                        "connection": {
                            "source_block_id": conn.source_block_id,
                            "source_output_name": conn.source_output_name,
                            "target_block_id": conn.target_block_id,
                            "target_input_name": conn.target_input_name
                        },
                        "upstream_block": None,
                        "upstream_local_state": None,
                        "upstream_data_items": None,
                        "pull_outputs_result": None,
                        "filter_applied": False
                    }
                    
                    # Get upstream block info for diagnostics
                    try:
                        upstream_block = self.block_service.get_block(self.current_project_id, conn.source_block_id) if self.current_project_id else None
                        if upstream_block:
                            connection_diagnostics["upstream_block"] = {
                                "id": upstream_block.id,
                                "name": upstream_block.name,
                                "type": upstream_block.type,
                                "has_outputs": bool(upstream_block.get_outputs())
                            }
                    except Exception as e:
                        connection_diagnostics["upstream_block"] = {"error": str(e)}
                    
                    # Single source of truth:
                    # upstream "output port" == upstream local data under that port name.
                    local_ref = None
                    try:
                        src_local = self.block_local_state_repo.get_inputs(conn.source_block_id) or {}
                        connection_diagnostics["upstream_local_state"] = {
                            "available_ports": list(src_local.keys()),
                            "requested_port": conn.source_output_name,
                            "has_requested_port": conn.source_output_name in src_local
                        }
                        local_ref = src_local.get(conn.source_output_name)
                        if local_ref:
                            connection_diagnostics["upstream_local_state"]["port_value"] = (
                                local_ref if isinstance(local_ref, str) else f"list({len(local_ref)} items)"
                            )
                    except Exception as e:
                        connection_diagnostics["upstream_local_state"] = {"error": str(e)}

                    if local_ref:
                        # Apply filter if filter_selections exist for this port
                        Log.debug(
                            f"ApplicationFacade: pull_block_inputs_overwrite - applying filter to {block.name}.{port_name} "
                            f"from connection {conn.source_block_id}.{conn.source_output_name}, "
                            f"local_ref type: {type(local_ref).__name__}"
                        )
                        filtered_ref = self._apply_filter_to_pull_result(
                            block, port_name, local_ref
                        )
                        connection_diagnostics["filter_applied"] = True

                        Log.debug(
                            f"ApplicationFacade: pull_block_inputs_overwrite - filter result for {block.name}.{port_name}: "
                            f"{'None (all filtered out)' if filtered_ref is None else f'{len(filtered_ref) if isinstance(filtered_ref, list) else 1} item(s)'}"
                        )
                        if filtered_ref is not None:
                            # Add to collection (handle both single ID and list)
                            if isinstance(filtered_ref, list):
                                all_item_ids.extend(filtered_ref)
                            else:
                                all_item_ids.append(filtered_ref)
                        else:
                            # Filter filtered out all items - skip this port gracefully
                            connection_diagnostics["filter_result"] = "all_items_filtered_out"
                            Log.debug(
                                f"ApplicationFacade: Filter for {block.name if block else block_id}.{port_name} "
                                f"filtered out all items from connection {conn.source_block_id}.{conn.source_output_name} - skipping"
                            )
                        continue

                    # Use pull_block_outputs to get upstream outputs (handles all execution modes)
                    upstream_result = self.pull_block_outputs(conn.source_block_id, conn.source_output_name)
                    connection_diagnostics["pull_outputs_result"] = {
                        "success": upstream_result.success,
                        "message": upstream_result.message if hasattr(upstream_result, "message") else None,
                        "has_data": bool(upstream_result.data if upstream_result.success else None),
                        "data_keys": list(upstream_result.data.keys()) if (upstream_result.success and upstream_result.data) else []
                    }
                    
                    # Check what data items exist for upstream block
                    try:
                        upstream_data_items = self.data_item_repo.list_by_block(conn.source_block_id)
                        connection_diagnostics["upstream_data_items"] = {
                            "total_count": len(upstream_data_items),
                            "by_port": {}
                        }
                        for item in upstream_data_items:
                            port = item.metadata.get("output_port", "unknown")
                            if port not in connection_diagnostics["upstream_data_items"]["by_port"]:
                                connection_diagnostics["upstream_data_items"]["by_port"][port] = []
                            connection_diagnostics["upstream_data_items"]["by_port"][port].append({
                                "id": item.id,
                                "name": item.name,
                                "type": item.type
                            })
                    except Exception as e:
                        connection_diagnostics["upstream_data_items"] = {"error": str(e)}
                    
                    if upstream_result.success and upstream_result.data:
                        upstream_output = upstream_result.data.get(conn.source_output_name)
                        if upstream_output:
                            # Apply filter if filter_selections exist for this port
                            if isinstance(upstream_output, list):
                                item_ids = [item.id for item in upstream_output if item]
                            else:
                                item_ids = upstream_output.id if upstream_output else None
                            
                            if item_ids:
                                Log.info(
                                    f"ApplicationFacade: pull_block_inputs_overwrite - applying filter to {block.name}.{port_name} "
                                    f"from pull_block_outputs result ({conn.source_block_id}.{conn.source_output_name}), "
                                    f"item_ids type: {type(item_ids).__name__}, "
                                    f"count: {len(item_ids) if isinstance(item_ids, list) else 1}"
                                )
                                filtered_ids = self._apply_filter_to_pull_result(
                                    block, port_name, item_ids
                                )
                                connection_diagnostics["filter_applied"] = True
                                Log.info(
                                    f"ApplicationFacade: pull_block_inputs_overwrite - filter result for {block.name}.{port_name} "
                                    f"from pull_block_outputs: "
                                    f"{'None (all filtered out)' if filtered_ids is None else f'{len(filtered_ids) if isinstance(filtered_ids, list) else 1} item(s)'}"
                                )
                                if filtered_ids is not None:
                                    # Add to collection (handle both single ID and list)
                                    if isinstance(filtered_ids, list):
                                        all_item_ids.extend(filtered_ids)
                                    else:
                                        all_item_ids.append(filtered_ids)
                                    continue
                                else:
                                    # Filter filtered out all items - skip this port gracefully
                                    connection_diagnostics["filter_result"] = "all_items_filtered_out"
                                    Log.debug(
                                        f"ApplicationFacade: Filter for {block.name if block else block_id}.{port_name} "
                                        f"filtered out all items from connection {conn.source_block_id}.{conn.source_output_name} - skipping"
                                    )
                                    continue
                            else:
                                # Upstream output is empty
                                missing_upstream.append({
                                    "source_block_id": conn.source_block_id,
                                    "source_output_name": conn.source_output_name,
                                    "target_input_name": port_name,
                                    "diagnostics": connection_diagnostics,
                                    "reason": "upstream_output_empty"
                                })
                        else:
                            # Port not in upstream data
                            missing_upstream.append({
                                "source_block_id": conn.source_block_id,
                                "source_output_name": conn.source_output_name,
                                "target_input_name": port_name,
                                "diagnostics": connection_diagnostics,
                                "reason": "port_not_in_upstream_data"
                            })
                    else:
                        # pull_block_outputs failed or returned no data
                        missing_upstream.append({
                            "source_block_id": conn.source_block_id,
                            "source_output_name": conn.source_output_name,
                            "target_input_name": port_name,
                            "diagnostics": connection_diagnostics,
                            "reason": "pull_outputs_failed" if not upstream_result.success else "no_data_returned"
                        })
                
                # Store collected item IDs for this port (single item or list)
                if all_item_ids:
                    # DEBUG: Log what we're about to store
                    Log.info(
                        f"ApplicationFacade: pull_block_inputs_overwrite - Storing {len(all_item_ids)} item(s) for port '{port_name}': "
                        f"{all_item_ids[:3]}{'...' if len(all_item_ids) > 3 else ''}"
                    )
                    # Check if any of these are sync layer items (should not happen)
                    for item_id in all_item_ids:
                        item = self.data_item_repo.get(item_id) if self.data_item_repo else None
                        if item:
                            is_sync = item.metadata.get("_synced_from_ma3") is True or (item.name and "ma3_sync" in item.name.lower())
                            if is_sync:
                                Log.error(
                                    f"ApplicationFacade: pull_block_inputs_overwrite - ERROR: Sync layer item '{item.name}' "
                                    f"(ID: {item_id[:8]}...) found in upstream pull for port '{port_name}'! "
                                    f"This should not happen - sync layers should not be in upstream blocks' outputs."
                                )
                    if len(all_item_ids) == 1:
                        inputs_map[port_name] = all_item_ids[0]
                    else:
                        inputs_map[port_name] = all_item_ids

            # Validate types before setting local state
            try:
                if block:
                    from src.application.processing.type_validation import validate_on_pull
                    from src.application.block_registry import get_block_registry
                    
                    block_metadata = get_block_registry().get(block.type)
                    if block_metadata:
                        # Build connection_map for validation (only first connection per port for backward compatibility)
                        # Note: Validation doesn't need all connections, just needs to verify types
                        connection_map = {}
                        for conn in incoming:
                            if conn.target_input_name not in connection_map:
                                connection_map[conn.target_input_name] = (conn.source_block_id, conn.source_output_name)
                        validation_errors = validate_on_pull(
                            block_id=block_id,
                            block_type=block.type,
                            connection_map=connection_map,
                            expected_inputs=block_metadata.inputs,
                            data_item_repo=self.data_item_repo,
                            block_local_state_repo=self.block_local_state_repo
                        )
                        
                        if validation_errors:
                            return CommandResult.error_result(
                                message="Type validation failed",
                                errors=validation_errors
                            )
            except Exception:
                # If validation fails, continue (non-blocking)
                pass

            # If we have missing upstream data, return error with detailed diagnostics
            if missing_upstream:
                # Build detailed diagnostic message
                diagnostic_lines = [
                    f"Block '{block.name if block else block_id}' (ID: {block_id}) failed to pull data from upstream:",
                    ""
                ]
                
                # Group by reason for better organization
                by_reason = {}
                for missing in missing_upstream:
                    reason = missing.get("reason", "unknown")
                    if reason not in by_reason:
                        by_reason[reason] = []
                    by_reason[reason].append(missing)
                
                for reason, items in by_reason.items():
                    diagnostic_lines.append(f"  Reason: {reason} ({len(items)} connection(s))")
                    for missing in items:
                        diag = missing.get("diagnostics", {})
                        conn_info = diag.get("connection", {})
                        upstream_info = diag.get("upstream_block", {})
                        local_state_info = diag.get("upstream_local_state", {})
                        data_items_info = diag.get("upstream_data_items", {})
                        pull_result_info = diag.get("pull_outputs_result", {})
                        
                        diagnostic_lines.append(f"    Connection: {conn_info.get('source_block_id', 'unknown')}.{conn_info.get('source_output_name', 'unknown')} -> {conn_info.get('target_block_id', 'unknown')}.{conn_info.get('target_input_name', 'unknown')}")
                        
                        if upstream_info:
                            if "error" in upstream_info:
                                diagnostic_lines.append(f"      Upstream Block: ERROR - {upstream_info['error']}")
                            else:
                                diagnostic_lines.append(f"      Upstream Block: '{upstream_info.get('name', 'unknown')}' (ID: {upstream_info.get('id', 'unknown')}, Type: {upstream_info.get('type', 'unknown')})")
                                diagnostic_lines.append(f"        Has outputs defined: {upstream_info.get('has_outputs', False)}")
                        
                        if local_state_info:
                            if "error" in local_state_info:
                                diagnostic_lines.append(f"      Upstream Local State: ERROR - {local_state_info['error']}")
                            else:
                                diagnostic_lines.append(f"      Upstream Local State: Available ports: {local_state_info.get('available_ports', [])}")
                                diagnostic_lines.append(f"        Requested port '{local_state_info.get('requested_port', 'unknown')}': {'FOUND' if local_state_info.get('has_requested_port', False) else 'NOT FOUND'}")
                                if local_state_info.get('has_requested_port'):
                                    diagnostic_lines.append(f"        Port value: {local_state_info.get('port_value', 'unknown')}")
                        
                        if data_items_info:
                            if "error" in data_items_info:
                                diagnostic_lines.append(f"      Upstream Data Items: ERROR - {data_items_info['error']}")
                            else:
                                diagnostic_lines.append(f"      Upstream Data Items: Total count: {data_items_info.get('total_count', 0)}")
                                by_port = data_items_info.get('by_port', {})
                                if by_port:
                                    for port, items in by_port.items():
                                        diagnostic_lines.append(f"        Port '{port}': {len(items)} item(s)")
                                        for item in items[:3]:  # Show first 3 items
                                            diagnostic_lines.append(f"          - {item.get('name', 'unknown')} (ID: {item.get('id', 'unknown')}, Type: {item.get('type', 'unknown')})")
                                        if len(items) > 3:
                                            diagnostic_lines.append(f"          ... and {len(items) - 3} more")
                                else:
                                    diagnostic_lines.append(f"        No data items found for any port")
                        
                        if pull_result_info:
                            diagnostic_lines.append(f"      Pull Outputs Result: Success={pull_result_info.get('success', False)}, Message='{pull_result_info.get('message', 'N/A')}'")
                            diagnostic_lines.append(f"        Has data: {pull_result_info.get('has_data', False)}, Data keys: {pull_result_info.get('data_keys', [])}")
                        
                        if diag.get("filter_applied"):
                            diagnostic_lines.append(f"      Filter: Applied (result: {diag.get('filter_result', 'unknown')})")
                        
                        diagnostic_lines.append("")
                
                diagnostic_message = "\n".join(diagnostic_lines)
                Log.error(f"ApplicationFacade: Pull failed with diagnostics:\n{diagnostic_message}")
                
                result = CommandResult.error_result(
                    message="Upstream has no data for one or more connections",
                    errors=[
                        "missing_upstream_outputs",
                        f"missing={missing_upstream}",
                        f"diagnostics={diagnostic_message}"
                    ]
                )
                result.data = {
                    "missing_upstream": missing_upstream,
                    "diagnostics": diagnostic_message,
                    "block_id": block_id,
                    "block_name": block.name if block else None
                }
                return result
            
            # If inputs_map is empty but block has inputs, that's an error
            # (blocks with inputs must have data pulled)
            input_ports = block.get_inputs() if block else {}
            if not inputs_map and block and input_ports:
                # Build diagnostic for empty inputs_map
                diagnostic_lines = [
                    f"Block '{block.name}' (ID: {block_id}) has inputs but no data was pulled:",
                    f"  Block type: {block.type}",
                    f"  Declared inputs: {list(input_ports.keys())}",
                    ""
                ]
                
                # Check connections
                if incoming:
                    diagnostic_lines.append(f"  Incoming connections: {len(incoming)}")
                    for conn in incoming:
                        diagnostic_lines.append(f"    {conn.source_block_id}.{conn.source_output_name} -> {conn.target_block_id}.{conn.target_input_name}")
                else:
                    diagnostic_lines.append("  No incoming connections found")
                
                diagnostic_message = "\n".join(diagnostic_lines)
                Log.error(f"ApplicationFacade: No data found with diagnostics:\n{diagnostic_message}")
                
                result = CommandResult.error_result(
                    message="No data found for any input connections",
                    errors=["no_data_found", f"diagnostics={diagnostic_message}"]
                )
                result.data = {
                    "diagnostics": diagnostic_message,
                    "block_id": block_id,
                    "block_name": block.name if block else None,
                    "incoming_connections": len(incoming) if incoming else 0
                }
                return result
            
            # Only set local state if we have data (or block has no inputs)
            if inputs_map or not (block and block.get_inputs()):
                self.block_local_state_repo.set_inputs(block_id, inputs_map)
                
                # DEBUG: Verify what was actually stored
                stored_state = self.block_local_state_repo.get_inputs(block_id) or {}
                Log.info(
                    f"ApplicationFacade: pull_block_inputs_overwrite - VERIFIED stored local state for '{block.name if block else block_id}': "
                    f"keys={list(stored_state.keys())}, "
                    f"events type={type(stored_state.get('events')).__name__}, "
                    f"events value={stored_state.get('events') if isinstance(stored_state.get('events'), str) else (len(stored_state.get('events')) if isinstance(stored_state.get('events'), list) else 'None')}"
                )
            
            # Log final inputs map
            Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - final inputs_map: {list(inputs_map.keys())}")
            for port_name, item_refs in inputs_map.items():
                if isinstance(item_refs, list):
                    Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - {port_name}: {len(item_refs)} item(s)")
                    for idx, item_id in enumerate(item_refs):
                        item = self.data_item_repo.get(item_id) if self.data_item_repo else None
                        if item:
                            output_name = item.metadata.get('output_name', 'NO_OUTPUT_NAME')
                            # Check if this is a sync layer item (should not be in local state references)
                            is_sync = item.metadata.get("_synced_from_ma3") is True or (item.name and "ma3_sync" in item.name.lower())
                            if is_sync:
                                Log.warning(
                                    f"ApplicationFacade: pull_block_inputs_overwrite - WARNING: Sync layer item '{item.name}' "
                                    f"(ID: {item_id[:8]}...) found in {port_name} reference! This should not happen."
                                )
                            Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - {port_name}[{idx}]: id={item_id}, name={item.name}, output_name={output_name}")
                else:
                    item = self.data_item_repo.get(item_refs) if self.data_item_repo else None
                    if item:
                        output_name = item.metadata.get('output_name', 'NO_OUTPUT_NAME')
                        # Check if this is a sync layer item
                        is_sync = item.metadata.get("_synced_from_ma3") is True or (item.name and "ma3_sync" in item.name.lower())
                        if is_sync:
                            Log.warning(
                                f"ApplicationFacade: pull_block_inputs_overwrite - WARNING: Sync layer item '{item.name}' "
                                f"(ID: {item_refs[:8]}...) found in {port_name} reference! This should not happen."
                            )
                        Log.info(f"ApplicationFacade: pull_block_inputs_overwrite - {port_name}: id={item_refs}, name={item.name}, output_name={output_name}")
            
            # Force UI refresh for panels that listen to BlockUpdated (e.g., Editor).
            try:
                if block:
                    # DEBUG: Check local state again before publishing BlockUpdated
                    state_before_block_updated = self.block_local_state_repo.get_inputs(block_id) or {}
                    Log.info(
                        f"ApplicationFacade: pull_block_inputs_overwrite - Local state BEFORE BlockUpdated: "
                        f"events={state_before_block_updated.get('events') if isinstance(state_before_block_updated.get('events'), str) else (len(state_before_block_updated.get('events')) if isinstance(state_before_block_updated.get('events'), list) else 'None')}"
                    )
                    self.block_service.update_block(self.current_project_id, block_id, block)
                    # DEBUG: Check local state again after publishing BlockUpdated
                    state_after_block_updated = self.block_local_state_repo.get_inputs(block_id) or {}
                    Log.info(
                        f"ApplicationFacade: pull_block_inputs_overwrite - Local state AFTER BlockUpdated: "
                        f"events={state_after_block_updated.get('events') if isinstance(state_after_block_updated.get('events'), str) else (len(state_after_block_updated.get('events')) if isinstance(state_after_block_updated.get('events'), list) else 'None')}"
                    )
            except Exception:
                pass
            return CommandResult.success_result(
                message=f"Pulled {len(inputs_map)} local input(s)",
                data=inputs_map
            )
        except Exception as e:
            return CommandResult.error_result(message=f"Failed to pull inputs: {e}", errors=[str(e)])
    
    def _apply_filter_to_pull_result(
        self,
        block: Optional[Any],
        port_name: str,
        item_ids: Any  # Can be str (single ID) or List[str] (list of IDs)
    ) -> Optional[Any]:  # Returns filtered IDs in same format (str or List[str])
        """
        Apply filter_selections to pulled input item IDs.
        
        OPTIMIZATION: Checks filter BEFORE loading items to avoid unnecessary
        database queries. Only loads items that might pass the filter.
        
        Args:
            block: Block entity (may be None)
            port_name: Input port name to check for filter
            item_ids: Single ID (str) or list of IDs (List[str])
            
        Returns:
            Filtered IDs in same format as input, or None if all items filtered out
        """
        if not block or not self.data_item_repo or not self.data_filter_manager:
            return item_ids
        
        # OPTIMIZATION: Check if filter would exclude all items BEFORE loading
        if self.data_filter_manager.is_filter_all_disabled(block, port_name):
            Log.debug(
                f"ApplicationFacade: Filter for {block.name}.{port_name} excludes all items - "
                f"skipping load of {len(item_ids) if isinstance(item_ids, list) else 1} item(s)"
            )
            return None if isinstance(item_ids, str) else []
        
        # Get enabled output_names for selective loading
        enabled_output_names = self.data_filter_manager.get_enabled_output_names(block, port_name)
        
        # Normalize item_ids to list for processing
        is_single_id = isinstance(item_ids, str)
        id_list = [item_ids] if is_single_id else list(item_ids) if isinstance(item_ids, list) else None
        
        if id_list is None:
            return item_ids
        
        # OPTIMIZATION: Only load items that might pass the filter
        # If enabled_output_names is None, no filter - load all
        # If enabled_output_names is set, only load items with matching output_name
        items = []
        skipped_by_filter = 0
        
        for item_id in id_list:
            item = self.data_item_repo.get(item_id)
            if item is None:
                continue
            
            # If we have a filter, check output_name BEFORE adding to list
            if enabled_output_names is not None:
                output_name = item.metadata.get('output_name')
                if output_name and output_name not in enabled_output_names:
                    skipped_by_filter += 1
                    continue
            
            items.append(item)
        
        if skipped_by_filter > 0:
            Log.debug(
                f"ApplicationFacade: Filtered out {skipped_by_filter} items during load for "
                f"{block.name}.{port_name} (only loaded items with enabled output_names)"
            )
        
        if not items:
            return None if is_single_id else []
        
        try:
            # Apply filter using DataFilterManager (handles edge cases like missing output_name)
            filtered_items = self.data_filter_manager.apply_filter(block, port_name, items)
        except ValueError as e:
            Log.error(f"ApplicationFacade: Invalid filter: {e}")
            return item_ids
        except Exception as e:
            Log.warning(f"ApplicationFacade: Failed to apply filter: {e}")
            return item_ids
        
        if not filtered_items:
            return None if is_single_id else []
        
        # Convert back to IDs
        filtered_ids = [item.id for item in filtered_items]
        
        if len(filtered_ids) != len(id_list):
            Log.debug(
                f"ApplicationFacade: Applied filter to {block.name}.{port_name}: "
                f"{len(id_list)} -> {len(filtered_ids)} items"
            )
        
        # Return in same format as input
        if is_single_id:
            return filtered_ids[0] if filtered_ids else None
        return filtered_ids
    
    def pull_block_outputs(
        self, 
        block_id: str, 
        output_port: Optional[str] = None
    ) -> CommandResult:
        """
        Pull outputs from a block's local database.
        
        For executable blocks: Returns stored execution outputs
        For live blocks: Returns current state of owned DataItems
        For pass-through blocks: Returns input references from local state
        
        Args:
            block_id: Block to pull from
            output_port: Optional port name (if None, returns all outputs)
            
        Returns:
            CommandResult with Dict[str, DataItem | List[DataItem]]
        """
        if not self.data_item_repo or not self.block_local_state_repo:
            return CommandResult.error_result(message="Required repositories not available")
        
        try:
            block = self.block_service.get_block(self.current_project_id, block_id)
            if not block:
                return CommandResult.error_result(message=f"Block {block_id} not found")
            
            from src.application.block_registry import get_block_registry
            registry = get_block_registry()
            block_metadata = registry.get(block.type)
            if not block_metadata:
                return CommandResult.error_result(message=f"Block type {block.type} not found in registry")
            
            execution_mode = block_metadata.execution_mode
            
            # Get outputs based on execution mode
            if execution_mode == "passthrough":
                # Return input references from local state
                local_state = self.block_local_state_repo.get_inputs(block_id) or {}
                outputs = {}
                for port_name, data_item_id in local_state.items():
                    if output_port and port_name != output_port:
                        continue
                    if isinstance(data_item_id, list):
                        items = [self.data_item_repo.get(id) for id in data_item_id if self.data_item_repo.get(id)]
                        if items:
                            outputs[port_name] = items[0] if len(items) == 1 else items
                    else:
                        item = self.data_item_repo.get(data_item_id)
                        if item:
                            outputs[port_name] = item
                port_msg = f" port '{output_port}'" if output_port else ""
                return CommandResult.success_result(
                    message=f"Pulled {len(outputs)} output(s){port_msg}",
                    data=outputs
                )
            
            elif execution_mode == "live":
                # Return current state of owned DataItems
                owned_items = self.data_item_repo.list_by_block(block_id)
                outputs = {}
                for port_name in block_metadata.outputs.keys():
                    if output_port and port_name != output_port:
                        continue
                    port_items = [
                        item for item in owned_items
                        if item.metadata.get("output_port") == port_name
                    ]
                    if port_items:
                        outputs[port_name] = port_items[0] if len(port_items) == 1 else port_items
                port_msg = f" port '{output_port}'" if output_port else ""
                return CommandResult.success_result(
                    message=f"Pulled {len(outputs)} output(s){port_msg} from live block",
                    data=outputs
                )
            
            else:  # executable
                # Return stored execution outputs
                owned_items = self.data_item_repo.list_by_block(block_id)
                outputs = {}
                for port_name in block_metadata.outputs.keys():
                    if output_port and port_name != output_port:
                        continue
                    port_items = [
                        item for item in owned_items
                        if item.metadata.get("output_port") == port_name
                    ]
                    if port_items:
                        outputs[port_name] = port_items[0] if len(port_items) == 1 else port_items
                port_msg = f" port '{output_port}'" if output_port else ""
                return CommandResult.success_result(
                    message=f"Pulled {len(outputs)} output(s){port_msg} from executable block",
                    data=outputs
                )
        
        except Exception as e:
            return CommandResult.error_result(message=f"Failed to pull outputs: {e}", errors=[str(e)])
    
    def has_cached_outputs(self, block_id: str) -> bool:
        """
        Check if a block has cached output data.
        
        Args:
            block_id: Block identifier
            
        Returns:
            True if block has cached data items
        """
        if not self.data_item_repo:
            return False
        
        data_items = self.data_item_repo.list_by_block(block_id)
        return len(data_items) > 0
    
    def get_cached_output_count(self, block_id: str) -> int:
        """
        Get count of cached output data items for a block.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Number of cached data items
        """
        if not self.data_item_repo:
            return 0
        
        data_items = self.data_item_repo.list_by_block(block_id)
        return len(data_items)
    
    def _save_block_outputs(self, block: Block, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save block outputs to database.
        
        Deletes any existing data items for the block before saving new ones
        to prevent duplicate data items from accumulating.
        
        Args:
            block: Block that produced outputs
            outputs: Dictionary mapping output port names to DataItem instances
            
        Returns:
            Dictionary mapping output port names to data item IDs
        """
        output_ids = {}
        
        if not self.data_item_repo:
            Log.warning("ApplicationFacade: No data_item_repo available, outputs not saved")
            return output_ids
        
        # Delete old data items for this block to prevent duplicates
        deleted_count = self.data_item_repo.delete_by_block(block.id)
        if deleted_count > 0:
            Log.info(f"ApplicationFacade: Cleared {deleted_count} old data item(s) for block '{block.name}'")
        
        # Import helper for default output names
        from src.application.processing.output_name_helpers import make_default_output_name
        
        for port_name, port_data in outputs.items():
            # Handle list of data items
            if isinstance(port_data, list):
                item_ids = []
                for data_item in port_data:
                    # Ensure data item has correct block_id
                    if data_item.block_id != block.id:
                        data_item.block_id = block.id
                    # Store port name in metadata for reconstruction later
                    if 'output_port' not in data_item.metadata:
                        data_item.metadata['output_port'] = port_name
                    # Set output_name if not already set by processor
                    if 'output_name' not in data_item.metadata:
                        data_item.metadata['output_name'] = make_default_output_name(port_name)
                    created_item = self.data_item_repo.create(data_item)
                    item_ids.append(created_item.id)
                output_ids[port_name] = item_ids
                Log.info(f"ApplicationFacade: Saved {len(item_ids)} data items on port '{port_name}'")
            # Handle single data item
            else:
                data_item = port_data
                if data_item.block_id != block.id:
                    data_item.block_id = block.id
                # Store port name in metadata for reconstruction later
                if 'output_port' not in data_item.metadata:
                    data_item.metadata['output_port'] = port_name
                # Set output_name if not already set by processor
                if 'output_name' not in data_item.metadata:
                    data_item.metadata['output_name'] = make_default_output_name(port_name)
                created_item = self.data_item_repo.create(data_item)
                output_ids[port_name] = created_item.id
                Log.info(f"ApplicationFacade: Saved data item '{data_item.name}' on port '{port_name}'")
        
        # Automatically update block_local_state with output references
        # This ensures outputs are persisted for later loading (especially for live blocks like Editor)
        if self.block_local_state_repo and output_ids:
            try:
                # Get existing local state (preserve inputs)
                current_state = self.block_local_state_repo.get_inputs(block.id) or {}
                # Update with output references (merge, don't overwrite inputs)
                for port_name, item_id in output_ids.items():
                    current_state[port_name] = item_id
                # Save updated state
                self.block_local_state_repo.set_inputs(block.id, current_state)
                Log.debug(f"ApplicationFacade: Updated block_local_state for '{block.name}' with {len(output_ids)} output reference(s)")
            except Exception as e:
                Log.warning(f"ApplicationFacade: Failed to update block_local_state for '{block.name}': {e}")
        
        return output_ids
    
    def validate_project(self, project_id: Optional[str] = None) -> CommandResult:
        """
        Validate project graph.
        
        Args:
            project_id: Project ID (uses current if None)
            
        Returns:
            CommandResult with validation result
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        if not self.execution_engine:
            return CommandResult.error_result(
                message="Execution engine not available"
            )
        
        try:
            is_valid, error = self.execution_engine.validate_project(project_id)
            
            if is_valid:
                return CommandResult.success_result(
                    message="Project graph is valid and executable"
                )
            else:
                return CommandResult.error_result(
                    message=f"Project graph is invalid: {error}",
                    errors=[error]
                )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to validate project: {e}")
            return CommandResult.error_result(
                message=f"Failed to validate project: {e}",
                errors=[str(e)]
            )
    
    def validate_data_items(self, project_id: Optional[str] = None) -> CommandResult:
        """
        Validate all data items in project and report status.
        Checks file paths, cache validity, and identifies issues.
        
        Args:
            project_id: Project ID (uses current if None)
            
        Returns:
            CommandResult with validation details
        """
        project_id = project_id or self.current_project_id
        if not project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        try:
            # Get all data items for project
            data_items = self.data_item_repo.list_by_project(project_id)
            
            if not data_items:
                return CommandResult.success_result(
                    message="No data items to validate (project not executed yet)",
                    data={'total': 0, 'valid': 0, 'invalid': 0, 'invalid_items': []}
                )
            
            valid = []
            invalid = []
            
            for item in data_items:
                if item.file_path:
                    file_valid = item.metadata.get('_file_valid', True)
                    
                    # Double-check file still exists
                    import os
                    if file_valid and not os.path.exists(item.file_path):
                        file_valid = False
                        item.metadata['_file_valid'] = False
                        self.data_item_repo.update(item)
                    
                    if file_valid:
                        valid.append(item)
                    else:
                        # Get block name for better error reporting
                        block = self.block_repo.get(project_id, item.block_id)
                        block_name = block.name if block else item.block_id
                        
                        invalid.append({
                            'block_id': item.block_id,
                            'block_name': block_name,
                            'data_item_name': item.name,
                            'file_path': item.file_path,
                            'reason': 'File not found'
                        })
                else:
                    # No file reference, assume valid (in-memory data)
                    valid.append(item)
            
            # Format message
            if invalid:
                message = f"Validated {len(data_items)} data items: {len(valid)} valid, {len(invalid)} invalid"
            else:
                message = f"All {len(data_items)} data items are valid"
            
            return CommandResult.success_result(
                message=message,
                data={
                    'total': len(data_items),
                    'valid': len(valid),
                    'invalid': len(invalid),
                    'invalid_items': invalid
                }
            )
            
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to validate data items: {e}")
            return CommandResult.error_result(
                message=f"Failed to validate data items: {e}",
                errors=[str(e)]
            )
    
    # ==================== Data State Operations ====================
    
    def get_block_data_state(self, block_id: str) -> CommandResult[DataState]:
        """
        Get overall data state for a block.
        
        Overall state = worst state across all input ports (NO_DATA > STALE > FRESH).
        
        Args:
            block_id: Block identifier
            
        Returns:
            CommandResult with DataState enum value
        """
        if not self.data_state_service:
            return CommandResult.error_result(
                message="Data state service not available"
            )
        
        try:
            state = self.data_state_service.get_block_data_state(
                block_id, 
                project_id=self.current_project_id
            )
            return CommandResult.success_result(
                message=f"Block data state: {state.display_name}",
                data=state
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get block data state: {e}")
            return CommandResult.error_result(
                message=f"Failed to get block data state: {e}",
                errors=[str(e)]
            )
    
    def get_port_data_state(
        self,
        block_id: str,
        port_name: str,
        is_input: bool = True
    ) -> CommandResult[DataState]:
        """
        Get data state for a specific port.
        
        Args:
            block_id: Block identifier
            port_name: Port name
            is_input: True for input port, False for output port
            
        Returns:
            CommandResult with DataState enum value
        """
        if not self.data_state_service:
            return CommandResult.error_result(
                message="Data state service not available"
            )
        
        try:
            state = self.data_state_service.get_port_data_state(
                block_id,
                port_name,
                is_input
            )
            return CommandResult.success_result(
                message=f"Port data state: {state.display_name}",
                data=state
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get port data state: {e}")
            return CommandResult.error_result(
                message=f"Failed to get port data state: {e}",
                errors=[str(e)]
            )
    
    def is_data_stale(self, block_id: str, port_name: str) -> CommandResult[bool]:
        """
        Check if data for a port is stale.
        
        Args:
            block_id: Block identifier
            port_name: Port name
            
        Returns:
            CommandResult with boolean (True if stale)
        """
        if not self.data_state_service:
            return CommandResult.error_result(
                message="Data state service not available"
            )
        
        try:
            is_stale = self.data_state_service.is_data_stale(block_id, port_name)
            return CommandResult.success_result(
                message=f"Data is {'stale' if is_stale else 'fresh'}",
                data=is_stale
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to check data staleness: {e}")
            return CommandResult.error_result(
                message=f"Failed to check data staleness: {e}",
                errors=[str(e)]
            )
    
    # ==================== DataItem Inspection Operations ====================
    
    def get_block_data(self, block_identifier: str) -> CommandResult:
        """
        Get all data items (outputs) for a specific block.
        
        Useful for UI to inspect execution results, display file info,
        and show what data a block has produced.
        
        Args:
            block_identifier: Block ID or name
            
        Returns:
            CommandResult with list of data items in .data
            Each item includes: id, name, type, file_path, metadata, created_at
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            # Find the block
            block = self._get_block_by_identifier(block_identifier)
            if not block:
                return CommandResult.error_result(
                    message=f"Block not found: '{block_identifier}'",
                    errors=[
                        f"No block with ID or name '{block_identifier}' found",
                        "Use 'listblocks' to see available blocks"
                    ]
                )
            
            # Get all data items for this block
            data_items = self.data_item_repo.list_by_block(block.id)
            
            # Convert to dictionaries for UI consumption
            data_items_list = []
            for item in data_items:
                item_dict = {
                    "id": item.id,
                    "name": item.name,
                    "type": item.type,
                    "file_path": item.file_path,
                    "created_at": item.created_at.isoformat() if item.created_at else None,
                    "metadata": item.metadata,
                    "file_valid": item.metadata.get('_file_valid', True),
                    "output_port": item.metadata.get('output_port', 'unknown')
                }
                
                # Add type-specific fields
                if item.type == "Audio":
                    from src.shared.domain.entities import AudioDataItem
                    if isinstance(item, AudioDataItem):
                        item_dict["sample_rate"] = item.sample_rate
                        item_dict["length_ms"] = item.length_ms
                        item_dict["original_path"] = item.metadata.get('original_path')
                
                elif item.type == "Event":
                    from src.shared.domain.entities import EventDataItem
                    if isinstance(item, EventDataItem):
                        item_dict["event_count"] = item.event_count
                        item_dict["has_events"] = len(item._events) > 0 if hasattr(item, '_events') else False
                
                data_items_list.append(item_dict)
            
            return CommandResult.success_result(
                message=f"Found {len(data_items_list)} data item(s) for block '{block.name}'",
                data={
                    "block_id": block.id,
                    "block_name": block.name,
                    "block_type": block.type,
                    "data_items": data_items_list
                }
            )
            
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get block data: {e}")
            return CommandResult.error_result(
                message=f"Failed to get block data: {e}",
                errors=[str(e)]
            )
    
    def get_data_item_details(self, data_item_id: str) -> CommandResult:
        """
        Get detailed information about a specific data item.
        
        Useful for UI to display data properties, file info,
        and metadata in detail panels.
        
        Args:
            data_item_id: DataItem identifier
            
        Returns:
            CommandResult with full data item details in .data
        """
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            # Get data item from repository
            data_item = self.data_item_repo.get(data_item_id)
            
            if not data_item:
                return CommandResult.error_result(
                    message=f"Data item not found: '{data_item_id}'",
                    errors=[
                        f"No data item with ID '{data_item_id}' found",
                        "The data item may have been deleted or the ID is incorrect"
                    ]
                )
            
            # Build comprehensive details dictionary
            details = {
                "id": data_item.id,
                "block_id": data_item.block_id,
                "name": data_item.name,
                "type": data_item.type,
                "created_at": data_item.created_at.isoformat() if data_item.created_at else None,
                "file_path": data_item.file_path,
                "metadata": data_item.metadata,
                "file_valid": data_item.metadata.get('_file_valid', True),
                "output_port": data_item.metadata.get('output_port', 'unknown')
            }
            
            # Add file system info if file exists
            if data_item.file_path:
                import os
                from pathlib import Path
                
                file_path = Path(data_item.file_path)
                details["file_exists"] = file_path.exists()
                
                if file_path.exists():
                    details["file_size_bytes"] = file_path.stat().st_size
                    details["file_size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)
                    details["file_extension"] = file_path.suffix
                    details["file_name"] = file_path.name
                    details["file_directory"] = str(file_path.parent)
            
            # Add type-specific details
            if data_item.type == "Audio":
                from src.shared.domain.entities import AudioDataItem
                if isinstance(data_item, AudioDataItem):
                    details["audio"] = {
                        "sample_rate": data_item.sample_rate,
                        "length_ms": data_item.length_ms,
                        "length_seconds": round(data_item.length_ms / 1000, 2) if data_item.length_ms else None,
                        "original_path": data_item.metadata.get('original_path'),
                        "channels": data_item.metadata.get('channels'),
                        "file_format": data_item.metadata.get('file_format')
                    }
            
            elif data_item.type == "Event":
                from src.shared.domain.entities import EventDataItem
                if isinstance(data_item, EventDataItem):
                    has_events = len(data_item._events) > 0 if hasattr(data_item, '_events') else False
                    details["events"] = {
                        "event_count": data_item.event_count,
                        "has_events_in_memory": has_events,
                        "source_audio": data_item.metadata.get('source_audio'),
                        "extractor": data_item.metadata.get('extractor')
                    }
                    
                    # Include first few events as samples if available
                    if has_events:
                        sample_events = []
                        for event in data_item._events[:5]:  # First 5 events
                            sample_events.append({
                                "time": event.time,
                                "classification": event.classification,
                                "duration": event.duration
                            })
                        details["events"]["sample_events"] = sample_events
            
            return CommandResult.success_result(
                message=f"Data item details for '{data_item.name}'",
                data=details
            )
            
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get data item details: {e}")
            return CommandResult.error_result(
                message=f"Failed to get data item details: {e}",
                errors=[str(e)]
            )
    
    def get_port_data_summary(self, block_identifier: str, port_name: str) -> CommandResult:
        """
        Get summary of data items on a specific block output port.
        
        Useful for UI to show what data is available on each port
        without loading full details.
        
        Args:
            block_identifier: Block ID or name
            port_name: Output port name (e.g., "audio", "events")
            
        Returns:
            CommandResult with port data summary in .data
        """
        if not self.current_project_id:
            return CommandResult.error_result(
                message="No project loaded"
            )
        
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            # Find the block
            block = self._get_block_by_identifier(block_identifier)
            if not block:
                return CommandResult.error_result(
                    message=f"Block not found: '{block_identifier}'",
                    errors=[
                        f"No block with ID or name '{block_identifier}' found"
                    ]
                )
            
            # Check if port exists on block
            output_ports = block.get_outputs()
            if port_name not in output_ports:
                return CommandResult.error_result(
                    message=f"Port '{port_name}' not found on block '{block.name}'",
                    errors=[
                        f"Block '{block.name}' has outputs: {list(output_ports.keys())}",
                        f"Port '{port_name}' does not exist"
                    ]
                )
            
            # Get all data items for this block
            all_data_items = self.data_item_repo.list_by_block(block.id)
            
            # Filter for items on this specific port
            port_items = [
                item for item in all_data_items
                if item.metadata.get('output_port') == port_name
            ]
            
            # Build summary
            port = output_ports[port_name]
            summary = {
                "block_id": block.id,
                "block_name": block.name,
                "block_type": block.type,
                "port_name": port_name,
                "port_type": str(port.port_type),
                "item_count": len(port_items),
                "items": []
            }
            
            # Add summary info for each item
            total_size = 0
            for item in port_items:
                item_summary = {
                    "id": item.id,
                    "name": item.name,
                    "type": item.type,
                    "file_path": item.file_path,
                    "file_valid": item.metadata.get('_file_valid', True)
                }
                
                # Add file size if available
                if item.file_path:
                    import os
                    if os.path.exists(item.file_path):
                        size = os.path.getsize(item.file_path)
                        item_summary["file_size_bytes"] = size
                        item_summary["file_size_mb"] = round(size / (1024 * 1024), 2)
                        total_size += size
                
                # Add type-specific summary
                if item.type == "Audio":
                    from src.shared.domain.entities import AudioDataItem
                    if isinstance(item, AudioDataItem):
                        item_summary["duration_ms"] = item.length_ms
                        item_summary["sample_rate"] = item.sample_rate
                
                elif item.type == "Event":
                    from src.shared.domain.entities import EventDataItem
                    if isinstance(item, EventDataItem):
                        item_summary["event_count"] = item.event_count
                
                summary["items"].append(item_summary)
            
            # Add aggregate statistics
            summary["total_size_bytes"] = total_size
            summary["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            message = f"Port '{port_name}' on '{block.name}' has {len(port_items)} item(s)"
            if total_size > 0:
                message += f" ({summary['total_size_mb']} MB)"
            
            return CommandResult.success_result(
                message=message,
                data=summary
            )
            
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get port data summary: {e}")
            return CommandResult.error_result(
                message=f"Failed to get port data summary: {e}",
                errors=[str(e)]
            )
    
    # ==================== UI State Management (Phase A) ====================
    
    def set_ui_state(self, state_type: str, entity_id: Optional[str], data: Dict[str, Any]) -> CommandResult:
        """
        Set UI state for a given type and entity.
        
        UI state is project-specific data like block positions, zoom levels, etc.
        Cleared when switching projects.
        
        Args:
            state_type: Type of UI state (e.g., 'block_position', 'zoom_level')
            entity_id: Optional entity ID (e.g., block_id for block_position)
            data: Dictionary of state data to store
            
        Returns:
            CommandResult indicating success or failure
        """
        if not self.ui_state_repo:
            return CommandResult.error_result(
                message="UI state repository not available"
            )
        
        try:
            self.ui_state_repo.set(state_type, entity_id, data)
            Log.debug(f"ApplicationFacade: Set UI state {state_type} for entity {entity_id}")
            
            # Publish event for UI state changes (allows UI components to react)
            self.event_bus.publish(UIStateChanged(
                project_id=self.get_current_project_id(),
                data={
                    "state_type": state_type,
                    "entity_id": entity_id,
                    "data": data
                }
            ))
            
            return CommandResult.success_result(
                message=f"UI state updated: {state_type}",
                data=data
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to set UI state: {e}")
            return CommandResult.error_result(
                message=f"Failed to set UI state: {e}",
                errors=[str(e)]
            )
    
    def get_ui_state(self, state_type: str, entity_id: Optional[str] = None) -> CommandResult[Dict[str, Any]]:
        """
        Get UI state for a given type and entity.
        
        Args:
            state_type: Type of UI state
            entity_id: Optional entity ID
            
        Returns:
            CommandResult with state data in .data (or empty dict if not found)
        """
        if not self.ui_state_repo:
            return CommandResult.error_result(
                message="UI state repository not available"
            )
        
        try:
            data = self.ui_state_repo.get(state_type, entity_id)
            if data is None:
                data = {}
            
            return CommandResult.success_result(
                message=f"UI state retrieved: {state_type}",
                data=data
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get UI state: {e}")
            return CommandResult.error_result(
                message=f"Failed to get UI state: {e}",
                errors=[str(e)]
            )
    
    def get_ui_states_by_type(self, state_type: str) -> CommandResult[List[Dict[str, Any]]]:
        """
        Get all UI state entries of a given type.
        
        Useful for loading all block positions at once.
        
        Args:
            state_type: Type of UI state (e.g., 'block_position')
            
        Returns:
            CommandResult with list of state data dictionaries in .data
        """
        if not self.ui_state_repo:
            return CommandResult.error_result(
                message="UI state repository not available"
            )
        
        try:
            states = self.ui_state_repo.get_by_type(state_type)
            return CommandResult.success_result(
                message=f"Retrieved {len(states)} UI state(s) of type '{state_type}'",
                data=states
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get UI states by type: {e}")
            return CommandResult.error_result(
                message=f"Failed to get UI states: {e}",
                errors=[str(e)]
            )
    
    # ==================== Preferences Management (Phase A) ====================
    
    def set_preference(self, key: str, value: Any) -> CommandResult:
        """
        Set a user preference.
        
        Preferences are application-wide settings that persist across sessions and projects.
        
        Args:
            key: Preference key (e.g., 'default_zoom', 'theme')
            value: Value to store (will be JSON serialized if dict/list)
            
        Returns:
            CommandResult indicating success or failure
        """
        if not self.preferences_repo:
            return CommandResult.error_result(
                message="Preferences repository not available"
            )
        
        try:
            self.preferences_repo.set(key, value)
            Log.debug(f"ApplicationFacade: Set preference '{key}'")
            return CommandResult.success_result(
                message=f"Preference set: {key}",
                data={"key": key, "value": value}
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to set preference: {e}")
            return CommandResult.error_result(
                message=f"Failed to set preference: {e}",
                errors=[str(e)]
            )
    
    def get_preference(self, key: str, default: Any = None) -> CommandResult[Any]:
        """
        Get a user preference.
        
        Args:
            key: Preference key
            default: Default value if preference not found
            
        Returns:
            CommandResult with preference value in .data
        """
        if not self.preferences_repo:
            return CommandResult.error_result(
                message="Preferences repository not available"
            )
        
        try:
            value = self.preferences_repo.get(key, default)
            return CommandResult.success_result(
                message=f"Preference retrieved: {key}",
                data=value
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get preference: {e}")
            return CommandResult.error_result(
                message=f"Failed to get preference: {e}",
                errors=[str(e)]
            )
    
    def get_all_preferences(self) -> CommandResult[Dict[str, Any]]:
        """
        Get all user preferences.
        
        Returns:
            CommandResult with all preferences as dictionary in .data
        """
        if not self.preferences_repo:
            return CommandResult.error_result(
                message="Preferences repository not available"
            )
        
        try:
            preferences = self.preferences_repo.get_all()
            return CommandResult.success_result(
                message=f"Retrieved {len(preferences)} preference(s)",
                data=preferences
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get preferences: {e}")
            return CommandResult.error_result(
                message=f"Failed to get preferences: {e}",
                errors=[str(e)]
            )
    
    # ==================== Session State Management (Phase A) ====================
    
    def set_session_state(self, key: str, value: Any) -> CommandResult:
        """
        Set session state.
        
        Session state persists across app restarts but is not project-specific.
        Examples: open panels, selected block, last opened project.
        
        Args:
            key: State key (e.g., 'open_panels', 'selected_block')
            value: Value to store (will be JSON serialized if dict/list)
            
        Returns:
            CommandResult indicating success or failure
        """
        if not self.session_state_repo:
            return CommandResult.error_result(
                message="Session state repository not available"
            )
        
        try:
            self.session_state_repo.set(key, value)
            Log.debug(f"ApplicationFacade: Set session state '{key}'")
            return CommandResult.success_result(
                message=f"Session state set: {key}",
                data={"key": key, "value": value}
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to set session state: {e}")
            return CommandResult.error_result(
                message=f"Failed to set session state: {e}",
                errors=[str(e)]
            )
    
    def get_session_state(self, key: str, default: Any = None) -> CommandResult[Any]:
        """
        Get session state.
        
        Args:
            key: State key
            default: Default value if state not found
            
        Returns:
            CommandResult with state value in .data
        """
        if not self.session_state_repo:
            return CommandResult.error_result(
                message="Session state repository not available"
            )
        
        try:
            value = self.session_state_repo.get(key, default)
            return CommandResult.success_result(
                message=f"Session state retrieved: {key}",
                data=value
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get session state: {e}")
            return CommandResult.error_result(
                message=f"Failed to get session state: {e}",
                errors=[str(e)]
            )
    
    def get_all_session_state(self) -> CommandResult[Dict[str, Any]]:
        """
        Get all session state.
        
        Returns:
            CommandResult with all session state as dictionary in .data
        """
        if not self.session_state_repo:
            return CommandResult.error_result(
                message="Session state repository not available"
            )
        
        try:
            state = self.session_state_repo.get_all()
            return CommandResult.success_result(
                message=f"Retrieved {len(state)} session state key(s)",
                data=state
            )
        except Exception as e:
            Log.error(f"ApplicationFacade: Failed to get session state: {e}")
            return CommandResult.error_result(
                message=f"Failed to get session state: {e}",
                errors=[str(e)]
            )
    
    # ==================== Event Data Operations ====================
    
    def update_event_in_data_item(
        self,
        data_item_id: str,
        event_index: int,
        time: Optional[float] = None,
        duration: Optional[float] = None,
        classification: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CommandResult:
        """
        Update a specific event within an EventDataItem.
        
        Args:
            data_item_id: ID of the EventDataItem
            event_index: Index of the event to update
            time: New start time (optional)
            duration: New duration (optional)
            classification: New classification (optional)
            metadata: New metadata dict (optional, merged with existing)
            
        Returns:
            CommandResult with updated event data
        """
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            from src.shared.domain.entities import EventDataItem
            
            # Get the data item
            data_item = self.data_item_repo.get(data_item_id)
            if not data_item:
                return CommandResult.error_result(
                    message=f"Data item not found: {data_item_id}"
                )
            
            if not isinstance(data_item, EventDataItem):
                return CommandResult.error_result(
                    message=f"Data item is not an EventDataItem: {data_item.type}"
                )
            
            # Get events
            events = data_item.get_events()
            if event_index < 0 or event_index >= len(events):
                return CommandResult.error_result(
                    message=f"Event index out of range: {event_index} (have {len(events)} events)"
                )
            
            # Update the event
            event = events[event_index]
            if time is not None:
                event.time = time
            if duration is not None:
                event.duration = duration
            if classification is not None:
                event.classification = classification
            if metadata is not None:
                event.metadata.update(metadata)
            
            # Save back to database
            self.data_item_repo.update(data_item)
            
            Log.debug(f"Updated event {event_index} in data item {data_item_id}")
            
            return CommandResult.success_result(
                message=f"Event updated",
                data={
                    "data_item_id": data_item_id,
                    "event_index": event_index,
                    "time": event.time,
                    "duration": event.duration,
                    "classification": event.classification
                }
            )
            
        except Exception as e:
            Log.error(f"Failed to update event: {e}")
            return CommandResult.error_result(
                message=f"Failed to update event: {str(e)}"
            )
    
    def delete_event_from_data_item(
        self,
        data_item_id: str,
        event_index: int
    ) -> CommandResult:
        """
        Delete a specific event from an EventDataItem.
        
        Args:
            data_item_id: ID of the EventDataItem
            event_index: Index of the event to delete
            
        Returns:
            CommandResult indicating success/failure
        """
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            from src.shared.domain.entities import EventDataItem
            
            # Get the data item
            data_item = self.data_item_repo.get(data_item_id)
            if not data_item:
                return CommandResult.error_result(
                    message=f"Data item not found: {data_item_id}"
                )
            
            if not isinstance(data_item, EventDataItem):
                return CommandResult.error_result(
                    message=f"Data item is not an EventDataItem: {data_item.type}"
                )
            
            # Get events and validate index
            events = data_item.get_events()
            if event_index < 0 or event_index >= len(events):
                return CommandResult.error_result(
                    message=f"Event index out of range: {event_index} (have {len(events)} events)"
                )
            
            # Remove the event
            event = events[event_index]
            data_item.remove_event(event)
            
            # Save back to database
            self.data_item_repo.update(data_item)
            
            Log.debug(f"Deleted event {event_index} from data item {data_item_id}")
            
            return CommandResult.success_result(
                message=f"Event deleted",
                data={
                    "data_item_id": data_item_id,
                    "deleted_index": event_index,
                    "remaining_count": data_item.event_count
                }
            )
            
        except Exception as e:
            Log.error(f"Failed to delete event: {e}")
            return CommandResult.error_result(
                message=f"Failed to delete event: {str(e)}"
            )
    
    def add_event_to_data_item(
        self,
        data_item_id: str,
        time: float,
        duration: float = 0.0,
        classification: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        layer_name: str = ""
    ) -> CommandResult:
        """
        Add a new event to an EventDataItem.
        
        Args:
            data_item_id: ID of the EventDataItem
            time: Start time in seconds
            duration: Duration in seconds (0 for markers)
            classification: Event classification/layer
            metadata: Optional event metadata
            layer_name: REQUIRED - Name of the layer to add the event to
            
        Returns:
            CommandResult with created event data
        """
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            from src.shared.domain.entities import EventDataItem
            
            # Get the data item
            data_item = self.data_item_repo.get(data_item_id)
            if not data_item:
                return CommandResult.error_result(
                    message=f"Data item not found: {data_item_id}"
                )
            
            if not isinstance(data_item, EventDataItem):
                return CommandResult.error_result(
                    message=f"Data item is not an EventDataItem: {data_item.type}"
                )
            
            # Add the event
            # layer_name is REQUIRED by EventDataItem.add_event()
            # Fall back to classification for backwards compatibility
            actual_layer_name = layer_name or classification or "event"
            event = data_item.add_event(
                time=time,
                duration=duration,
                classification=classification,
                metadata=metadata,
                layer_name=actual_layer_name
            )
            
            # Save back to database
            self.data_item_repo.update(data_item)
            
            Log.debug(f"Added event to data item {data_item_id} at time {time}")
            
            return CommandResult.success_result(
                message=f"Event created",
                data={
                    "data_item_id": data_item_id,
                    "event_index": data_item.event_count - 1,
                    "time": event.time,
                    "duration": event.duration,
                    "classification": event.classification
                }
            )
            
        except Exception as e:
            Log.error(f"Failed to add event: {e}")
            return CommandResult.error_result(
                message=f"Failed to add event: {str(e)}"
            )
    
    def batch_update_events(
        self,
        data_item_id: str,
        updates: List[Dict[str, Any]]
    ) -> CommandResult:
        """
        Batch update multiple events in a single transaction.
        
        More efficient than calling update_event_in_data_item multiple times.
        
        Args:
            data_item_id: ID of the EventDataItem
            updates: List of update dicts, each with:
                - event_index: int (required)
                - time: float (optional)
                - duration: float (optional)
                - classification: str (optional)
                - metadata: dict (optional)
                
        Returns:
            CommandResult with update summary
        """
        if not self.data_item_repo:
            return CommandResult.error_result(
                message="Data item repository not available"
            )
        
        try:
            from src.shared.domain.entities import EventDataItem
            
            # Get the data item
            data_item = self.data_item_repo.get(data_item_id)
            if not data_item:
                return CommandResult.error_result(
                    message=f"Data item not found: {data_item_id}"
                )
            
            if not isinstance(data_item, EventDataItem):
                return CommandResult.error_result(
                    message=f"Data item is not an EventDataItem: {data_item.type}"
                )
            
            events = data_item.get_events()
            updated_count = 0
            errors = []
            
            for update in updates:
                event_index = update.get('event_index')
                if event_index is None:
                    errors.append("Update missing 'event_index'")
                    continue
                
                if event_index < 0 or event_index >= len(events):
                    errors.append(f"Event index out of range: {event_index}")
                    continue
                
                event = events[event_index]
                
                if 'time' in update:
                    event.time = update['time']
                if 'duration' in update:
                    event.duration = update['duration']
                if 'classification' in update:
                    event.classification = update['classification']
                if 'metadata' in update and update['metadata']:
                    event.metadata.update(update['metadata'])
                
                updated_count += 1
            
            # Single save for all updates
            self.data_item_repo.update(data_item)
            
            Log.debug(f"Batch updated {updated_count} events in data item {data_item_id}")
            
            result_data = {
                "data_item_id": data_item_id,
                "updated_count": updated_count,
                "total_events": len(events)
            }
            
            if errors:
                result_data["errors"] = errors
                return CommandResult.success_result(
                    message=f"Updated {updated_count} events with {len(errors)} errors",
                    data=result_data
                )
            
            return CommandResult.success_result(
                message=f"Updated {updated_count} events",
                data=result_data
            )
            
        except Exception as e:
            Log.error(f"Failed to batch update events: {e}")
            return CommandResult.error_result(
                message=f"Failed to batch update events: {str(e)}"
            )
    
    # ==================== Utility Methods ====================
    
    def get_current_project_id(self) -> Optional[str]:
        """Get current project ID"""
        return self.current_project_id
    
    def set_current_project(self, project_id: str):
        """Set current project ID"""
        self.current_project_id = project_id
        Log.info(f"ApplicationFacade: Current project set to {project_id}")
    
    def _validate_save_directory(self, directory: str) -> Optional[str]:
        """
        Validate save directory path.
        
        Args:
            directory: Directory path to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        if not directory or not directory.strip():
            return "Save directory cannot be empty"
        
        # Check if path exists and is a file (not a directory)
        if os.path.exists(directory):
            if not os.path.isdir(directory):
                return f"Path '{directory}' exists but is not a directory"
        else:
            # Path doesn't exist - check if parent directory exists and is writable
            parent_dir = os.path.dirname(directory)
            if parent_dir and not os.path.exists(parent_dir):
                return f"Parent directory '{parent_dir}' does not exist"
            
            if parent_dir and not os.path.isdir(parent_dir):
                return f"Parent path '{parent_dir}' is not a directory"
            
            if parent_dir and not os.access(parent_dir, os.W_OK):
                return f"Parent directory '{parent_dir}' is not writable"
        
        # Check for invalid characters
        if os.name == 'nt':  # Windows
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        else:  # Unix-like
            invalid_chars = ['\x00']
        
        if any(char in directory for char in invalid_chars):
            return f"Directory path contains invalid characters"
        
        return None

