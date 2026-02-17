"""
Setlist Processing Service

Handles song processing, action execution, pre/post hooks, and cleanup.
Extracted from SetlistService for single-responsibility.
"""
from typing import Optional, Dict, Any, List, Callable, Tuple
from pathlib import Path

from src.features.setlists.domain import Setlist, SetlistSong
from src.features.setlists.domain import SetlistRepository, SetlistSongRepository
from src.features.blocks.domain import BlockRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.features.projects.application import SnapshotService, ProjectService
from src.features.execution.application import BlockExecutionEngine
from src.application.api.application_facade import ApplicationFacade
from src.application.blocks.quick_actions import get_quick_actions
from src.shared.application.services.progress_context import get_progress_context
from src.utils.message import Log


class SetlistProcessingService:
    """
    Handles song-level and setlist-level processing logic.
    
    Responsibilities:
    - Processing individual songs through the action pipeline
    - Processing entire setlists with error recovery
    - Executing action items (including pre/post hooks)
    - Resolving dynamic placeholders in action args
    - Validating action items before execution
    - Publishing setlist events
    - Data cleanup between songs
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
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._block_repo = block_repo
        self._data_item_repo = data_item_repo
        self._block_local_state_repo = block_local_state_repo
        self._snapshot_service = snapshot_service
        self._project_service = project_service
        self._execution_engine = execution_engine
        self._facade = facade
        self._progress = get_progress_context()
        Log.info("SetlistProcessingService: Initialized")
    
    def process_song(
        self,
        setlist_id: str,
        song_id: str,
        progress_callback: Optional[Any] = None,
        action_progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> bool:
        """
        Process a single song through the project action pipeline.
        
        Steps:
        1. Verify project matches setlist
        2. Find audio input block (SetlistAudioInput or LoadAudio)
        3. Execute pre-song hooks (from song.action_overrides)
        4. Execute main action items
        5. Execute post-song hooks (from song.action_overrides)
        6. Save snapshot
        7. Clear data for next song
        8. Mark song as completed
        
        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier
            progress_callback: Optional progress callback (message, current, total)
            action_progress_callback: Optional callback for action-level progress
            
        Returns:
            True if processing succeeded, False otherwise
        """
        setlist = self._setlist_repo.get(setlist_id)
        if not setlist:
            Log.error(f"SetlistProcessingService: Setlist {setlist_id} not found")
            return False
        
        song = self._setlist_song_repo.get(song_id)
        if not song:
            Log.error(f"SetlistProcessingService: Song {song_id} not found")
            return False
        
        if song.setlist_id != setlist_id:
            Log.error(f"SetlistProcessingService: Song {song_id} does not belong to setlist {setlist_id}")
            return False
        
        Log.info(f"SetlistProcessingService: Processing song {song.audio_path} (song_id: {song_id})")
        
        # Publish song processing event
        self._publish_event("SetlistSongProcessing", {
            "setlist_id": setlist_id,
            "song_id": song_id,
            "song_index": song.order_index,
            "song_name": Path(song.audio_path).name,
            "audio_path": song.audio_path,
        })
        
        try:
            song.mark_processing()
            song.error_message = None
            self._setlist_song_repo.update(song)
            
            # 1. Verify project
            if progress_callback:
                progress_callback("Verifying project...", 0, 100)
            
            if not self._facade.current_project_id:
                raise Exception("No current project. Please open the project used to create this setlist.")
            
            if self._facade.current_project_id != setlist.project_id:
                raise Exception(
                    f"Current project ({self._facade.current_project_id}) does not match "
                    f"setlist's project ({setlist.project_id}). Please open the correct project."
                )
            
            project = self._project_service.load_project(setlist.project_id)
            if not project:
                raise Exception(f"Project not found: {setlist.project_id}")
            
            # 2. Find audio input block
            if progress_callback:
                progress_callback("Finding audio input block...", 10, 100)
            
            blocks = self._block_repo.list_by_project(project.id)
            audio_input_block = None
            for block in blocks:
                if block.type == "SetlistAudioInput":
                    audio_input_block = block
                    break
            if not audio_input_block:
                for block in blocks:
                    if block.type == "LoadAudio":
                        audio_input_block = block
                        break
            if not audio_input_block:
                raise Exception("No SetlistAudioInput or LoadAudio block found in project.")
            
            # Build setlist context
            all_songs = self._setlist_song_repo.list_by_setlist(setlist_id)
            total_songs = len(all_songs)
            setlist_name = Path(setlist.audio_folder_path).name if setlist.audio_folder_path else ""
            
            setlist_context = {
                "song_audio_path": song.audio_path,
                "song_name": Path(song.audio_path).stem,
                "song_full_name": Path(song.audio_path).name,
                "song_index": song.order_index,
                "song_index_1": song.order_index + 1,
                "song_count": total_songs,
                "setlist_id": setlist.id,
                "setlist_name": setlist_name,
            }
            
            # 3. Execute pre-song hooks
            if progress_callback:
                progress_callback("Running pre-song hooks...", 15, 100)
            
            pre_actions = self._get_hook_actions(song, "pre_actions")
            if pre_actions:
                Log.info(f"SetlistProcessingService: Executing {len(pre_actions)} pre-song hook(s)")
                self._execute_hook_actions(
                    project.id, pre_actions, setlist_context, action_progress_callback,
                    phase_label="Pre"
                )
            
            # 4. Load and execute main ActionItems
            if progress_callback:
                progress_callback("Applying actions...", 20, 100)
            
            action_items_result = self._facade.list_action_items_by_project(project_id=project.id)
            if not action_items_result.success:
                raise Exception(f"Failed to load action items: {action_items_result.message}")
            
            action_items = action_items_result.data or []
            
            # Ensure audio file action is included
            from src.features.projects.domain import ActionItem
            
            has_audio_action = False
            for action_item in action_items:
                if (action_item.block_id == audio_input_block.id and 
                    action_item.action_name == "Set Audio File"):
                    has_audio_action = True
                    if not action_item.action_args or not isinstance(action_item.action_args, dict):
                        action_item.action_args = {}
                    if "file_path" not in action_item.action_args or not action_item.action_args.get("file_path"):
                        action_item.action_args["file_path"] = "{song_audio_path}"
                    break
            
            if not has_audio_action:
                audio_action = ActionItem(
                    action_type="block",
                    action_name="Set Audio File",
                    block_id=audio_input_block.id,
                    block_name=audio_input_block.name,
                    action_args={"file_path": "{song_audio_path}"}
                )
                action_items.insert(0, audio_action)
            
            self._execute_action_items(
                project.id, action_items,
                setlist_context=setlist_context,
                action_progress_callback=action_progress_callback
            )
            
            # 5. Execute post-song hooks
            if progress_callback:
                progress_callback("Running post-song hooks...", 65, 100)
            
            post_actions = self._get_hook_actions(song, "post_actions")
            if post_actions:
                Log.info(f"SetlistProcessingService: Executing {len(post_actions)} post-song hook(s)")
                self._execute_hook_actions(
                    project.id, post_actions, setlist_context, action_progress_callback,
                    phase_label="Post"
                )
            
            # 6. Save snapshot
            if progress_callback:
                progress_callback("Saving snapshot...", 70, 100)
            
            snapshot = self._snapshot_service.save_snapshot(
                project_id=project.id,
                song_id=song_id,
                block_settings_overrides=None
            )
            snapshot_dict = snapshot.to_dict()
            self._project_service.set_snapshot(song_id, snapshot_dict, project)
            
            # 7. Clear data
            if progress_callback:
                progress_callback("Clearing data for next song...", 85, 100)
            self._clear_all_project_data(project.id)
            
            # 8. Cleanup
            if progress_callback:
                progress_callback("Cleaning up...", 90, 100)
            self._cleanup_blocks_after_execution(project.id)
            
            song.mark_completed()
            self._setlist_song_repo.update(song)
            
            if progress_callback:
                progress_callback("Completed", 100, 100)
            
            # Publish completion event
            self._publish_event("SetlistSongCompleted", {
                "setlist_id": setlist_id,
                "song_id": song_id,
                "song_index": song.order_index,
                "success": True,
            })
            
            Log.info(f"SetlistProcessingService: Successfully processed song {song.audio_path}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            Log.error(f"SetlistProcessingService: Failed to process song {song.audio_path}: {error_msg}")
            song.mark_failed(error_message=error_msg)
            self._setlist_song_repo.update(song)
            
            self._publish_event("SetlistSongCompleted", {
                "setlist_id": setlist_id,
                "song_id": song_id,
                "song_index": song.order_index,
                "success": False,
                "error_message": error_msg,
            })
            
            return False
    
    def process_setlist(
        self,
        setlist_id: str,
        error_callback: Optional[Callable[[str, str], None]] = None,
        action_progress_callback: Optional[Callable[[str, int, int, str, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ) -> Dict[str, bool]:
        """
        Process all songs in a setlist with error recovery.
        
        Publishes SetlistProcessingStarted and SetlistProcessingCompleted events.
        Continues processing even if individual songs fail.
        """
        setlist = self._setlist_repo.get(setlist_id)
        if not setlist:
            Log.error(f"SetlistProcessingService: Setlist {setlist_id} not found")
            return {}
        
        songs = self._setlist_song_repo.list_by_setlist(setlist_id)
        results = {}
        errors = []
        total_songs = len(songs)
        
        setlist_name = "Setlist"
        if setlist.audio_folder_path:
            setlist_name = Path(setlist.audio_folder_path).name
        
        # Publish start event
        self._publish_event("SetlistProcessingStarted", {
            "setlist_id": setlist_id,
            "song_count": total_songs,
            "setlist_name": setlist_name,
        })
        
        with self._progress.setlist_processing(setlist_id, setlist_name, total_songs=total_songs) as op:
            for index, song in enumerate(songs):
                if cancel_check and cancel_check():
                    Log.info(f"SetlistProcessingService: Processing cancelled after {index}/{total_songs} songs")
                    break
                
                song_name = Path(song.audio_path).name
                
                try:
                    with op.song(song.id, song_name, audio_path=song.audio_path) as song_ctx:
                        song_ctx.update(message=f"Processing ({index + 1}/{total_songs})")
                        
                        song_action_progress = None
                        if action_progress_callback:
                            def make_song_action_callback(song_id_param, song_context):
                                def callback(action_idx, total_actions, action_name, status):
                                    song_context.update(
                                        current=action_idx + 1 if status == "completed" else action_idx,
                                        total=total_actions,
                                        message=f"{action_name}: {status}"
                                    )
                                    action_progress_callback(song_id_param, action_idx, total_actions, action_name, status)
                                return callback
                            song_action_progress = make_song_action_callback(song.id, song_ctx)
                        
                        success = self.process_song(
                            setlist_id=setlist_id,
                            song_id=song.id,
                            action_progress_callback=song_action_progress
                        )
                        results[song.id] = success
                        
                        if not success:
                            error_msg = song.error_message or "Processing failed - see logs"
                            errors.append({"song": song.audio_path, "error": error_msg})
                            song_ctx.update(message=f"Failed: {error_msg}")
                            if error_callback:
                                error_callback(song.audio_path, error_msg)
                            raise Exception(error_msg)
                        else:
                            song_ctx.update(message="Completed")
                            
                except Exception as e:
                    error_msg = str(e)
                    if song.id not in results:
                        results[song.id] = False
                    if not any(err["song"] == song.audio_path for err in errors):
                        errors.append({"song": song.audio_path, "error": error_msg})
                        if error_callback:
                            error_callback(song.audio_path, error_msg)
                    continue
        
        success_count = sum(1 for s in results.values() if s)
        failed_count = total_songs - success_count
        
        # Publish completion event
        self._publish_event("SetlistProcessingCompleted", {
            "setlist_id": setlist_id,
            "results": results,
            "total_songs": total_songs,
            "successful_songs": success_count,
            "failed_songs": failed_count,
        })
        
        Log.info(
            f"SetlistProcessingService: Processed {success_count}/{total_songs} song(s) successfully. "
            f"{len(errors)} error(s) occurred."
        )
        
        if errors:
            Log.warning(f"SetlistProcessingService: Errors occurred during processing:")
            for error in errors:
                Log.warning(f"  - {Path(error['song']).name}: {error['error']}")
        
        return results
    
    def validate_action_items(self, project_id: str) -> List[str]:
        """
        Validate action items before processing begins.
        
        Checks:
        - All referenced blocks exist
        - All referenced actions exist in the quick_actions registry
        - Required action parameters are present (non-empty)
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        action_items_result = self._facade.list_action_items_by_project(project_id=project_id)
        if not action_items_result.success:
            errors.append(f"Failed to load action items: {action_items_result.message}")
            return errors
        
        action_items = action_items_result.data or []
        if not action_items:
            return errors  # No actions to validate
        
        for idx, action_item in enumerate(action_items):
            action_label = f"Action {idx + 1} ({action_item.action_name})"
            
            if action_item.action_type == "project":
                valid_project_actions = {"validate_project", "save_project", "save_as"}
                if action_item.action_name not in valid_project_actions:
                    errors.append(f"{action_label}: Unknown project action '{action_item.action_name}'")
                continue
            
            # Block-level validation
            if not action_item.block_id:
                errors.append(f"{action_label}: Missing block_id")
                continue
            
            block = self._block_repo.get(project_id, action_item.block_id)
            if not block:
                errors.append(f"{action_label}: Block '{action_item.block_name}' not found in project")
                continue
            
            quick_actions = get_quick_actions(block.type)
            action = next((a for a in quick_actions if a.name == action_item.action_name), None)
            if not action:
                errors.append(
                    f"{action_label}: Action '{action_item.action_name}' not available "
                    f"for block type '{block.type}'"
                )
        
        return errors
    
    # -- Hook Actions --
    
    def _get_hook_actions(self, song: SetlistSong, phase: str) -> List[Dict[str, Any]]:
        """
        Get pre- or post-song hook actions from song.action_overrides.
        
        Args:
            song: SetlistSong entity
            phase: "pre_actions" or "post_actions"
            
        Returns:
            List of action dicts, or empty list if none configured
        """
        overrides = song.action_overrides or {}
        actions = overrides.get(phase, [])
        if not isinstance(actions, list):
            return []
        return actions
    
    def _execute_hook_actions(
        self,
        project_id: str,
        hook_actions: List[Dict[str, Any]],
        setlist_context: Optional[Dict[str, Any]] = None,
        action_progress_callback: Optional[Callable] = None,
        phase_label: str = ""
    ) -> None:
        """
        Execute hook actions (pre- or post-song).
        
        Hook actions are lightweight dicts (not full ActionItem entities)
        with keys: action_type, block_id, block_name, action_name, action_args.
        """
        for idx, hook in enumerate(hook_actions):
            action_type = hook.get("action_type", "block")
            action_name = hook.get("action_name", "Unknown")
            block_id = hook.get("block_id")
            block_name = hook.get("block_name", "")
            action_args = hook.get("action_args", {})
            
            display_name = f"[{phase_label}] {block_name} -> {action_name}" if block_name else f"[{phase_label}] {action_name}"
            
            try:
                resolved_args = self._resolve_action_args(action_args, setlist_context)
                
                if action_type == "project":
                    self._execute_project_action(project_id, action_name, resolved_args)
                elif block_id:
                    block = self._block_repo.get(project_id, block_id)
                    if not block:
                        Log.warning(f"SetlistProcessingService: Hook block {block_id} not found, skipping")
                        continue
                    
                    quick_actions = get_quick_actions(block.type)
                    action = next((a for a in quick_actions if a.name == action_name), None)
                    if action:
                        if isinstance(resolved_args, dict):
                            action.handler(self._facade, block_id, **resolved_args)
                        else:
                            action.handler(self._facade, block_id, value=resolved_args)
                    else:
                        Log.warning(f"SetlistProcessingService: Hook action '{action_name}' not found for block type '{block.type}'")
                
                Log.info(f"SetlistProcessingService: {phase_label} hook completed: {display_name}")
            except Exception as e:
                Log.warning(f"SetlistProcessingService: {phase_label} hook failed: {display_name} - {e}")
    
    def _execute_project_action(
        self,
        project_id: str,
        action_name: str,
        resolved_args: Any
    ) -> None:
        """Execute a project-level action."""
        if action_name == "validate_project":
            self._facade.validate_project(project_id)
        elif action_name == "save_project":
            self._facade.save_project()
        elif action_name == "save_as":
            if isinstance(resolved_args, dict):
                save_directory = resolved_args.get("save_directory") or resolved_args.get("directory")
                name = resolved_args.get("name")
            elif isinstance(resolved_args, str):
                save_directory = resolved_args
                name = None
            else:
                return
            if save_directory:
                self._facade.save_project_as(save_directory, name=name)
        else:
            Log.warning(f"SetlistProcessingService: Unknown project action '{action_name}'")
    
    # -- Action Execution --
    
    def _execute_action_items(
        self,
        project_id: str,
        action_items: List,
        setlist_context: Optional[Dict[str, Any]] = None,
        action_progress_callback: Optional[Callable[[int, int, str, str], None]] = None
    ) -> None:
        """
        Execute all ActionItems sequentially in order.
        """
        from src.features.projects.domain import ActionItem
        
        total_actions = len(action_items)
        for idx, action_item in enumerate(action_items):
            action_name = action_item.action_name
            action_display_name = f"{action_item.block_name} -> {action_name}" if action_item.block_name else action_name
            
            if action_progress_callback:
                action_progress_callback(idx, total_actions, action_display_name, "running")
            
            # Project-level actions
            if action_item.action_type == "project":
                try:
                    resolved_args = self._resolve_action_args(action_item.action_args, setlist_context)
                    self._execute_project_action(project_id, action_item.action_name, resolved_args)
                    if action_progress_callback:
                        action_progress_callback(idx, total_actions, action_display_name, "completed")
                except Exception as e:
                    Log.warning(f"SetlistProcessingService: Project action '{action_name}' failed: {e}")
                    if action_progress_callback:
                        action_progress_callback(idx, total_actions, action_display_name, "failed")
                continue
            
            # Block-level actions
            if not action_item.block_id:
                Log.warning(f"SetlistProcessingService: Block action missing block_id, skipping")
                continue
            
            block = self._block_repo.get(project_id, action_item.block_id)
            if not block:
                Log.warning(f"SetlistProcessingService: Block {action_item.block_id} not found, skipping")
                continue
            
            quick_actions = get_quick_actions(block.type)
            action = next((a for a in quick_actions if a.name == action_item.action_name), None)
            
            if action:
                try:
                    resolved_args = self._resolve_action_args(action_item.action_args, setlist_context)
                    if isinstance(resolved_args, dict):
                        result = action.handler(self._facade, action_item.block_id, **resolved_args)
                    else:
                        result = action.handler(self._facade, action_item.block_id, value=resolved_args)
                    
                    if isinstance(result, dict) and result.get("success") is False:
                        Log.warning(
                            f"SetlistProcessingService: Action '{action_name}' returned error: "
                            f"{result.get('error', 'Unknown error')}"
                        )
                        if action_progress_callback:
                            action_progress_callback(idx, total_actions, action_display_name, "failed")
                    else:
                        if action_progress_callback:
                            action_progress_callback(idx, total_actions, action_display_name, "completed")
                except Exception as e:
                    Log.warning(f"SetlistProcessingService: Action '{action_name}' failed: {e}")
                    if action_progress_callback:
                        action_progress_callback(idx, total_actions, action_display_name, "failed")
            else:
                Log.warning(f"SetlistProcessingService: Action '{action_name}' not found for block type '{block.type}'")
                if action_progress_callback:
                    action_progress_callback(idx, total_actions, action_display_name, "failed")
    
    def _resolve_action_args(self, action_args: Any, setlist_context: Optional[Dict[str, Any]] = None) -> Any:
        """Resolve dynamic values in action arguments using setlist context."""
        if not setlist_context:
            return action_args
        
        if isinstance(action_args, str):
            resolved = action_args
            for key, value in setlist_context.items():
                placeholder = f"{{{key}}}"
                if placeholder in resolved:
                    resolved = resolved.replace(placeholder, str(value))
            return resolved
        elif isinstance(action_args, dict):
            return {key: self._resolve_action_args(value, setlist_context) for key, value in action_args.items()}
        elif isinstance(action_args, list):
            return [self._resolve_action_args(item, setlist_context) for item in action_args]
        else:
            return action_args
    
    # -- Cleanup --
    
    def _clear_all_project_data(self, project_id: str) -> None:
        """Clear all data items and block local state for a project."""
        deleted_count = self._data_item_repo.delete_by_project(project_id)
        if deleted_count > 0:
            Log.info(f"SetlistProcessingService: Cleared {deleted_count} data item(s)")
        
        blocks = self._block_repo.list_by_project(project_id)
        cleared_blocks = 0
        for block in blocks:
            try:
                self._block_local_state_repo.clear_inputs(block.id)
                cleared_blocks += 1
            except Exception as e:
                Log.warning(f"SetlistProcessingService: Failed to clear local state for block '{block.name}': {e}")
        
        event_bus = getattr(self._facade, 'event_bus', None)
        if event_bus:
            from src.application.events import BlockUpdated
            for block in blocks:
                event_bus.publish(BlockUpdated(
                    project_id=project_id,
                    data={
                        "id": block.id,
                        "name": block.name,
                        "type": block.type,
                        "data_items_cleared": True,
                        "local_state_cleared": True
                    }
                ))
    
    def _cleanup_blocks_after_execution(self, project_id: str) -> None:
        """Clean up all blocks and their libraries after execution."""
        blocks = self._block_repo.list_by_project(project_id)
        for block in blocks:
            processor = self._execution_engine.get_processor(block)
            if processor and hasattr(processor, 'cleanup'):
                try:
                    processor.cleanup(block)
                except Exception as e:
                    Log.warning(f"SetlistProcessingService: Failed to cleanup block {block.name}: {e}")
        
        import gc
        gc.collect()
    
    # -- Event Publishing --
    
    def _publish_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Publish a setlist event via the event bus."""
        event_bus = getattr(self._facade, 'event_bus', None)
        if not event_bus:
            return
        
        try:
            from src.application.events.events import (
                SetlistProcessingStarted,
                SetlistSongProcessing,
                SetlistSongCompleted,
                SetlistProcessingCompleted,
            )
            
            event_classes = {
                "SetlistProcessingStarted": SetlistProcessingStarted,
                "SetlistSongProcessing": SetlistSongProcessing,
                "SetlistSongCompleted": SetlistSongCompleted,
                "SetlistProcessingCompleted": SetlistProcessingCompleted,
            }
            
            event_cls = event_classes.get(event_name)
            if event_cls:
                event = event_cls(data=data)
                event_bus.publish(event)
        except Exception as e:
            Log.debug(f"SetlistProcessingService: Failed to publish event {event_name}: {e}")
