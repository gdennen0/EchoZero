"""
Setlist Processing Service

Handles song processing, action execution, pre/post hooks, and cleanup.
Extracted from SetlistService for single-responsibility.

The facade is passed as a method parameter (not stored as instance state)
to avoid the circular dependency that occurs when services are constructed
before the facade exists.
"""
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path

from src.features.setlists.domain import Setlist, SetlistSong, SongProcessingResult, SetlistProcessingResult
from src.features.setlists.domain import SetlistRepository, SetlistSongRepository
from src.features.blocks.domain import BlockRepository
from src.shared.domain.repositories import DataItemRepository
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository
from src.features.projects.application import SnapshotService, ProjectService
from src.features.execution.application import BlockExecutionEngine
from src.application.blocks.quick_actions import get_quick_actions
from src.shared.application.services.progress_context import get_progress_context
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


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

    The facade is accepted as a method parameter rather than stored as
    instance state, eliminating the bootstrap circular dependency.
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
    ):
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._block_repo = block_repo
        self._data_item_repo = data_item_repo
        self._block_local_state_repo = block_local_state_repo
        self._snapshot_service = snapshot_service
        self._project_service = project_service
        self._execution_engine = execution_engine
        self._progress = get_progress_context()
        Log.info("SetlistProcessingService: Initialized")

    def process_song(
        self,
        setlist_id: str,
        song_id: str,
        facade: "ApplicationFacade",
        progress_callback: Optional[Any] = None,
        song_progress_ctx: Optional[Any] = None,
    ) -> SongProcessingResult:
        """
        Process a single song through the project action pipeline.

        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier
            facade: ApplicationFacade for project context and action execution
            progress_callback: Optional progress callback (message, current, total)
            song_progress_ctx: Optional LevelContext from the parent setlist operation.
                If provided, action-level progress is written to the ProgressStore.

        Returns:
            SongProcessingResult with success/failure details
        """
        setlist = self._setlist_repo.get(setlist_id)
        if not setlist:
            Log.error(f"SetlistProcessingService: Setlist {setlist_id} not found")
            return SongProcessingResult(success=False, song_id=song_id, failed_step="Lookup", error_message=f"Setlist {setlist_id} not found")

        song = self._setlist_song_repo.get(song_id)
        if not song:
            Log.error(f"SetlistProcessingService: Song {song_id} not found")
            return SongProcessingResult(success=False, song_id=song_id, failed_step="Lookup", error_message=f"Song {song_id} not found")

        if song.setlist_id != setlist_id:
            Log.error(f"SetlistProcessingService: Song {song_id} does not belong to setlist {setlist_id}")
            return SongProcessingResult(success=False, song_id=song_id, failed_step="Lookup", error_message=f"Song {song_id} does not belong to setlist {setlist_id}")

        Log.info(f"SetlistProcessingService: Processing song {song.audio_path} (song_id: {song_id})")

        self._publish_event("SetlistSongProcessing", {
            "setlist_id": setlist_id,
            "song_id": song_id,
            "song_index": song.order_index,
            "song_name": Path(song.audio_path).name,
            "audio_path": song.audio_path,
        }, facade)

        owns_progress_ctx = song_progress_ctx is None
        _progress_cm = None
        _song_cm = None

        try:
            facade._force_in_process = True

            if owns_progress_ctx:
                song_name = Path(song.audio_path).name
                _progress_cm = self._progress.setlist_processing(
                    setlist_id, song_name, total_songs=1
                )
                op = _progress_cm.__enter__()
                _song_cm = op.song(song.id, song_name, audio_path=song.audio_path)
                song_progress_ctx = _song_cm.__enter__()
                # #region agent log
                import json as _dj1, time as _dt1; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj1.dumps({"timestamp":int(_dt1.time()*1000),"location":"setlist_processing_service.py:128","message":"Created standalone progress ctx for single song","data":{"song_id":song_id,"song_name":song_name,"setlist_id":setlist_id},"hypothesisId":"H_PROGRESS"})+'\n')
                # #endregion

            song.mark_processing()
            song.error_message = None
            self._setlist_song_repo.update(song)
            current_step = "Verify project"

            # 1. Verify project
            if progress_callback:
                progress_callback("Verifying project...", 0, 100)

            if not facade.current_project_id:
                raise Exception("No current project. Please open the project used to create this setlist.")

            if facade.current_project_id != setlist.project_id:
                raise Exception(
                    f"Current project ({facade.current_project_id}) does not match "
                    f"setlist's project ({setlist.project_id}). Please open the correct project."
                )

            project = self._project_service.load_project(setlist.project_id)
            if not project:
                raise Exception(f"Project not found: {setlist.project_id}")

            # 2. Find audio input block
            current_step = "Find audio input block"
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
                "set_audio_path": song.audio_path,
                "song_name": Path(song.audio_path).stem,
                "song_full_name": Path(song.audio_path).name,
                "song_index": song.order_index,
                "song_index_1": song.order_index + 1,
                "song_count": total_songs,
                "setlist_id": setlist.id,
                "setlist_name": setlist_name,
            }

            # 3. Execute pre-song hooks
            current_step = "Pre-song hooks"
            if progress_callback:
                progress_callback("Running pre-song hooks...", 15, 100)

            pre_actions = self._get_hook_actions(song, "pre_actions")
            if pre_actions:
                Log.info(f"SetlistProcessingService: Executing {len(pre_actions)} pre-song hook(s)")
                self._execute_hook_actions(
                    project.id, pre_actions, facade,
                    setlist_context=setlist_context,
                    phase_label="Pre"
                )

            # 4. Load and execute main ActionItems (from the active action set only)
            current_step = "Execute actions"
            if progress_callback:
                progress_callback("Applying actions...", 20, 100)

            action_set_id = self._get_active_action_set_id(project.id, facade)
            if action_set_id:
                action_items_result = facade.list_action_items(action_set_id)
            else:
                action_items_result = facade.list_action_items_by_project(project_id=project.id)

            if not action_items_result.success:
                raise Exception(f"Failed to load action items: {action_items_result.message}")

            action_items = action_items_result.data or []

            from src.features.projects.domain import ActionItem

            has_audio_action = False
            for action_item in action_items:
                if (action_item.block_id == audio_input_block.id and
                    action_item.action_name == "Set Audio File"):
                    has_audio_action = True
                    if not action_item.action_args or not isinstance(action_item.action_args, dict):
                        action_item.action_args = {}
                    action_item.action_args["file_path"] = "{song_audio_path}"
                    break

            if not has_audio_action:
                audio_action = ActionItem(
                    action_type="block",
                    action_name="Set Audio File",
                    action_description="Set audio file for setlist song",
                    block_id=audio_input_block.id,
                    block_name=audio_input_block.name,
                    action_args={"file_path": "{song_audio_path}"}
                )
                action_items.insert(0, audio_action)

            # Ensure audio input block is executed after "Set Audio File"
            # so it produces output data for downstream blocks
            set_audio_idx = next(
                (i for i, a in enumerate(action_items)
                 if a.block_id == audio_input_block.id and a.action_name == "Set Audio File"),
                0
            )
            has_audio_execute = any(
                a.block_id == audio_input_block.id and a.action_name == "Execute"
                for a in action_items
            )
            if not has_audio_execute:
                execute_action = ActionItem(
                    action_type="block",
                    action_name="Execute",
                    action_description="Execute audio input block",
                    block_id=audio_input_block.id,
                    block_name=audio_input_block.name,
                    action_args={}
                )
                action_items.insert(set_audio_idx + 1, execute_action)

            # #region agent log
            import json as _dj, time as _dt; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj.dumps({"timestamp":int(_dt.time()*1000),"location":"setlist_processing_service.py:action_list","message":"Final action list before execution","data":{"song_audio_path":song.audio_path,"action_set_id":action_set_id,"total_actions":len(action_items),"action_names":[(a.block_name or "")+"->"+a.action_name for a in action_items],"has_audio_execute":has_audio_execute,"audio_block_id":audio_input_block.id},"hypothesisId":"H1"})+'\n')
            # #endregion

            self._execute_action_items(
                project.id, action_items, facade,
                setlist_context=setlist_context,
                song_progress_ctx=song_progress_ctx
            )

            # 5. Execute post-song hooks
            current_step = "Post-song hooks"
            if progress_callback:
                progress_callback("Running post-song hooks...", 65, 100)

            post_actions = self._get_hook_actions(song, "post_actions")
            if post_actions:
                Log.info(f"SetlistProcessingService: Executing {len(post_actions)} post-song hook(s)")
                self._execute_hook_actions(
                    project.id, post_actions, facade,
                    setlist_context=setlist_context,
                    phase_label="Post"
                )

            # 6. Save snapshot
            current_step = "Save snapshot"
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
            current_step = "Clear data"
            if progress_callback:
                progress_callback("Clearing data for next song...", 85, 100)
            self._clear_all_project_data(project.id, facade)

            # 8. Cleanup
            current_step = "Cleanup"
            if progress_callback:
                progress_callback("Cleaning up...", 90, 100)
            self._cleanup_blocks_after_execution(project.id)

            song.mark_completed()
            self._setlist_song_repo.update(song)

            if progress_callback:
                progress_callback("Completed", 100, 100)

            self._publish_event("SetlistSongCompleted", {
                "setlist_id": setlist_id,
                "song_id": song_id,
                "song_index": song.order_index,
                "success": True,
            }, facade)

            # #region agent log
            import json as _dj, time as _dt; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj.dumps({"timestamp":int(_dt.time()*1000),"location":"setlist_processing_service.py:song_success","message":"Song processing succeeded","data":{"song_path":song.audio_path},"hypothesisId":"H1"})+'\n')
            # #endregion
            Log.info(f"SetlistProcessingService: Successfully processed song {song.audio_path}")
            return SongProcessingResult(success=True, song_id=song_id)

        except Exception as e:
            error_msg = f"[{current_step}] {e}"
            # #region agent log
            import json as _dj, time as _dt; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj.dumps({"timestamp":int(_dt.time()*1000),"location":"setlist_processing_service.py:song_failed","message":"Song processing failed","data":{"song_path":song.audio_path,"current_step":current_step,"error":str(e)[:500]},"hypothesisId":"H3"})+'\n')
            # #endregion
            Log.error(f"SetlistProcessingService: Failed to process song {song.audio_path}: {error_msg}")
            song.mark_failed(error_message=error_msg)
            self._setlist_song_repo.update(song)

            self._publish_event("SetlistSongCompleted", {
                "setlist_id": setlist_id,
                "song_id": song_id,
                "song_index": song.order_index,
                "success": False,
                "error_message": error_msg,
            }, facade)

            return SongProcessingResult(
                success=False,
                song_id=song_id,
                failed_step=current_step,
                error_message=str(e),
            )
        finally:
            facade._force_in_process = False
            if owns_progress_ctx:
                try:
                    if _song_cm:
                        _song_cm.__exit__(None, None, None)
                    if _progress_cm:
                        _progress_cm.__exit__(None, None, None)
                except Exception:
                    pass

    def process_setlist(
        self,
        setlist_id: str,
        facade: "ApplicationFacade",
        error_callback: Optional[Any] = None,
        cancel_check: Optional[Any] = None
    ) -> SetlistProcessingResult:
        """
        Process all songs in a setlist with error recovery.

        Progress is written to the ProgressStore. The UI reads it via polling.

        Args:
            setlist_id: Setlist identifier
            facade: ApplicationFacade for project context and action execution
            error_callback: Optional callback (song_path, error_message)
            cancel_check: Optional callable that returns True to cancel

        Returns:
            SetlistProcessingResult with per-song results
        """
        setlist = self._setlist_repo.get(setlist_id)
        if not setlist:
            Log.error(f"SetlistProcessingService: Setlist {setlist_id} not found")
            return SetlistProcessingResult()

        songs = self._setlist_song_repo.list_by_setlist(setlist_id)
        result = SetlistProcessingResult()
        total_songs = len(songs)

        setlist_name = "Setlist"
        if setlist.audio_folder_path:
            setlist_name = Path(setlist.audio_folder_path).name

        self._publish_event("SetlistProcessingStarted", {
            "setlist_id": setlist_id,
            "song_count": total_songs,
            "setlist_name": setlist_name,
        }, facade)

        with self._progress.setlist_processing(setlist_id, setlist_name, total_songs=total_songs) as op:
            for index, song in enumerate(songs):
                if cancel_check and cancel_check():
                    Log.info(f"SetlistProcessingService: Processing cancelled after {index}/{total_songs} songs")
                    break

                song_name = Path(song.audio_path).name

                try:
                    with op.song(song.id, song_name, audio_path=song.audio_path) as song_ctx:
                        song_ctx.update(message=f"Processing ({index + 1}/{total_songs})")

                        song_result = self.process_song(
                            setlist_id=setlist_id,
                            song_id=song.id,
                            facade=facade,
                            song_progress_ctx=song_ctx,
                        )
                        result.song_results.append(song_result)

                        if not song_result.success:
                            error_msg = song_result.error_message or "Processing failed"
                            song_ctx.update(message=f"Failed: {error_msg}")
                            if error_callback:
                                error_callback(song.audio_path, error_msg)
                            raise Exception(error_msg)
                        else:
                            song_ctx.update(message="Completed")

                except Exception:
                    if not any(r.song_id == song.id for r in result.song_results):
                        result.song_results.append(SongProcessingResult(
                            success=False, song_id=song.id,
                            failed_step="Unknown", error_message="Unexpected error"
                        ))
                    continue

        self._publish_event("SetlistProcessingCompleted", {
            "setlist_id": setlist_id,
            "results": {r.song_id: r.success for r in result.song_results},
            "total_songs": total_songs,
            "successful_songs": result.success_count,
            "failed_songs": result.failed_count,
        }, facade)

        Log.info(
            f"SetlistProcessingService: Processed {result.success_count}/{total_songs} song(s) successfully. "
            f"{result.failed_count} error(s) occurred."
        )

        if result.errors:
            Log.warning(f"SetlistProcessingService: Errors occurred during processing:")
            for err in result.errors:
                Log.warning(f"  - {err.song_id}: [{err.failed_step}] {err.error_message}")

        return result

    def validate_action_items(self, project_id: str, facade: "ApplicationFacade") -> List[str]:
        """
        Validate action items before processing begins.

        Args:
            project_id: Project identifier
            facade: ApplicationFacade for loading action items

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        action_set_id = self._get_active_action_set_id(project_id, facade)
        if action_set_id:
            action_items_result = facade.list_action_items(action_set_id)
        else:
            action_items_result = facade.list_action_items_by_project(project_id=project_id)

        if not action_items_result.success:
            errors.append(f"Failed to load action items: {action_items_result.message}")
            return errors

        action_items = action_items_result.data or []
        if not action_items:
            return errors

        for idx, action_item in enumerate(action_items):
            action_label = f"Action {idx + 1} ({action_item.action_name})"

            if action_item.action_type == "project":
                valid_project_actions = {"validate_project", "save_project", "save_as"}
                if action_item.action_name not in valid_project_actions:
                    errors.append(f"{action_label}: Unknown project action '{action_item.action_name}'")
                continue

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
        """Get pre- or post-song hook actions from song.action_overrides."""
        overrides = song.action_overrides or {}
        actions = overrides.get(phase, [])
        if not isinstance(actions, list):
            return []
        return actions

    def _execute_hook_actions(
        self,
        project_id: str,
        hook_actions: List[Dict[str, Any]],
        facade: "ApplicationFacade",
        setlist_context: Optional[Dict[str, Any]] = None,
        phase_label: str = ""
    ) -> None:
        """Execute hook actions (pre- or post-song)."""
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
                    self._execute_project_action(project_id, action_name, resolved_args, facade)
                elif block_id:
                    block = self._block_repo.get(project_id, block_id)
                    if not block:
                        Log.warning(f"SetlistProcessingService: Hook block {block_id} not found, skipping")
                        continue

                    quick_actions = get_quick_actions(block.type)
                    action = next((a for a in quick_actions if a.name == action_name), None)
                    if action:
                        if isinstance(resolved_args, dict):
                            action.handler(facade, block_id, **resolved_args)
                        else:
                            action.handler(facade, block_id, value=resolved_args)
                    else:
                        Log.warning(f"SetlistProcessingService: Hook action '{action_name}' not found for block type '{block.type}'")

                Log.info(f"SetlistProcessingService: {phase_label} hook completed: {display_name}")
            except Exception as e:
                Log.warning(f"SetlistProcessingService: {phase_label} hook failed: {display_name} - {e}")

    def _execute_project_action(
        self,
        project_id: str,
        action_name: str,
        resolved_args: Any,
        facade: "ApplicationFacade"
    ) -> None:
        """Execute a project-level action."""
        if action_name == "validate_project":
            facade.validate_project(project_id)
        elif action_name == "save_project":
            facade.save_project()
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
                facade.save_project_as(save_directory, name=name)
        else:
            Log.warning(f"SetlistProcessingService: Unknown project action '{action_name}'")

    # -- Action Execution --

    def _execute_action_items(
        self,
        project_id: str,
        action_items: List,
        facade: "ApplicationFacade",
        setlist_context: Optional[Dict[str, Any]] = None,
        song_progress_ctx: Optional[Any] = None
    ) -> None:
        """Execute all ActionItems sequentially. Progress is written to ProgressStore.

        Raises on the first failed Execute action, since downstream blocks
        depend on upstream outputs. Non-execute failures are collected and
        raised as a summary after all actions run.
        """
        from src.features.projects.domain import ActionItem

        total_actions = len(action_items)
        if song_progress_ctx:
            song_progress_ctx.set_total(total_actions)

        non_critical_failures: List[str] = []

        for idx, action_item in enumerate(action_items):
            action_name = action_item.action_name
            action_display_name = f"{action_item.block_name} -> {action_name}" if action_item.block_name else action_name
            is_execute = (action_name == "Execute")

            if song_progress_ctx:
                song_progress_ctx.update(current=idx, total=total_actions, message=f"{action_display_name}: running")

            # Project-level actions
            if action_item.action_type == "project":
                try:
                    resolved_args = self._resolve_action_args(action_item.action_args, setlist_context)
                    self._execute_project_action(project_id, action_item.action_name, resolved_args, facade)
                    if song_progress_ctx:
                        song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: completed")
                except Exception as e:
                    Log.warning(f"SetlistProcessingService: Project action '{action_name}' failed: {e}")
                    if song_progress_ctx:
                        song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: failed")
                    non_critical_failures.append(f"{action_display_name}: {e}")
                continue

            # Block-level actions
            if not action_item.block_id:
                Log.warning(f"SetlistProcessingService: Block action missing block_id, skipping")
                continue

            block = self._block_repo.get(project_id, action_item.block_id)
            if not block:
                msg = f"Block {action_item.block_id} not found"
                Log.warning(f"SetlistProcessingService: {msg}, skipping")
                if is_execute:
                    raise Exception(f"{action_display_name}: {msg}")
                non_critical_failures.append(f"{action_display_name}: {msg}")
                continue

            quick_actions = get_quick_actions(block.type)
            action = next((a for a in quick_actions if a.name == action_item.action_name), None)

            if not action:
                msg = f"Action '{action_name}' not found for block type '{block.type}'"
                Log.warning(f"SetlistProcessingService: {msg}")
                if song_progress_ctx:
                    song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: failed")
                non_critical_failures.append(f"{action_display_name}: {msg}")
                continue

            try:
                resolved_args = self._resolve_action_args(action_item.action_args, setlist_context)
                # #region agent log
                import json as _dj, time as _dt; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj.dumps({"timestamp":int(_dt.time()*1000),"location":"setlist_processing_service.py:action_exec","message":"Executing action","data":{"idx":idx,"action_display_name":action_display_name,"is_execute":is_execute,"resolved_args":str(resolved_args)[:200],"block_type":block.type},"hypothesisId":"H2"})+'\n')
                # #endregion
                if isinstance(resolved_args, dict):
                    result = action.handler(facade, action_item.block_id, **resolved_args)
                else:
                    result = action.handler(facade, action_item.block_id, value=resolved_args)

                # #region agent log
                import json as _dj, time as _dt; open('/Users/gdennen/Projects/EchoZero/.cursor/debug.log','a').write(_dj.dumps({"timestamp":int(_dt.time()*1000),"location":"setlist_processing_service.py:action_result","message":"Action result","data":{"idx":idx,"action_display_name":action_display_name,"is_execute":is_execute,"result_type":type(result).__name__,"result_summary":str(result)[:300] if result else "None"},"hypothesisId":"H1"})+'\n')
                # #endregion

                from src.application.api.result_types import CommandResult as _CR
                action_failed = False
                error_detail = ""
                if isinstance(result, _CR) and result.failed:
                    action_failed = True
                    error_detail = result.message or "; ".join(result.errors) or "Unknown error"
                elif isinstance(result, dict) and result.get("success") is False:
                    action_failed = True
                    error_detail = result.get("error", "Unknown error")

                if action_failed:
                    Log.warning(
                        f"SetlistProcessingService: Action '{action_name}' returned error: {error_detail}"
                    )
                    if song_progress_ctx:
                        song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: failed")
                    if is_execute:
                        raise Exception(f"{action_display_name} failed: {error_detail}")
                    non_critical_failures.append(f"{action_display_name}: {error_detail}")
                else:
                    if song_progress_ctx:
                        song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: completed")
            except Exception as e:
                if is_execute:
                    raise Exception(f"{action_display_name} failed: {e}") from e
                Log.warning(f"SetlistProcessingService: Action '{action_name}' failed: {e}")
                if song_progress_ctx:
                    song_progress_ctx.update(current=idx + 1, message=f"{action_display_name}: failed")
                non_critical_failures.append(f"{action_display_name}: {e}")

        if non_critical_failures:
            summary = "; ".join(non_critical_failures[:5])
            if len(non_critical_failures) > 5:
                summary += f" (+{len(non_critical_failures) - 5} more)"
            raise Exception(f"{len(non_critical_failures)} action(s) failed: {summary}")

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

    def _get_active_action_set_id(self, project_id: str, facade: "ApplicationFacade") -> Optional[str]:
        """Return the ID of the first action set for this project, or None."""
        repo = getattr(facade, "action_set_repo", None)
        if not repo:
            return None
        try:
            sets = repo.list_by_project(project_id)
            if sets:
                return sets[0].id
        except Exception as e:
            Log.warning(f"SetlistProcessingService: Could not resolve action set: {e}")
        return None

    # -- Cleanup --

    def _clear_all_project_data(self, project_id: str, facade: "ApplicationFacade") -> None:
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

        event_bus = getattr(facade, 'event_bus', None)
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

    def _publish_event(self, event_name: str, data: Dict[str, Any], facade: "ApplicationFacade") -> None:
        """Publish a setlist event via the event bus."""
        event_bus = getattr(facade, 'event_bus', None)
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
