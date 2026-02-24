"""
Setlist Snapshot Service

Handles song switching via snapshot save/restore.
Extracted from SetlistService for single-responsibility.

The facade is passed as a method parameter (not stored as instance state)
to avoid the circular dependency that occurs when services are constructed
before the facade exists.
"""
from typing import Optional, Tuple, TYPE_CHECKING, Dict, Any, Set, List
import threading

from src.features.setlists.domain import SetlistRepository, SetlistSongRepository
from src.shared.domain.entities import DataStateSnapshot
from src.features.projects.application import SnapshotService, ProjectService
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class SetlistSnapshotService:
    """
    Handles song switching by saving and restoring data state snapshots.

    Thread-safe: uses a lock to prevent concurrent switches.
    """

    def __init__(
        self,
        setlist_repo: SetlistRepository,
        setlist_song_repo: SetlistSongRepository,
        snapshot_service: SnapshotService,
        project_service: ProjectService,
    ):
        self._setlist_repo = setlist_repo
        self._setlist_song_repo = setlist_song_repo
        self._snapshot_service = snapshot_service
        self._project_service = project_service
        self._switching_lock = threading.Lock()
        self._is_switching = False
        self._active_song_id: Optional[str] = None
        Log.info("SetlistSnapshotService: Initialized")

    @property
    def is_switching(self) -> bool:
        """Whether a song switch is currently in progress."""
        return self._is_switching

    def _effective_state_scope(self, block) -> str:
        """
        Resolve effective state scope for a block.

        Defaults to per-song when metadata is missing. ShowManager is always per-song.
        """
        if getattr(block, "type", "") == "ShowManager":
            return "per_song"
        metadata = getattr(block, "metadata", {}) or {}
        scope = str(metadata.get("state_scope", "per_song")).strip().lower()
        if scope not in ("per_song", "global"):
            return "per_song"
        return scope

    def _resolve_song_relevant_block_ids(
        self,
        facade: "ApplicationFacade",
        snapshot: DataStateSnapshot
    ) -> Set[str]:
        """
        Resolve song-relevant blocks from snapshot payload.

        Approach A contract:
        - all per-song blocks are relevant by default
        - payload blocks are relevant (even if global) so saved snapshots remain replayable
        """
        relevant: Set[str] = set()
        blocks_result = facade.list_blocks()
        if blocks_result.success and blocks_result.data:
            for block in blocks_result.data:
                if self._effective_state_scope(block) == "per_song":
                    relevant.add(block.id)

        # Snapshot payload blocks are always relevant to replay that snapshot.
        relevant.update(snapshot.block_local_state.keys())
        relevant.update(snapshot.block_settings_overrides.keys())
        for item_data in snapshot.data_items:
            if isinstance(item_data, dict):
                block_id = item_data.get("block_id")
                if block_id:
                    relevant.add(block_id)
        return relevant

    def _expand_with_upstream_dependencies(
        self,
        facade: "ApplicationFacade",
        block_ids: Set[str],
        payload_block_ids: Set[str]
    ) -> Set[str]:
        """
        Expand relevant blocks by traversing upstream dependencies through connections.

        Global blocks are excluded from expansion unless explicitly present in snapshot payload.
        """
        connection_repo = getattr(facade, "connection_repo", None)
        if not connection_repo:
            return set(block_ids)

        block_scope_by_id: Dict[str, str] = {}
        blocks_result = facade.list_blocks()
        if blocks_result.success and blocks_result.data:
            for block in blocks_result.data:
                block_scope_by_id[block.id] = self._effective_state_scope(block)

        expanded = set(block_ids)
        queue = list(block_ids)
        while queue:
            block_id = queue.pop()
            try:
                connections = connection_repo.list_by_block(block_id) or []
            except Exception:
                continue
            for conn in connections:
                source_block_id = getattr(conn, "source_block_id", None)
                target_block_id = getattr(conn, "target_block_id", None)
                # Include upstream blocks feeding relevant targets
                if target_block_id == block_id and source_block_id and source_block_id not in expanded:
                    source_scope = block_scope_by_id.get(source_block_id, "per_song")
                    if source_scope == "global" and source_block_id not in payload_block_ids:
                        continue
                    expanded.add(source_block_id)
                    queue.append(source_block_id)
        return expanded

    def _resolve_relevant_show_manager_ids(self, facade: "ApplicationFacade", relevant_block_ids: Set[str]) -> Set[str]:
        """Get ShowManager block IDs within the relevant scope."""
        show_manager_ids: Set[str] = set()
        blocks_result = facade.list_blocks()
        if not blocks_result.success:
            return show_manager_ids
        for block in blocks_result.data:
            if block.type == "ShowManager" and block.id in relevant_block_ids:
                show_manager_ids.add(block.id)
        return show_manager_ids

    def _prepare_show_manager_switch(
        self,
        facade: "ApplicationFacade",
        show_manager_ids: Set[str]
    ) -> List[Dict[str, str]]:
        """
        Unhook/pause ShowManager sync before song switch restore.
        Returns a list of per-block errors for fail-loud reporting.
        """
        errors: List[Dict[str, str]] = []
        for block_id in sorted(show_manager_ids):
            try:
                ssm = facade.sync_system_manager(block_id)
                if not ssm:
                    errors.append({"block_id": block_id, "error": "SyncSystemManager unavailable"})
                    continue
                ssm.prepare_for_song_switch()
            except Exception as e:
                errors.append({"block_id": block_id, "error": str(e)})
        return errors

    def _restore_show_manager_after_switch(
        self,
        facade: "ApplicationFacade",
        show_manager_ids: Set[str]
    ) -> List[Dict[str, str]]:
        """
        Rehook/resume ShowManager sync after restore.
        Returns a list of per-block errors for degraded-mode reporting.
        """
        errors: List[Dict[str, str]] = []
        for block_id in sorted(show_manager_ids):
            try:
                ssm = facade.sync_system_manager(block_id)
                if not ssm:
                    errors.append({"block_id": block_id, "error": "SyncSystemManager unavailable"})
                    continue
                ssm.restore_after_song_switch()
            except Exception as e:
                errors.append({"block_id": block_id, "error": str(e)})
        return errors

    def _backfill_empty_local_state(
        self,
        facade: "ApplicationFacade",
        snapshot: DataStateSnapshot,
        relevant_block_ids: Set[str],
    ) -> None:
        """
        After snapshot restore, pull inputs for relevant blocks whose local
        state was not in the snapshot.

        This covers terminal/UI blocks (e.g. Editor) that have upstream
        connections but were not part of the original processing pipeline.
        Their upstream data is now restored; pulling inputs writes the
        correct references into local state so the UI can render them.
        """
        if not hasattr(facade, "pull_block_inputs_overwrite"):
            return
        connection_repo = getattr(facade, "connection_repo", None)
        if not connection_repo:
            return

        snapshot_local_state_blocks = set(snapshot.block_local_state.keys())

        for block_id in sorted(relevant_block_ids):
            if block_id in snapshot_local_state_blocks:
                continue
            try:
                connections = connection_repo.list_by_block(block_id) or []
            except Exception:
                continue
            has_inbound = any(
                getattr(c, "target_block_id", None) == block_id for c in connections
            )
            if not has_inbound:
                continue
            try:
                result = facade.pull_block_inputs_overwrite(block_id)
                if getattr(result, "success", False):
                    Log.info(f"SetlistSnapshotService: Backfilled local state for block {block_id}")
                else:
                    Log.warning(
                        f"SetlistSnapshotService: Failed to backfill block {block_id}: "
                        f"{getattr(result, 'message', '')}"
                    )
            except Exception as e:
                Log.warning(f"SetlistSnapshotService: Backfill error for block {block_id}: {e}")

    def _save_outgoing_song_snapshot(
        self,
        outgoing_song_id: str,
        project_id: str,
        facade: "ApplicationFacade",
    ) -> Optional[str]:
        """
        Save a full snapshot for the outgoing song before switching away.

        Reuses the same SnapshotService.save_snapshot path that setlist
        processing uses, so the saved state is identical in format.

        Returns None on success, or an error message string on failure.
        The caller should log the error but NOT block the switch.
        """
        try:
            project = self._project_service.get_project(project_id)
            if not project:
                return f"Could not load project {project_id} for outgoing save"

            snapshot = self._snapshot_service.save_snapshot(
                project_id=project_id,
                song_id=outgoing_song_id,
                block_settings_overrides=None,
            )
            snapshot_dict = snapshot.to_dict()

            # Cache-only save: update in-memory snapshot store so the next
            # get_snapshot() returns current data without touching the
            # 138 MB project ZIP file (which takes ~22 s to rewrite).
            # Dirty snapshots are flushed to disk on project save / app close.
            self._project_service.cache_snapshot(outgoing_song_id, snapshot_dict)

            Log.info(
                f"SetlistSnapshotService: Auto-saved outgoing song {outgoing_song_id} "
                f"({len(snapshot.data_items)} data items, "
                f"{len(snapshot.block_local_state)} block states)"
            )
            return None
        except Exception as e:
            error_msg = f"Failed to auto-save outgoing song {outgoing_song_id}: {e}"
            Log.error(f"SetlistSnapshotService: {error_msg}")
            return error_msg

    def switch_active_song(
        self,
        setlist_id: str,
        song_id: str,
        facade: "ApplicationFacade"
    ) -> Tuple[bool, Optional[str]]:
        """
        Switch active song by auto-saving the outgoing song then restoring
        the target song's snapshot.

        Args:
            setlist_id: Setlist identifier
            song_id: Song identifier to switch to
            facade: ApplicationFacade for project context

        Returns:
            (success, error_message_or_degraded_warning)
        """
        if not self._switching_lock.acquire(blocking=False):
            return False, "Another song switch is already in progress. Please wait."

        try:
            self._is_switching = True
            degraded_parts: List[str] = []

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

            if not facade.current_project_id:
                return False, "No current project. Please open the project used to create this setlist."

            if facade.current_project_id != setlist.project_id:
                return False, (
                    f"Current project ({facade.current_project_id}) does not match "
                    f"setlist's project ({setlist.project_id}). Please open the correct project."
                )

            project = self._project_service.get_project(setlist.project_id)
            if not project:
                return False, f"Project not found: {setlist.project_id}"

            # --- Auto-save outgoing song ---
            outgoing_id = self._active_song_id
            if outgoing_id is None:
                outgoing_id = (getattr(setlist, "metadata", None) or {}).get("active_song_id")

            if outgoing_id and outgoing_id != song_id:
                save_error = self._save_outgoing_song_snapshot(
                    outgoing_song_id=outgoing_id,
                    project_id=project.id,
                    facade=facade,
                )
                if save_error:
                    degraded_parts.append(f"Auto-save outgoing song failed: {save_error}")

            # --- Load target snapshot ---
            snapshot_data = self._project_service.get_snapshot(song_id, project)
            if not snapshot_data:
                return False, "Song has not been processed yet. Please process the song first."

            try:
                snapshot = DataStateSnapshot.from_dict(snapshot_data)
            except Exception as e:
                return False, f"Failed to parse snapshot: {e}"

            event_bus = getattr(facade, 'event_bus', None)
            if not event_bus:
                Log.warning("SetlistSnapshotService: No event_bus available, UI may not refresh correctly")

            from src.utils.paths import get_project_workspace_dir
            project_dir = get_project_workspace_dir(project.id) if project.id else None
            payload_block_ids = set(snapshot.block_local_state.keys())
            payload_block_ids.update(snapshot.block_settings_overrides.keys())
            payload_block_ids.update(
                item_data.get("block_id")
                for item_data in snapshot.data_items
                if isinstance(item_data, dict) and item_data.get("block_id")
            )
            relevant_block_ids = self._resolve_song_relevant_block_ids(facade, snapshot)
            relevant_block_ids = self._expand_with_upstream_dependencies(
                facade,
                relevant_block_ids,
                payload_block_ids
            )
            relevant_show_manager_ids = self._resolve_relevant_show_manager_ids(facade, relevant_block_ids)

            # --- ShowManager teardown ---
            pre_switch_show_manager_errors = self._prepare_show_manager_switch(
                facade, relevant_show_manager_ids
            )

            # --- Restore target snapshot ---
            try:
                Log.info(
                    f"SetlistSnapshotService: Restoring snapshot for song {snapshot.song_id} "
                    f"({len(snapshot.data_items)} data items, {len(snapshot.block_local_state)} block states)"
                )
                restore_report = self._snapshot_service.restore_snapshot(
                    project_id=project.id,
                    snapshot=snapshot,
                    project_dir=project_dir,
                    event_bus=event_bus,
                    progress_callback=None,
                    relevant_block_ids=relevant_block_ids
                )
                Log.info(f"SetlistSnapshotService: Successfully restored snapshot for song {snapshot.song_id}")

                self._backfill_empty_local_state(facade, snapshot, relevant_block_ids)
            except Exception as e:
                error_msg = str(e)
                Log.error(f"SetlistSnapshotService: Failed to restore snapshot: {error_msg}")
                import traceback
                Log.error(traceback.format_exc())
                return False, f"Failed to restore snapshot: {error_msg}"

            # --- ShowManager rebuild ---
            post_switch_show_manager_errors = self._restore_show_manager_after_switch(
                facade, relevant_show_manager_ids
            )

            # --- Collect degraded-mode issues ---
            restore_failures = []
            if isinstance(restore_report, dict):
                restore_failures = restore_report.get("failed_blocks", []) or []

            if restore_failures:
                degraded_parts.append(f"{len(restore_failures)} block restore failure(s)")
            if pre_switch_show_manager_errors:
                degraded_parts.append(f"{len(pre_switch_show_manager_errors)} ShowManager prepare failure(s)")
            if post_switch_show_manager_errors:
                degraded_parts.append(f"{len(post_switch_show_manager_errors)} ShowManager rehook failure(s)")

            degraded_message: Optional[str] = None
            if degraded_parts:
                degraded_message = "Song switch completed with issues: " + "; ".join(degraded_parts) + ". See logs for details."
                Log.error(f"SetlistSnapshotService: {degraded_message}")
                for failure in restore_failures:
                    Log.error(
                        f"SetlistSnapshotService: Restore failure "
                        f"block_id={failure.get('block_id')} error={failure.get('error')}"
                    )
                for failure in pre_switch_show_manager_errors:
                    Log.error(
                        f"SetlistSnapshotService: ShowManager prepare failure "
                        f"block_id={failure.get('block_id')} error={failure.get('error')}"
                    )
                for failure in post_switch_show_manager_errors:
                    Log.error(
                        f"SetlistSnapshotService: ShowManager rehook failure "
                        f"block_id={failure.get('block_id')} error={failure.get('error')}"
                    )

            self._active_song_id = song_id
            Log.info(f"SetlistSnapshotService: Successfully switched to song {song.audio_path}")
            return True, degraded_message

        except Exception as e:
            error_msg = f"Unexpected error during switch: {e}"
            Log.error(f"SetlistSnapshotService: {error_msg}")
            import traceback
            Log.error(traceback.format_exc())
            return False, error_msg

        finally:
            self._is_switching = False
            self._switching_lock.release()
