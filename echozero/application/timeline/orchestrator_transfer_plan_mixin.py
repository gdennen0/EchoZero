"""Transfer plan helpers for the timeline orchestrator.
Exists to isolate batch transfer plan preview, apply, reset, and row-building flows from intent routing.
Connects manual push/pull plan state to typed preview/apply execution across the canonical orchestrator boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any, Protocol, cast

from echozero.application.session.models import (
    BatchTransferPlanRowState,
    BatchTransferPlanState,
    ManualPullEventOption,
    ManualPullTrackOption,
    ManualPushTrackOption,
    Session,
)
from echozero.application.session.service import SessionService
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, TakeId
from echozero.application.sync.diff_service import SyncDiffService
from echozero.application.timeline.models import Event, EventRef, Layer, Timeline
from echozero.application.timeline.orchestrator_transfer_lookup_mixin import (
    _PULL_TARGET_CREATE_NEW_LAYER_ID,
    _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
    _PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
)

_KEEP_TRANSFER_PLAN_ISSUE = object()


class _TransferPlanHost(Protocol):
    diff_service: SyncDiffService
    session_service: SessionService
    sync_service: Any

    @staticmethod
    def _main_take(layer: Layer) -> Any: ...

    def _apply_manual_pull_import(
        self,
        *,
        target_layer: Layer,
        source_track: ManualPullTrackOption,
        selected_events: list[ManualPullEventOption],
        import_mode: str,
    ) -> tuple[TakeId, list[EventId]]: ...

    def _load_manual_pull_event_options(
        self, source_track_coord: str
    ) -> list[ManualPullEventOption]: ...

    def _load_manual_pull_track_options(self) -> list[ManualPullTrackOption]: ...

    def _load_manual_push_track_options(self) -> list[ManualPushTrackOption]: ...

    @staticmethod
    def _manual_pull_selected_events_by_ids(
        available_events: list[ManualPullEventOption],
        selected_ids: list[str],
        *,
        action_name: str,
    ) -> list[ManualPullEventOption]: ...

    @staticmethod
    def _manual_pull_track_by_coord(
        available_tracks: list[ManualPullTrackOption],
        source_track_coord: str,
        action_name: str,
    ) -> ManualPullTrackOption: ...

    @staticmethod
    def _manual_push_track_by_coord(
        available_tracks: list[ManualPushTrackOption],
        target_track_coord: str,
    ) -> ManualPushTrackOption: ...

    def _resolve_event_refs_by_ids(
        self,
        timeline: Timeline,
        event_ids: list[EventId],
        *,
        preferred_layer_ids: list[str] | None = None,
        preferred_take_id: str | None = None,
    ) -> list[EventRef]: ...

    def _resolve_manual_pull_target_layer(
        self,
        timeline: Timeline,
        *,
        target_layer_id: str,
        source_track: ManualPullTrackOption,
    ) -> Layer: ...

    def _selected_event_records_by_layer(self, timeline: Timeline) -> list[tuple[Layer, list[Any]]]: ...

    def _selected_events_by_ids(self, timeline: Timeline, event_ids: list[EventId]) -> list[Event]: ...

    def _set_selected_event_refs(self, timeline: Timeline, event_refs: list[EventRef]) -> None: ...


class TimelineOrchestratorTransferPlanMixin:
    @staticmethod
    def _manual_pull_import_mode_for_target_layer(
        timeline: Timeline,
        target_layer_id: str | None,
    ) -> str:
        if target_layer_id in {
            _PULL_TARGET_CREATE_NEW_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
        }:
            return "main"
        if target_layer_id is not None:
            for layer in timeline.layers:
                if layer.id == target_layer_id and layer.kind is LayerKind.SECTION:
                    return "main"
        return "new_take"

    def _require_active_transfer_plan(
        self, session: Session, plan_id: str, *, action_name: str
    ) -> BatchTransferPlanState:
        plan = session.batch_transfer_plan
        if plan is None:
            raise ValueError(f"{action_name} requires an active batch transfer plan")
        if plan.plan_id != plan_id:
            raise ValueError(
                f"{action_name} plan_id does not match active batch transfer plan: "
                f"expected {plan.plan_id}, got {plan_id}"
            )
        return plan

    def _preview_transfer_plan(
        self, timeline: Timeline, session: Session, plan: BatchTransferPlanState
    ) -> None:
        preview_rows: list[BatchTransferPlanRowState] = []
        for row in plan.rows:
            if row.status == "blocked":
                preview_rows.append(self._copy_plan_row(row))
                continue
            if row.direction == "push":
                preview_rows.append(self._preview_push_plan_row(timeline, row))
                continue
            if row.direction == "pull":
                preview_rows.append(self._preview_pull_plan_row(row))
                continue
            preview_rows.append(
                self._copy_plan_row(row, issue=f"Unsupported transfer direction: {row.direction}")
            )

        plan.rows = preview_rows
        self._refresh_plan_counters(plan)

    def _apply_transfer_plan(
        self, timeline: Timeline, session: Session, plan: BatchTransferPlanState
    ) -> None:
        applied_rows: list[BatchTransferPlanRowState] = []
        stop_execution = False
        for row in plan.rows:
            if row.status == "blocked":
                applied_rows.append(self._copy_plan_row(row))
                continue
            if stop_execution:
                applied_rows.append(self._copy_plan_row(row))
                continue
            if row.status != "ready":
                applied_rows.append(self._copy_plan_row(row))
                continue
            try:
                if row.direction == "pull":
                    applied_row = self._apply_pull_plan_row(timeline, row)
                elif row.direction == "push":
                    applied_row = self._apply_push_plan_row(timeline, row)
                else:
                    applied_row = self._copy_plan_row(
                        row,
                        status="failed",
                        issue=f"Unsupported transfer direction: {row.direction}",
                    )
                applied_rows.append(applied_row)
                if applied_row.status == "failed":
                    stop_execution = True
            except Exception as exc:
                applied_rows.append(
                    self._copy_plan_row(
                        row,
                        status="failed",
                        issue=self._deterministic_issue_text(exc),
                    )
                )
                stop_execution = True

        plan.rows = applied_rows
        self._refresh_plan_counters(plan)
        self._clear_plan_diff_gates(session)

    def _cancel_transfer_plan(self, session: Session, plan: BatchTransferPlanState) -> None:
        if plan.operation_type in {"push", "mixed"}:
            self._reset_manual_push_flow(session)
        if plan.operation_type in {"pull", "mixed"}:
            self._reset_manual_pull_flow(session)
        session.batch_transfer_plan = None

    def _preview_push_plan_row(
        self, timeline: Timeline, row: BatchTransferPlanRowState
    ) -> BatchTransferPlanRowState:
        host = cast(_TransferPlanHost, self)
        target_track = host._manual_push_track_by_coord(
            host._load_manual_push_track_options(),
            row.target_track_coord or "",
        )
        selected_events = host._selected_events_by_ids(timeline, list(row.selected_event_ids))
        host.diff_service.build_push_preview_rows(
            selected_events=selected_events,
            target_track_name=target_track.name,
            target_track_coord=target_track.coord,
        )
        return self._copy_plan_row(row)

    def _preview_pull_plan_row(self, row: BatchTransferPlanRowState) -> BatchTransferPlanRowState:
        host = cast(_TransferPlanHost, self)
        if row.target_layer_id is None:
            return self._copy_plan_row(row)
        selected_events = host._manual_pull_selected_events_by_ids(
            available_events=host._load_manual_pull_event_options(row.source_track_coord or ""),
            selected_ids=list(row.selected_ma3_event_ids),
            action_name="PreviewTransferPlan",
        )
        host.diff_service.build_pull_preview_rows(
            selected_events=selected_events,
            target_layer_name=str(row.target_label or row.target_layer_id),
        )
        return self._copy_plan_row(row)

    def _apply_pull_plan_row(
        self, timeline: Timeline, row: BatchTransferPlanRowState
    ) -> BatchTransferPlanRowState:
        host = cast(_TransferPlanHost, self)
        if row.target_layer_id is None:
            return self._copy_plan_row(
                row,
                status="failed",
                issue="Transfer plan row is missing a target layer",
            )
        source_track = host._manual_pull_track_by_coord(
            host._load_manual_pull_track_options(),
            row.source_track_coord or "",
            action_name="ApplyTransferPlan",
        )
        target_layer = host._resolve_manual_pull_target_layer(
            timeline,
            target_layer_id=row.target_layer_id,
            source_track=source_track,
        )
        selected_events = host._manual_pull_selected_events_by_ids(
            available_events=host._load_manual_pull_event_options(source_track.coord),
            selected_ids=list(row.selected_ma3_event_ids),
            action_name="ApplyTransferPlan",
        )
        selected_take_id, selected_event_ids = host._apply_manual_pull_import(
            target_layer=target_layer,
            source_track=source_track,
            selected_events=selected_events,
            import_mode=row.import_mode,
        )

        timeline.selection.selected_layer_id = target_layer.id
        timeline.selection.selected_layer_ids = [target_layer.id]
        timeline.selection.selected_take_id = selected_take_id
        host._set_selected_event_refs(
            timeline,
            host._resolve_event_refs_by_ids(
                timeline,
                selected_event_ids,
                preferred_layer_ids=[target_layer.id],
                preferred_take_id=selected_take_id,
            ),
        )

        session = host.session_service.get_session()
        session.manual_pull_flow.target_layer_id_by_source_track[source_track.coord] = target_layer.id
        active_source_coord = (
            session.manual_pull_flow.active_source_track_coord
            or session.manual_pull_flow.source_track_coord
        )
        if active_source_coord == source_track.coord:
            session.manual_pull_flow.target_layer_id = target_layer.id
        refresh_targets = getattr(host, "_refresh_manual_pull_target_options", None)
        if callable(refresh_targets):
            refresh_targets(timeline, session)

        applied_row = self._copy_plan_row(row, status="applied", issue=None)
        applied_row.target_layer_id = target_layer.id
        applied_row.target_label = target_layer.name
        return applied_row

    def _apply_push_plan_row(
        self, timeline: Timeline, row: BatchTransferPlanRowState
    ) -> BatchTransferPlanRowState:
        host = cast(_TransferPlanHost, self)
        selected_events = host._selected_events_by_ids(timeline, list(row.selected_event_ids))
        if not selected_events:
            return self._copy_plan_row(
                row,
                status="failed",
                issue="No main-take events selected for push",
            )
        apply_push = getattr(host.sync_service, "apply_push_transfer", None)
        if callable(apply_push):
            self._invoke_push_apply(
                apply_push,
                target_track_coord=row.target_track_coord,
                selected_events=selected_events,
            )
            return self._copy_plan_row(row, status="applied", issue=None)
        execute_push = getattr(host.sync_service, "execute_push_transfer", None)
        if callable(execute_push):
            self._invoke_push_apply(
                execute_push,
                target_track_coord=row.target_track_coord,
                selected_events=selected_events,
            )
            return self._copy_plan_row(row, status="applied", issue=None)
        return self._copy_plan_row(
            row,
            status="failed",
            issue="Push execution endpoint unavailable",
        )

    def _invoke_push_apply(
        self,
        callback: Callable[..., object],
        *,
        target_track_coord: str | None,
        selected_events: list[Event],
    ) -> None:
        host = cast(_TransferPlanHost, self)
        kwargs: dict[str, object] = {
            "target_track_coord": target_track_coord,
            "selected_events": selected_events,
        }
        transfer_mode = host.session_service.get_session().manual_push_flow.transfer_mode
        try:
            parameters: Mapping[str, inspect.Parameter] | None = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            parameters = None
        if parameters is not None and "transfer_mode" in parameters:
            kwargs["transfer_mode"] = transfer_mode
        elif parameters is not None and "mode" in parameters:
            kwargs["mode"] = transfer_mode
        callback(**kwargs)

    @staticmethod
    def _copy_plan_row(
        row: BatchTransferPlanRowState,
        *,
        status: str | None = None,
        issue: str | None | object = _KEEP_TRANSFER_PLAN_ISSUE,
    ) -> BatchTransferPlanRowState:
        copied = BatchTransferPlanRowState(
            row_id=row.row_id,
            direction=row.direction,
            source_label=row.source_label,
            target_label=row.target_label,
            source_layer_id=row.source_layer_id,
            source_track_coord=row.source_track_coord,
            target_track_coord=row.target_track_coord,
            target_layer_id=row.target_layer_id,
            import_mode=row.import_mode,
            selected_event_ids=list(row.selected_event_ids),
            selected_ma3_event_ids=list(row.selected_ma3_event_ids),
            selected_count=row.selected_count,
            status=row.status if status is None else status,
            issue=row.issue,
        )
        if issue is not _KEEP_TRANSFER_PLAN_ISSUE:
            copied.issue = cast(str | None, issue)
        return copied

    @staticmethod
    def _refresh_plan_counters(plan: BatchTransferPlanState) -> None:
        plan.draft_count = sum(1 for row in plan.rows if row.status == "draft")
        plan.ready_count = sum(1 for row in plan.rows if row.status == "ready")
        plan.blocked_count = sum(1 for row in plan.rows if row.status == "blocked")
        plan.applied_count = sum(1 for row in plan.rows if row.status == "applied")
        plan.failed_count = sum(1 for row in plan.rows if row.status == "failed")

    @staticmethod
    def _deterministic_issue_text(exc: Exception) -> str:
        message = str(exc).strip()
        return message or exc.__class__.__name__

    @staticmethod
    def _plan_counters_locked(plan: BatchTransferPlanState | None) -> bool:
        if plan is None:
            return False
        return (plan.applied_count + plan.failed_count) > 0

    @staticmethod
    def _clear_plan_diff_gates(session: Session) -> None:
        session.manual_push_flow.diff_gate_open = False
        session.manual_push_flow.diff_preview = None
        session.manual_pull_flow.diff_gate_open = False
        session.manual_pull_flow.diff_preview = None

    @staticmethod
    def _reset_manual_push_flow(session: Session) -> None:
        session.manual_push_flow.dialog_open = False
        session.manual_push_flow.push_mode_active = False
        session.manual_push_flow.selected_event_ids = []
        session.manual_push_flow.available_tracks = []
        session.manual_push_flow.target_track_coord = None
        session.manual_push_flow.transfer_mode = "merge"
        session.manual_push_flow.diff_gate_open = False
        session.manual_push_flow.diff_preview = None

    @staticmethod
    def _reset_manual_pull_flow(session: Session) -> None:
        session.manual_pull_flow.dialog_open = False
        session.manual_pull_flow.workspace_active = False
        session.manual_pull_flow.available_tracks = []
        session.manual_pull_flow.selected_source_track_coords = []
        session.manual_pull_flow.active_source_track_coord = None
        session.manual_pull_flow.source_track_coord = None
        session.manual_pull_flow.available_events = []
        session.manual_pull_flow.selected_ma3_event_ids = []
        session.manual_pull_flow.selected_ma3_event_ids_by_track = {}
        session.manual_pull_flow.import_mode = "new_take"
        session.manual_pull_flow.import_mode_by_source_track = {}
        session.manual_pull_flow.available_target_layers = []
        session.manual_pull_flow.target_layer_id = None
        session.manual_pull_flow.target_layer_id_by_source_track = {}
        session.manual_pull_flow.diff_gate_open = False
        session.manual_pull_flow.diff_preview = None

    def _rebuild_push_transfer_plan(self, timeline: Timeline, session: Session) -> None:
        rows = self._build_push_transfer_plan_rows(timeline)
        if not rows:
            session.batch_transfer_plan = None
            return

        ready_count = sum(1 for row in rows if row.status == "ready")
        blocked_count = sum(1 for row in rows if row.status == "blocked")
        session.batch_transfer_plan = BatchTransferPlanState(
            plan_id=f"push:{timeline.id}",
            operation_type="push",
            rows=rows,
            ready_count=ready_count,
            blocked_count=blocked_count,
        )

    def _build_push_transfer_plan_rows(
        self, timeline: Timeline
    ) -> list[BatchTransferPlanRowState]:
        host = cast(_TransferPlanHost, self)
        grouped_records = host._selected_event_records_by_layer(timeline)
        if not grouped_records:
            return []

        available_tracks = host._load_manual_push_track_options()
        rows: list[BatchTransferPlanRowState] = []
        for layer, records in grouped_records:
            main_take = host._main_take(layer)
            main_event_ids = {
                event.id for event in (main_take.events if main_take is not None else [])
            }
            selected_event_ids = [
                record.event.id for record in records if record.event.id in main_event_ids
            ]
            target_track_coord = layer.sync.ma3_track_coord
            target_track = None
            if target_track_coord:
                target_track = self._manual_push_track_option_by_coord(
                    available_tracks,
                    target_track_coord,
                )
            if not selected_event_ids:
                status = "blocked"
                issue = "Select main-take events to push"
            elif not target_track_coord:
                status = "blocked"
                issue = "Select an MA3 target track"
            else:
                status = "ready"
                issue = None
            target_label = (
                self._format_manual_push_target_label(target_track)
                if target_track is not None
                else (target_track_coord or "Unmapped")
            )
            rows.append(
                BatchTransferPlanRowState(
                    row_id=f"push:{layer.id}",
                    direction="push",
                    source_label=layer.name,
                    target_label=target_label,
                    source_layer_id=layer.id,
                    target_track_coord=target_track_coord,
                    selected_event_ids=selected_event_ids,
                    selected_count=len(selected_event_ids),
                    status=status,
                    issue=issue,
                )
            )

        rows.sort(key=lambda row: (row.source_label.lower(), row.row_id))
        return rows

    def _rebuild_pull_transfer_plan(self, timeline: Timeline, session: Session) -> None:
        rows = self._build_pull_transfer_plan_rows(timeline, session)
        if not rows:
            if (
                session.batch_transfer_plan is not None
                and session.batch_transfer_plan.operation_type == "pull"
            ):
                session.batch_transfer_plan = None
            return

        ready_count = sum(1 for row in rows if row.status == "ready")
        blocked_count = sum(1 for row in rows if row.status == "blocked")
        draft_count = sum(1 for row in rows if row.status == "draft")
        session.batch_transfer_plan = BatchTransferPlanState(
            plan_id=f"pull:{timeline.id}",
            operation_type="pull",
            rows=rows,
            draft_count=draft_count,
            ready_count=ready_count,
            blocked_count=blocked_count,
        )

    def _build_pull_transfer_plan_rows(
        self, timeline: Timeline, session: Session
    ) -> list[BatchTransferPlanRowState]:
        flow = session.manual_pull_flow
        if not flow.workspace_active:
            return []

        tracks_by_coord = {track.coord: track for track in flow.available_tracks}
        track_order = {track.coord: index for index, track in enumerate(flow.available_tracks)}
        target_labels = {target.layer_id: target.name for target in flow.available_target_layers}
        selected_coords = [
            coord for coord in flow.selected_source_track_coords if coord in tracks_by_coord
        ]
        selected_coords.sort(key=lambda coord: (track_order.get(coord, 0), coord))

        rows: list[BatchTransferPlanRowState] = []
        for coord in selected_coords:
            track = tracks_by_coord[coord]
            selected_event_ids = list(flow.selected_ma3_event_ids_by_track.get(coord, []))
            target_layer_id = flow.target_layer_id_by_source_track.get(coord)
            import_mode = self._manual_pull_import_mode_for_target_layer(
                timeline,
                target_layer_id,
            )
            if not selected_event_ids and target_layer_id is None:
                status = "blocked"
                issue = "Select source events and target layer mapping"
            elif not selected_event_ids:
                status = "blocked"
                issue = "Select source events"
            elif target_layer_id is None:
                status = "blocked"
                issue = "Select target layer mapping"
            else:
                status = "ready"
                issue = None
            target_label = (
                target_labels.get(target_layer_id, str(target_layer_id))
                if target_layer_id is not None
                else "Unmapped"
            )
            rows.append(
                BatchTransferPlanRowState(
                    row_id=f"pull:{coord}",
                    direction="pull",
                    source_label=f"{track.name} ({track.coord})",
                    target_label=target_label,
                    source_track_coord=coord,
                    target_layer_id=target_layer_id,
                    import_mode=import_mode,
                    selected_ma3_event_ids=selected_event_ids,
                    selected_count=len(selected_event_ids),
                    status=status,
                    issue=issue,
                )
            )

        return rows

    @staticmethod
    def _manual_push_track_option_by_coord(
        available_tracks: list[ManualPushTrackOption],
        target_track_coord: str | None,
    ) -> ManualPushTrackOption | None:
        if not target_track_coord:
            return None
        for track in available_tracks:
            if track.coord == target_track_coord:
                return track
        return None

    @staticmethod
    def _format_manual_push_target_label(track: ManualPushTrackOption) -> str:
        if track.note:
            return f"{track.name} ({track.coord}) - {track.note}"
        return f"{track.name} ({track.coord})"
