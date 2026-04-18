"""
TimelineAssembler: Builds UI-facing timeline presentation objects from app state.
Exists to keep presentation shaping out of widgets and inside the application contract.
Connects timeline/session models to the Qt shell without inventing truth in the UI.
"""

from dataclasses import dataclass, field

from echozero.application.presentation.models import (
    BatchTransferPlanPresentation,
    BatchTransferPlanRowPresentation,
    EventPresentation,
    LayerHeaderControlPresentation,
    LayerPresentation,
    ManualPullDiffPreviewPresentation,
    ManualPullEventOptionPresentation,
    ManualPullFlowPresentation,
    ManualPullTargetOptionPresentation,
    ManualPullTrackOptionPresentation,
    LayerStatusPresentation,
    ManualPushDiffPreviewPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    TransferPresetPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    SyncDiffRowPresentation,
    SyncDiffSummaryPresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.shared.enums import SyncMode
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.perf import timed


@dataclass(slots=True)
class TimelineAssembler:
    """Build a UI-facing timeline presentation from application state."""

    _last_signature: tuple | None = field(default=None, init=False, repr=False)
    _last_layers: list[LayerPresentation] | None = field(default=None, init=False, repr=False)

    def assemble(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        with timed("timeline.assemble"):
            selected_layer_id = timeline.selection.selected_layer_id
            selected_layer_ids = list(timeline.selection.selected_layer_ids)
            if not selected_layer_ids and selected_layer_id is not None:
                selected_layer_ids = [selected_layer_id]
            selected_take_id = timeline.selection.selected_take_id
            active_playback_layer_id = timeline.playback_target.layer_id
            active_playback_take_id = timeline.playback_target.take_id
            selected_event_ids = set(timeline.selection.selected_event_ids)

            ordered_layers = sorted(timeline.layers, key=lambda value: value.order_index)
            signature = self._layer_signature(
                timeline,
                ordered_layers,
                selected_layer_id,
                selected_layer_ids,
                selected_take_id,
                active_playback_layer_id,
                active_playback_take_id,
                selected_event_ids,
                session,
            )

            if signature == self._last_signature and self._last_layers is not None:
                layers = self._last_layers
            else:
                layers = [
                    self._assemble_layer(
                        layer,
                        session,
                        selected_layer_id,
                        selected_layer_ids,
                        selected_take_id,
                        active_playback_layer_id,
                        active_playback_take_id,
                        selected_event_ids,
                    )
                    for layer in ordered_layers
                ]
                self._last_signature = signature
                self._last_layers = layers

            return TimelinePresentation(
                timeline_id=timeline.id,
                title=f"Timeline {timeline.id}",
                layers=layers,
                playhead=session.transport_state.playhead,
                is_playing=session.transport_state.is_playing,
                loop_region=timeline.loop_region,
                follow_mode=session.transport_state.follow_mode,
                selected_layer_id=selected_layer_id,
                selected_layer_ids=selected_layer_ids,
                selected_take_id=selected_take_id,
                active_playback_layer_id=active_playback_layer_id,
                active_playback_take_id=active_playback_take_id,
                selected_event_ids=list(timeline.selection.selected_event_ids),
                pixels_per_second=timeline.viewport.pixels_per_second,
                scroll_x=timeline.viewport.scroll_x,
                scroll_y=timeline.viewport.scroll_y,
                experimental_live_sync_enabled=session.sync_state.experimental_live_sync_enabled,
                manual_push_flow=self._assemble_manual_push_flow(session),
                manual_pull_flow=self._assemble_manual_pull_flow(session),
                batch_transfer_plan=self._assemble_batch_transfer_plan(session),
                transfer_presets=self._assemble_transfer_presets(session),
            )

    def _assemble_layer(
        self,
        layer: Layer,
        session: Session,
        selected_layer_id,
        selected_layer_ids: list,
        selected_take_id,
        active_playback_layer_id,
        active_playback_take_id,
        selected_event_ids: set,
    ) -> LayerPresentation:
        main_take = self._main_take(layer)
        main_events = self._assemble_events(main_take.events if main_take else [], selected_event_ids)
        take_rows = self._assemble_take_rows(
            layer,
            selected_layer_id=selected_layer_id,
            selected_layer_ids=selected_layer_ids,
            selected_take_id=selected_take_id,
            active_playback_layer_id=active_playback_layer_id,
            active_playback_take_id=active_playback_take_id,
            selected_event_ids=selected_event_ids,
        )

        badges: list[str] = ["main", layer.kind.value]
        if layer.sync.connected:
            badges.append("sync")

        sync_mode = self._coerce_sync_mode(layer.sync.mode)

        status = LayerStatusPresentation(
            stale=layer.status.stale,
            manually_modified=layer.status.manually_modified,
            source_label=self._source_label(layer, main_take),
            sync_label="Connected" if layer.sync.connected else "No sync",
            stale_reason=layer.status.stale_reason or "",
            source_layer_id=str(layer.provenance.source_layer_id or ""),
            source_song_version_id=str(layer.provenance.source_song_version_id or ""),
            pipeline_id=layer.provenance.pipeline_id or "",
            output_name=layer.provenance.output_name or "",
            source_run_id=layer.provenance.source_run_id or "",
        )

        return LayerPresentation(
            layer_id=layer.id,
            main_take_id=main_take.id if main_take is not None else None,
            title=layer.name,
            subtitle="",
            kind=layer.kind,
            is_selected=layer.id in selected_layer_ids or (not selected_layer_ids and layer.id == selected_layer_id),
            is_playback_active=layer.id == active_playback_layer_id,
            is_expanded=layer.presentation_hints.expanded,
            events=main_events,
            takes=take_rows,
            visible=layer.presentation_hints.visible,
            locked=layer.presentation_hints.locked,
            gain_db=layer.mixer.gain_db,
            pan=layer.mixer.pan,
            playback_mode=layer.playback.mode,
            playback_enabled=layer.playback.enabled,
            sync_mode=sync_mode,
            sync_connected=layer.sync.connected,
            live_sync_state=layer.sync.live_sync_state,
            live_sync_pause_reason=layer.sync.live_sync_pause_reason,
            live_sync_divergent=layer.sync.live_sync_divergent,
            sync_target_label=self._sync_target_label(layer),
            push_target_label=self._push_target_label(session, layer),
            push_selection_count=self._push_selection_count(session, layer),
            push_row_status=self._push_row_status(session, layer),
            push_row_issue=self._push_row_issue(session, layer),
            pull_target_label=self._pull_target_label(session, layer),
            pull_selection_count=self._pull_selection_count(session, layer),
            pull_row_status=self._pull_row_status(session, layer),
            pull_row_issue=self._pull_row_issue(session, layer),
            color=layer.presentation_hints.color,
            badges=badges,
            header_controls=self._assemble_header_controls(
                layer,
                is_playback_active=layer.id == active_playback_layer_id,
            ),
            playback_source_ref=layer.playback.armed_source_ref,
            status=status,
        )

    @staticmethod
    def _assemble_header_controls(
        layer: Layer,
        *,
        is_playback_active: bool,
    ) -> list[LayerHeaderControlPresentation]:
        controls = [
            LayerHeaderControlPresentation(
                control_id="set_active_playback_target",
                label="ACTIVE",
                kind="toggle",
                active=is_playback_active,
            )
        ]
        if layer.kind.name == "EVENT" and layer.takes:
            controls.extend(
                [
                    LayerHeaderControlPresentation(
                        control_id="push_to_ma3",
                        label="Push",
                        kind="action",
                    ),
                    LayerHeaderControlPresentation(
                        control_id="pull_from_ma3",
                        label="Pull",
                        kind="action",
                    ),
                ]
            )
        return controls

    def _assemble_take_rows(
        self,
        layer: Layer,
        *,
        selected_layer_id,
        selected_layer_ids: list,
        selected_take_id,
        active_playback_layer_id,
        active_playback_take_id,
        selected_event_ids: set,
    ) -> list[TakeLanePresentation]:
        layer_selected = layer.id in selected_layer_ids or (
            not selected_layer_ids and layer.id == selected_layer_id
        )
        return [
            TakeLanePresentation(
                take_id=take.id,
                name=take.name,
                is_main=False,
                kind=layer.kind,
                is_selected=layer_selected and take.id == selected_take_id,
                is_playback_active=(
                    layer.id == active_playback_layer_id and take.id == active_playback_take_id
                ),
                events=self._assemble_events(take.events, selected_event_ids),
                source_ref=take.source_ref,
                playback_source_ref=layer.playback.armed_source_ref,
                actions=self._take_actions(),
            )
            for take in layer.takes[1:]
        ]

    @staticmethod
    def _layer_signature(
        timeline: Timeline,
        ordered_layers: list[Layer],
        selected_layer_id,
        selected_layer_ids: list,
        selected_take_id,
        active_playback_layer_id,
        active_playback_take_id,
        selected_event_ids: set,
        session: Session,
    ) -> tuple:
        def _events_sig(events: list[Event]) -> tuple:
            return (
                id(events),
                tuple(
                    (
                        str(event.id),
                        str(event.take_id),
                        float(event.start),
                        float(event.end),
                        bool(event.muted),
                    )
                    for event in events
                ),
            )

        layer_sigs: list[tuple] = []
        for layer in ordered_layers:
            take_sigs = tuple(
                (
                    str(take.id),
                    idx == 0,
                    take.name,
                    take.source_ref,
                    _events_sig(take.events),
                )
                for idx, take in enumerate(layer.takes)
            )
            layer_sigs.append(
                (
                    str(layer.id),
                    int(layer.order_index),
                    bool(layer.presentation_hints.expanded),
                    bool(layer.presentation_hints.visible),
                    bool(layer.presentation_hints.locked),
                    layer.presentation_hints.color,
                    bool(layer.status.stale),
                    bool(layer.status.manually_modified),
                    layer.status.stale_reason,
                    str(layer.provenance.source_layer_id) if layer.provenance.source_layer_id is not None else None,
                    str(layer.provenance.source_song_version_id) if layer.provenance.source_song_version_id is not None else None,
                    layer.provenance.source_run_id,
                    layer.provenance.pipeline_id,
                    layer.provenance.output_name,
                    bool(layer.mixer.mute),
                    bool(layer.mixer.solo),
                    float(layer.mixer.gain_db),
                    float(layer.mixer.pan),
                    str(layer.sync.mode),
                    bool(layer.sync.connected),
                    layer.sync.target_ref,
                    layer.sync.show_manager_block_id,
                    layer.sync.ma3_track_coord,
                    str(layer.sync.live_sync_state.value),
                    layer.sync.live_sync_pause_reason,
                    bool(layer.sync.live_sync_divergent),
                    take_sigs,
                )
            )

        return (
            str(timeline.id),
            tuple(layer_sigs),
            str(selected_layer_id) if selected_layer_id is not None else None,
            tuple(str(layer_id) for layer_id in selected_layer_ids),
            str(selected_take_id) if selected_take_id is not None else None,
            str(active_playback_layer_id) if active_playback_layer_id is not None else None,
            str(active_playback_take_id) if active_playback_take_id is not None else None,
            tuple(sorted(str(event_id) for event_id in selected_event_ids)),
            TimelineAssembler._session_transfer_signature(session),
        )

    @staticmethod
    def _session_transfer_signature(session: Session) -> tuple:
        plan = session.batch_transfer_plan
        plan_sig = None
        if plan is not None:
            plan_sig = (
                plan.plan_id,
                plan.operation_type,
                tuple(
                    (
                        row.row_id,
                        row.direction,
                        str(row.source_layer_id) if row.source_layer_id is not None else None,
                        row.source_track_coord,
                        row.target_track_coord,
                        str(row.target_layer_id) if row.target_layer_id is not None else None,
                        row.import_mode,
                        tuple(str(event_id) for event_id in row.selected_event_ids),
                        tuple(row.selected_ma3_event_ids),
                        row.selected_count,
                        row.status,
                        row.issue,
                        row.target_label,
                    )
                    for row in plan.rows
                ),
            )

        return (
            session.manual_push_flow.push_mode_active,
            tuple(str(layer_id) for layer_id in session.manual_push_flow.selected_layer_ids),
            session.manual_push_flow.transfer_mode,
            session.manual_pull_flow.workspace_active,
            session.manual_pull_flow.active_source_track_coord,
            session.manual_pull_flow.source_track_coord,
            tuple(session.manual_pull_flow.selected_source_track_coords),
            tuple(
                (coord, tuple(event_ids))
                for coord, event_ids in sorted(session.manual_pull_flow.selected_ma3_event_ids_by_track.items())
            ),
            session.manual_pull_flow.import_mode,
            tuple(
                (coord, mode)
                for coord, mode in sorted(session.manual_pull_flow.import_mode_by_source_track.items())
            ),
            tuple(
                (coord, str(layer_id))
                for coord, layer_id in sorted(session.manual_pull_flow.target_layer_id_by_source_track.items())
            ),
            plan_sig,
        )

    @staticmethod
    def _assemble_events(events: list[Event], selected_event_ids: set) -> list[EventPresentation]:
        ordered = events
        if len(events) > 1:
            for i in range(1, len(events)):
                prev = events[i - 1]
                cur = events[i]
                if (prev.start, prev.end, str(prev.id)) > (cur.start, cur.end, str(cur.id)):
                    ordered = sorted(events, key=lambda value: (value.start, value.end, str(value.id)))
                    break

        return [
            EventPresentation(
                event_id=event.id,
                start=event.start,
                end=event.end,
                label=event.label,
                color=event.color,
                muted=event.muted,
                is_selected=event.id in selected_event_ids,
                badges=["muted"] if event.muted else [],
            )
            for event in ordered
        ]

    @staticmethod
    def _take_actions() -> list[TakeActionPresentation]:
        return [
            TakeActionPresentation(action_id="overwrite_main", label="Overwrite Main"),
            TakeActionPresentation(action_id="merge_main", label="Merge Main"),
        ]

    @staticmethod
    def _coerce_sync_mode(raw_mode: object) -> SyncMode:
        if isinstance(raw_mode, SyncMode):
            return raw_mode

        raw_value = str(raw_mode)
        for sync_mode in SyncMode:
            if sync_mode.value == raw_value:
                return sync_mode
        return SyncMode.NONE

    def _assemble_manual_push_flow(self, session: Session) -> ManualPushFlowPresentation:
        flow = session.manual_push_flow
        diff_preview = None
        if flow.diff_preview is not None:
            diff_preview = ManualPushDiffPreviewPresentation(
                selected_count=flow.diff_preview.selected_count,
                target_track_coord=flow.diff_preview.target_track_coord,
                target_track_name=flow.diff_preview.target_track_name,
                target_track_note=flow.diff_preview.target_track_note,
                target_track_event_count=flow.diff_preview.target_track_event_count,
                diff_summary=self._assemble_sync_diff_summary(flow.diff_preview.diff_summary),
                diff_rows=self._assemble_sync_diff_rows(flow.diff_preview.diff_rows),
            )

        return ManualPushFlowPresentation(
            dialog_open=flow.dialog_open,
            push_mode_active=flow.push_mode_active,
            selected_layer_ids=list(flow.selected_layer_ids),
            available_tracks=[
                ManualPushTrackOptionPresentation(
                    coord=track.coord,
                    name=track.name,
                    note=track.note,
                    event_count=track.event_count,
                )
                for track in flow.available_tracks
            ],
            target_track_coord=flow.target_track_coord,
            transfer_mode=flow.transfer_mode,
            diff_gate_open=flow.diff_gate_open,
            diff_preview=diff_preview,
        )

    def _assemble_manual_pull_flow(self, session: Session) -> ManualPullFlowPresentation:
        flow = session.manual_pull_flow
        diff_preview = None
        if flow.diff_preview is not None:
            diff_preview = ManualPullDiffPreviewPresentation(
                selected_count=flow.diff_preview.selected_count,
                source_track_coord=flow.diff_preview.source_track_coord,
                source_track_name=flow.diff_preview.source_track_name,
                source_track_note=flow.diff_preview.source_track_note,
                source_track_event_count=flow.diff_preview.source_track_event_count,
                target_layer_id=flow.diff_preview.target_layer_id,
                target_layer_name=flow.diff_preview.target_layer_name,
                import_mode=flow.diff_preview.import_mode,
                diff_summary=self._assemble_sync_diff_summary(flow.diff_preview.diff_summary),
                diff_rows=self._assemble_sync_diff_rows(flow.diff_preview.diff_rows),
            )

        return ManualPullFlowPresentation(
            dialog_open=flow.dialog_open,
            workspace_active=flow.workspace_active,
            available_tracks=[
                ManualPullTrackOptionPresentation(
                    coord=track.coord,
                    name=track.name,
                    note=track.note,
                    event_count=track.event_count,
                )
                for track in flow.available_tracks
            ],
            selected_source_track_coords=list(flow.selected_source_track_coords),
            active_source_track_coord=flow.active_source_track_coord,
            source_track_coord=flow.source_track_coord,
            available_events=[
                ManualPullEventOptionPresentation(
                    event_id=event.event_id,
                    label=event.label,
                    start=event.start,
                    end=event.end,
                )
                for event in flow.available_events
            ],
            selected_ma3_event_ids=list(flow.selected_ma3_event_ids),
            selected_ma3_event_ids_by_track={
                coord: list(event_ids)
                for coord, event_ids in flow.selected_ma3_event_ids_by_track.items()
            },
            import_mode=flow.import_mode,
            import_mode_by_source_track=dict(flow.import_mode_by_source_track),
            available_target_layers=[
                ManualPullTargetOptionPresentation(
                    layer_id=target.layer_id,
                    name=target.name,
                )
                for target in flow.available_target_layers
            ],
            target_layer_id=flow.target_layer_id,
            target_layer_id_by_source_track=dict(flow.target_layer_id_by_source_track),
            diff_gate_open=flow.diff_gate_open,
            diff_preview=diff_preview,
        )

    @staticmethod
    def _assemble_batch_transfer_plan(session: Session) -> BatchTransferPlanPresentation | None:
        plan = session.batch_transfer_plan
        if plan is None:
            return None
        return BatchTransferPlanPresentation(
            plan_id=plan.plan_id,
            operation_type=plan.operation_type,
            rows=[
                BatchTransferPlanRowPresentation(
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
                    status=row.status,
                    issue=row.issue,
                )
                for row in plan.rows
            ],
            draft_count=plan.draft_count,
            ready_count=plan.ready_count,
            blocked_count=plan.blocked_count,
            applied_count=plan.applied_count,
            failed_count=plan.failed_count,
        )

    @staticmethod
    def _assemble_transfer_presets(session: Session) -> list[TransferPresetPresentation]:
        return [
            TransferPresetPresentation(
                preset_id=preset.preset_id,
                name=preset.name,
                push_target_mapping_by_layer_id=dict(preset.push_target_mapping_by_layer_id),
                pull_target_mapping_by_source_track=dict(preset.pull_target_mapping_by_source_track),
            )
            for preset in session.transfer_presets
        ]

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if not layer.takes:
            return None
        return layer.takes[0]

    @staticmethod
    def _assemble_sync_diff_summary(summary) -> SyncDiffSummaryPresentation | None:
        if summary is None:
            return None
        return SyncDiffSummaryPresentation(
            added_count=summary.added_count,
            removed_count=summary.removed_count,
            modified_count=summary.modified_count,
            unchanged_count=summary.unchanged_count,
            row_count=summary.row_count,
        )

    @staticmethod
    def _assemble_sync_diff_rows(rows) -> list[SyncDiffRowPresentation]:
        return [
            SyncDiffRowPresentation(
                row_id=row.row_id,
                action=row.action,
                start=row.start,
                end=row.end,
                label=row.label,
                before=row.before,
                after=row.after,
            )
            for row in rows
        ]

    @staticmethod
    def _source_label(layer: Layer, main_take: Take | None) -> str:
        if layer.provenance.pipeline_id and layer.provenance.output_name:
            return f"{layer.provenance.pipeline_id} · {layer.provenance.output_name}"
        if main_take and main_take.source_ref:
            return main_take.source_ref
        return ""

    @staticmethod
    def _sync_target_label(layer: Layer) -> str:
        if layer.sync.ma3_track_coord:
            return layer.sync.ma3_track_coord
        if layer.sync.target_ref and layer.sync.show_manager_block_id:
            return f"{layer.sync.target_ref} · {layer.sync.show_manager_block_id}"
        if layer.sync.target_ref:
            return layer.sync.target_ref
        if layer.sync.show_manager_block_id:
            return layer.sync.show_manager_block_id
        return ""

    @staticmethod
    def _push_plan_row(session: Session, layer: Layer):
        plan = session.batch_transfer_plan
        if plan is None:
            return None
        for row in plan.rows:
            if row.direction == "push" and row.source_layer_id == layer.id:
                return row
        return None

    @classmethod
    def _push_target_label(cls, session: Session, layer: Layer) -> str:
        row = cls._push_plan_row(session, layer)
        if row is not None and row.target_label:
            return row.target_label
        return cls._sync_target_label(layer)

    @classmethod
    def _push_selection_count(cls, session: Session, layer: Layer) -> int:
        row = cls._push_plan_row(session, layer)
        return 0 if row is None else row.selected_count

    @classmethod
    def _push_row_status(cls, session: Session, layer: Layer) -> str:
        row = cls._push_plan_row(session, layer)
        return "" if row is None else row.status

    @classmethod
    def _push_row_issue(cls, session: Session, layer: Layer) -> str:
        row = cls._push_plan_row(session, layer)
        return "" if row is None or row.issue is None else row.issue

    @staticmethod
    def _pull_plan_row(session: Session, layer: Layer):
        plan = session.batch_transfer_plan
        if plan is None:
            return None
        for row in plan.rows:
            if row.direction == "pull" and row.target_layer_id == layer.id:
                return row
        return None

    @classmethod
    def _pull_target_label(cls, session: Session, layer: Layer) -> str:
        row = cls._pull_plan_row(session, layer)
        return "" if row is None else row.target_label

    @classmethod
    def _pull_selection_count(cls, session: Session, layer: Layer) -> int:
        row = cls._pull_plan_row(session, layer)
        return 0 if row is None else row.selected_count

    @classmethod
    def _pull_row_status(cls, session: Session, layer: Layer) -> str:
        row = cls._pull_plan_row(session, layer)
        return "" if row is None else row.status

    @classmethod
    def _pull_row_issue(cls, session: Session, layer: Layer) -> str:
        row = cls._pull_plan_row(session, layer)
        return "" if row is None or row.issue is None else row.issue
