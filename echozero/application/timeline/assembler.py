"""Timeline presentation assembly for the new EchoZero application layer."""

from dataclasses import dataclass, field

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    ManualPushDiffPreviewPresentation,
    ManualPushFlowPresentation,
    ManualPushTrackOptionPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.shared.enums import SyncMode
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.perf import timed


@dataclass(slots=True)
class TimelineAssembler:
    _last_signature: tuple | None = field(default=None, init=False, repr=False)
    _last_layers: list[LayerPresentation] | None = field(default=None, init=False, repr=False)
    """Builds a UI-facing timeline presentation from application state.

    Contract: main take (index 0) is truth. Non-main takes render as subordinate lanes.
    """

    def assemble(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        with timed("timeline.assemble"):
            selected_layer_id = timeline.selection.selected_layer_id
            selected_take_id = timeline.selection.selected_take_id
            selected_event_ids = set(timeline.selection.selected_event_ids)

            ordered_layers = sorted(timeline.layers, key=lambda value: value.order_index)
            signature = self._layer_signature(
                timeline,
                ordered_layers,
                selected_layer_id,
                selected_take_id,
                selected_event_ids,
            )

            if signature == self._last_signature and self._last_layers is not None:
                layers = self._last_layers
            else:
                layers = [
                    self._assemble_layer(layer, selected_layer_id, selected_take_id, selected_event_ids)
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
                selected_take_id=selected_take_id,
                selected_event_ids=list(timeline.selection.selected_event_ids),
                pixels_per_second=timeline.viewport.pixels_per_second,
                scroll_x=timeline.viewport.scroll_x,
                scroll_y=timeline.viewport.scroll_y,
                manual_push_flow=self._assemble_manual_push_flow(session),
            )

    def _assemble_layer(
        self,
        layer: Layer,
        selected_layer_id,
        selected_take_id,
        selected_event_ids: set,
    ) -> LayerPresentation:
        main_take = self._main_take(layer)
        main_events = self._assemble_events(main_take.events if main_take else [], selected_event_ids)

        take_rows: list[TakeLanePresentation] = []
        for take in layer.takes[1:]:
            take_rows.append(
                TakeLanePresentation(
                    take_id=take.id,
                    name=take.name,
                    is_main=False,
                    kind=layer.kind,
                    events=self._assemble_events(take.events, selected_event_ids),
                    source_ref=take.source_ref,
                    playback_source_ref=layer.playback.armed_source_ref,
                    actions=[
                        TakeActionPresentation(action_id="overwrite_main", label="Overwrite Main"),
                        TakeActionPresentation(action_id="merge_main", label="Merge Main"),
                    ],
                )
            )

        badges: list[str] = ["main", layer.kind.value]
        if layer.sync.connected:
            badges.append("sync")
        if layer.mixer.mute:
            badges.append("muted")
        if layer.mixer.solo:
            badges.append("solo")

        sync_mode = layer.sync.mode if isinstance(layer.sync.mode, SyncMode) else SyncMode(str(layer.sync.mode)) if str(layer.sync.mode) in {m.value for m in SyncMode} else SyncMode.NONE

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
            is_selected=layer.id == selected_layer_id,
            is_expanded=layer.presentation_hints.expanded,
            events=main_events,
            takes=take_rows,
            visible=layer.presentation_hints.visible,
            locked=layer.presentation_hints.locked,
            muted=layer.mixer.mute,
            soloed=layer.mixer.solo,
            gain_db=layer.mixer.gain_db,
            pan=layer.mixer.pan,
            playback_mode=layer.playback.mode,
            playback_enabled=layer.playback.enabled,
            sync_mode=sync_mode,
            sync_connected=layer.sync.connected,
            color=layer.presentation_hints.color,
            badges=badges,
            playback_source_ref=layer.playback.armed_source_ref,
            status=status,
        )

    @staticmethod
    def _layer_signature(
        timeline: Timeline,
        ordered_layers: list[Layer],
        selected_layer_id,
        selected_take_id,
        selected_event_ids: set,
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
                    take_sigs,
                )
            )

        return (
            str(timeline.id),
            tuple(layer_sigs),
            str(selected_layer_id) if selected_layer_id is not None else None,
            str(selected_take_id) if selected_take_id is not None else None,
            tuple(sorted(str(event_id) for event_id in selected_event_ids)),
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
    def _assemble_manual_push_flow(session: Session) -> ManualPushFlowPresentation:
        flow = session.manual_push_flow
        diff_preview = None
        if flow.diff_preview is not None:
            diff_preview = ManualPushDiffPreviewPresentation(
                selected_count=flow.diff_preview.selected_count,
                target_track_coord=flow.diff_preview.target_track_coord,
                target_track_name=flow.diff_preview.target_track_name,
                target_track_note=flow.diff_preview.target_track_note,
                target_track_event_count=flow.diff_preview.target_track_event_count,
            )

        return ManualPushFlowPresentation(
            dialog_open=flow.dialog_open,
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
            diff_gate_open=flow.diff_gate_open,
            diff_preview=diff_preview,
        )

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if not layer.takes:
            return None
        return layer.takes[0]

    @staticmethod
    def _source_label(layer: Layer, main_take: Take | None) -> str:
        if layer.provenance.pipeline_id and layer.provenance.output_name:
            return f"{layer.provenance.pipeline_id} · {layer.provenance.output_name}"
        if main_take and main_take.source_ref:
            return main_take.source_ref
        return ""
