"""Timeline presentation assembly for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.shared.enums import SyncMode
from echozero.application.timeline.models import Event, Layer, Take, Timeline


@dataclass(slots=True)
class TimelineAssembler:
    """Builds a UI-facing timeline presentation from application state.

    Contract: main take (index 0) is truth. Non-main takes render as subordinate lanes.
    """

    def assemble(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        selected_layer_id = timeline.selection.selected_layer_id
        selected_take_id = timeline.selection.selected_take_id
        selected_event_ids = set(timeline.selection.selected_event_ids)

        layers = [
            self._assemble_layer(layer, selected_layer_id, selected_take_id, selected_event_ids)
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
        ]

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
            stale=False,
            manually_modified=False,
            source_label=main_take.source_ref if main_take and main_take.source_ref else "",
            sync_label="Connected" if layer.sync.connected else "No sync",
        )

        return LayerPresentation(
            layer_id=layer.id,
            title=layer.name,
            subtitle="",
            kind=layer.kind,
            is_selected=layer.id == selected_layer_id,
            is_expanded=layer.presentation_hints.take_selector_expanded,
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
            status=status,
        )

    @staticmethod
    def _assemble_events(events: list[Event], selected_event_ids: set) -> list[EventPresentation]:
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
            for event in sorted(events, key=lambda value: (value.start, value.end, str(value.id)))
        ]

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if not layer.takes:
            return None
        return layer.takes[0]
