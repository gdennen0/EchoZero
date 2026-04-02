"""Timeline presentation assembly for the new EchoZero application layer."""

from dataclasses import dataclass

from echozero.application.presentation.models import (
    TimelinePresentation,
    LayerPresentation,
    EventPresentation,
    TakeSummaryPresentation,
)
from echozero.application.timeline.models import Timeline, Layer, Take, Event
from echozero.application.session.models import Session


@dataclass(slots=True)
class TimelineAssembler:
    """Builds a simple UI-facing timeline presentation from application state."""

    def assemble(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        selected_layer_id = timeline.selection.selected_layer_id
        selected_take_id = timeline.selection.selected_take_id
        selected_event_ids = set(timeline.selection.selected_event_ids)

        layers = [
            self._assemble_layer(layer, selected_layer_id, selected_take_id, selected_event_ids)
            for layer in sorted(timeline.layers, key=lambda layer: layer.order_index)
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
        active_take = self._get_active_take(layer)
        take_summary = self._assemble_take_summary(layer, active_take)
        events = self._assemble_events(active_take, selected_event_ids)

        badges: list[str] = []
        if layer.sync.connected:
            badges.append("sync")
        if layer.presentation_hints.locked:
            badges.append("locked")
        if layer.mixer.mute:
            badges.append("muted")
        if layer.mixer.solo:
            badges.append("solo")

        return LayerPresentation(
            layer_id=layer.id,
            title=layer.name,
            subtitle=active_take.name if active_take else "",
            kind=layer.kind,
            is_selected=layer.id == selected_layer_id,
            is_expanded=layer.presentation_hints.take_selector_expanded,
            active_take_id=layer.active_take_id,
            take_summary=take_summary,
            events=events,
            visible=layer.presentation_hints.visible,
            locked=layer.presentation_hints.locked,
            muted=layer.mixer.mute,
            soloed=layer.mixer.solo,
            gain_db=layer.mixer.gain_db,
            pan=layer.mixer.pan,
            playback_mode=layer.playback.mode,
            playback_enabled=layer.playback.enabled,
            sync_mode=layer.sync.mode,
            sync_connected=layer.sync.connected,
            color=layer.presentation_hints.color,
            badges=badges,
        )

    def _assemble_take_summary(self, layer: Layer, active_take: Take | None) -> TakeSummaryPresentation:
        take_names = [take.name for take in layer.takes if take.available]
        active_name = active_take.name if active_take else None
        compact_label = active_name or "No active take"

        return TakeSummaryPresentation(
            total_take_count=len(layer.takes),
            active_take_id=layer.active_take_id,
            active_take_name=active_name,
            available_take_names=take_names,
            compact_label=compact_label,
            can_expand=len(layer.takes) > 1,
        )

    def _assemble_events(self, active_take: Take | None, selected_event_ids: set) -> list[EventPresentation]:
        if active_take is None:
            return []

        return [
            self._assemble_event(event, selected_event_ids)
            for event in sorted(active_take.events, key=lambda event: (event.start, event.end, event.id))
        ]

    def _assemble_event(self, event: Event, selected_event_ids: set) -> EventPresentation:
        badges: list[str] = []
        if event.muted:
            badges.append("muted")

        return EventPresentation(
            event_id=event.id,
            start=event.start,
            end=event.end,
            label=event.label,
            color=event.color,
            muted=event.muted,
            is_selected=event.id in selected_event_ids,
            badges=badges,
        )

    def _get_active_take(self, layer: Layer) -> Take | None:
        if layer.active_take_id is not None:
            for take in layer.takes:
                if take.id == layer.active_take_id:
                    return take

        if layer.takes:
            return layer.takes[0]

        return None
