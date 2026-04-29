"""
TimelineAssembler: Builds UI-facing timeline presentation objects from app state.
Exists to keep presentation shaping out of widgets and inside the application contract.
Connects timeline/session models to the Qt shell without inventing truth in the UI.
"""

from dataclasses import dataclass, field

from echozero.application.presentation.models import (
    LayerPresentation,
    RegionPresentation,
    SectionCuePresentation,
    SectionRegionPresentation,
    TimelinePresentation,
)
from echozero.application.session.models import Session
from echozero.application.timeline.assembler_layers import assemble_layer
from echozero.application.timeline.assembler_signature import build_layer_signature
from echozero.application.timeline.assembler_state import (
    AssemblerSignature,
    build_timeline_assembly_state,
)
from echozero.application.timeline.assembler_transfers import (
    assemble_batch_transfer_plan,
    assemble_manual_pull_flow,
    assemble_manual_push_flow,
    assemble_pipeline_run_banner,
    assemble_transfer_presets,
)
from echozero.application.timeline.models import Timeline, derive_section_regions
from echozero.perf import timed

__all__ = ["TimelineAssembler"]


@dataclass(slots=True)
class TimelineAssembler:
    """Build a UI-facing timeline presentation from application state."""

    _last_signature: AssemblerSignature | None = field(default=None, init=False, repr=False)
    _last_layers: list[LayerPresentation] | None = field(default=None, init=False, repr=False)

    def assemble(self, timeline: Timeline, session: Session) -> TimelinePresentation:
        with timed("timeline.assemble"):
            state = build_timeline_assembly_state(timeline)
            ordered_layers = sorted(timeline.layers, key=lambda value: value.order_index)
            signature = build_layer_signature(timeline, ordered_layers, state, session)

            if signature == self._last_signature and self._last_layers is not None:
                layers = self._last_layers
            else:
                layers = [
                    assemble_layer(layer, session=session, state=state)
                    for layer in ordered_layers
                ]
                self._last_signature = signature
                self._last_layers = layers

            return TimelinePresentation(
                timeline_id=timeline.id,
                title=f"Timeline {timeline.id}",
                layers=layers,
                section_cues=self._assemble_section_cues(timeline),
                section_regions=self._assemble_section_regions(timeline),
                playhead=session.transport_state.playhead,
                is_playing=session.transport_state.is_playing,
                loop_region=timeline.loop_region,
                follow_mode=session.transport_state.follow_mode,
                selected_layer_id=state.selected_layer_id,
                selected_layer_ids=list(state.selected_layer_ids),
                selected_take_id=state.selected_take_id,
                selected_event_refs=list(state.selected_event_refs),
                active_playback_layer_id=state.active_playback_layer_id,
                active_playback_take_id=state.active_playback_take_id,
                playback_output_channels=max(0, int(session.playback_state.output_channels)),
                selected_event_ids=list(state.selected_event_ids),
                selected_region_id=timeline.selection.selected_region_id,
                regions=self._assemble_regions(timeline),
                pixels_per_second=timeline.viewport.pixels_per_second,
                scroll_x=timeline.viewport.scroll_x,
                scroll_y=timeline.viewport.scroll_y,
                experimental_live_sync_enabled=session.sync_state.experimental_live_sync_enabled,
                manual_push_flow=assemble_manual_push_flow(session),
                manual_pull_flow=assemble_manual_pull_flow(session),
                batch_transfer_plan=assemble_batch_transfer_plan(session),
                transfer_presets=assemble_transfer_presets(session),
                pipeline_run_banner=assemble_pipeline_run_banner(
                    session,
                    song_version_id=str(timeline.song_version_id),
                ),
            )

    @staticmethod
    def _assemble_regions(timeline: Timeline) -> list[RegionPresentation]:
        ordered = sorted(
            timeline.regions,
            key=lambda region: (
                float(region.start),
                float(region.end),
                int(region.order_index),
                str(region.id),
            ),
        )
        selected_region_id = timeline.selection.selected_region_id
        return [
            RegionPresentation(
                region_id=region.id,
                start=float(region.start),
                end=float(region.end),
                label=region.label,
                color=region.color,
                kind=region.kind,
                is_selected=region.id == selected_region_id,
            )
            for region in ordered
        ]

    @staticmethod
    def _assemble_section_cues(timeline: Timeline) -> list[SectionCuePresentation]:
        ordered = sorted(
            enumerate(timeline.section_cues),
            key=lambda item: (float(item[1].start), int(item[0]), str(item[1].id)),
        )
        return [
            SectionCuePresentation(
                cue_id=cue.id,
                start=float(cue.start),
                cue_ref=cue.cue_ref,
                name=cue.name,
                color=cue.color,
                notes=cue.notes,
                payload_ref=cue.payload_ref,
            )
            for _index, cue in ordered
        ]

    @staticmethod
    def _assemble_section_regions(timeline: Timeline) -> list[SectionRegionPresentation]:
        return [
            SectionRegionPresentation(
                cue_id=region.cue_id,
                start=float(region.start),
                end=float(region.end),
                cue_ref=region.cue_ref,
                name=region.name,
                color=region.color,
                notes=region.notes,
                payload_ref=region.payload_ref,
            )
            for region in derive_section_regions(timeline.section_cues, timeline_end=timeline.end)
        ]
