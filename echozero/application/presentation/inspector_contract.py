"""Inspector contract builders for the canonical timeline presentation layer.
Exists to turn selected timeline objects into stable inspector sections and actions.
Connects presentation models to the Qt inspector surface without widget-owned truth.
"""

from __future__ import annotations

from echozero.application.presentation.inspector_contract_context_actions import (
    event_context_sections as _event_context_sections,
    format_seconds as _format_seconds,
    shared_context_sections as _shared_context_sections,
)
from echozero.application.presentation.inspector_contract_lookup import (
    find_event as _find_event,
    find_layer as _find_layer,
    find_selected_event as _find_selected_event,
    find_take as _find_take,
)
from echozero.application.presentation.models import (
    EventPresentation,
    LayerPresentation,
    TakeLanePresentation,
    TimelinePresentation,
)
from echozero.application.presentation.inspector_contract_support import (
    layer_transfer_rows as _layer_transfer_rows,
    playback_state_label as _playback_state_label,
    selection_playback_context_rows as _selection_playback_context_rows,
)
from echozero.application.presentation.inspector_contract_types import (
    InspectorAction,
    InspectorContextSection,
    InspectorContract,
    InspectorFactRow,
    InspectorObjectIdentity,
    InspectorSection,
    TimelineInspectorHitTarget,
)

__all__ = [
    "InspectorAction",
    "InspectorContextSection",
    "InspectorContract",
    "InspectorFactRow",
    "InspectorObjectIdentity",
    "InspectorSection",
    "TimelineInspectorHitTarget",
    "build_timeline_inspector_contract",
    "render_inspector_contract_text",
]


def build_timeline_inspector_contract(
    presentation: TimelinePresentation,
    *,
    hit_target: TimelineInspectorHitTarget | None = None,
) -> InspectorContract:
    if hit_target is not None:
        if hit_target.event_id is not None:
            event_match = _find_event(
                presentation,
                layer_id=hit_target.layer_id,
                take_id=hit_target.take_id,
                event_id=hit_target.event_id,
            )
            if event_match is not None:
                layer, take, event = event_match
                return _event_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    event=event,
                    hit_target=hit_target,
                )
        if hit_target.take_id is not None:
            take_match = _find_take(
                presentation, layer_id=hit_target.layer_id, take_id=hit_target.take_id
            )
            if take_match is not None:
                layer, take = take_match
                return _take_contract(
                    presentation,
                    layer=layer,
                    take=take,
                    hit_target=hit_target,
                )
        if hit_target.layer_id is not None:
            selected_layer = _find_layer(presentation, hit_target.layer_id)
            if selected_layer is not None:
                return _layer_contract(
                    presentation,
                    layer=selected_layer,
                    hit_target=hit_target,
                )
        if presentation.active_song_version_id:
            return _song_version_contract(
                presentation,
                hit_target=hit_target,
                has_selected_events=bool(presentation.selected_event_ids),
            )
        return _empty_contract(
            presentation,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
        )

    if presentation.selected_event_refs:
        selected_ref = presentation.selected_event_refs[-1]
        event_match = _find_event(
            presentation,
            layer_id=selected_ref.layer_id,
            take_id=selected_ref.take_id,
            event_id=selected_ref.event_id,
        )
        if event_match is not None:
            layer, take, event = event_match
            return _event_contract(
                presentation,
                layer=layer,
                take=take,
                event=event,
                hit_target=None,
            )

    if presentation.selected_event_ids and presentation.selected_layer_id is not None:
        selected_event_id = presentation.selected_event_ids[0]
        event_match = _find_selected_event(presentation, selected_event_id)
        if event_match is not None:
            layer, take, event = event_match
            return _event_contract(
                presentation,
                layer=layer,
                take=take,
                event=event,
                hit_target=None,
            )

    if presentation.selected_layer_id is not None and not presentation.selected_event_ids:
        selected_layer = _find_layer(presentation, presentation.selected_layer_id)
        if selected_layer is not None:
            return _layer_contract(
                presentation,
                layer=selected_layer,
                hit_target=None,
            )

    if presentation.active_song_version_id:
        return _song_version_contract(
            presentation,
            hit_target=None,
            has_selected_events=bool(presentation.selected_event_ids),
        )

    return _empty_contract(
        presentation,
        hit_target=None,
        has_selected_events=bool(presentation.selected_event_ids),
    )


def render_inspector_contract_text(contract: InspectorContract) -> str:
    if contract.identity is None and not contract.sections:
        return contract.empty_state

    lines: list[str] = [contract.title]
    for section in contract.sections:
        for row in section.rows:
            lines.append(f"{row.label}: {row.value}")
    return "\n".join(lines)


def _empty_contract(
    presentation: TimelinePresentation,
    *,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> InspectorContract:
    context_rows = _selection_playback_context_rows(presentation)
    return InspectorContract(
        title="Timeline" if context_rows else "No timeline object selected.",
        sections=(
            (
                InspectorSection(
                    section_id="timeline-context",
                    label="Timeline",
                    rows=context_rows,
                ),
            )
            if context_rows
            else ()
        ),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=None,
            take=None,
            hit_target=hit_target,
            has_selected_events=has_selected_events,
            include_layer_transfer_controls=False,
        ),
    )


def _song_version_contract(
    presentation: TimelinePresentation,
    *,
    hit_target: TimelineInspectorHitTarget | None,
    has_selected_events: bool,
) -> InspectorContract:
    context_rows = _selection_playback_context_rows(presentation)
    version_count = len(presentation.available_song_versions)
    title = (
        f"Song {presentation.active_song_title}"
        if presentation.active_song_title
        else "Song Version"
    )
    label = (
        presentation.active_song_title
        if presentation.active_song_title
        else presentation.active_song_version_label or "Song Version"
    )
    sections = [
        InspectorSection(
            section_id="song-version-core",
            label="Song Version",
            rows=(
                InspectorFactRow("song id", presentation.active_song_id or "none"),
                InspectorFactRow(
                    "song title",
                    presentation.active_song_title or "Untitled Song",
                ),
                InspectorFactRow(
                    "version id",
                    presentation.active_song_version_id or "none",
                ),
                InspectorFactRow(
                    "version label",
                    presentation.active_song_version_label or "Unlabeled",
                ),
                InspectorFactRow(
                    "ma3 tc pool",
                    (
                        f"TC{presentation.active_song_version_ma3_timecode_pool_no}"
                        if presentation.active_song_version_ma3_timecode_pool_no is not None
                        else "unconfigured"
                    ),
                ),
                InspectorFactRow("versions", str(version_count or 1)),
                InspectorFactRow("timeline duration", presentation.end_time_label),
                InspectorFactRow("layers", str(len(presentation.layers))),
            ),
        )
    ]
    if context_rows:
        sections.append(
            InspectorSection(
                section_id="song-version-context",
                label="Context",
                rows=context_rows,
            )
        )
    return InspectorContract(
        title=title,
        identity=InspectorObjectIdentity(
            object_id=presentation.active_song_version_id,
            object_type="song_version",
            label=label,
        ),
        sections=tuple(sections),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=None,
            take=None,
            hit_target=hit_target,
            has_selected_events=has_selected_events,
            include_layer_transfer_controls=False,
        ),
    )


def _layer_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    flags: list[str] = []
    if layer.status.stale:
        flags.append("stale")
    if layer.status.manually_modified:
        flags.append("edited")

    take_count = 0 if layer.main_take_id is None else 1 + len(layer.takes)
    rows = [
        InspectorFactRow("id", str(layer.layer_id)),
        InspectorFactRow("kind", layer.kind.name),
        InspectorFactRow("main take", str(layer.main_take_id or "none")),
        InspectorFactRow("takes", str(take_count) if take_count else "none"),
        InspectorFactRow("status flags", ", ".join(flags) if flags else "none"),
        InspectorFactRow(
            "playback state", _playback_state_label(presentation, layer=layer, take=None)
        ),
    ]
    rows.extend(_layer_transfer_rows(presentation, layer))
    if presentation.experimental_live_sync_enabled:
        rows.append(InspectorFactRow("live sync state", layer.live_sync_state.value))
        if layer.live_sync_pause_reason:
            rows.append(InspectorFactRow("live sync pause", layer.live_sync_pause_reason))
        if layer.live_sync_divergent:
            rows.append(InspectorFactRow("live sync divergence", "diverged"))

    return InspectorContract(
        title=f"Layer {layer.title}",
        identity=InspectorObjectIdentity(
            object_id=str(layer.layer_id),
            object_type="layer",
            label=layer.title,
        ),
        sections=(
            InspectorSection(
                section_id="layer-core",
                label="Layer",
                rows=tuple(rows),
            ),
            InspectorSection(
                section_id="layer-context",
                label="Context",
                rows=_selection_playback_context_rows(presentation),
            ),
        ),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=layer,
            take=None,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
            include_layer_transfer_controls=True,
        ),
    )


def _take_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    return InspectorContract(
        title=f"Take {take.name}",
        identity=InspectorObjectIdentity(
            object_id=str(take.take_id),
            object_type="take",
            label=take.name,
        ),
        sections=(
            InspectorSection(
                section_id="take-core",
                label="Take",
                rows=(
                    InspectorFactRow("id", str(take.take_id)),
                    InspectorFactRow("layer", layer.title),
                    InspectorFactRow("kind", take.kind.name),
                    InspectorFactRow("main truth", "no"),
                    InspectorFactRow("events", str(len(take.events))),
                    InspectorFactRow(
                        "playback state",
                        _playback_state_label(presentation, layer=layer, take=take),
                    ),
                ),
            ),
            InspectorSection(
                section_id="take-transfer",
                label="Sync & Transfer",
                rows=_layer_transfer_rows(presentation, layer),
            ),
            InspectorSection(
                section_id="take-context",
                label="Context",
                rows=_selection_playback_context_rows(presentation),
            ),
        ),
        context_sections=_shared_context_sections(
            presentation=presentation,
            layer=layer,
            take=take,
            hit_target=hit_target,
            has_selected_events=bool(presentation.selected_event_ids),
            include_layer_transfer_controls=False,
        ),
    )


def _event_contract(
    presentation: TimelinePresentation,
    *,
    layer: LayerPresentation,
    take: TakeLanePresentation | None,
    event: EventPresentation,
    hit_target: TimelineInspectorHitTarget | None,
) -> InspectorContract:
    take_name = "Main take" if take is None else take.name
    take_id = layer.main_take_id if take is None else take.take_id
    sections = (
        InspectorSection(
            section_id="event-core",
            label="Event",
            rows=(
                InspectorFactRow("id", str(event.event_id)),
                InspectorFactRow("start", _format_seconds(event.start)),
                InspectorFactRow("end", _format_seconds(event.end)),
                InspectorFactRow("duration", _format_seconds(event.duration)),
                InspectorFactRow("layer", layer.title),
                InspectorFactRow("take", f"{take_name} ({take_id or 'none'})"),
                InspectorFactRow(
                    "playback state", _playback_state_label(presentation, layer=layer, take=take)
                ),
            ),
        ),
        InspectorSection(
            section_id="event-transfer",
            label="Sync & Transfer",
            rows=_layer_transfer_rows(presentation, layer),
        ),
        InspectorSection(
            section_id="event-context",
            label="Context",
            rows=_selection_playback_context_rows(presentation),
        ),
    )
    return InspectorContract(
        title=f"Event {event.label}",
        identity=InspectorObjectIdentity(
            object_id=str(event.event_id),
            object_type="event",
            label=event.label,
        ),
        sections=sections,
        context_sections=(
            *_event_context_sections(
                presentation=presentation,
                layer=layer,
                take=take,
                event=event,
            ),
            *_shared_context_sections(
                presentation=presentation,
                layer=layer,
                take=take,
                hit_target=hit_target,
                has_selected_events=bool(presentation.selected_event_ids),
                include_layer_transfer_controls=False,
            ),
        ),
    )
