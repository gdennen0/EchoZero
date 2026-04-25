"""
Timeline layer presentation builders for the canonical app contract.
Exists to keep layer, take, and event shaping out of the TimelineAssembler facade.
Connects timeline truth and transfer-plan state to LayerPresentation rows.
"""

from echozero.application.presentation.models import (
    EventPresentation,
    LayerHeaderControlPresentation,
    LayerPresentation,
    LayerStatusPresentation,
    TakeActionPresentation,
    TakeLanePresentation,
    default_take_actions,
)
from echozero.application.session.models import BatchTransferPlanRowState, Session
from echozero.application.shared.enums import LayerKind, SyncMode
from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.timeline.assembler_state import TimelineAssemblyState
from echozero.application.timeline.models import Event, Layer, Take
from echozero.application.timeline.object_actions import pipeline_actions_for_audio_layer

__all__ = ["assemble_layer"]


def assemble_layer(
    layer: Layer,
    *,
    session: Session,
    state: TimelineAssemblyState,
) -> LayerPresentation:
    """Build one UI-facing layer row from canonical timeline truth."""

    primary_take = _main_take(layer)
    main_events = _assemble_events(
        layer.id,
        primary_take.id if primary_take is not None else None,
        primary_take.events if primary_take is not None else [],
        state,
    )
    take_rows = _assemble_take_rows(layer, state=state)

    badges: list[str] = ["main", layer.kind.value]
    if layer.sync.connected:
        badges.append("sync")

    status = LayerStatusPresentation(
        stale=layer.status.stale,
        manually_modified=layer.status.manually_modified,
        source_label=_source_label(layer, primary_take),
        sync_label="Connected" if layer.sync.connected else "No sync",
        stale_reason=layer.status.stale_reason or "",
        source_layer_id=str(layer.provenance.source_layer_id or ""),
        source_song_version_id=str(layer.provenance.source_song_version_id or ""),
        pipeline_id=layer.provenance.pipeline_id or "",
        output_name=layer.provenance.output_name or "",
        source_run_id=layer.provenance.source_run_id or "",
    )

    layer_selected = _is_layer_selected(layer.id, state)
    return LayerPresentation(
        layer_id=layer.id,
        main_take_id=primary_take.id if primary_take is not None else None,
        title=layer.name,
        subtitle="",
        kind=layer.kind,
        is_selected=layer_selected,
        is_playback_active=layer.id == state.active_playback_layer_id,
        is_expanded=layer.presentation_hints.expanded,
        events=main_events,
        takes=take_rows,
        visible=layer.presentation_hints.visible,
        locked=layer.presentation_hints.locked,
        gain_db=layer.mixer.gain_db,
        pan=layer.mixer.pan,
        playback_mode=layer.playback.mode,
        playback_enabled=layer.playback.enabled,
        sync_mode=_coerce_sync_mode(layer.sync.mode),
        sync_connected=layer.sync.connected,
        live_sync_state=layer.sync.live_sync_state,
        live_sync_pause_reason=layer.sync.live_sync_pause_reason,
        live_sync_divergent=layer.sync.live_sync_divergent,
        sync_target_label=_sync_target_label(layer),
        push_target_label=_push_target_label(session, layer),
        push_selection_count=_push_selection_count(session, layer),
        push_row_status=_push_row_status(session, layer),
        push_row_issue=_push_row_issue(session, layer),
        pull_target_label=_pull_target_label(session, layer),
        pull_selection_count=_pull_selection_count(session, layer),
        pull_row_status=_pull_row_status(session, layer),
        pull_row_issue=_pull_row_issue(session, layer),
        color=layer.presentation_hints.color,
        badges=badges,
        header_controls=_assemble_header_controls(
            layer,
            is_playback_active=layer.id == state.active_playback_layer_id,
            is_selected=layer_selected,
        ),
        playback_source_ref=layer.playback.armed_source_ref,
        status=status,
    )


def _assemble_header_controls(
    layer: Layer,
    *,
    is_playback_active: bool,
    is_selected: bool,
) -> list[LayerHeaderControlPresentation]:
    controls = [
        LayerHeaderControlPresentation(
            control_id="set_active_playback_target",
            label="ACTIVE",
            kind="toggle",
            active=is_playback_active,
        )
    ]
    if is_selected and _layer_pipeline_action_count(layer) > 0:
        controls.append(
            LayerHeaderControlPresentation(
                control_id="layer_pipeline_actions",
                label="Pipelines",
                kind="action",
            )
        )
    return controls


def _assemble_take_rows(
    layer: Layer,
    *,
    state: TimelineAssemblyState,
) -> list[TakeLanePresentation]:
    layer_selected = _is_layer_selected(layer.id, state)
    return [
        TakeLanePresentation(
            take_id=take.id,
            name=take.name,
            is_main=False,
            kind=layer.kind,
            is_selected=layer_selected and take.id == state.selected_take_id,
            is_playback_active=(
                layer.id == state.active_playback_layer_id
                and take.id == state.active_playback_take_id
            ),
            events=_assemble_events(layer.id, take.id, take.events, state),
            source_ref=take.source_ref,
            playback_source_ref=layer.playback.armed_source_ref,
            actions=_take_actions(
                has_selection=_take_has_selected_events(
                    layer.id,
                    take.id,
                    state.selected_event_ref_keys,
                )
            ),
        )
        for take in layer.takes[1:]
    ]


def _assemble_events(
    layer_id: LayerId,
    take_id: TakeId | None,
    events: list[Event],
    state: TimelineAssemblyState,
) -> list[EventPresentation]:
    ordered = events
    if len(events) > 1:
        for index in range(1, len(events)):
            previous = events[index - 1]
            current = events[index]
            if (previous.start, previous.end, str(previous.id)) > (
                current.start,
                current.end,
                str(current.id),
            ):
                ordered = sorted(
                    events,
                    key=lambda value: (value.start, value.end, str(value.id)),
                )
                break

    return [
        EventPresentation(
            event_id=event.id,
            start=event.start,
            end=event.end,
            label=event.label,
            color=event.color,
            muted=event.muted,
            source_event_id=event.source_event_id,
            payload_ref=event.payload_ref,
            is_selected=(
                (
                    str(layer_id),
                    str(take_id),
                    str(event.id),
                )
                in state.selected_event_ref_keys
            )
            or (
                not state.selected_event_ref_keys and event.id in state.selected_event_ids_set
            ),
            badges=["muted"] if event.muted else [],
        )
        for event in ordered
    ]


def _take_actions(*, has_selection: bool) -> list[TakeActionPresentation]:
    return default_take_actions(has_selection=has_selection)


def _take_has_selected_events(
    layer_id: LayerId,
    take_id: TakeId,
    selected_event_ref_keys: frozenset[tuple[str, str, str]],
) -> bool:
    return any(
        key_layer_id == str(layer_id) and key_take_id == str(take_id)
        for key_layer_id, key_take_id, _event_id in selected_event_ref_keys
    )


def _is_layer_selected(layer_id: LayerId, state: TimelineAssemblyState) -> bool:
    return layer_id in state.selected_layer_ids or (
        not state.selected_layer_ids and layer_id == state.selected_layer_id
    )


def _coerce_sync_mode(raw_mode: object) -> SyncMode:
    if isinstance(raw_mode, SyncMode):
        return raw_mode

    raw_value = str(raw_mode)
    for sync_mode in SyncMode:
        if sync_mode.value == raw_value:
            return sync_mode
    return SyncMode.NONE


def _layer_pipeline_action_count(layer: Layer) -> int:
    return len(
        pipeline_actions_for_audio_layer(
            is_stem_capable=_is_stem_capable_layer(layer),
            is_drum_capable=_is_drum_capable_layer(layer),
            is_song_drum_capable=_is_song_drum_capable_layer(layer),
        )
    )


def _is_stem_capable_layer(layer: Layer) -> bool:
    return layer.kind is LayerKind.AUDIO


def _is_drum_capable_layer(layer: Layer) -> bool:
    if not _is_stem_capable_layer(layer):
        return False
    title = layer.name.strip().lower()
    pipeline_id = (layer.provenance.pipeline_id or "").strip().lower()
    output_name = (layer.provenance.output_name or "").strip().lower()
    return "drum" in title or "drum" in pipeline_id or "drum" in output_name


def _is_song_drum_capable_layer(layer: Layer) -> bool:
    if not _is_stem_capable_layer(layer) or _is_drum_capable_layer(layer):
        return False
    pipeline_id = (layer.provenance.pipeline_id or "").strip()
    return layer.provenance.source_layer_id is None and not pipeline_id


def _main_take(layer: Layer) -> Take | None:
    if not layer.takes:
        return None
    return layer.takes[0]


def _source_label(layer: Layer, primary_take: Take | None) -> str:
    if layer.provenance.pipeline_id and layer.provenance.output_name:
        return f"{layer.provenance.pipeline_id} · {layer.provenance.output_name}"
    if primary_take is not None and primary_take.source_ref:
        return primary_take.source_ref
    return ""


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


def _push_target_label(session: Session, layer: Layer) -> str:
    row = _push_plan_row(session, layer)
    if row is not None and row.target_label:
        return row.target_label
    return _sync_target_label(layer)


def _push_selection_count(session: Session, layer: Layer) -> int:
    row = _push_plan_row(session, layer)
    return 0 if row is None else row.selected_count


def _push_row_status(session: Session, layer: Layer) -> str:
    row = _push_plan_row(session, layer)
    return "" if row is None else row.status


def _push_row_issue(session: Session, layer: Layer) -> str:
    row = _push_plan_row(session, layer)
    return "" if row is None or row.issue is None else row.issue


def _pull_target_label(session: Session, layer: Layer) -> str:
    row = _pull_plan_row(session, layer)
    return "" if row is None else row.target_label


def _pull_selection_count(session: Session, layer: Layer) -> int:
    row = _pull_plan_row(session, layer)
    return 0 if row is None else row.selected_count


def _pull_row_status(session: Session, layer: Layer) -> str:
    row = _pull_plan_row(session, layer)
    return "" if row is None else row.status


def _pull_row_issue(session: Session, layer: Layer) -> str:
    row = _pull_plan_row(session, layer)
    return "" if row is None or row.issue is None else row.issue


def _push_plan_row(session: Session, layer: Layer) -> BatchTransferPlanRowState | None:
    plan = session.batch_transfer_plan
    if plan is None:
        return None
    for row in plan.rows:
        if row.direction == "push" and row.source_layer_id == layer.id:
            return row
    return None


def _pull_plan_row(session: Session, layer: Layer) -> BatchTransferPlanRowState | None:
    plan = session.batch_transfer_plan
    if plan is None:
        return None
    for row in plan.rows:
        if row.direction == "pull" and row.target_layer_id == layer.id:
            return row
    return None
