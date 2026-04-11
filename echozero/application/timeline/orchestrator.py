"""Timeline orchestration for the new EchoZero application layer."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from echozero.application.mixer.service import MixerService
from echozero.application.playback.service import PlaybackService
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.session.models import ManualPushDiffPreview, ManualPushTrackOption
from echozero.application.session.service import SessionService
from echozero.application.sync.service import SyncService
if TYPE_CHECKING:
    from echozero.application.timeline.assembler import TimelineAssembler
from echozero.application.timeline.intents import (
    ClearSelection,
    ConfirmPushToMA3,
    DuplicateSelectedEvents,
    DisableSync,
    MoveSelectedEvents,
    NudgeSelectedEvents,
    SelectAllEvents,
    SelectEvent,
    SelectLayer,
    SelectTake,
    EnableSync,
    OpenPushToMA3Dialog,
    Play,
    Pause,
    Seek,
    SelectPushTargetTrack,
    SetGain,
    SetPushTrackOptions,
    Stop,
    TimelineIntent,
    ToggleLayerExpanded,
    ToggleMute,
    ToggleSolo,
    TriggerTakeAction,
)
from echozero.application.timeline.models import Timeline, Layer, Take, Event
from echozero.application.transport.service import TransportService

_KEYBOARD_STEP_SECONDS = 1.0 / 30.0


@dataclass(slots=True)
class TimelineOrchestrator:
    """Coordinates timeline intents across sibling application services."""

    session_service: SessionService
    transport_service: TransportService
    mixer_service: MixerService
    playback_service: PlaybackService
    sync_service: SyncService
    assembler: 'TimelineAssembler'

    def handle(self, timeline: Timeline, intent: TimelineIntent) -> TimelinePresentation:
        if isinstance(intent, SelectLayer):
            timeline.selection.selected_layer_id = intent.layer_id
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_ids = []

        elif isinstance(intent, SelectTake):
            self._handle_select_take(timeline, intent.layer_id, intent.take_id)

        elif isinstance(intent, SelectEvent):
            self._handle_select_event(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                event_id=intent.event_id,
                mode=intent.mode,
            )

        elif isinstance(intent, ClearSelection):
            timeline.selection.selected_take_id = None
            timeline.selection.selected_event_ids = []

        elif isinstance(intent, SelectAllEvents):
            self._handle_select_all_events(timeline)

        elif isinstance(intent, MoveSelectedEvents):
            self._handle_move_selected_events(
                timeline,
                delta_seconds=float(intent.delta_seconds),
                target_layer_id=intent.target_layer_id,
            )

        elif isinstance(intent, ToggleLayerExpanded):
            layer = self._find_layer(timeline, intent.layer_id)
            layer.presentation_hints.expanded = not layer.presentation_hints.expanded

        elif isinstance(intent, TriggerTakeAction):
            self._handle_trigger_take_action(
                timeline,
                layer_id=intent.layer_id,
                take_id=intent.take_id,
                action_id=intent.action_id,
            )

        elif isinstance(intent, NudgeSelectedEvents):
            self._handle_nudge_selected_events(
                timeline,
                direction=intent.direction,
                steps=intent.steps,
            )

        elif isinstance(intent, DuplicateSelectedEvents):
            self._handle_duplicate_selected_events(timeline, steps=intent.steps)

        elif isinstance(intent, Play):
            self.transport_service.play()

        elif isinstance(intent, Pause):
            self.transport_service.pause()

        elif isinstance(intent, Stop):
            self.transport_service.stop()

        elif isinstance(intent, Seek):
            self.transport_service.seek(intent.position)

        elif isinstance(intent, ToggleMute):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_mute(intent.layer_id, not layer.mixer.mute)
            layer.mixer.mute = not layer.mixer.mute

        elif isinstance(intent, ToggleSolo):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_solo(intent.layer_id, not layer.mixer.solo)
            layer.mixer.solo = not layer.mixer.solo

        elif isinstance(intent, SetGain):
            layer = self._find_layer(timeline, intent.layer_id)
            self.mixer_service.set_gain(intent.layer_id, intent.gain_db)
            layer.mixer.gain_db = intent.gain_db

        elif isinstance(intent, EnableSync):
            sync_state = self.sync_service.set_mode(intent.mode)
            session = self.session_service.get_session()
            try:
                sync_state = self.sync_service.connect()
            except Exception:
                # Keep session state in lockstep with sync service error state.
                session.sync_state = self.sync_service.get_state()
                raise
            session.sync_state = sync_state

        elif isinstance(intent, DisableSync):
            session = self.session_service.get_session()
            session.sync_state = self.sync_service.disconnect()

        elif isinstance(intent, OpenPushToMA3Dialog):
            session = self.session_service.get_session()
            session.manual_push_flow.dialog_open = True
            session.manual_push_flow.selected_event_ids = list(intent.selection_event_ids)
            session.manual_push_flow.target_track_coord = None
            session.manual_push_flow.diff_gate_open = False
            session.manual_push_flow.diff_preview = None
            session.manual_push_flow.available_tracks = self._load_manual_push_track_options()

        elif isinstance(intent, SetPushTrackOptions):
            session = self.session_service.get_session()
            session.manual_push_flow.available_tracks = list(intent.tracks)

        elif isinstance(intent, SelectPushTargetTrack):
            session = self.session_service.get_session()
            available_coords = {
                track.coord
                for track in session.manual_push_flow.available_tracks
            }
            if intent.target_track_coord not in available_coords:
                raise ValueError(
                    f"SelectPushTargetTrack target_track_coord not found in available_tracks: "
                    f"{intent.target_track_coord}"
                )
            session.manual_push_flow.target_track_coord = intent.target_track_coord

        elif isinstance(intent, ConfirmPushToMA3):
            session = self.session_service.get_session()
            target_track = self._manual_push_track_by_coord(
                session.manual_push_flow.available_tracks,
                intent.target_track_coord,
            )
            session.manual_push_flow.dialog_open = False
            session.manual_push_flow.selected_event_ids = list(intent.selected_event_ids)
            session.manual_push_flow.target_track_coord = intent.target_track_coord
            session.manual_push_flow.diff_gate_open = True
            session.manual_push_flow.diff_preview = ManualPushDiffPreview(
                selected_count=len(intent.selected_event_ids),
                target_track_coord=target_track.coord,
                target_track_name=target_track.name,
                target_track_note=target_track.note,
                target_track_event_count=target_track.event_count,
            )

        session = self.session_service.get_session()
        audibility = self.mixer_service.resolve_audibility(timeline.layers)
        self.playback_service.update_runtime(
            timeline=timeline,
            transport=session.transport_state,
            audibility=audibility,
            sync=session.sync_state,
        )
        return self.assembler.assemble(timeline=timeline, session=session)

    def _handle_select_take(self, timeline: Timeline, layer_id, take_id) -> None:
        # Selection only. Selecting a take must never change timeline truth.
        self._find_layer(timeline, layer_id)
        timeline.selection.selected_layer_id = layer_id
        timeline.selection.selected_take_id = take_id
        timeline.selection.selected_event_ids = []

    def _handle_select_event(self, timeline: Timeline, layer_id, take_id, event_id, mode: str) -> None:
        layer = self._find_layer(timeline, layer_id)
        if event_id is None:
            timeline.selection.selected_layer_id = layer.id
            timeline.selection.selected_take_id = take_id
            timeline.selection.selected_event_ids = []
            return

        mode_normalized = (mode or "replace").strip().lower()
        selected_ids = list(timeline.selection.selected_event_ids)

        if mode_normalized == "replace":
            selected_ids = [event_id]
        elif mode_normalized == "additive":
            if event_id not in selected_ids:
                selected_ids.append(event_id)
        elif mode_normalized == "toggle":
            if event_id in selected_ids:
                selected_ids = [selected_id for selected_id in selected_ids if selected_id != event_id]
            else:
                selected_ids.append(event_id)
        else:
            raise ValueError(f"Unsupported selection mode: {mode}")

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_event_ids = selected_ids
        timeline.selection.selected_take_id = take_id if selected_ids else None

    def _handle_select_all_events(self, timeline: Timeline) -> None:
        selected_layer_id = timeline.selection.selected_layer_id
        target_layers: list[Layer]
        if selected_layer_id is not None:
            layer = self._find_layer(timeline, selected_layer_id)
            target_layers = [layer]
        else:
            target_layers = [layer for layer in timeline.layers if layer.presentation_hints.visible and not layer.presentation_hints.locked]

        selected_event_ids: list = []
        selected_take_id = None
        for layer in target_layers:
            if not layer.presentation_hints.visible or layer.presentation_hints.locked:
                continue
            for take in layer.takes:
                if take.events and selected_take_id is None:
                    selected_take_id = take.id
                selected_event_ids.extend(event.id for event in take.events)

        timeline.selection.selected_event_ids = selected_event_ids
        timeline.selection.selected_take_id = selected_take_id

    def _handle_trigger_take_action(self, timeline: Timeline, layer_id, take_id, action_id: str) -> None:
        layer = self._find_layer(timeline, layer_id)
        source_take = self._find_take(layer, take_id)
        main_take = self._main_take(layer)
        if source_take is None or main_take is None:
            return

        normalized = (action_id or "").strip().lower()
        if normalized in {"overwrite_main", "promote_take"}:
            if source_take.id == main_take.id:
                return
            main_take.events = self._clone_events_for_target(source_take.events, main_take)
        elif normalized == "merge_main":
            if source_take.id == main_take.id:
                return
            merged = list(main_take.events)
            merged.extend(self._clone_events_for_target(source_take.events, main_take))
            main_take.events = sorted(merged, key=lambda event: (event.start, event.end, str(event.id)))
        else:
            return

        timeline.selection.selected_layer_id = layer.id
        timeline.selection.selected_take_id = main_take.id
        timeline.selection.selected_event_ids = []

    def _handle_move_selected_events(self, timeline: Timeline, delta_seconds: float, target_layer_id) -> None:
        selected_ids = list(timeline.selection.selected_event_ids)
        if not selected_ids:
            return

        records = self._selected_event_records(timeline, selected_ids)
        if not records:
            timeline.selection.selected_event_ids = []
            timeline.selection.selected_take_id = None
            return

        applied_delta = max(float(delta_seconds), -min(record.event.start for record in records))
        source_layer_ids = {record.layer.id for record in records}
        source_layer_id = timeline.selection.selected_layer_id
        transfer_target = None

        if target_layer_id is not None and target_layer_id not in source_layer_ids:
            transfer_target = self._find_layer(timeline, target_layer_id)
            if (
                transfer_target.kind != records[0].layer.kind
                or transfer_target.kind.value != "event"
                or transfer_target.presentation_hints.locked
                or not transfer_target.presentation_hints.visible
            ):
                return

        affected_takes: dict[object, Take] = {}
        if transfer_target is None:
            for record in records:
                record.event.start += applied_delta
                record.event.end += applied_delta
                affected_takes[record.take.id] = record.take
            self._sort_take_events(*affected_takes.values())
            if source_layer_id is None and records:
                source_layer_id = records[0].layer.id
            timeline.selection.selected_layer_id = source_layer_id
            timeline.selection.selected_take_id = self._resolve_selected_take_id(
                self._find_layer(timeline, source_layer_id) if source_layer_id is not None else records[0].layer,
                selected_ids,
                fallback_take_id=timeline.selection.selected_take_id,
            )
            return

        target_take = self._main_take(transfer_target)
        if target_take is None:
            return

        for record in records:
            record.take.events = [candidate for candidate in record.take.events if candidate.id != record.event.id]
            affected_takes[record.take.id] = record.take
            record.event.start += applied_delta
            record.event.end += applied_delta
            record.event.take_id = target_take.id
            target_take.events.append(record.event)

        affected_takes[target_take.id] = target_take
        self._sort_take_events(*affected_takes.values())
        timeline.selection.selected_layer_id = transfer_target.id
        timeline.selection.selected_take_id = target_take.id
        timeline.selection.selected_event_ids = [record.event.id for record in records]

    @dataclass(slots=True)
    class _EventRecord:
        layer: Layer
        take: Take
        event: Event

    def _selected_event_records(self, timeline: Timeline, selected_ids: list) -> list[_EventRecord]:
        selected_lookup = set(selected_ids)
        order = {event_id: idx for idx, event_id in enumerate(selected_ids)}
        records: list[TimelineOrchestrator._EventRecord] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    if event.id in selected_lookup:
                        records.append(self._EventRecord(layer=layer, take=take, event=event))
        records.sort(key=lambda record: order.get(record.event.id, len(order)))
        return records

    @staticmethod
    def _resolve_selected_take_id(layer: Layer, selected_ids: list, fallback_take_id=None):
        selected_lookup = set(selected_ids)
        if fallback_take_id is not None:
            fallback = TimelineOrchestrator._find_take(layer, fallback_take_id)
            if fallback is not None and any(event.id in selected_lookup for event in fallback.events):
                return fallback_take_id

        for take in layer.takes:
            if any(event.id in selected_lookup for event in take.events):
                return take.id
        return None

    @staticmethod
    def _sort_take_events(*takes: Take) -> None:
        for take in takes:
            take.events = sorted(take.events, key=lambda event: (event.start, event.end, str(event.id)))

    def _handle_nudge_selected_events(self, timeline: Timeline, direction: int, steps: int) -> None:
        if direction == 0 or steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(direction) * float(steps) * _KEYBOARD_STEP_SECONDS
        for _layer, take, event in selected:
            duration = event.duration
            next_start = max(0.0, event.start + delta)
            event.start = next_start
            event.end = next_start + duration
            take.events = self._sorted_events(take.events)

    def _handle_duplicate_selected_events(self, timeline: Timeline, steps: int) -> None:
        if steps <= 0:
            return

        selected = self._selected_events(timeline)
        if not selected:
            return

        delta = float(steps) * _KEYBOARD_STEP_SECONDS
        existing_ids = self._all_event_ids(timeline)
        duplicated_ids: list = []
        selected_layer_id = timeline.selection.selected_layer_id
        selected_take_id = timeline.selection.selected_take_id

        for layer, take, event in selected:
            duplicate_id = self._next_duplicate_event_id(take, event, existing_ids)
            duplicate = Event(
                id=duplicate_id,
                take_id=take.id,
                start=event.start + delta,
                end=event.end + delta,
                payload_ref=event.payload_ref,
                label=event.label,
                color=event.color,
                muted=event.muted,
            )
            take.events = self._sorted_events([*take.events, duplicate])
            existing_ids.add(str(duplicate.id))
            duplicated_ids.append(duplicate.id)
            selected_layer_id = layer.id
            selected_take_id = take.id

        timeline.selection.selected_layer_id = selected_layer_id
        timeline.selection.selected_take_id = selected_take_id
        timeline.selection.selected_event_ids = duplicated_ids

    @staticmethod
    def _clone_events_for_target(events: list[Event], target_take: Take) -> list[Event]:
        clones: list[Event] = []
        for idx, event in enumerate(events, start=1):
            clones.append(
                Event(
                    id=f"{target_take.id}:from:{event.id}:{idx}",
                    take_id=target_take.id,
                    start=event.start,
                    end=event.end,
                    payload_ref=event.payload_ref,
                    label=event.label,
                    color=event.color,
                    muted=event.muted,
                )
            )
        return clones

    @staticmethod
    def _main_take(layer: Layer) -> Take | None:
        if layer.takes:
            return layer.takes[0]
        return None

    @staticmethod
    def _find_take(layer: Layer, take_id) -> Take | None:
        for take in layer.takes:
            if take.id == take_id:
                return take
        return None

    @staticmethod
    def _sorted_events(events: list[Event]) -> list[Event]:
        return sorted(events, key=lambda event: (event.start, event.end, str(event.id)))

    @staticmethod
    def _all_event_ids(timeline: Timeline) -> set[str]:
        return {
            str(event.id)
            for layer in timeline.layers
            for take in layer.takes
            for event in take.events
        }

    @staticmethod
    def _next_duplicate_event_id(take: Take, event: Event, existing_ids: set[str]):
        index = 1
        while True:
            candidate = f"{take.id}:dup:{event.id}:{index}"
            if candidate not in existing_ids:
                return candidate
            index += 1

    def _selected_events(self, timeline: Timeline) -> list[tuple[Layer, Take, Event]]:
        selected_ids = set(timeline.selection.selected_event_ids)
        if not selected_ids:
            return []

        selected_order = {
            str(event_id): index for index, event_id in enumerate(timeline.selection.selected_event_ids)
        }
        selected: list[tuple[Layer, Take, Event]] = []
        for layer in timeline.layers:
            for take in layer.takes:
                for event in take.events:
                    if event.id in selected_ids:
                        selected.append((layer, take, event))

        selected.sort(key=lambda item: selected_order[str(item[2].id)])
        return selected

    def _load_manual_push_track_options(self) -> list[ManualPushTrackOption]:
        provider = self.sync_service
        if hasattr(provider, "list_push_track_options"):
            raw_tracks = provider.list_push_track_options()
        elif hasattr(provider, "get_available_ma3_tracks"):
            raw_tracks = provider.get_available_ma3_tracks()
        else:
            return []

        return [
            self._normalize_manual_push_track_option(raw_track)
            for raw_track in raw_tracks or []
        ]

    @staticmethod
    def _normalize_manual_push_track_option(raw_track: Any) -> ManualPushTrackOption:
        if isinstance(raw_track, ManualPushTrackOption):
            return raw_track

        coord = TimelineOrchestrator._track_option_value(raw_track, "coord")
        name = TimelineOrchestrator._track_option_value(raw_track, "name")
        note = TimelineOrchestrator._track_option_value(raw_track, "note")
        event_count = TimelineOrchestrator._track_option_value(raw_track, "event_count")

        return ManualPushTrackOption(
            coord=str(coord or ""),
            name=str(name or ""),
            note=None if note is None else str(note),
            event_count=TimelineOrchestrator._coerce_optional_int(event_count),
        )

    @staticmethod
    def _track_option_value(raw_track: Any, key: str) -> Any:
        if isinstance(raw_track, dict):
            return raw_track.get(key)
        return getattr(raw_track, key, None)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _manual_push_track_by_coord(
        available_tracks: list[ManualPushTrackOption],
        target_track_coord: str,
    ) -> ManualPushTrackOption:
        for track in available_tracks:
            if track.coord == target_track_coord:
                return track
        raise ValueError(
            f"ConfirmPushToMA3 target_track_coord not found in available_tracks: "
            f"{target_track_coord}"
        )

    def _find_layer(self, timeline: Timeline, layer_id):
        for layer in timeline.layers:
            if layer.id == layer_id:
                return layer
        raise ValueError(f"Layer not found: {layer_id}")
