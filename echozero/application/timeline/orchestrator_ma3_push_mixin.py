"""MA3 push helpers for the canonical timeline orchestrator.
Exists to keep operator-grade MA3 routing and push execution out of the batch transfer mixins.
Connects typed push intents to layer main-take data and the sync-service push boundary.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol, cast

from echozero.application.session.models import (
    ManualPushSequenceOption,
    ManualPushTimecodeOption,
    ManualPushTrackGroupOption,
    ManualPushTrackOption,
)
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.cue_numbers import cue_number_from_ref_text, cue_number_text
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import LayerId
from echozero.application.timeline.ma3_push_intents import (
    AssignMA3TrackSequence,
    CreateMA3Timecode,
    CreateMA3Track,
    CreateMA3TrackGroup,
    CreateMA3Sequence,
    MA3PushScope,
    MA3PushTargetMode,
    PollMA3PushOperation,
    MA3SequenceCreationMode,
    MA3SequenceRefreshRangeMode,
    MA3TrackSequenceAction,
    PrepareMA3TrackForPush,
    PushLayerToMA3,
    RefreshMA3Sequences,
)
from echozero.application.timeline.models import Event, Layer, Timeline


class _MA3PushSessionService(Protocol):
    def get_session(self) -> Any: ...


class _MA3PushHost(Protocol):
    session_service: _MA3PushSessionService
    sync_service: Any

    def _find_layer(self, timeline: Timeline, layer_id: LayerId) -> Layer: ...

    @staticmethod
    def _main_take(layer: Layer): ...

    def _load_manual_push_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[ManualPushTrackOption]: ...

    def _load_manual_push_timecode_options(self) -> list[ManualPushTimecodeOption]: ...

    def _load_manual_push_track_group_options(
        self,
        *,
        timecode_no: int,
    ) -> list[ManualPushTrackGroupOption]: ...

    def _default_manual_push_track_group_no(
        self,
        *,
        selected_timecode_no: int | None,
        available_track_groups: list[ManualPushTrackGroupOption],
        preferred_track_coord: str | None = None,
    ) -> int | None: ...

    def _coerce_optional_positive_int(self, value: Any) -> int | None: ...

    def _load_manual_push_sequence_options(
        self,
        *,
        range_mode: MA3SequenceRefreshRangeMode,
    ): ...

    def _normalize_manual_push_timecode_option(
        self, raw_timecode: Any
    ) -> ManualPushTimecodeOption: ...

    def _normalize_manual_push_track_group_option(
        self, raw_group: Any
    ) -> ManualPushTrackGroupOption: ...

    def _normalize_manual_push_track_option(self, raw_track: Any) -> ManualPushTrackOption: ...

    def _normalize_manual_push_sequence_option(
        self, raw_sequence: Any
    ) -> ManualPushSequenceOption: ...


class TimelineOrchestratorMA3PushMixin:
    """Handles saved-route MA3 updates and direct layer push execution."""

    def _handle_refresh_ma3_push_tracks(self, intent) -> None:
        self._call_sync_capability(
            "refresh_push_track_options",
            error_message="Sync service does not support MA3 push-track refresh",
            target_track_coord=intent.target_track_coord,
            timecode_no=intent.timecode_no,
            track_group_no=intent.track_group_no,
        )
        self._refresh_manual_push_tracks(
            target_track_coord=intent.target_track_coord,
            timecode_no=intent.timecode_no,
            track_group_no=intent.track_group_no,
        )

    def _handle_refresh_ma3_sequences(self, intent: RefreshMA3Sequences) -> None:
        session = cast(_MA3PushHost, self).session_service.get_session()
        available_sequences, current_song_range = self._load_manual_push_sequence_options(
            range_mode=intent.range_mode
        )
        session.manual_push_flow.available_sequences = list(available_sequences)
        session.manual_push_flow.current_song_sequence_range = current_song_range

    def _handle_assign_ma3_track_sequence(self, intent: AssignMA3TrackSequence) -> None:
        self._assign_track_sequence(
            target_track_coord=intent.target_track_coord,
            sequence_no=intent.sequence_no,
        )

    def _handle_create_ma3_sequence(self, intent: CreateMA3Sequence) -> None:
        self._create_ma3_sequence(intent)

    def _handle_create_ma3_timecode(self, intent: CreateMA3Timecode) -> None:
        self._create_ma3_timecode(intent)

    def _handle_create_ma3_track_group(self, intent: CreateMA3TrackGroup) -> None:
        self._create_ma3_track_group(intent)

    def _handle_create_ma3_track(self, intent: CreateMA3Track) -> None:
        self._create_ma3_track(intent)

    def _handle_prepare_ma3_track_for_push(self, intent: PrepareMA3TrackForPush) -> None:
        self._prepare_track_for_push(intent.target_track_coord)

    def _handle_set_layer_ma3_route(
        self,
        timeline: Timeline,
        *,
        layer_id: LayerId,
        target_track_coord: str,
        ma3_channel_no: int | None = None,
        sequence_action: MA3TrackSequenceAction | None = None,
    ) -> None:
        layer = self._find_layer(timeline, layer_id)
        self._require_ma3_event_layer(layer, action_name="SetLayerMA3Route")
        route_events_for_sequence = self._events_for_layer_sequence_prep(layer)
        self._prepare_target_track_for_push_if_needed(
            target_track_coord=target_track_coord,
            sequence_action=sequence_action,
            action_name="SetLayerMA3Route",
            selected_events=route_events_for_sequence,
            allow_hitmaker_auto=True,
            fallback_event_type="hit",
        )
        layer.sync.ma3_track_coord = target_track_coord
        layer.sync.ma3_channel_no = ma3_channel_no

    def _handle_push_layer_to_ma3(self, timeline: Timeline, intent: PushLayerToMA3) -> None:
        layer = self._find_layer(timeline, intent.layer_id)
        self._require_ma3_event_layer(layer, action_name="PushLayerToMA3")

        selected_events = self._resolve_push_events_for_layer(timeline, layer, intent)
        transfer_events = self._prepare_events_for_ma3_push(
            layer,
            selected_events,
            skip_cue_1=intent.skip_cue_1,
        )
        target_track_coord = self._resolve_push_target_coord(layer, intent)
        ma3_channel_no = self._resolve_push_target_channel_no(layer, intent)
        self._prepare_target_track_for_push_if_needed(
            target_track_coord=target_track_coord,
            sequence_action=intent.sequence_action,
            action_name="PushLayerToMA3",
            selected_events=transfer_events,
            allow_hitmaker_auto=True,
            fallback_event_type="hit",
        )
        session = cast(_MA3PushHost, self).session_service.get_session()
        push_offset_seconds = self._resolve_project_ma3_push_offset_seconds(session)
        session.manual_push_flow.target_track_coord = target_track_coord
        session.manual_push_flow.selected_event_ids = [event.id for event in selected_events]
        session.manual_push_flow.diff_gate_open = False
        session.manual_push_flow.diff_preview = None
        start_push = getattr(cast(_MA3PushHost, self).sync_service, "start_push", None)
        if callable(start_push):
            operation_id = str(
                self._call_sync_capability(
                    "start_push",
                    error_message="Sync service does not support MA3 async push operations",
                    target_track_coord=target_track_coord,
                    ma3_channel_no=ma3_channel_no,
                    selected_events=transfer_events,
                    transfer_mode=intent.apply_mode.value,
                    start_offset_seconds=push_offset_seconds,
                )
                or ""
            ).strip()
            if not operation_id:
                raise RuntimeError("Sync service returned an empty MA3 operation id")
            session.manual_push_flow.operation_id = operation_id
            session.manual_push_flow.operation_status = "running"
            session.manual_push_flow.operation_message = (
                f"Sending {len(selected_events)} event(s) to {target_track_coord}"
            )
            return

        self._call_sync_capability(
            "apply_push_transfer",
            error_message="Sync service does not support canonical MA3 push apply",
            target_track_coord=target_track_coord,
            ma3_channel_no=ma3_channel_no,
            selected_events=transfer_events,
            transfer_mode=intent.apply_mode.value,
            start_offset_seconds=push_offset_seconds,
        )
        session.manual_push_flow.operation_id = None
        session.manual_push_flow.operation_status = "success"
        session.manual_push_flow.operation_message = (
            f"Sent {len(selected_events)} event(s) to {target_track_coord}"
        )
        self._call_sync_capability(
            "refresh_push_track_options",
            error_message="Sync service does not support MA3 push-track refresh",
            target_track_coord=target_track_coord,
        )
        self._refresh_manual_push_tracks(target_track_coord=target_track_coord)

    def _handle_poll_ma3_push_operation(
        self,
        timeline: Timeline,
        intent: PollMA3PushOperation,
    ) -> None:
        del timeline
        session = cast(_MA3PushHost, self).session_service.get_session()
        flow = session.manual_push_flow
        if flow.operation_id != intent.operation_id:
            return
        operation = self._call_sync_capability(
            "get_operation",
            error_message="Sync service does not support MA3 operation polling",
            operation_id=intent.operation_id,
        )
        if not isinstance(operation, dict):
            raise RuntimeError("MA3 operation polling returned a non-dict payload")

        status = str(operation.get("status") or "").strip().lower()
        message = str(operation.get("message") or "").strip()
        if status in {"running", "queued"}:
            flow.operation_status = "running"
            if message:
                flow.operation_message = message
            return

        if status == "success":
            flow.operation_status = "success"
            flow.operation_message = message or "MA3 push completed"
            self._call_sync_capability(
                "refresh_push_track_options",
                error_message="Sync service does not support MA3 push-track refresh",
                target_track_coord=flow.target_track_coord,
            )
            self._refresh_manual_push_tracks(target_track_coord=flow.target_track_coord)
            flow.operation_id = None
            return

        if status in {"error", "failed"}:
            error_text = str(operation.get("error") or message or "MA3 push failed").strip()
            flow.operation_status = "error"
            flow.operation_message = error_text
            flow.operation_id = None
            return

        if status == "cancelled":
            flow.operation_status = "idle"
            flow.operation_message = "Push cancelled"
            flow.operation_id = None
            return

        raise RuntimeError(f"Unsupported MA3 operation status: {status or 'unknown'}")

    def _refresh_manual_push_tracks(
        self,
        *,
        target_track_coord: str | None = None,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[ManualPushTrackOption]:
        session = cast(_MA3PushHost, self).session_service.get_session()
        flow = session.manual_push_flow

        explicit_timecode_no = cast(_MA3PushHost, self)._coerce_optional_positive_int(timecode_no)
        explicit_track_group_no = cast(_MA3PushHost, self)._coerce_optional_positive_int(
            track_group_no
        )
        resolved_timecode_no = explicit_timecode_no
        if resolved_timecode_no is None:
            resolved_timecode_no = self._resolve_manual_push_timecode_no(
                target_track_coord=target_track_coord
            )
        if resolved_timecode_no is None:
            resolved_timecode_no = flow.selected_timecode_no

        flow.available_timecodes = list(self._load_manual_push_timecode_options())
        available_timecode_numbers = {timecode.number for timecode in flow.available_timecodes}
        if resolved_timecode_no not in available_timecode_numbers:
            resolved_timecode_no = (
                None if not flow.available_timecodes else flow.available_timecodes[0].number
            )
        flow.selected_timecode_no = resolved_timecode_no

        if resolved_timecode_no is None:
            flow.available_track_groups = []
            flow.selected_track_group_no = None
            flow.available_tracks = []
            return []

        flow.available_track_groups = list(
            self._load_manual_push_track_group_options(timecode_no=resolved_timecode_no)
        )
        if explicit_track_group_no is None:
            preferred_track_coord = str(target_track_coord or "").strip() or None
            if preferred_track_coord is None:
                preferred_track_coord = (
                    flow.target_track_coord
                    if self._ma3_track_coord_timecode_no(flow.target_track_coord)
                    == resolved_timecode_no
                    else None
                )
            resolved_track_group_no = self._default_manual_push_track_group_no(
                selected_timecode_no=resolved_timecode_no,
                available_track_groups=flow.available_track_groups,
                preferred_track_coord=preferred_track_coord,
            )
        else:
            resolved_track_group_no = explicit_track_group_no

        available_group_numbers = {group.number for group in flow.available_track_groups}
        if resolved_track_group_no not in available_group_numbers:
            resolved_track_group_no = None
        flow.selected_track_group_no = resolved_track_group_no

        filter_track_group_no = explicit_track_group_no
        flow.available_tracks = list(
            self._load_manual_push_track_options(
                timecode_no=resolved_timecode_no,
                track_group_no=filter_track_group_no,
            )
        )
        return list(flow.available_tracks)

    def _assign_track_sequence(self, *, target_track_coord: str, sequence_no: int) -> None:
        self._call_sync_capability(
            "assign_track_sequence",
            error_message="Sync service does not support MA3 track sequence assignment",
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )
        self._update_cached_track_sequence_no(
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )
        self._refresh_manual_push_tracks(target_track_coord=target_track_coord)

    def _create_ma3_sequence(self, intent: CreateMA3Sequence) -> ManualPushSequenceOption:
        if intent.creation_mode is MA3SequenceCreationMode.CURRENT_SONG_RANGE:
            _available_sequences, current_song_range = self._load_manual_push_sequence_options(
                range_mode=MA3SequenceRefreshRangeMode.ALL
            )
            session = cast(_MA3PushHost, self).session_service.get_session()
            session.manual_push_flow.current_song_sequence_range = current_song_range
            if current_song_range is None:
                raise ValueError(
                    "CreateMA3Sequence requires an available current-song MA3 sequence range"
                )

        method_name = (
            "create_sequence_in_current_song_range"
            if intent.creation_mode is MA3SequenceCreationMode.CURRENT_SONG_RANGE
            else "create_sequence_next_available"
        )
        raw_sequence = self._call_sync_capability(
            method_name,
            error_message="Sync service does not support MA3 sequence creation",
            preferred_name=intent.preferred_name,
        )
        created_sequence = self._normalize_manual_push_sequence_option(raw_sequence)
        session = cast(_MA3PushHost, self).session_service.get_session()
        for index, existing in enumerate(session.manual_push_flow.available_sequences):
            if existing.number == created_sequence.number:
                session.manual_push_flow.available_sequences[index] = created_sequence
                break
        else:
            session.manual_push_flow.available_sequences.append(created_sequence)
        session.manual_push_flow.available_sequences.sort(key=lambda value: value.number)
        return created_sequence

    def _create_ma3_timecode(self, intent: CreateMA3Timecode) -> ManualPushTimecodeOption:
        raw_timecode = self._call_sync_capability(
            "create_timecode_next_available",
            error_message="Sync service does not support MA3 timecode creation",
            preferred_name=intent.preferred_name,
        )
        created_timecode = self._normalize_manual_push_timecode_option(raw_timecode)
        self._refresh_manual_push_tracks(timecode_no=created_timecode.number)
        return created_timecode

    def _create_ma3_track_group(self, intent: CreateMA3TrackGroup) -> ManualPushTrackGroupOption:
        raw_track_group = self._call_sync_capability(
            "create_track_group_next_available",
            error_message="Sync service does not support MA3 track-group creation",
            timecode_no=int(intent.timecode_no),
            preferred_name=intent.preferred_name,
        )
        created_track_group = self._normalize_manual_push_track_group_option(raw_track_group)
        self._refresh_manual_push_tracks(
            timecode_no=int(intent.timecode_no),
            track_group_no=created_track_group.number,
        )
        return created_track_group

    def _create_ma3_track(self, intent: CreateMA3Track) -> ManualPushTrackOption:
        raw_track = self._call_sync_capability(
            "create_track",
            error_message="Sync service does not support MA3 track creation",
            timecode_no=int(intent.timecode_no),
            track_group_no=int(intent.track_group_no),
            preferred_name=intent.preferred_name,
        )
        created_track = self._normalize_manual_push_track_option(raw_track)
        self._refresh_manual_push_tracks(
            target_track_coord=created_track.coord,
            timecode_no=int(intent.timecode_no),
            track_group_no=int(intent.track_group_no),
        )
        session = cast(_MA3PushHost, self).session_service.get_session()
        session.manual_push_flow.target_track_coord = created_track.coord
        return created_track

    def _prepare_track_for_push(self, target_track_coord: str) -> None:
        self._call_sync_capability(
            "prepare_track_for_events",
            error_message="Sync service does not support MA3 track preparation",
            target_track_coord=target_track_coord,
        )

    def _prepare_target_track_for_push_if_needed(
        self,
        *,
        target_track_coord: str,
        sequence_action: MA3TrackSequenceAction | None,
        action_name: str,
        selected_events: list[Event] | None,
        allow_hitmaker_auto: bool,
        fallback_event_type: str | None = None,
    ) -> None:
        target_track = self._cached_manual_push_track_option_by_coord(
            target_track_coord,
            refresh=True,
        )
        if (target_track is None or target_track.sequence_no is None) and sequence_action is None:
            self._call_sync_capability(
                "refresh_push_track_options",
                error_message="Sync service does not support MA3 push-track refresh",
                target_track_coord=target_track_coord,
                required=False,
            )
            target_track = self._cached_manual_push_track_option_by_coord(
                target_track_coord,
                refresh=True,
            )
        if target_track is not None and target_track.sequence_no is not None:
            try:
                self._prepare_track_for_push(target_track_coord)
                return
            except Exception as exc:
                if not (allow_hitmaker_auto and self._is_no_sequence_assigned_error(exc)):
                    raise
                auto_created_sequence = self._create_hitmaker_sequence_for_events(
                    selected_events=selected_events or [],
                    preferred_name=self._preferred_hitmaker_sequence_name_for_track(target_track),
                    fallback_event_type=fallback_event_type,
                )
                if auto_created_sequence is None:
                    raise
                sequence_no = auto_created_sequence.number
                self._assign_track_sequence(
                    target_track_coord=target_track_coord,
                    sequence_no=sequence_no,
                )
                self._prepare_track_for_push(target_track_coord)
                self._update_cached_track_sequence_no(
                    target_track_coord=target_track_coord,
                    sequence_no=sequence_no,
                )
            return

        if target_track is None and sequence_action is None:
            return

        if sequence_action is None:
            if allow_hitmaker_auto:
                auto_created_sequence = self._create_hitmaker_sequence_for_events(
                    selected_events=selected_events or [],
                    preferred_name=self._preferred_hitmaker_sequence_name_for_track(target_track),
                    fallback_event_type=fallback_event_type,
                )
                if auto_created_sequence is not None:
                    sequence_no = auto_created_sequence.number
                    self._assign_track_sequence(
                        target_track_coord=target_track_coord,
                        sequence_no=sequence_no,
                    )
                    self._prepare_track_for_push(target_track_coord)
                    self._update_cached_track_sequence_no(
                        target_track_coord=target_track_coord,
                        sequence_no=sequence_no,
                    )
                    return
            raise ValueError(
                f"{action_name} target track {target_track_coord} has no assigned MA3 sequence"
            )

        if isinstance(sequence_action, AssignMA3TrackSequence):
            if sequence_action.target_track_coord != target_track_coord:
                raise ValueError(
                    f"{action_name} sequence assignment target {sequence_action.target_track_coord} "
                    f"does not match push target {target_track_coord}"
                )
            sequence_no = sequence_action.sequence_no
        else:
            if allow_hitmaker_auto:
                created_sequence = self._create_hitmaker_sequence_for_events(
                    selected_events=selected_events or [],
                    preferred_name=sequence_action.preferred_name,
                    fallback_event_type=fallback_event_type,
                )
                if created_sequence is None:
                    raise ValueError(
                        f"{action_name} requires a hit-capable event type to create sequences via HitMaker"
                    )
            else:
                created_sequence = self._create_ma3_sequence(sequence_action)
            sequence_no = created_sequence.number

        self._assign_track_sequence(
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )
        self._prepare_track_for_push(target_track_coord)
        self._update_cached_track_sequence_no(
            target_track_coord=target_track_coord,
            sequence_no=sequence_no,
        )

    def _create_hitmaker_sequence_for_events(
        self,
        *,
        selected_events: list[Event],
        preferred_name: str | None = None,
        fallback_event_type: str | None = None,
    ) -> ManualPushSequenceOption | None:
        event_type = self._resolve_hit_event_type(selected_events)
        if event_type is None:
            normalized_fallback = str(fallback_event_type or "").strip().lower()
            if normalized_fallback:
                event_type = normalized_fallback
        if event_type is None:
            return None
        resolved_preferred_name = preferred_name
        if resolved_preferred_name is None or not str(resolved_preferred_name).strip():
            resolved_preferred_name = self._derive_hit_sequence_name(
                selected_events,
                event_type=event_type,
            )
        raw_sequence = self._call_sync_capability(
            "create_sequence_for_event_type",
            error_message="Sync service does not support HitMaker sequence creation",
            required=False,
            event_type=event_type,
            sequence_type="go_hit",
            preferred_name=resolved_preferred_name,
        )
        if raw_sequence is None:
            return None
        created_sequence = cast(_MA3PushHost, self)._normalize_manual_push_sequence_option(
            raw_sequence
        )
        session = cast(_MA3PushHost, self).session_service.get_session()
        for index, existing in enumerate(session.manual_push_flow.available_sequences):
            if existing.number == created_sequence.number:
                session.manual_push_flow.available_sequences[index] = created_sequence
                break
        else:
            session.manual_push_flow.available_sequences.append(created_sequence)
        session.manual_push_flow.available_sequences.sort(key=lambda value: value.number)
        return created_sequence

    def _events_for_layer_sequence_prep(self, layer: Layer) -> list[Event]:
        main_take = cast(_MA3PushHost, self)._main_take(layer)
        if main_take is None or not main_take.events:
            return []
        promoted_main_events = [event for event in main_take.events if event.is_promoted]
        if promoted_main_events:
            return promoted_main_events
        return list(main_take.events)

    @staticmethod
    def _resolve_hit_event_type(selected_events: list[Event]) -> str | None:
        hit_tokens = {
            "hit",
            "kick",
            "snare",
            "clap",
            "rim",
            "rimshot",
            "tom",
            "hat",
            "hihat",
            "hi-hat",
            "ride",
            "crash",
            "perc",
            "percussion",
        }
        token_score: dict[str, int] = {}

        def push_token(raw_value: object) -> None:
            if raw_value is None:
                return
            text = str(raw_value).strip().lower()
            if not text:
                return
            text = text.replace("-", " ").replace("_", " ")
            for part in text.split():
                token = part.strip()
                if not token:
                    continue
                if token in {"event", "events", "main", "take", "cue"}:
                    continue
                token_score[token] = token_score.get(token, 0) + 1

        for event in selected_events:
            push_token(event.label)
            for key in ("class", "label", "type", "instrument", "event_type"):
                push_token(event.classifications.get(key))
            detection = event.detection_metadata
            for key in ("class", "label", "type", "instrument"):
                push_token(detection.get(key))

        ranked = sorted(token_score.items(), key=lambda item: (-item[1], item[0]))
        for token, _score in ranked:
            if token in hit_tokens or "hit" in token:
                return token
        return None

    @staticmethod
    def _derive_hit_sequence_name(
        selected_events: list[Event],
        *,
        event_type: str,
    ) -> str | None:
        cleaned_event_type = str(event_type or "").strip()
        if not cleaned_event_type:
            return None
        if selected_events:
            source_label = str(selected_events[0].label or "").strip()
            if source_label and source_label.lower() not in {"event", "events"}:
                return source_label
        return cleaned_event_type.replace("_", " ").replace("-", " ").title()

    @staticmethod
    def _is_no_sequence_assigned_error(exc: Exception) -> bool:
        text = str(exc or "").strip().lower()
        if "no_sequence_assigned" in text:
            return True
        if "no sequence assigned" in text:
            return True
        return False

    @staticmethod
    def _preferred_hitmaker_sequence_name_for_track(
        track: ManualPushTrackOption | None,
    ) -> str | None:
        if track is None:
            return None
        resolved = str(track.name or "").strip()
        return resolved or None

    def _cached_manual_push_track_option_by_coord(
        self,
        target_track_coord: str,
        *,
        refresh: bool = False,
    ) -> ManualPushTrackOption | None:
        session = cast(_MA3PushHost, self).session_service.get_session()
        available_tracks = session.manual_push_flow.available_tracks
        if refresh or not available_tracks:
            available_tracks = self._refresh_manual_push_tracks(
                target_track_coord=target_track_coord
            )
        for track in available_tracks:
            if track.coord == target_track_coord:
                return track
        if refresh:
            return None
        return self._cached_manual_push_track_option_by_coord(target_track_coord, refresh=True)

    def _update_cached_track_sequence_no(
        self,
        *,
        target_track_coord: str,
        sequence_no: int,
    ) -> None:
        session = cast(_MA3PushHost, self).session_service.get_session()
        for track in session.manual_push_flow.available_tracks:
            if track.coord == target_track_coord:
                track.sequence_no = sequence_no
                break

    def _call_sync_capability(
        self,
        capability_name: str,
        *,
        error_message: str,
        required: bool = True,
        **kwargs: Any,
    ) -> Any:
        sync_service = cast(_MA3PushHost, self).sync_service
        capability = getattr(sync_service, capability_name, None)
        if not callable(capability):
            if not required:
                return None
            raise RuntimeError(error_message)
        return capability(**kwargs)

    def _resolve_push_target_coord(self, layer: Layer, intent: PushLayerToMA3) -> str:
        if intent.target_mode is MA3PushTargetMode.SAVED_ROUTE:
            target_track_coord = str(layer.sync.ma3_track_coord or "").strip()
            if not target_track_coord:
                raise ValueError("PushLayerToMA3 requires a saved MA3 route for the layer")
            return target_track_coord
        assert intent.target_mode is MA3PushTargetMode.DIFFERENT_TRACK_ONCE
        assert intent.target_track_coord is not None
        return intent.target_track_coord

    @staticmethod
    def _resolve_push_target_channel_no(layer: Layer, intent: PushLayerToMA3) -> int | None:
        if intent.target_mode is MA3PushTargetMode.SAVED_ROUTE:
            if intent.ma3_channel_no is not None:
                return intent.ma3_channel_no
            return layer.sync.ma3_channel_no
        return intent.ma3_channel_no

    def _resolve_push_events_for_layer(
        self,
        timeline: Timeline,
        layer: Layer,
        intent: PushLayerToMA3,
    ) -> list[Event]:
        main_take = self._main_take(layer)
        if main_take is None or not main_take.events:
            raise ValueError("PushLayerToMA3 requires at least one main event on the layer")
        promoted_main_events = [event for event in main_take.events if event.is_promoted]
        if not promoted_main_events:
            raise ValueError(
                "PushLayerToMA3 requires at least one promoted main event on the layer"
            )

        if intent.scope is MA3PushScope.LAYER_MAIN:
            return promoted_main_events

        event_lookup = {str(event.id): event for event in main_take.events}
        selected_events: list[Event] = []
        missing_ids: list[str] = []
        for event_id in intent.selected_event_ids:
            event = event_lookup.get(str(event_id))
            if event is None:
                missing_ids.append(str(event_id))
                continue
            selected_events.append(event)

        if missing_ids:
            raise ValueError(
                "PushLayerToMA3 selected_event_ids must belong to the layer main take: "
                + ", ".join(missing_ids)
            )
        promoted_selected_events = [event for event in selected_events if event.is_promoted]
        if not promoted_selected_events:
            raise ValueError("PushLayerToMA3 requires selected promoted main events to push")
        return promoted_selected_events

    @staticmethod
    def _prepare_events_for_ma3_push(
        layer: Layer,
        selected_events: list[Event],
        *,
        skip_cue_1: bool = False,
    ) -> list[Event]:
        """Translate clean EZ section markers into MA3 cue-addressed events at the boundary."""

        if layer.kind is not LayerKind.SECTION:
            return selected_events
        main_take = layer.takes[0] if layer.takes else None
        section_events = [
            event
            for event in (main_take.events if main_take is not None else selected_events)
            if event.is_promoted
        ]
        ordered_events = sorted(
            section_events,
            key=lambda event: (float(event.start), float(event.end), str(event.id)),
        )
        internal_numbers = {
            str(event.id): index for index, event in enumerate(ordered_events, start=1)
        }
        transfer_events: list[Event] = []
        selected_lookup = {str(event.id): event for event in selected_events}
        selected_ordered_events = [
            event for event in ordered_events if str(event.id) in selected_lookup
        ]
        if not selected_ordered_events:
            selected_ordered_events = sorted(
                selected_events,
                key=lambda event: (float(event.start), float(event.end), str(event.id)),
            )
        for fallback_index, event in enumerate(selected_ordered_events, start=1):
            cue_number = cue_number_from_ref_text(event.cue_ref)
            if cue_number is None:
                cue_number = internal_numbers.get(str(event.id), fallback_index)
            if cue_number == 1 and skip_cue_1:
                continue
            cue_ref = event.cue_ref or cue_number_text(cue_number) or str(cue_number)
            transfer_events.append(
                replace(
                    event,
                    cue_number=cue_number,
                    cue_ref=cue_ref,
                    label=str(event.label or "").strip() or f"Section {index}",
                )
            )
        return transfer_events

    @staticmethod
    def _resolve_project_ma3_push_offset_seconds(session: Any) -> float:
        raw_offset_seconds = getattr(session, "project_ma3_push_offset_seconds", -1.0)
        try:
            return float(raw_offset_seconds)
        except (TypeError, ValueError):
            return -1.0

    @staticmethod
    def _require_ma3_event_layer(layer: Layer, *, action_name: str) -> None:
        if not is_event_like_layer_kind(layer.kind):
            raise ValueError(f"{action_name} requires an event layer")
