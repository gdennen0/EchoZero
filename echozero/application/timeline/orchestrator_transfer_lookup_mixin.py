"""Manual transfer lookup helpers for the timeline orchestrator.
Exists to isolate sync-provider option loading and raw-option normalization from transfer execution.
Connects manual push/pull UI flows to typed provider options and target lookups.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, cast

from echozero.application.session.models import (
    ManualPullEventOption,
    ManualPullTargetOption,
    ManualPullTimecodeOption,
    ManualPullTrackGroupOption,
    ManualPullTrackOption,
    ManualPushSequenceOption,
    ManualPushSequenceRange,
    ManualPushTimecodeOption,
    ManualPushTrackGroupOption,
    ManualPushTrackOption,
)
from echozero.application.shared.cue_numbers import parse_positive_cue_number
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.shared.ids import LayerId
from echozero.application.timeline.ma3_push_intents import MA3SequenceRefreshRangeMode
from echozero.application.timeline.models import Timeline

_PULL_TARGET_CREATE_NEW_LAYER_ID = LayerId("__manual_pull__:create_new_layer")
_PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID = LayerId("__manual_pull__:create_new_section_layer")
_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID = LayerId(
    "__manual_pull__:create_new_layer_per_source_track"
)
_PULL_TARGET_CREATE_NEW_LAYER_NAME = "+ Create New Layer..."
_PULL_TARGET_CREATE_NEW_SECTION_LAYER_NAME = "+ Create Section Layer..."
_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_NAME = "+ Create New Layer Per Source Track..."


class _TransferLookupHost(Protocol):
    sync_service: Any
    session_service: Any


class TimelineOrchestratorTransferLookupMixin:
    def _load_manual_push_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[ManualPushTrackOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        if hasattr(provider, "list_push_track_options"):
            raw_tracks = self._call_lookup_capability(
                provider.list_push_track_options,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        elif hasattr(provider, "get_available_ma3_tracks"):
            raw_tracks = self._call_lookup_capability(
                provider.get_available_ma3_tracks,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        else:
            return []

        tracks = [
            self._normalize_manual_push_track_option(raw_track) for raw_track in raw_tracks or []
        ]
        if timecode_no is None:
            filtered_tracks = tracks
        else:
            filtered_tracks = [
                track
                for track in tracks
                if self._ma3_track_coord_timecode_no(track.coord) == int(timecode_no)
            ]
        if track_group_no is None:
            return filtered_tracks
        return [
            track
            for track in filtered_tracks
            if self._ma3_track_coord_track_group_no(track.coord) == int(track_group_no)
        ]

    def _load_manual_push_timecode_options(self) -> list[ManualPushTimecodeOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        method = getattr(provider, "list_timecodes", None)
        if callable(method):
            raw_timecodes = method()
            return [
                self._normalize_manual_push_timecode_option(raw_timecode)
                for raw_timecode in raw_timecodes or []
            ]

        tracks = self._load_manual_push_track_options()
        by_number: dict[int, ManualPushTimecodeOption] = {}
        for track in tracks:
            timecode_no = self._ma3_track_coord_timecode_no(track.coord)
            if timecode_no is None:
                continue
            by_number[timecode_no] = ManualPushTimecodeOption(
                number=timecode_no,
                name=track.timecode_name,
            )
        return [by_number[number] for number in sorted(by_number)]

    def _load_manual_push_track_group_options(
        self,
        *,
        timecode_no: int,
    ) -> list[ManualPushTrackGroupOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        method = getattr(provider, "list_track_groups", None)
        if callable(method):
            raw_track_groups = self._call_lookup_capability(method, timecode_no=timecode_no)
            groups = [
                self._normalize_manual_push_track_group_option(raw_group)
                for raw_group in raw_track_groups or []
            ]
            return sorted(groups, key=lambda value: value.number)

        tracks = self._load_manual_push_track_options(timecode_no=timecode_no)
        counts: dict[int, int] = {}
        for track in tracks:
            group_no = self._ma3_track_coord_track_group_no(track.coord)
            if group_no is None:
                continue
            counts[group_no] = counts.get(group_no, 0) + 1
        return [
            ManualPushTrackGroupOption(
                number=group_no,
                name=f"Group {group_no}",
                track_count=count,
            )
            for group_no, count in sorted(counts.items())
        ]

    def _load_manual_pull_timecode_options(self) -> list[ManualPullTimecodeOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        method = getattr(provider, "list_timecodes", None)
        if callable(method):
            raw_timecodes = method()
            return [
                self._normalize_manual_pull_timecode_option(raw_timecode)
                for raw_timecode in raw_timecodes or []
            ]

        tracks = self._load_manual_pull_track_options()
        by_number: dict[int, ManualPullTimecodeOption] = {}
        for track in tracks:
            timecode_no = self._ma3_track_coord_timecode_no(track.coord)
            if timecode_no is None:
                continue
            by_number[timecode_no] = ManualPullTimecodeOption(
                number=timecode_no,
                name=track.timecode_name,
            )
        return [by_number[number] for number in sorted(by_number)]

    def _load_manual_pull_track_group_options(
        self,
        *,
        timecode_no: int,
    ) -> list[ManualPullTrackGroupOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        method = getattr(provider, "list_track_groups", None)
        if callable(method):
            raw_track_groups = self._call_lookup_capability(method, timecode_no=timecode_no)
            groups = [
                self._normalize_manual_pull_track_group_option(raw_group)
                for raw_group in raw_track_groups or []
            ]
            return sorted(groups, key=lambda value: value.number)

        tracks = self._load_manual_pull_track_options(timecode_no=timecode_no)
        counts: dict[int, int] = {}
        for track in tracks:
            group_no = self._ma3_track_coord_track_group_no(track.coord)
            if group_no is None:
                continue
            counts[group_no] = counts.get(group_no, 0) + 1
        return [
            ManualPullTrackGroupOption(
                number=group_no,
                name=f"Group {group_no}",
                track_count=count,
            )
            for group_no, count in sorted(counts.items())
        ]

    def _load_manual_push_sequence_options(
        self,
        *,
        range_mode: MA3SequenceRefreshRangeMode = MA3SequenceRefreshRangeMode.ALL,
    ) -> tuple[list[ManualPushSequenceOption], ManualPushSequenceRange | None]:
        provider = cast(_TransferLookupHost, self).sync_service
        current_song_range = self._load_current_song_sequence_range()
        start_no: int | None = None
        end_no: int | None = None
        if range_mode is MA3SequenceRefreshRangeMode.CURRENT_SONG:
            if current_song_range is None:
                return [], None
            start_no = current_song_range.start
            end_no = current_song_range.end

        method = getattr(provider, "list_sequences", None)
        if not callable(method):
            return [], current_song_range

        raw_sequences = method(start_no=start_no, end_no=end_no)
        sequences = [
            self._normalize_manual_push_sequence_option(raw_sequence)
            for raw_sequence in raw_sequences or []
        ]
        return sorted(sequences, key=lambda value: value.number), current_song_range

    def _load_current_song_sequence_range(self) -> ManualPushSequenceRange | None:
        provider = cast(_TransferLookupHost, self).sync_service
        method = getattr(provider, "get_current_song_sequence_range", None)
        if not callable(method):
            return None
        raw_range = method()
        if raw_range is None:
            return None
        return self._normalize_manual_push_sequence_range(raw_range)

    def _load_manual_pull_track_options(
        self,
        *,
        timecode_no: int | None = None,
        track_group_no: int | None = None,
    ) -> list[ManualPullTrackOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        if hasattr(provider, "list_pull_track_options"):
            raw_tracks = self._call_lookup_capability(
                provider.list_pull_track_options,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        elif hasattr(provider, "get_available_ma3_tracks"):
            raw_tracks = self._call_lookup_capability(
                provider.get_available_ma3_tracks,
                timecode_no=timecode_no,
                track_group_no=track_group_no,
            )
        else:
            return []

        tracks = [
            self._normalize_manual_pull_track_option(raw_track) for raw_track in raw_tracks or []
        ]
        if timecode_no is not None:
            tracks = [
                track
                for track in tracks
                if self._ma3_track_coord_timecode_no(track.coord) == int(timecode_no)
            ]
        if track_group_no is not None:
            tracks = [
                track
                for track in tracks
                if self._ma3_track_coord_track_group_no(track.coord) == int(track_group_no)
            ]
        return tracks

    def _load_manual_pull_event_options(
        self, source_track_coord: str
    ) -> list[ManualPullEventOption]:
        provider = cast(_TransferLookupHost, self).sync_service
        if hasattr(provider, "list_pull_source_events"):
            raw_events = provider.list_pull_source_events(source_track_coord)
        elif hasattr(provider, "list_ma3_track_events"):
            raw_events = provider.list_ma3_track_events(source_track_coord)
        elif hasattr(provider, "get_available_ma3_events"):
            raw_events = provider.get_available_ma3_events(source_track_coord)
        else:
            return []

        return [
            self._normalize_manual_pull_event_option(raw_event) for raw_event in raw_events or []
        ]

    @staticmethod
    def _load_manual_pull_target_options(
        timeline: Timeline,
        *,
        include_create_per_source_track: bool = False,
    ) -> list[ManualPullTargetOption]:
        targets = [
            ManualPullTargetOption(layer_id=layer.id, name=layer.name, kind=layer.kind)
            for layer in sorted(timeline.layers, key=lambda value: value.order_index)
            if is_event_like_layer_kind(layer.kind)
            and layer.presentation_hints.visible
            and not layer.presentation_hints.locked
        ]
        targets.append(
            ManualPullTargetOption(
                layer_id=_PULL_TARGET_CREATE_NEW_LAYER_ID,
                name=_PULL_TARGET_CREATE_NEW_LAYER_NAME,
                kind=LayerKind.EVENT,
            )
        )
        targets.append(
            ManualPullTargetOption(
                layer_id=_PULL_TARGET_CREATE_NEW_SECTION_LAYER_ID,
                name=_PULL_TARGET_CREATE_NEW_SECTION_LAYER_NAME,
                kind=LayerKind.SECTION,
            )
        )
        if include_create_per_source_track:
            targets.append(
                ManualPullTargetOption(
                    layer_id=_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
                    name=_PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_NAME,
                    kind=LayerKind.EVENT,
                )
            )
        return targets

    @staticmethod
    def _normalize_manual_push_track_option(raw_track: Any) -> ManualPushTrackOption:
        if isinstance(raw_track, ManualPushTrackOption):
            return raw_track

        coord = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "coord")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "name")
        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "no")
        timecode_name = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_track, "timecode_name"
        )
        if timecode_name in {None, ""}:
            timecode_name = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_track, "timecodeName"
            )
        note = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "note")
        event_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_track, "event_count"
        )
        sequence_no = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_track, "sequence_no"
        )
        if sequence_no in {None, ""}:
            sequence_no = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_track, "sequenceNo"
            )
        if sequence_no in {None, ""}:
            sequence_no = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_track, "seq_no"
            )
        if sequence_no in {None, ""}:
            sequence_no = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_track, "seqNo"
            )

        return ManualPushTrackOption(
            coord=str(coord or ""),
            name=str(name or ""),
            number=TimelineOrchestratorTransferLookupMixin._coerce_optional_positive_int(number),
            timecode_name=None if timecode_name in {None, ""} else str(timecode_name),
            note=None if note is None else str(note),
            event_count=TimelineOrchestratorTransferLookupMixin._coerce_optional_int(
                event_count
            ),
            sequence_no=TimelineOrchestratorTransferLookupMixin._coerce_optional_positive_int(
                sequence_no
            ),
        )

    @staticmethod
    def _normalize_manual_push_sequence_option(raw_sequence: Any) -> ManualPushSequenceOption:
        if isinstance(raw_sequence, ManualPushSequenceOption):
            return raw_sequence

        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_sequence, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_sequence, "sequence_no"
            )
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_sequence, "name")
        return ManualPushSequenceOption(
            number=int(number),
            name=str(name or number or ""),
        )

    @staticmethod
    def _normalize_manual_push_sequence_range(raw_range: Any) -> ManualPushSequenceRange:
        if isinstance(raw_range, ManualPushSequenceRange):
            return raw_range

        start = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_range, "start")
        end = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_range, "end")
        song_label = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_range, "song_label"
        )
        if song_label in {None, ""}:
            song_label = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_range, "songLabel"
            )
        return ManualPushSequenceRange(
            start=int(start),
            end=int(end),
            song_label=None if song_label is None else str(song_label),
        )

    @staticmethod
    def _normalize_manual_push_timecode_option(raw_timecode: Any) -> ManualPushTimecodeOption:
        if isinstance(raw_timecode, ManualPushTimecodeOption):
            return raw_timecode

        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "no")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "name")
        return ManualPushTimecodeOption(
            number=int(number or 0),
            name=None if name in {None, ""} else str(name),
        )

    @staticmethod
    def _normalize_manual_push_track_group_option(raw_group: Any) -> ManualPushTrackGroupOption:
        if isinstance(raw_group, ManualPushTrackGroupOption):
            return raw_group

        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "no")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "name")
        track_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_group, "track_count"
        )
        if track_count in {None, ""}:
            track_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_group, "trackCount"
            )
        return ManualPushTrackGroupOption(
            number=int(number or 0),
            name=str(name or ""),
            track_count=TimelineOrchestratorTransferLookupMixin._coerce_optional_int(track_count),
        )

    @staticmethod
    def _normalize_manual_pull_timecode_option(raw_timecode: Any) -> ManualPullTimecodeOption:
        if isinstance(raw_timecode, ManualPullTimecodeOption):
            return raw_timecode

        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "no")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_timecode, "name")
        return ManualPullTimecodeOption(
            number=int(number or 0),
            name=None if name in {None, ""} else str(name),
        )

    @staticmethod
    def _normalize_manual_pull_track_group_option(raw_group: Any) -> ManualPullTrackGroupOption:
        if isinstance(raw_group, ManualPullTrackGroupOption):
            return raw_group

        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "no")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_group, "name")
        track_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_group, "track_count"
        )
        if track_count in {None, ""}:
            track_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_group, "trackCount"
            )
        return ManualPullTrackGroupOption(
            number=int(number or 0),
            name=str(name or ""),
            track_count=TimelineOrchestratorTransferLookupMixin._coerce_optional_int(track_count),
        )

    @staticmethod
    def _normalize_manual_pull_track_option(raw_track: Any) -> ManualPullTrackOption:
        if isinstance(raw_track, ManualPullTrackOption):
            return raw_track

        coord = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "coord")
        name = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "name")
        number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "number")
        if number in {None, ""}:
            number = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "no")
        timecode_name = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_track, "timecode_name"
        )
        if timecode_name in {None, ""}:
            timecode_name = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_track, "timecodeName"
            )
        note = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_track, "note")
        event_count = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_track, "event_count"
        )

        return ManualPullTrackOption(
            coord=str(coord or ""),
            name=str(name or ""),
            number=TimelineOrchestratorTransferLookupMixin._coerce_optional_positive_int(number),
            timecode_name=None if timecode_name in {None, ""} else str(timecode_name),
            note=None if note is None else str(note),
            event_count=TimelineOrchestratorTransferLookupMixin._coerce_optional_int(
                event_count
            ),
        )

    @staticmethod
    def _normalize_manual_pull_event_option(raw_event: Any) -> ManualPullEventOption:
        if isinstance(raw_event, ManualPullEventOption):
            return raw_event

        event_id = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_event, "event_id"
        )
        label = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "label")
        start = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "start")
        end = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "end")
        cue_number = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_event, "cue_number"
        )
        if cue_number in {None, ""}:
            cue_number = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_event, "cue_no"
            )
        if cue_number in {None, ""}:
            cue_number = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_event, "cueno"
            )
        if cue_number in {None, ""}:
            cue_number = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_event, "cueNo"
            )
        cue_ref = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "cue_ref")
        if cue_ref in {None, ""}:
            cue_ref = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_event, "cueRef"
            )
        color = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "color")
        notes = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "notes")
        if notes in {None, ""}:
            notes = TimelineOrchestratorTransferLookupMixin._track_option_value(raw_event, "note")
        payload_ref = TimelineOrchestratorTransferLookupMixin._track_option_value(
            raw_event, "payload_ref"
        )
        if payload_ref in {None, ""}:
            payload_ref = TimelineOrchestratorTransferLookupMixin._track_option_value(
                raw_event, "payloadRef"
            )

        return ManualPullEventOption(
            event_id=str(event_id or ""),
            label=str(label or event_id or ""),
            start=None if start is None else float(start),
            end=None if end is None else float(end),
            cue_number=parse_positive_cue_number(cue_number),
            cue_ref=None if cue_ref in {None, ""} else str(cue_ref),
            color=None if color in {None, ""} else str(color),
            notes=None if notes in {None, ""} else str(notes),
            payload_ref=None if payload_ref in {None, ""} else str(payload_ref),
        )

    @staticmethod
    def _track_option_value(raw_track: Any, key: str) -> Any:
        if isinstance(raw_track, dict):
            return raw_track.get(key)
        return getattr(raw_track, key, None)

    def _active_song_version_ma3_timecode_pool_no(self) -> int | None:
        session = cast(_TransferLookupHost, self).session_service.get_session()
        value = getattr(session, "active_song_version_ma3_timecode_pool_no", None)
        return self._coerce_optional_positive_int(value)

    def _resolve_manual_push_timecode_no(
        self,
        *,
        target_track_coord: str | None = None,
    ) -> int | None:
        coord_timecode_no = self._ma3_track_coord_timecode_no(target_track_coord)
        if coord_timecode_no is not None:
            return coord_timecode_no
        return self._active_song_version_ma3_timecode_pool_no()

    def _default_manual_push_track_group_no(
        self,
        *,
        selected_timecode_no: int | None,
        available_track_groups: list[ManualPushTrackGroupOption],
        preferred_track_coord: str | None = None,
    ) -> int | None:
        if not available_track_groups:
            return None
        preferred_timecode_no = self._ma3_track_coord_timecode_no(preferred_track_coord)
        preferred_group_no = (
            self._ma3_track_coord_track_group_no(preferred_track_coord)
            if preferred_timecode_no == selected_timecode_no
            else None
        )
        available_numbers = {group.number for group in available_track_groups}
        if preferred_group_no in available_numbers:
            return preferred_group_no
        for group in available_track_groups:
            if (group.track_count or 0) > 0:
                return group.number
        return available_track_groups[0].number

    @staticmethod
    def _call_lookup_capability(capability, **kwargs):
        try:
            parameters = inspect.signature(capability).parameters
        except (TypeError, ValueError):
            parameters = {}
        if not parameters:
            return capability()
        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in parameters and value is not None
        }
        return capability(**supported_kwargs)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _coerce_optional_positive_int(value: Any) -> int | None:
        if value is None or value == "":
            return None
        parsed = int(value)
        return parsed if parsed >= 1 else None

    @staticmethod
    def _ma3_track_coord_timecode_no(raw_coord: Any) -> int | None:
        coord = str(raw_coord or "").strip().lower()
        if not coord.startswith("tc"):
            return None
        tc_text = coord[2:].split("_", 1)[0]
        try:
            parsed = int(tc_text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

    @staticmethod
    def _ma3_track_coord_track_group_no(raw_coord: Any) -> int | None:
        coord = str(raw_coord or "").strip().lower()
        if "_tg" not in coord:
            return None
        tg_text = coord.split("_tg", 1)[1].split("_", 1)[0]
        try:
            parsed = int(tg_text)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 1 else None

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

    @staticmethod
    def _manual_pull_track_by_coord(
        available_tracks: list[ManualPullTrackOption],
        source_track_coord: str,
        action_name: str,
    ) -> ManualPullTrackOption:
        for track in available_tracks:
            if track.coord == source_track_coord:
                return track
        raise ValueError(
            f"{action_name} source_track_coord not found in available_tracks: "
            f"{source_track_coord}"
        )

    @staticmethod
    def _manual_pull_target_layer_by_id(
        available_targets: list[ManualPullTargetOption],
        target_layer_id: LayerId,
        action_name: str,
    ) -> ManualPullTargetOption:
        for target in available_targets:
            if target.layer_id == target_layer_id:
                return target
        raise ValueError(
            f"{action_name} target_layer_id not found in available_target_layers: "
            f"{target_layer_id}"
        )
