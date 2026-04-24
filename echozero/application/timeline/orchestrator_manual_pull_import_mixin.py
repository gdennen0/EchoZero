"""Manual pull import helpers for the timeline orchestrator.
Exists to isolate pull-target resolution and imported layer/take creation from transfer-plan coordination.
Connects manual MA3 pull actions to canonical timeline layer and take mutations.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol, cast

from echozero.application.session.models import ManualPullEventOption, ManualPullTrackOption
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId, LayerId, TakeId
from echozero.application.timeline.models import Event, Layer, Take, Timeline
from echozero.application.timeline.orchestrator_transfer_lookup_mixin import (
    _PULL_TARGET_CREATE_NEW_LAYER_ID,
    _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
)


class _ManualPullImportHost(Protocol):
    @staticmethod
    def _main_take(layer: Layer) -> Take | None: ...

    def _build_manual_pull_take(
        self,
        *,
        layer: Layer,
        source_track: ManualPullTrackOption,
        selected_events: list[ManualPullEventOption],
    ) -> Take: ...

    def _build_manual_pull_event(
        self,
        *,
        take_id: TakeId,
        source_track: ManualPullTrackOption,
        source_event: ManualPullEventOption,
        order_index: int,
        existing_event_ids: set[str],
    ) -> Event: ...

    def _find_layer(self, timeline: Timeline, layer_id: LayerId) -> Layer: ...

    def _sort_take_events(self, take: Take) -> None: ...


class TimelineOrchestratorManualPullImportMixin:
    @staticmethod
    def _next_manual_pull_take_id(layer: Layer) -> TakeId:
        existing_ids = {str(take.id) for take in layer.takes}
        index = 1
        while True:
            candidate = f"{layer.id}:ma3_pull:{index}"
            if candidate not in existing_ids:
                return TakeId(candidate)
            index += 1

    def _apply_manual_pull_import(
        self,
        *,
        target_layer: Layer,
        source_track: ManualPullTrackOption,
        selected_events: list[ManualPullEventOption],
        import_mode: str,
    ) -> tuple[TakeId, list[EventId]]:
        host = cast(_ManualPullImportHost, self)
        self._maybe_link_manual_pull_target_layer(
            target_layer=target_layer,
            source_track=source_track,
        )
        if import_mode == "main":
            target_take = self._resolve_or_create_manual_pull_main_take(target_layer)
            existing_event_ids = {str(event.id) for event in target_take.events}
            imported_events = [
                host._build_manual_pull_event(
                    take_id=target_take.id,
                    source_track=source_track,
                    source_event=source_event,
                    order_index=index,
                    existing_event_ids=existing_event_ids,
                )
                for index, source_event in enumerate(selected_events, start=1)
            ]
            target_take.events.extend(imported_events)
            host._sort_take_events(target_take)
            return target_take.id, [event.id for event in imported_events]

        imported_take = host._build_manual_pull_take(
            layer=target_layer,
            source_track=source_track,
            selected_events=selected_events,
        )
        target_layer.takes.append(imported_take)
        host._sort_take_events(imported_take)
        return imported_take.id, [event.id for event in imported_take.events]

    def _resolve_or_create_manual_pull_main_take(self, layer: Layer) -> Take:
        main_take = cast(_ManualPullImportHost, self)._main_take(layer)
        if main_take is not None:
            return main_take
        main_take = Take(
            id=TakeId(f"{layer.id}:main"),
            layer_id=layer.id,
            name="Main",
        )
        layer.takes.insert(0, main_take)
        return main_take

    @staticmethod
    def _is_manual_pull_synthetic_target(target_layer_id: LayerId) -> bool:
        return target_layer_id in {
            _PULL_TARGET_CREATE_NEW_LAYER_ID,
            _PULL_TARGET_CREATE_NEW_LAYER_PER_SOURCE_TRACK_ID,
        }

    def _resolve_manual_pull_target_layer(
        self,
        timeline: Timeline,
        *,
        target_layer_id: LayerId,
        source_track: ManualPullTrackOption,
    ) -> Layer:
        host = cast(_ManualPullImportHost, self)
        if self._is_manual_pull_synthetic_target(target_layer_id):
            return self._create_manual_pull_target_layer(timeline, source_track=source_track)
        return host._find_layer(timeline, target_layer_id)

    def _create_manual_pull_target_layer(
        self, timeline: Timeline, *, source_track: ManualPullTrackOption
    ) -> Layer:
        layer_name = self._next_manual_pull_layer_name(timeline, source_track.name)
        layer_id = self._next_manual_pull_layer_id(timeline, source_track)
        main_take_id = TakeId(f"{layer_id}:main")
        new_layer = Layer(
            id=layer_id,
            timeline_id=timeline.id,
            name=layer_name,
            kind=LayerKind.EVENT,
            order_index=self._next_timeline_layer_order_index(timeline),
            takes=[
                Take(
                    id=main_take_id,
                    layer_id=layer_id,
                    name="Main",
                )
            ],
        )
        new_layer.sync.ma3_track_coord = source_track.coord
        timeline.layers.append(new_layer)
        return new_layer

    @staticmethod
    def _maybe_link_manual_pull_target_layer(
        *,
        target_layer: Layer,
        source_track: ManualPullTrackOption,
    ) -> None:
        current_track_coord = str(target_layer.sync.ma3_track_coord or "").strip()
        if current_track_coord and current_track_coord != source_track.coord:
            return
        target_layer.sync.ma3_track_coord = source_track.coord

    @staticmethod
    def _next_timeline_layer_order_index(timeline: Timeline) -> int:
        if not timeline.layers:
            return 0
        return max(int(layer.order_index) for layer in timeline.layers) + 1

    def _next_manual_pull_layer_id(
        self,
        timeline: Timeline,
        source_track: ManualPullTrackOption,
    ) -> LayerId:
        base_slug = self._manual_pull_layer_slug(source_track)
        existing_ids = {str(layer.id) for layer in timeline.layers}
        index = 1
        while True:
            suffix = "" if index == 1 else f"_{index}"
            candidate = LayerId(f"layer_ma3_pull_{base_slug}{suffix}")
            if str(candidate) not in existing_ids:
                return candidate
            index += 1

    def _next_manual_pull_layer_name(self, timeline: Timeline, source_name: str) -> str:
        base_name = source_name.strip() or "Imported Layer"
        existing_names = {layer.name for layer in timeline.layers}
        if base_name not in existing_names:
            return base_name
        index = 2
        while True:
            candidate = f"{base_name} {index}"
            if candidate not in existing_names:
                return candidate
            index += 1

    @staticmethod
    def _manual_pull_layer_slug(source_track: ManualPullTrackOption) -> str:
        raw = f"{source_track.name}_{source_track.coord}".strip().lower()
        normalized = unicodedata.normalize("NFKD", raw)
        ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
        return slug or "import"
