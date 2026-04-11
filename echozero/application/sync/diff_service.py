"""Shared diff preview service for manual MA3 push/pull flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from echozero.application.timeline.models import Event


@dataclass(frozen=True, slots=True)
class SyncDiffSummary:
    added_count: int = 0
    removed_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    row_count: int = 0


@dataclass(frozen=True, slots=True)
class SyncDiffRow:
    row_id: str
    action: str
    start: float
    end: float
    label: str
    before: str
    after: str


class _PullEventOptionLike(Protocol):
    event_id: str
    label: str
    start: float | None
    end: float | None


@dataclass(slots=True)
class SyncDiffService:
    """Build deterministic diff previews for manual push/pull confirmation."""

    def build_push_preview_rows(
        self,
        *,
        selected_events: list[Event],
        target_track_name: str,
        target_track_coord: str,
    ) -> tuple[SyncDiffSummary, list[SyncDiffRow]]:
        rows = [
            SyncDiffRow(
                row_id=str(event.id),
                action="add",
                start=float(event.start),
                end=float(event.end),
                label=event.label,
                before="Not present in MA3 target",
                after=f"{target_track_name} ({target_track_coord})",
            )
            for event in sorted(selected_events, key=self._event_sort_key)
        ]
        return self._added_only_summary(rows), rows

    def build_pull_preview_rows(
        self,
        *,
        selected_events: list[_PullEventOptionLike],
        target_layer_name: str,
    ) -> tuple[SyncDiffSummary, list[SyncDiffRow]]:
        rows = [
            self._build_pull_row(
                source_event=event,
                order_index=index,
                target_layer_name=target_layer_name,
            )
            for index, event in enumerate(selected_events, start=1)
        ]
        rows.sort(key=self._row_sort_key)
        return self._added_only_summary(rows), rows

    @staticmethod
    def resolve_pull_event_range(source_event: _PullEventOptionLike, *, order_index: int) -> tuple[float, float]:
        default_duration = 0.25
        if source_event.start is not None and source_event.end is not None:
            return float(source_event.start), float(source_event.end)
        if source_event.start is not None:
            start = max(0.0, float(source_event.start))
            return start, start + default_duration
        if source_event.end is not None:
            end = max(0.0, float(source_event.end))
            return max(0.0, end - default_duration), end

        start = float(order_index - 1) * default_duration
        return start, start + default_duration

    def _build_pull_row(
        self,
        *,
        source_event: _PullEventOptionLike,
        order_index: int,
        target_layer_name: str,
    ) -> SyncDiffRow:
        start, end = self.resolve_pull_event_range(source_event, order_index=order_index)
        return SyncDiffRow(
            row_id=str(source_event.event_id),
            action="add",
            start=start,
            end=end,
            label=source_event.label,
            before="Not present in EZ target layer",
            after=target_layer_name,
        )

    @staticmethod
    def _added_only_summary(rows: list[SyncDiffRow]) -> SyncDiffSummary:
        return SyncDiffSummary(
            added_count=len(rows),
            removed_count=0,
            modified_count=0,
            unchanged_count=0,
            row_count=len(rows),
        )

    @staticmethod
    def _event_sort_key(event: Event) -> tuple[float, float, str, str]:
        return (float(event.start), float(event.end), str(event.label), str(event.id))

    @staticmethod
    def _row_sort_key(row: SyncDiffRow) -> tuple[float, float, str, str]:
        return (float(row.start), float(row.end), str(row.label), str(row.row_id))
