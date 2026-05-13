"""MA3 show sync contract validation for EchoZero.
Exists so EchoZero can preflight clean show intent before MA3SongManager mutates a showfile.
Connects setlist planning, automator events, and MA3 snapshots through typed DTOs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from echozero.application.ma3_show.models import (
    EchoZeroShowPlan,
    MA3SongManagerSnapshot,
    SequenceBlockPolicy,
    Setlist,
)
from echozero.application.ma3_show.planning import build_automator_events


@dataclass(frozen=True, slots=True)
class ShowPlanPreflight:
    """Validation result for an EchoZero show plan before MA3 sync."""

    is_ready: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ShowPlanDiff:
    """Diff between EchoZero show intent and an MA3SongManager snapshot."""

    missing_sequences: tuple[int, ...] = field(default_factory=tuple)
    missing_automator_events: tuple[str, ...] = field(default_factory=tuple)
    placement_changes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_changes(self) -> bool:
        """Return whether MA3 differs from EchoZero show intent."""

        return bool(
            self.missing_sequences or self.missing_automator_events or self.placement_changes
        )


def build_show_plan(
    *,
    project_id: str,
    setlist: Setlist,
    sequence_policy: SequenceBlockPolicy | None = None,
) -> EchoZeroShowPlan:
    """Build a serializable show plan from a mapped setlist."""

    policy = sequence_policy or SequenceBlockPolicy()
    return EchoZeroShowPlan(
        project_id=project_id,
        setlist=setlist,
        automator_events=build_automator_events(setlist),
        sequence_policy=policy,
    )


def preflight_show_plan(plan: EchoZeroShowPlan) -> ShowPlanPreflight:
    """Validate that a show plan is safe to send to MA3SongManager."""

    issues: list[str] = []
    warnings: list[str] = []
    sequence_numbers: set[int] = set()
    event_ids: set[str] = set()
    for song in plan.setlist.songs:
        mapping = song.ma3_mapping
        if mapping is None:
            issues.append(f"Song {song.id} has no MA3 sequence mapping")
            continue
        if mapping.main_sequence_no in sequence_numbers:
            issues.append(f"Sequence {mapping.main_sequence_no} is assigned to multiple songs")
        sequence_numbers.add(mapping.main_sequence_no)
        if not song.sections:
            warnings.append(f"Song {song.id} has no sections for automator generation")

    for event in plan.automator_events:
        if event.id in event_ids:
            issues.append(f"Automator event id {event.id} is duplicated")
        event_ids.add(event.id)
        if "record timecode" in event.command.lower():
            issues.append(f"Automator event {event.id} uses Record Timecode")
        if "cursor" in event.command.lower():
            issues.append(f"Automator event {event.id} depends on cursor state")

    return ShowPlanPreflight(
        is_ready=not issues,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


def diff_show_plan_snapshot(
    plan: EchoZeroShowPlan,
    snapshot: MA3SongManagerSnapshot,
) -> ShowPlanDiff:
    """Compare EchoZero show intent with an MA3SongManager snapshot."""

    snapshot_sequences = _snapshot_sequences(snapshot)
    snapshot_event_ids = _snapshot_event_ids(snapshot)
    snapshot_placements = _snapshot_placements(snapshot)

    missing_sequences: list[int] = []
    placement_changes: list[str] = []
    for song in plan.setlist.songs:
        mapping = song.ma3_mapping
        if mapping is None:
            continue
        sequence_no = mapping.main_sequence_no
        if sequence_no not in snapshot_sequences:
            missing_sequences.append(sequence_no)
        for placement in mapping.executor_placements:
            placement_key = (
                sequence_no,
                int(placement.page_no),
                int(placement.executor_no),
            )
            if placement_key not in snapshot_placements:
                placement_changes.append(
                    f"Sequence {sequence_no} placement {placement.page_no}.{placement.executor_no}"
                )

    missing_events = [
        event.id for event in plan.automator_events if event.id not in snapshot_event_ids
    ]
    return ShowPlanDiff(
        missing_sequences=tuple(sorted(set(missing_sequences))),
        missing_automator_events=tuple(missing_events),
        placement_changes=tuple(placement_changes),
    )


def _snapshot_sequences(snapshot: MA3SongManagerSnapshot) -> set[int]:
    sequences: set[int] = set()
    for song in snapshot.songs:
        value = song.get("main_sequence_no")
        if value is None:
            value = song.get("sequence_no")
        try:
            sequences.add(int(value))
        except (TypeError, ValueError):
            continue
    return sequences


def _snapshot_event_ids(snapshot: MA3SongManagerSnapshot) -> set[str]:
    event_ids: set[str] = set()
    for event in snapshot.automator_events:
        event_id = str(event.get("id") or event.get("event_id") or "").strip()
        if event_id:
            event_ids.add(event_id)
    return event_ids


def _snapshot_placements(snapshot: MA3SongManagerSnapshot) -> set[tuple[int, int, int]]:
    placements: set[tuple[int, int, int]] = set()
    for song in snapshot.songs:
        sequence_value = song.get("main_sequence_no")
        if sequence_value is None:
            sequence_value = song.get("sequence_no")
        try:
            sequence_no = int(sequence_value)
        except (TypeError, ValueError):
            continue
        for placement in song.get("executor_placements") or ():
            if not isinstance(placement, dict):
                continue
            try:
                placements.add(
                    (
                        sequence_no,
                        int(placement.get("page_no")),
                        int(placement.get("executor_no")),
                    )
                )
            except (TypeError, ValueError):
                continue
    return placements
