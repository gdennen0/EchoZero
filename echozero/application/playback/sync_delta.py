"""
Playback sync delta classifier for app-side runtime audio decisions.
Exists because the UI must decide playback sync work locally without blocking on runtime IPC.
Connects compact playback projections to one shared none/mix/structure change model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

from echozero.application.playback.audible_intent import audible_intent_from_sync_payload
from echozero.application.playback.sync_projection import PlaybackSyncPayload
from echozero.application.presentation.models import TimelinePresentation


class PlaybackChangeKind(StrEnum):
    """Kinds of app-side playback runtime change."""

    NONE = "none"
    MIX_ONLY = "mix_only"
    STRUCTURE = "structure"


@dataclass(slots=True, frozen=True)
class PlaybackSyncDelta:
    """One classified playback-sync transition."""

    payload: PlaybackSyncPayload
    change_kind: PlaybackChangeKind
    projection_build_ms: float
    classify_ms: float


def classify_playback_sync_change(
    previous_payload: PlaybackSyncPayload | None,
    next_source: TimelinePresentation | PlaybackSyncPayload,
) -> PlaybackSyncDelta:
    """Classify one playback change without touching the runtime-audio backend."""

    build_started = perf_counter()
    if isinstance(next_source, PlaybackSyncPayload):
        next_payload = next_source
        projection_build_ms = 0.0
    else:
        next_payload = PlaybackSyncPayload.from_presentation(next_source)
        projection_build_ms = max(0.0, (perf_counter() - build_started) * 1000.0)

    classify_started = perf_counter()
    if previous_payload is None:
        next_structure = _structure_signature(next_payload)
        next_mix = _mix_signature(next_payload)
        change_kind = (
            PlaybackChangeKind.STRUCTURE if next_structure or next_mix else PlaybackChangeKind.NONE
        )
    else:
        previous_structure = _structure_signature(previous_payload)
        next_structure = _structure_signature(next_payload)
        if previous_structure != next_structure:
            change_kind = PlaybackChangeKind.STRUCTURE
        else:
            previous_mix = _mix_signature(previous_payload)
            next_mix = _mix_signature(next_payload)
            if previous_mix != next_mix:
                change_kind = PlaybackChangeKind.MIX_ONLY
            else:
                change_kind = PlaybackChangeKind.NONE
    classify_ms = max(0.0, (perf_counter() - classify_started) * 1000.0)

    return PlaybackSyncDelta(
        payload=next_payload,
        change_kind=change_kind,
        projection_build_ms=projection_build_ms,
        classify_ms=classify_ms,
    )


def playback_structure_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    """Return the shared structure signature used for playback graph decisions."""

    return _structure_signature(payload)


def playback_mix_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    """Return the shared mix signature used for playback graph decisions."""

    return _mix_signature(payload)


def _structure_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    return audible_intent_from_sync_payload(payload).structure_signature


def _mix_signature(payload: PlaybackSyncPayload) -> tuple[tuple[str, str], ...]:
    return audible_intent_from_sync_payload(payload).mix_signature
