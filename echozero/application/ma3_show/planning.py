"""Planning helpers for EchoZero-owned MA3 song and automator intent.
Exists to allocate canonical sequence ranges and generated direct events.
Connects imported setlists to MA3-ready show plans.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.ma3_show.models import (
    AutomatorEvent,
    MA3ObjectMapping,
    SequenceBlockPolicy,
    Setlist,
    ShowSong,
)


def command_for_sequence_cue(sequence_no: int, cue_ref: str) -> str:
    """Return the direct MA3 command used by automator timecode events."""

    if sequence_no < 1:
        raise ValueError("sequence_no must be >= 1")
    normalized_cue_ref = str(cue_ref or "").strip()
    if not normalized_cue_ref:
        raise ValueError("cue_ref is required")
    return f"Goto Sequence {int(sequence_no)} Cue {normalized_cue_ref}"


def allocate_song_sequence_mappings(
    setlist: Setlist,
    *,
    policy: SequenceBlockPolicy | None = None,
) -> Setlist:
    """Assign canonical MA3 sequence ranges to songs that do not already have one."""

    resolved_policy = policy or SequenceBlockPolicy()
    planned_songs: list[ShowSong] = []
    for index, song in enumerate(setlist.songs):
        if song.ma3_mapping is not None:
            planned_songs.append(song)
            continue
        main_sequence_no = resolved_policy.sequence_for_index(index)
        mapping = MA3ObjectMapping(
            main_sequence_no=main_sequence_no,
            sequence_range_start=main_sequence_no,
            sequence_range_end=main_sequence_no + resolved_policy.block_size - 1,
        )
        planned_songs.append(replace(song, ma3_mapping=mapping))
    return replace(setlist, songs=tuple(planned_songs))


def build_automator_events(setlist: Setlist) -> tuple[AutomatorEvent, ...]:
    """Generate direct MA3 automator event plans from song sections."""

    events: list[AutomatorEvent] = []
    for song in setlist.songs:
        if song.ma3_mapping is None:
            raise ValueError(f"Song {song.id} requires an MA3 mapping before automator planning")
        for section_index, section in enumerate(song.sections, start=1):
            cue_ref = section.cue_ref or section.cue_number or str(section_index)
            command = command_for_sequence_cue(song.ma3_mapping.main_sequence_no, cue_ref)
            events.append(
                AutomatorEvent(
                    id=f"{song.id}:{section.id}:automator",
                    song_id=song.id,
                    section_id=section.id,
                    start_seconds=section.start_seconds,
                    target_sequence_no=song.ma3_mapping.main_sequence_no,
                    cue_ref=cue_ref,
                    command=command,
                    label=section.label,
                    sync_status="ready",
                    metadata={
                        "source": "song_section",
                        "canonical_identity": song.ma3_mapping.canonical_identity,
                    },
                )
            )
    return tuple(events)
