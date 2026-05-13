"""Domain models for EchoZero-owned MA3 show intent.
Exists to keep song identity, setlist order, and automator events explicit.
Connects clean-sheet EchoZero planning to MA3SongManager payloads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


AutomatorEventStatus = Literal["generated", "manual", "locked"]
SyncStatus = Literal["draft", "ready", "synced", "blocked", "failed"]


@dataclass(frozen=True, slots=True)
class SequenceBlockPolicy:
    """Defines how EchoZero assigns MA3 sequence ranges to setlist songs."""

    first_sequence_no: int = 1100
    block_size: int = 100

    def __post_init__(self) -> None:
        if self.first_sequence_no < 1:
            raise ValueError("SequenceBlockPolicy.first_sequence_no must be >= 1")
        if self.block_size < 1:
            raise ValueError("SequenceBlockPolicy.block_size must be >= 1")

    def sequence_for_index(self, index: int) -> int:
        if index < 0:
            raise ValueError("sequence index must be >= 0")
        return self.first_sequence_no + (index * self.block_size)


@dataclass(frozen=True, slots=True)
class MA3ExecutorPlacement:
    """Movable MA3 page/executor placement for a song sequence."""

    page_no: int
    executor_no: int
    label: str = ""

    def __post_init__(self) -> None:
        if self.page_no < 1:
            raise ValueError("MA3ExecutorPlacement.page_no must be >= 1")
        if self.executor_no < 1:
            raise ValueError("MA3ExecutorPlacement.executor_no must be >= 1")
        object.__setattr__(self, "label", str(self.label or "").strip())


@dataclass(frozen=True, slots=True)
class MA3MasterOutputAssignment:
    """Single-channel assignment for a show-level MA3 master output."""

    id: str
    label: str
    channel_no: int
    universe_no: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        output_id = str(self.id or "").strip()
        if not output_id:
            raise ValueError("MA3MasterOutputAssignment.id is required")
        if self.channel_no < 1:
            raise ValueError("MA3MasterOutputAssignment.channel_no must be >= 1")
        if self.universe_no is not None and self.universe_no < 1:
            raise ValueError("MA3MasterOutputAssignment.universe_no must be >= 1 when provided")
        object.__setattr__(self, "id", output_id)
        object.__setattr__(self, "label", str(self.label or "").strip() or output_id)
        object.__setattr__(self, "notes", str(self.notes or "").strip())

    @property
    def channel_address(self) -> str:
        """Return the compact MA3 channel address used for assignment."""

        if self.universe_no is None:
            return f"channel:{self.channel_no}"
        return f"universe:{self.universe_no}/channel:{self.channel_no}"


@dataclass(frozen=True, slots=True)
class MA3ObjectMapping:
    """Canonical and view-only MA3 mappings for one EchoZero song."""

    main_sequence_no: int
    sequence_range_start: int
    sequence_range_end: int
    timecode_pool_no: int | None = None
    track_coords: tuple[str, ...] = field(default_factory=tuple)
    executor_placements: tuple[MA3ExecutorPlacement, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.main_sequence_no < 1:
            raise ValueError("MA3ObjectMapping.main_sequence_no must be >= 1")
        if self.sequence_range_start < 1:
            raise ValueError("MA3ObjectMapping.sequence_range_start must be >= 1")
        if self.sequence_range_end < self.sequence_range_start:
            raise ValueError(
                "MA3ObjectMapping.sequence_range_end must be >= sequence_range_start"
            )
        if not self.sequence_range_start <= self.main_sequence_no <= self.sequence_range_end:
            raise ValueError("main_sequence_no must be inside the sequence range")
        if self.timecode_pool_no is not None and self.timecode_pool_no < 1:
            raise ValueError("timecode_pool_no must be >= 1 when provided")
        object.__setattr__(
            self,
            "track_coords",
            tuple(str(coord).strip() for coord in self.track_coords if str(coord).strip()),
        )
        object.__setattr__(
            self,
            "executor_placements",
            tuple(self.executor_placements or ()),
        )

    @property
    def canonical_identity(self) -> str:
        """Return the stable MA3 song identity."""

        return f"sequence:{self.main_sequence_no}"


@dataclass(frozen=True, slots=True)
class SongSection:
    """Song-local section/cue plan used to generate automator events."""

    id: str
    label: str
    start_seconds: float
    end_seconds: float | None = None
    cue_number: str | None = None
    cue_ref: str | None = None
    notes: str = ""
    locked: bool = False

    def __post_init__(self) -> None:
        section_id = str(self.id or "").strip()
        if not section_id:
            raise ValueError("SongSection.id is required")
        if self.start_seconds < 0:
            raise ValueError("SongSection.start_seconds must be >= 0")
        if self.end_seconds is not None and self.end_seconds < self.start_seconds:
            raise ValueError("SongSection.end_seconds must be >= start_seconds")
        object.__setattr__(self, "id", section_id)
        object.__setattr__(self, "label", str(self.label or "").strip() or "Section")
        cue_number = None if self.cue_number is None else str(self.cue_number).strip()
        cue_ref = None if self.cue_ref is None else str(self.cue_ref).strip()
        object.__setattr__(self, "cue_number", cue_number or None)
        object.__setattr__(self, "cue_ref", cue_ref or cue_number or None)
        object.__setattr__(self, "notes", str(self.notes or "").strip())


@dataclass(frozen=True, slots=True)
class AutomatorEvent:
    """EchoZero-owned direct MA3 timecode command event plan."""

    id: str
    song_id: str
    section_id: str | None
    start_seconds: float
    target_sequence_no: int
    cue_ref: str
    command: str
    label: str
    status: AutomatorEventStatus = "generated"
    sync_status: SyncStatus = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        event_id = str(self.id or "").strip()
        song_id = str(self.song_id or "").strip()
        if not event_id:
            raise ValueError("AutomatorEvent.id is required")
        if not song_id:
            raise ValueError("AutomatorEvent.song_id is required")
        if self.start_seconds < 0:
            raise ValueError("AutomatorEvent.start_seconds must be >= 0")
        if self.target_sequence_no < 1:
            raise ValueError("AutomatorEvent.target_sequence_no must be >= 1")
        cue_ref = str(self.cue_ref or "").strip()
        if not cue_ref:
            raise ValueError("AutomatorEvent.cue_ref is required")
        command = str(self.command or "").strip()
        if not command:
            raise ValueError("AutomatorEvent.command is required")
        object.__setattr__(self, "id", event_id)
        object.__setattr__(self, "song_id", song_id)
        object.__setattr__(
            self,
            "section_id",
            None if self.section_id is None else str(self.section_id).strip() or None,
        )
        object.__setattr__(self, "cue_ref", cue_ref)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "label", str(self.label or "").strip() or command)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True, slots=True)
class ShowSong:
    """Setlist song with clean-sheet MA3 planning data."""

    id: str
    title: str
    artist: str = ""
    order: int = 0
    notes: str = ""
    bpm: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: tuple[SongSection, ...] = field(default_factory=tuple)
    ma3_mapping: MA3ObjectMapping | None = None

    def __post_init__(self) -> None:
        song_id = str(self.id or "").strip()
        if not song_id:
            raise ValueError("ShowSong.id is required")
        if self.order < 0:
            raise ValueError("ShowSong.order must be >= 0")
        if self.bpm is not None and self.bpm <= 0:
            raise ValueError("ShowSong.bpm must be > 0 when provided")
        object.__setattr__(self, "id", song_id)
        object.__setattr__(self, "title", str(self.title or "").strip() or "Untitled")
        object.__setattr__(self, "artist", str(self.artist or "").strip())
        object.__setattr__(self, "notes", str(self.notes or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "sections", tuple(self.sections or ()))


@dataclass(frozen=True, slots=True)
class Setlist:
    """One EchoZero show setlist."""

    id: str
    name: str
    songs: tuple[ShowSong, ...] = field(default_factory=tuple)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        setlist_id = str(self.id or "").strip()
        if not setlist_id:
            raise ValueError("Setlist.id is required")
        songs = tuple(sorted(self.songs or (), key=lambda song: song.order))
        object.__setattr__(self, "id", setlist_id)
        object.__setattr__(self, "name", str(self.name or "").strip() or "Setlist")
        object.__setattr__(self, "songs", songs)
        object.__setattr__(self, "notes", str(self.notes or "").strip())
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True, slots=True)
class EchoZeroShowPlan:
    """Serializable EchoZero-to-MA3SongManager show intent."""

    project_id: str
    setlist: Setlist
    automator_events: tuple[AutomatorEvent, ...] = field(default_factory=tuple)
    master_outputs: tuple[MA3MasterOutputAssignment, ...] = field(default_factory=tuple)
    sequence_policy: SequenceBlockPolicy = field(default_factory=SequenceBlockPolicy)

    def __post_init__(self) -> None:
        project_id = str(self.project_id or "").strip()
        if not project_id:
            raise ValueError("EchoZeroShowPlan.project_id is required")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "automator_events", tuple(self.automator_events or ()))
        object.__setattr__(self, "master_outputs", tuple(self.master_outputs or ()))

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible payload for MA3SongManager."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MA3SongManagerSnapshot:
    """Snapshot returned by MA3SongManager after preflight/apply/refresh."""

    project_id: str
    songs: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    automator_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        project_id = str(self.project_id or "").strip()
        if not project_id:
            raise ValueError("MA3SongManagerSnapshot.project_id is required")
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "songs", tuple(dict(song) for song in self.songs))
        object.__setattr__(
            self,
            "automator_events",
            tuple(dict(event) for event in self.automator_events),
        )
        object.__setattr__(
            self,
            "validation_errors",
            tuple(str(error) for error in self.validation_errors if str(error).strip()),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
