"""Project timeline shared types for the Qt app shell.
Exists to keep storage and presentation helpers typed without circular imports.
Connects baseline assembly and presentation enrichment through shared containers.
"""

from __future__ import annotations

from dataclasses import dataclass

from echozero.application.presentation.models import (
    SongOptionPresentation,
    SongVersionOptionPresentation,
)
from echozero.application.shared.ids import LayerId, TakeId


@dataclass(slots=True)
class AudioPresentationFields:
    """Audio-specific presentation fields layered onto timeline rows and takes."""

    waveform_key: str | None = None
    source_audio_path: str | None = None
    playback_source_ref: str | None = None


@dataclass(slots=True)
class TimelinePresentationOverlay:
    """Presentation overlay fields applied after baseline timeline assembly."""

    project_title: str
    end_time_label: str
    bpm: float | None
    layer_audio: dict[LayerId, AudioPresentationFields]
    take_audio: dict[TakeId, AudioPresentationFields]
    active_song_id: str = ""
    active_song_title: str = ""
    active_song_version_id: str = ""
    active_song_version_label: str = ""
    active_song_version_ma3_timecode_pool_no: int | None = None
    available_songs: list[SongOptionPresentation] | None = None
    available_song_versions: list[SongVersionOptionPresentation] | None = None

    def __post_init__(self) -> None:
        if self.available_songs is None:
            self.available_songs = []
        if self.available_song_versions is None:
            self.available_song_versions = []
