"""Project timeline presentation overlay helpers for the Qt app shell.
Exists to keep presentation enrichment and selector option formatting out of the storage root.
Connects baseline timeline assembly to the timeline widget-facing presentation surface.
"""

from __future__ import annotations

from dataclasses import replace

from echozero.application.playback.timecode import format_clock_label
from echozero.application.presentation.models import (
    LayerPresentation,
    SongOptionPresentation,
    SongVersionOptionPresentation,
    TimelinePresentation,
)
from echozero.application.shared.enums import LayerKind
from echozero.ui.qt.app_shell_project_timeline_types import (
    AudioPresentationFields,
    TimelinePresentationOverlay,
)


def empty_overlay(
    project_title: str,
    *,
    active_song_id: str = "",
    active_song_title: str = "",
    active_song_version_id: str = "",
    active_song_version_label: str = "",
    active_song_version_ma3_timecode_pool_no: int | None = None,
    available_songs: list[SongOptionPresentation] | None = None,
    available_song_versions: list[SongVersionOptionPresentation] | None = None,
) -> TimelinePresentationOverlay:
    """Build the empty-state presentation overlay for a project shell."""

    return TimelinePresentationOverlay(
        project_title=project_title,
        end_time_label="00:00.00",
        bpm=None,
        active_song_id=active_song_id,
        active_song_title=active_song_title,
        active_song_version_id=active_song_version_id,
        active_song_version_label=active_song_version_label,
        active_song_version_ma3_timecode_pool_no=active_song_version_ma3_timecode_pool_no,
        available_songs=list(available_songs or []),
        available_song_versions=list(available_song_versions or []),
        layer_audio={},
        take_audio={},
    )


def apply_timeline_presentation_overlay(
    presentation: TimelinePresentation,
    *,
    overlay: TimelinePresentationOverlay,
) -> TimelinePresentation:
    """Apply audio and waveform overlay fields onto a presentation snapshot."""

    layers: list[LayerPresentation] = []
    for layer in presentation.layers:
        layer_fields = overlay.layer_audio.get(layer.layer_id, AudioPresentationFields())
        takes = [
            replace(
                take,
                waveform_key=overlay.take_audio.get(
                    take.take_id,
                    AudioPresentationFields(),
                ).waveform_key,
                source_audio_path=overlay.take_audio.get(
                    take.take_id,
                    AudioPresentationFields(),
                ).source_audio_path,
                playback_source_ref=overlay.take_audio.get(
                    take.take_id,
                    AudioPresentationFields(),
                ).playback_source_ref,
            )
            for take in layer.takes
        ]
        layers.append(
            replace(
                layer,
                badges=layer_badges(layer.title, layer.kind),
                waveform_key=layer_fields.waveform_key,
                source_audio_path=layer_fields.source_audio_path,
                playback_source_ref=layer_fields.playback_source_ref,
                takes=takes,
            )
        )

    return replace(
        presentation,
        title=overlay.project_title,
        bpm=overlay.bpm,
        active_song_id=overlay.active_song_id,
        active_song_title=overlay.active_song_title,
        active_song_version_id=overlay.active_song_version_id,
        active_song_version_label=overlay.active_song_version_label,
        active_song_version_ma3_timecode_pool_no=overlay.active_song_version_ma3_timecode_pool_no,
        available_songs=list(overlay.available_songs),
        available_song_versions=list(overlay.available_song_versions),
        current_time_label=format_time(presentation.playhead),
        end_time_label=overlay.end_time_label,
        layers=layers,
    )


def layer_badges(name: str, kind: LayerKind) -> list[str]:
    """Compute default layer badges for runtime presentation."""

    badges = ["main", kind.value]
    if kind is LayerKind.AUDIO and name.strip().lower() != "imported song":
        badges.append("stem")
    if "drum" in name.strip().lower():
        badges.append("drums")
    return badges


def format_time(seconds: float) -> str:
    """Format seconds for the timeline shell time labels."""

    return format_clock_label(seconds)


def available_song_options(
    songs,
    *,
    active_song_id: str | None,
    active_versions_by_song_id: dict[str, object | None],
    versions_by_song_id: dict[str, list[object]],
    version_counts_by_song_id: dict[str, int],
) -> list[SongOptionPresentation]:
    """Build selector options for songs in the current project."""

    options: list[SongOptionPresentation] = []
    for song in songs:
        active_version = active_versions_by_song_id.get(song.id)
        song_versions = versions_by_song_id.get(song.id, [])
        options.append(
            SongOptionPresentation(
                song_id=song.id,
                title=song.title,
                is_active=song.id == active_song_id,
                active_version_id=song.active_version_id or "",
                active_version_label=active_version.label if active_version is not None else "",
                version_count=version_counts_by_song_id.get(song.id, 0),
                versions=available_song_version_options(
                    song_versions,
                    active_song_version_id=song.active_version_id,
                ),
            )
        )
    return options


def available_song_version_options(
    versions,
    *,
    active_song_version_id: str | None,
) -> list[SongVersionOptionPresentation]:
    """Build selector options for versions of the currently active song."""

    return [
        SongVersionOptionPresentation(
            song_version_id=version.id,
            label=version.label,
            is_active=version.id == active_song_version_id,
            ma3_timecode_pool_no=version.ma3_timecode_pool_no,
        )
        for version in versions
    ]
