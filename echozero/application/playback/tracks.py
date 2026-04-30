"""
Playback track planning for EZ runtime playback.
Exists because every playable EZ layer should resolve into one simple DAW-style track surface.
Connects timeline presentation state to engine-ready `PlaybackTrack` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import PlaybackMode
from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.audio.layer import AudioTrack


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _normalize_output_bus(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    output_bus = value.strip()
    return output_bus or None


def _event_start_seconds(event: object) -> float:
    try:
        return float(getattr(event, "start", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _event_is_muted(event: object) -> bool:
    return bool(getattr(event, "muted", False))


def _event_badges(event: object) -> tuple[str, ...]:
    badges = getattr(event, "badges", ())
    if not isinstance(badges, (list, tuple, set)):
        return ()
    return tuple(str(badge) for badge in badges)


def _event_signature(events: list[object]) -> str:
    return ",".join(
        f"{_event_start_seconds(event):.6f}:{int(_event_is_muted(event))}:{int('demoted' in _event_badges(event))}"
        for event in events
    )


@dataclass(slots=True)
class PlaybackTrack:
    """One EZ playback track resolved from a presentation layer or take."""

    track_id: str
    source_layer_id: LayerId
    source_take_id: TakeId | None
    name: str
    gain_db: float
    output_bus: str | None
    source_key: str
    cache_keys: tuple[str, ...]
    muted: bool = False
    buffer: np.ndarray | None = None
    sample_rate: int = 0
    source_ref: str | None = None

    @property
    def mode(self) -> PlaybackMode:
        if self.source_key.startswith("event:"):
            return PlaybackMode.EVENT_SLICE
        return PlaybackMode.CONTINUOUS_AUDIO

    @property
    def signature_token(self) -> str:
        return f"{self.source_key}|{self.output_bus or 'outputs_1_2'}"

    def to_audio_track(
        self,
        *,
        engine_track_id: str,
        engine_sample_rate: int | None = None,
    ) -> AudioTrack:
        """Build one engine-ready audio track from this playback track."""

        if self.buffer is None or self.sample_rate <= 0:
            raise ValueError(f"Playback track '{self.track_id}' is missing resolved audio.")
        return AudioTrack(
            layer_id=engine_track_id,
            name=self.name,
            buffer=self.buffer,
            sample_rate=self.sample_rate,
            volume=_db_to_linear(self.gain_db),
            engine_sample_rate=engine_sample_rate,
            output_bus=self.output_bus,
        )


@dataclass(slots=True, frozen=True)
class PlaybackTrackPlan:
    """One selected playback plan ready to sync into the engine."""

    tracks: tuple[PlaybackTrack, ...]
    signature: tuple[tuple[str, str], ...]
    cache_keys: frozenset[str]
    uses_track_routing: bool


class PlaybackTrackBuilder:
    """Builds DAW-style playback tracks from one timeline presentation."""

    def __init__(
        self,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]],
    ) -> None:
        self._audio_loader = audio_loader
        self._buffer_cache: dict[str, tuple[np.ndarray, int]] = {}

    def prune_cache(self, keep_keys: set[str] | frozenset[str]) -> None:
        """Drop cached decoded buffers that are no longer needed."""

        stale_keys = [key for key in self._buffer_cache if key not in keep_keys]
        for stale_key in stale_keys:
            self._buffer_cache.pop(stale_key, None)

    def load_source_buffer(self, cache_key: str, source_ref: str) -> tuple[np.ndarray, int]:
        """Load one decoded buffer through the planner cache."""

        cached = self._buffer_cache.get(cache_key)
        if cached is None:
            cached = self._audio_loader(source_ref)
            self._buffer_cache[cache_key] = cached
        return cached

    def resolve_audio(self, playback_track: PlaybackTrack) -> tuple[np.ndarray, int]:
        """Resolve decoded audio for one playback track."""

        if playback_track.buffer is not None and playback_track.sample_rate > 0:
            return playback_track.buffer, playback_track.sample_rate
        source_ref = str(playback_track.source_ref or "").strip()
        if not source_ref:
            raise ValueError(f"Playback track '{playback_track.track_id}' has no source audio ref.")
        buffer, sample_rate = self.load_source_buffer(playback_track.source_key, source_ref)
        playback_track.buffer = buffer
        playback_track.sample_rate = sample_rate
        return buffer, sample_rate

    def build_track_plan(self, presentation: TimelinePresentation) -> PlaybackTrackPlan:
        """Build the current selected playback tracks for one presentation."""

        tracks, uses_track_routing = self._selected_tracks(
            presentation,
            resolve_audio=True,
        )
        cache_keys = frozenset(
            cache_key
            for playback_track in tracks
            for cache_key in playback_track.cache_keys
        )
        signature = tuple(
            (playback_track.track_id, playback_track.signature_token)
            for playback_track in tracks
        )
        return PlaybackTrackPlan(
            tracks=tracks,
            signature=signature,
            cache_keys=cache_keys,
            uses_track_routing=uses_track_routing,
        )

    def build_track_signature(
        self,
        presentation: TimelinePresentation,
    ) -> tuple[tuple[str, str], ...]:
        """Describe the selected playback tracks without decoding audio."""

        tracks, _ = self._selected_tracks(
            presentation,
            resolve_audio=False,
        )
        return tuple(
            (playback_track.track_id, playback_track.signature_token)
            for playback_track in tracks
        )

    def describe_selected_tracks(
        self,
        presentation: TimelinePresentation,
    ) -> tuple[PlaybackTrack, ...]:
        """Describe the current selected playback tracks without decoding audio."""

        tracks, _ = self._selected_tracks(
            presentation,
            resolve_audio=False,
        )
        return tracks

    def _selected_tracks(
        self,
        presentation: TimelinePresentation,
        *,
        resolve_audio: bool,
    ) -> tuple[tuple[PlaybackTrack, ...], bool]:
        tracks = self._select_mix_tracks(
            presentation,
            resolve_audio=resolve_audio,
        )
        uses_track_routing = len(tracks) > 1 or any(
            playback_track.output_bus is not None for playback_track in tracks
        )
        return tuple(tracks), uses_track_routing

    def _select_mix_tracks(
        self,
        presentation: TimelinePresentation,
        *,
        resolve_audio: bool,
    ) -> list[PlaybackTrack]:
        layer_candidates = [
            layer for layer in presentation.layers if self._layer_has_playable_source(layer)
        ]
        if not layer_candidates:
            return []
        has_soloed_layers = any(
            bool(getattr(layer, "soloed", False))
            for layer in layer_candidates
        )
        selected_layer_id = (
            str(presentation.selected_layer_id)
            if presentation.selected_layer_id is not None
            else None
        )
        selected_take_id = (
            str(presentation.selected_take_id)
            if presentation.selected_take_id is not None
            else None
        )
        tracks: list[PlaybackTrack] = []
        seen_track_ids: set[str] = set()
        for layer in layer_candidates:
            layer_id = str(getattr(layer, "layer_id"))
            if layer_id == selected_layer_id and selected_take_id is not None:
                playback_track = self._track_for_target(
                    presentation,
                    layer_id=layer_id,
                    take_id=selected_take_id,
                    resolve_audio=resolve_audio,
                )
            else:
                playback_track = self._track_from_layer(layer, resolve_audio=resolve_audio)
            if playback_track is None or playback_track.track_id in seen_track_ids:
                continue
            layer_muted = bool(getattr(layer, "muted", False))
            layer_soloed = bool(getattr(layer, "soloed", False))
            playback_track.muted = layer_muted or (has_soloed_layers and not layer_soloed)
            tracks.append(playback_track)
            seen_track_ids.add(playback_track.track_id)
        return tracks

    def _track_for_target(
        self,
        presentation: TimelinePresentation,
        *,
        layer_id: str,
        take_id: str | None,
        resolve_audio: bool,
    ) -> PlaybackTrack | None:
        for layer in presentation.layers:
            if str(layer.layer_id) != layer_id:
                continue
            if take_id is not None:
                for take in layer.takes:
                    if str(take.take_id) == take_id:
                        playback_track = self._track_from_take(
                            layer,
                            take,
                            resolve_audio=resolve_audio,
                        )
                        if playback_track is not None:
                            return playback_track
            return self._track_from_layer(layer, resolve_audio=resolve_audio)
        return None

    def _track_from_layer(
        self,
        layer: object,
        *,
        resolve_audio: bool,
    ) -> PlaybackTrack | None:
        source_audio_path = getattr(layer, "source_audio_path", None)
        if source_audio_path:
            return PlaybackTrack(
                track_id=str(getattr(layer, "layer_id")),
                source_layer_id=getattr(layer, "layer_id"),
                source_take_id=None,
                name=str(getattr(layer, "title")),
                gain_db=float(getattr(layer, "gain_db", 0.0)),
                output_bus=_normalize_output_bus(getattr(layer, "output_bus", None)),
                muted=bool(getattr(layer, "muted", False)),
                source_key=f"audio:{source_audio_path}",
                cache_keys=(f"audio:{source_audio_path}",),
                source_ref=str(source_audio_path),
            )
        if not self._is_event_track_source(layer):
            return None
        return self._build_event_track(
            track_id=str(getattr(layer, "layer_id")),
            source_layer_id=getattr(layer, "layer_id"),
            source_take_id=None,
            title=str(getattr(layer, "title")),
            gain_db=float(getattr(layer, "gain_db", 0.0)),
            output_bus=_normalize_output_bus(getattr(layer, "output_bus", None)),
            muted=bool(getattr(layer, "muted", False)),
            playback_source_ref=str(getattr(layer, "playback_source_ref")),
            events=list(getattr(layer, "events")),
            resolve_audio=resolve_audio,
        )

    def _track_from_take(
        self,
        layer: object,
        take: object,
        *,
        resolve_audio: bool,
    ) -> PlaybackTrack | None:
        layer_id = str(getattr(layer, "layer_id"))
        take_id = str(getattr(take, "take_id"))
        source_audio_path = getattr(take, "source_audio_path", None)
        if source_audio_path:
            return PlaybackTrack(
                track_id=f"{layer_id}:{take_id}",
                source_layer_id=getattr(layer, "layer_id"),
                source_take_id=getattr(take, "take_id"),
                name=f"{getattr(layer, 'title')} · {getattr(take, 'name')}",
                gain_db=float(getattr(layer, "gain_db", 0.0)),
                output_bus=_normalize_output_bus(getattr(layer, "output_bus", None)),
                muted=bool(getattr(layer, "muted", False)),
                source_key=f"audio:{source_audio_path}",
                cache_keys=(f"audio:{source_audio_path}",),
                source_ref=str(source_audio_path),
            )
        if not self._is_event_track_source(take):
            return None
        return self._build_event_track(
            track_id=f"{layer_id}:{take_id}",
            source_layer_id=getattr(layer, "layer_id"),
            source_take_id=getattr(take, "take_id"),
            title=f"{getattr(layer, 'title')} · {getattr(take, 'name')}",
            gain_db=float(getattr(layer, "gain_db", 0.0)),
            output_bus=_normalize_output_bus(getattr(layer, "output_bus", None)),
            muted=bool(getattr(layer, "muted", False)),
            playback_source_ref=str(getattr(take, "playback_source_ref")),
            events=list(getattr(take, "events")),
            resolve_audio=resolve_audio,
        )

    def _build_event_track(
        self,
        *,
        track_id: str,
        source_layer_id: LayerId,
        source_take_id: TakeId | None,
        title: str,
        gain_db: float,
        output_bus: str | None,
        muted: bool,
        playback_source_ref: str,
        events: list[object],
        resolve_audio: bool,
    ) -> PlaybackTrack | None:
        sample_source_key = f"event-sample:{playback_source_ref}"
        rendered_source_key = f"event:{playback_source_ref}:{_event_signature(events)}"
        if not resolve_audio:
            return PlaybackTrack(
                track_id=track_id,
                source_layer_id=source_layer_id,
                source_take_id=source_take_id,
                name=title,
                gain_db=gain_db,
                output_bus=output_bus,
                muted=muted,
                source_key=rendered_source_key,
                cache_keys=(sample_source_key, rendered_source_key),
                source_ref=playback_source_ref,
            )
        event_buffer, sample_rate = self.load_source_buffer(sample_source_key, playback_source_ref)
        cached_render = self._buffer_cache.get(rendered_source_key)
        if cached_render is None:
            rendered = self._render_event_track_buffer(
                event_buffer,
                sample_rate,
                events=events,
            )
            if rendered.size == 0:
                return None
            self._buffer_cache[rendered_source_key] = (rendered, sample_rate)
        else:
            rendered, sample_rate = cached_render
        return PlaybackTrack(
            track_id=track_id,
            source_layer_id=source_layer_id,
            source_take_id=source_take_id,
            name=title,
            gain_db=gain_db,
            output_bus=output_bus,
            muted=muted,
            source_key=rendered_source_key,
            cache_keys=(sample_source_key, rendered_source_key),
            buffer=rendered,
            sample_rate=sample_rate,
            source_ref=playback_source_ref,
        )

    @staticmethod
    def _layer_has_playable_source(layer: object) -> bool:
        return bool(
            getattr(layer, "source_audio_path", None)
            or PlaybackTrackBuilder._is_event_track_source(layer)
        )

    @staticmethod
    def _is_event_track_source(layer: object) -> bool:
        return bool(
            is_event_like_layer_kind(getattr(layer, "kind", None))
            and getattr(layer, "playback_enabled", False)
            and getattr(layer, "playback_mode", None) == PlaybackMode.EVENT_SLICE
            and getattr(layer, "playback_source_ref", None)
        )

    @staticmethod
    def _render_event_track_buffer(
        event_buffer: np.ndarray,
        sample_rate: int,
        *,
        events: list[object],
    ) -> np.ndarray:
        if event_buffer.size == 0:
            return np.zeros(0, dtype=np.float32)
        active_events = [
            event
            for event in events
            if not _event_is_muted(event) and "demoted" not in _event_badges(event)
        ]
        if not active_events:
            return np.zeros(0, dtype=np.float32)
        start_samples = [
            max(0, int(round(_event_start_seconds(event) * sample_rate)))
            for event in active_events
        ]
        total_samples = max(start_samples) + int(event_buffer.shape[0])
        if event_buffer.ndim == 1:
            rendered = np.zeros(total_samples, dtype=np.float32)
        else:
            rendered = np.zeros((total_samples, event_buffer.shape[1]), dtype=np.float32)
        for start_sample in start_samples:
            end_sample = start_sample + int(event_buffer.shape[0])
            rendered[start_sample:end_sample] += event_buffer
        np.clip(rendered, -1.0, 1.0, out=rendered)
        return rendered


__all__ = [
    "PlaybackTrack",
    "PlaybackTrackBuilder",
    "PlaybackTrackPlan",
]
