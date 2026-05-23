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
from echozero.application.shared.enums import LayerKind, PlaybackMode
from echozero.application.shared.ids import LayerId, TakeId
from echozero.application.shared.layer_kinds import is_event_like_layer_kind
from echozero.application.playback.track_identity import (
    event_slice_signature,
    sanitize_output_bus_for_channels,
)
from echozero.audio.layer import AudioTrack


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _event_start_seconds(event: object) -> float:
    try:
        return float(getattr(event, "start", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _event_end_seconds(event: object) -> float:
    try:
        return float(getattr(event, "end", _event_start_seconds(event)))
    except (TypeError, ValueError):
        return _event_start_seconds(event)


def _event_is_muted(event: object) -> bool:
    return bool(getattr(event, "muted", False))


def _event_badges(event: object) -> tuple[str, ...]:
    badges = getattr(event, "badges", ())
    if not isinstance(badges, (list, tuple, set)):
        return ()
    return tuple(str(badge) for badge in badges)


def _event_slice_fade_samples(sample_rate: int, total_samples: int) -> int:
    if total_samples <= 1:
        return 0
    requested = max(2, int(round(float(sample_rate) * 0.0015)))
    if total_samples < 16:
        return min(requested, max(0, total_samples // 2))
    return min(requested, max(0, total_samples // 8))


def _shape_event_slice_for_click_suppression(
    source: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray:
    total_samples = int(source.shape[0]) if source.ndim > 0 else 0
    fade_samples = _event_slice_fade_samples(sample_rate, total_samples)
    if fade_samples < 2:
        return np.asarray(source, dtype=np.float32)
    shaped = np.asarray(source, dtype=np.float32).copy()
    ramp_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    if shaped.ndim == 1:
        shaped[:fade_samples] *= ramp_in
        shaped[-fade_samples:] *= ramp_out
    else:
        shaped[:fade_samples, :] *= ramp_in[:, None]
        shaped[-fade_samples:, :] *= ramp_out[:, None]
    return shaped


def _uses_timeline_aligned_source(
    source: np.ndarray,
    *,
    sample_rate: int,
    events: list[object],
) -> bool:
    if sample_rate <= 0 or source.size == 0:
        return False
    source_frames = int(source.shape[0])
    event_starts = [
        max(0, int(round(_event_start_seconds(event) * sample_rate))) for event in events
    ]
    if not event_starts or max(event_starts) <= 0:
        return False
    source_seconds = source_frames / float(sample_rate)
    latest_event_seconds = max(event_starts) / float(sample_rate)
    return source_seconds >= latest_event_seconds + 2.0


def _event_slice_length_samples(event: object, *, sample_rate: int) -> int:
    duration_seconds = max(0.0, _event_end_seconds(event) - _event_start_seconds(event))
    if duration_seconds <= 0.0:
        duration_seconds = 0.75
    return max(1, int(round(duration_seconds * sample_rate)))


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
    soloed: bool = False
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
        track = AudioTrack(
            layer_id=engine_track_id,
            name=self.name,
            buffer=self.buffer,
            sample_rate=self.sample_rate,
            volume=_db_to_linear(self.gain_db),
            engine_sample_rate=engine_sample_rate,
            output_bus=self.output_bus,
        )
        track.muted = bool(self.muted)
        track.solo = bool(self.soloed)
        return track


@dataclass(slots=True, frozen=True)
class PlaybackTrackPlan:
    """One selected playback plan ready to sync into the engine."""

    tracks: tuple[PlaybackTrack, ...]
    signature: tuple[tuple[str, str], ...]
    cache_keys: frozenset[str]
    uses_track_routing: bool


@dataclass(slots=True, frozen=True)
class PlaybackMixPlan:
    """One mix-only playback projection resolved without audio decoding."""

    tracks: tuple[PlaybackTrack, ...]
    signature: tuple[tuple[str, str], ...]
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

    def clear_cache(self) -> None:
        """Drop every decoded source buffer owned by this builder."""

        self._buffer_cache.clear()

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
            raise ValueError(
                f"Playback track '{playback_track.track_id}' has no source audio ref."
            )
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
            cache_key for playback_track in tracks for cache_key in playback_track.cache_keys
        )
        signature = tuple(
            (playback_track.track_id, playback_track.signature_token) for playback_track in tracks
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
            (playback_track.track_id, playback_track.signature_token) for playback_track in tracks
        )

    def build_mix_plan(self, presentation: TimelinePresentation) -> PlaybackMixPlan:
        """Build the selected playback tracks without decoding audio buffers."""

        tracks, uses_track_routing = self._selected_tracks(
            presentation,
            resolve_audio=False,
        )
        signature = tuple(
            (playback_track.track_id, playback_track.signature_token) for playback_track in tracks
        )
        return PlaybackMixPlan(
            tracks=tracks,
            signature=signature,
            uses_track_routing=uses_track_routing,
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
            playback_output_channels=max(1, int(presentation.playback_output_channels)),
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
        playback_output_channels: int,
    ) -> list[PlaybackTrack]:
        layer_candidates = [
            layer for layer in presentation.layers if self._layer_has_playable_source(layer)
        ]
        if not layer_candidates:
            return []
        has_soloed_layers = any(
            bool(getattr(layer, "soloed", False)) for layer in layer_candidates
        )
        tracks: list[PlaybackTrack] = []
        seen_track_ids: set[str] = set()
        for layer in layer_candidates:
            playback_track = self._track_from_layer(
                layer,
                resolve_audio=resolve_audio,
                playback_output_channels=playback_output_channels,
            )
            if playback_track is None or playback_track.track_id in seen_track_ids:
                continue
            layer_soloed = bool(getattr(layer, "soloed", False))
            layer_muted = bool(getattr(layer, "muted", False)) and not layer_soloed
            playback_track.muted = layer_muted or (has_soloed_layers and not layer_soloed)
            playback_track.soloed = layer_soloed
            tracks.append(playback_track)
            seen_track_ids.add(playback_track.track_id)
        return tracks

    def _track_from_layer(
        self,
        layer: object,
        *,
        resolve_audio: bool,
        playback_output_channels: int,
    ) -> PlaybackTrack | None:
        source_audio_path = self._audio_source_ref(layer)
        if source_audio_path and self._is_continuous_audio_layer(layer):
            return PlaybackTrack(
                track_id=str(getattr(layer, "layer_id")),
                source_layer_id=getattr(layer, "layer_id"),
                source_take_id=None,
                name=str(getattr(layer, "title")),
                gain_db=float(getattr(layer, "gain_db", 0.0)),
                output_bus=sanitize_output_bus_for_channels(
                    getattr(layer, "output_bus", None),
                    playback_output_channels=playback_output_channels,
                ),
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
            output_bus=sanitize_output_bus_for_channels(
                getattr(layer, "output_bus", None),
                playback_output_channels=playback_output_channels,
            ),
            muted=bool(getattr(layer, "muted", False)),
            playback_source_ref=self._event_source_ref(layer),
            events=list(getattr(layer, "events")),
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
        rendered_source_key = f"event:{playback_source_ref}:{event_slice_signature(events)}"
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
        has_continuous_source = bool(
            PlaybackTrackBuilder._audio_source_ref(layer)
            and PlaybackTrackBuilder._is_continuous_audio_layer(layer)
        )
        return bool(has_continuous_source or PlaybackTrackBuilder._is_event_track_source(layer))

    @staticmethod
    def _is_event_like_layer(layer: object) -> bool:
        return is_event_like_layer_kind(getattr(layer, "kind", None))

    @staticmethod
    def _is_continuous_audio_layer(layer: object) -> bool:
        kind = getattr(layer, "kind", None)
        if isinstance(kind, LayerKind):
            return kind is LayerKind.AUDIO
        return str(kind or "").strip().lower() == LayerKind.AUDIO.value

    @staticmethod
    def _is_event_track_source(layer: object) -> bool:
        return bool(
            is_event_like_layer_kind(getattr(layer, "kind", None))
            and getattr(layer, "playback_enabled", False)
            and getattr(layer, "playback_mode", None) == PlaybackMode.EVENT_SLICE
            and PlaybackTrackBuilder._event_source_ref(layer)
        )

    @staticmethod
    def _audio_source_ref(item: object) -> str | None:
        source_audio_path = getattr(item, "source_audio_path", None)
        if source_audio_path:
            return str(source_audio_path)
        source_content_ref = getattr(item, "source_content_ref", None)
        locator = getattr(source_content_ref, "locator", None)
        if locator:
            return str(locator)
        return None

    @staticmethod
    def _event_source_ref(item: object, *, fallback_layer: object | None = None) -> str:
        source_content_ref = getattr(item, "source_content_ref", None)
        locator = getattr(source_content_ref, "locator", None)
        if locator:
            return str(locator)
        playback_source_ref = getattr(item, "playback_source_ref", None)
        if playback_source_ref:
            return str(playback_source_ref)
        source_audio_path = getattr(item, "source_audio_path", None)
        if source_audio_path:
            return str(source_audio_path)
        if fallback_layer is not None:
            return PlaybackTrackBuilder._event_source_ref(fallback_layer)
        return ""

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
        if _uses_timeline_aligned_source(
            event_buffer,
            sample_rate=sample_rate,
            events=active_events,
        ):
            return PlaybackTrackBuilder._render_timeline_aligned_event_buffer(
                event_buffer,
                sample_rate,
                events=active_events,
            )
        start_samples = [
            max(0, int(round(_event_start_seconds(event) * sample_rate)))
            for event in active_events
        ]
        shaped_event_buffer = _shape_event_slice_for_click_suppression(
            event_buffer,
            sample_rate=sample_rate,
        )
        total_samples = max(start_samples) + int(shaped_event_buffer.shape[0])
        if event_buffer.ndim == 1:
            rendered = np.zeros(total_samples, dtype=np.float32)
        else:
            rendered = np.zeros((total_samples, event_buffer.shape[1]), dtype=np.float32)
        for start_sample in start_samples:
            end_sample = start_sample + int(shaped_event_buffer.shape[0])
            rendered[start_sample:end_sample] += shaped_event_buffer
        peak = float(np.max(np.abs(rendered))) if rendered.size > 0 else 0.0
        if peak > 1.0:
            rendered *= np.float32(0.98 / peak)
        return rendered

    @staticmethod
    def _render_timeline_aligned_event_buffer(
        event_buffer: np.ndarray,
        sample_rate: int,
        *,
        events: list[object],
    ) -> np.ndarray:
        event_spans: list[tuple[int, np.ndarray]] = []
        source_frames = int(event_buffer.shape[0])
        for event in events:
            start_sample = max(0, int(round(_event_start_seconds(event) * sample_rate)))
            if start_sample >= source_frames:
                continue
            requested_length = _event_slice_length_samples(event, sample_rate=sample_rate)
            end_sample = min(source_frames, start_sample + requested_length)
            if end_sample <= start_sample:
                continue
            event_slice = _shape_event_slice_for_click_suppression(
                event_buffer[start_sample:end_sample],
                sample_rate=sample_rate,
            )
            event_spans.append((start_sample, event_slice))
        if not event_spans:
            return np.zeros(0, dtype=np.float32)

        total_samples = max(start + int(event_slice.shape[0]) for start, event_slice in event_spans)
        if event_buffer.ndim == 1:
            rendered = np.zeros(total_samples, dtype=np.float32)
        else:
            rendered = np.zeros((total_samples, event_buffer.shape[1]), dtype=np.float32)
        for start_sample, event_slice in event_spans:
            end_sample = start_sample + int(event_slice.shape[0])
            rendered[start_sample:end_sample] += event_slice
        peak = float(np.max(np.abs(rendered))) if rendered.size > 0 else 0.0
        if peak > 1.0:
            rendered *= np.float32(0.98 / peak)
        return rendered


__all__ = [
    "PlaybackMixPlan",
    "PlaybackTrack",
    "PlaybackTrackBuilder",
    "PlaybackTrackPlan",
]
