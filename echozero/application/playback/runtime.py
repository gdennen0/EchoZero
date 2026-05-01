"""
Playback controller for EZ runtime audio.
Exists because the app should sync DAW-style playback tracks without exposing engine internals.
Connects timeline presentation state to one backend-agnostic runtime playback controller.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from echozero.application.playback.models import (
    PlaybackDiagnostics,
    PlaybackSource,
    PlaybackState,
    PlaybackTimingSnapshot,
)
from echozero.application.playback.tracks import (
    PlaybackMixPlan,
    PlaybackTrack,
    PlaybackTrackBuilder,
    PlaybackTrackPlan,
)
from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import PlaybackStatus
from echozero.audio.engine import AudioEngine
from echozero.audio.file_cache import load_audio_file


_SOUNDDEVICE_BACKEND = "sounddevice"


def _db_to_linear(gain_db: float) -> float:
    return float(10.0 ** (float(gain_db) / 20.0))


def _load_runtime_audio(path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = load_audio_file(path)
    return samples.astype(np.float32, copy=False), int(sample_rate)


class PlaybackController:
    """Playback facade that keeps EZ on one simple track-based runtime surface."""

    _PRIMARY_TRACK_ID = "__ez_primary_track__"
    _ROUTED_TRACK_PREFIX = "__ez_route__"
    _PREVIEW_TRACK_ID = "__ez_preview_track__"

    def __init__(
        self,
        engine: AudioEngine | None = None,
        *,
        engine_factory: Callable[[], AudioEngine] | None = None,
        preview_engine: AudioEngine | None = None,
        preview_engine_factory: Callable[[], AudioEngine] | None = None,
        audio_loader: Callable[[str | Path], tuple[np.ndarray, int]] = _load_runtime_audio,
    ) -> None:
        self._engine = engine or (
            engine_factory() if engine_factory is not None else AudioEngine()
        )
        resolved_preview_factory = preview_engine_factory or engine_factory
        self._preview_engine = preview_engine or (
            resolved_preview_factory() if resolved_preview_factory is not None else AudioEngine()
        )
        self._track_builder = PlaybackTrackBuilder(audio_loader)
        self._async_track_builder = PlaybackTrackBuilder(audio_loader)
        self._loaded_track_signature: tuple[tuple[str, str], ...] = ()
        self._loaded_uses_track_routing = False
        self._structure_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ez-structure-sync",
        )
        self._pending_structure_futures: dict[Future[PlaybackTrackPlan], int] = {}
        self._generation_terminal_outcomes: dict[int, str] = {}
        self._next_generation = 1
        self._latest_requested_generation = 0
        self._latest_ready_generation = 0
        self._owner_thread_id = threading.get_ident()
        self._shutdown_state = "running"
        self._preview_active = False
        self._last_transition = ""
        self._last_track_sync_reason = ""
        self._structural_rebuild_count = 0
        self._coalesced_edit_count = 0
        self._last_structural_rebuild_ms = 0.0
        self._max_structural_rebuild_ms = 0.0

    @property
    def engine(self) -> AudioEngine:
        return self._engine

    def sync_presentation(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `sync_presentation`."""

        self.sync_structure_state(presentation)

    def sync_structure_state(self, presentation: TimelinePresentation) -> None:
        """Sync the currently playable EZ presentation into engine tracks."""

        if self._shutdown_state != "running":
            self._last_track_sync_reason = "structure-async-shutdown-ignored"
            return
        self.drain_pending_structure_sync()
        resolved = self._with_resolved_output_channels(presentation)
        if self.is_playing():
            self._queue_async_structure_sync(resolved)
            return
        self._cancel_pending_structure_sync(reason="structure-sync-superseded")
        self._sync_track_plan(self._track_builder.build_track_plan(resolved))

    def sync_mix_state(self, presentation: TimelinePresentation) -> None:
        """Sync track gain and routing changes for the current presentation."""

        if self._shutdown_state != "running":
            return
        self.drain_pending_structure_sync()
        resolved = self._with_resolved_output_channels(presentation)
        mix_plan = self._track_builder.build_mix_plan(resolved)
        if self._apply_track_mix_state(mix_plan):
            self._last_track_sync_reason = "mix-state-pending-structure-sync"
            return
        self._last_track_sync_reason = "mix-state-applied"

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `build_for_presentation`."""

        self.sync_structure_state(presentation)

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        """Compatibility alias for callers that still say `apply_mix_state`."""

        self.sync_mix_state(presentation)

    def record_coalesced_structural_edits(self, count: int = 1) -> None:
        self._coalesced_edit_count += max(0, int(count))

    def play(self) -> None:
        self._sync_preview_state()
        if self._preview_active:
            self.stop_preview()
        self._last_transition = "play"
        self._engine.play()

    def pause(self) -> None:
        self._last_transition = "pause"
        self._engine.pause()

    def stop(self) -> None:
        self._last_transition = "stop"
        self._engine.stop()

    def seek(self, position_seconds: float) -> None:
        self._last_transition = "seek"
        self._engine.seek_seconds(position_seconds)

    def current_time_seconds(self) -> float:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        return float(self._engine.audible_time_seconds)

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        snapshot_time = getattr(self._engine, "_last_audible_time_seconds", None)
        snapshot_monotonic = getattr(self._engine, "_last_audible_monotonic_seconds", None)
        clock_time = float(self._engine.clock.position_seconds)
        is_playing = bool(self._engine.transport.is_playing)
        if snapshot_time is None:
            snapshot_time = float(self._engine.audible_time_seconds)
            snapshot_monotonic = None
        return PlaybackTimingSnapshot(
            audible_time_seconds=max(0.0, float(snapshot_time)),
            clock_time_seconds=max(0.0, clock_time),
            snapshot_monotonic_seconds=(
                max(0.0, float(snapshot_monotonic)) if snapshot_monotonic is not None else None
            ),
            is_playing=is_playing,
        )

    def is_playing(self) -> bool:
        self.drain_pending_structure_sync()
        self._sync_preview_state()
        return bool(self._engine.transport.is_playing)

    def shutdown(self) -> None:
        if self._shutdown_state == "shutdown":
            return
        self._shutdown_state = "shutting_down"
        self._cancel_pending_structure_sync(reason="shutdown")
        if self._pending_structure_futures:
            wait(list(self._pending_structure_futures.keys()), timeout=0.05)
            self._drain_completed_structure_outcomes_without_apply(outcome="cancelled")
        self._structure_executor.shutdown(wait=False, cancel_futures=True)
        self._shutdown_state = "shutdown"
        self._engine.clear_tracks()
        self._loaded_track_signature = ()
        self._loaded_uses_track_routing = False
        self._engine.shutdown()
        self.stop_preview()
        self._preview_engine.shutdown()

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        source_path = str(source_ref).strip()
        if not source_path:
            return False
        source_buffer, sample_rate = self._track_builder.load_source_buffer(
            f"preview:{source_path}",
            source_path,
        )
        if source_buffer.size == 0:
            return False
        start_sample = max(0, int(round(float(start_seconds) * sample_rate)))
        end_sample = max(start_sample, int(round(float(end_seconds) * sample_rate)))
        end_sample = min(end_sample, int(source_buffer.shape[0]))
        if end_sample <= start_sample:
            return False
        preview_buffer = np.asarray(source_buffer[start_sample:end_sample], dtype=np.float32)
        if preview_buffer.size == 0:
            return False
        self.stop_preview()
        self._last_transition = "preview_start"
        self._preview_engine.load_track(
            self._PREVIEW_TRACK_ID,
            preview_buffer,
            sample_rate,
            name="Event Preview",
            volume=_db_to_linear(gain_db),
        )
        self._preview_engine.seek_seconds(0.0)
        self._preview_engine.play()
        self._preview_active = True
        return True

    def stop_preview(self) -> None:
        self._last_transition = "preview_stop"
        self._preview_active = False
        self._preview_engine.stop()
        self._preview_engine.remove_track(self._PREVIEW_TRACK_ID)
        if self._preview_engine.is_active:
            self._preview_engine.shutdown()

    def presentation_signature(
        self,
        presentation: TimelinePresentation,
    ) -> tuple[tuple[str, str], ...]:
        resolved = self._with_resolved_output_channels(presentation)
        return self._track_builder.build_track_signature(resolved)

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        self.drain_pending_structure_sync()
        resolved = self._with_resolved_output_channels(presentation)
        active_tracks = self._track_builder.describe_selected_tracks(resolved)
        active_sources = [
            PlaybackSource(
                layer_id=playback_track.source_layer_id,
                take_id=playback_track.source_take_id,
                source_ref=playback_track.source_ref,
                mode=playback_track.mode,
            )
            for playback_track in active_tracks
        ]
        return PlaybackState(
            status=PlaybackStatus.PLAYING if self.is_playing() else PlaybackStatus.STOPPED,
            active_sources=active_sources,
            latency_ms=float(self._engine.reported_output_latency_seconds) * 1000.0,
            backend_name=self._engine.backend_name or _SOUNDDEVICE_BACKEND,
            active_layer_id=presentation.selected_layer_id,
            active_take_id=presentation.selected_take_id,
            output_sample_rate=int(self._engine.sample_rate),
            output_channels=int(self._engine.output_channels),
            diagnostics=PlaybackDiagnostics(
                glitch_count=int(self._engine.glitch_count),
                last_audio_status=self._format_audio_status(self._engine.last_audio_status),
                output_device=self._format_output_device(self._engine.output_device),
                stream_latency=self._engine.stream_latency,
                stream_blocksize=int(self._engine.stream_blocksize),
                prime_output_buffers_using_stream_callback=bool(
                    self._engine.prime_output_buffers_using_stream_callback
                ),
                last_transition=self._last_transition,
                last_track_sync_reason=self._last_track_sync_reason,
                structural_rebuild_count=self._structural_rebuild_count,
                coalesced_edit_count=self._coalesced_edit_count,
                last_structural_rebuild_ms=self._last_structural_rebuild_ms,
                max_structural_rebuild_ms=self._max_structural_rebuild_ms,
            ),
        )

    def _with_resolved_output_channels(
        self,
        presentation: TimelinePresentation,
    ) -> TimelinePresentation:
        configured_channels = int(getattr(presentation, "playback_output_channels", 0) or 0)
        if configured_channels > 0:
            return presentation
        return replace(
            presentation,
            playback_output_channels=max(1, int(self._engine.output_channels)),
        )

    def _sync_preview_state(self) -> None:
        if self._preview_active and self._preview_engine.reached_end:
            self.stop_preview()

    def _sync_track_plan(self, track_plan: PlaybackTrackPlan) -> None:
        self._track_builder.prune_cache(track_plan.cache_keys)
        resume_seconds = self._current_engine_time_seconds()
        resume_playing = bool(self._engine.transport.is_playing)
        should_reload_tracks = (
            track_plan.signature != self._loaded_track_signature
            or track_plan.uses_track_routing != self._loaded_uses_track_routing
        )
        if should_reload_tracks:
            if (
                track_plan.signature != self._loaded_track_signature
                and track_plan.uses_track_routing != self._loaded_uses_track_routing
            ):
                reason = "track-signature-and-routing-changed"
            elif track_plan.signature != self._loaded_track_signature:
                reason = "track-signature-changed"
            else:
                reason = "routing-mode-changed"
            self._replace_engine_tracks(
                track_plan,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
                reason=reason,
            )
            return
        mix_plan = PlaybackMixPlan(
            tracks=track_plan.tracks,
            signature=track_plan.signature,
            uses_track_routing=track_plan.uses_track_routing,
        )
        if self._apply_track_mix_state(mix_plan):
            self._last_track_sync_reason = "structure-sync-track-gap"
            return
        self._last_track_sync_reason = "structure-sync-mix-applied"

    def drain_pending_structure_sync(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            return
        if self._shutdown_state != "running":
            self._drain_completed_structure_outcomes_without_apply(outcome="cancelled")
            return
        if not self._pending_structure_futures:
            return
        ready: list[tuple[int, PlaybackTrackPlan]] = []
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                continue
            self._pending_structure_futures.pop(future, None)
            if future.cancelled():
                self._mark_generation_terminal_outcome(generation, "cancelled")
                continue
            try:
                plan = future.result()
            except Exception:
                if generation >= self._latest_requested_generation:
                    self._last_track_sync_reason = "structure-async-failed"
                self._mark_generation_terminal_outcome(generation, "failed")
                continue
            ready.append((generation, plan))

        if not ready:
            return

        for generation, plan in sorted(ready, key=lambda entry: entry[0]):
            if generation < self._latest_requested_generation:
                self._last_track_sync_reason = "structure-async-stale-dropped"
                self._mark_generation_terminal_outcome(generation, "stale-dropped")
                continue
            resume_seconds = self._current_engine_time_seconds()
            resume_playing = bool(self._engine.transport.is_playing)
            self._replace_engine_tracks(
                plan,
                resume_seconds=resume_seconds,
                resume_playing=resume_playing,
                reason="structure-async-applied",
                resolve_audio=False,
            )
            self._latest_ready_generation = generation
            self._mark_generation_terminal_outcome(generation, "applied")

    def _queue_async_structure_sync(self, presentation: TimelinePresentation) -> None:
        if self._shutdown_state != "running":
            self._last_track_sync_reason = "structure-async-shutdown-ignored"
            return
        generation = self._next_generation
        self._next_generation += 1
        self._latest_requested_generation = generation

        superseded_generations = [
            (future, pending_generation)
            for future, pending_generation in self._pending_structure_futures.items()
            if pending_generation < generation and not future.done()
        ]
        if superseded_generations:
            self._coalesced_edit_count += len(superseded_generations)
            for future, pending_generation in superseded_generations:
                if future.cancel():
                    self._mark_generation_terminal_outcome(pending_generation, "cancelled")
                    self._pending_structure_futures.pop(future, None)
            self._last_track_sync_reason = "structure-async-cancelled"

        future = self._structure_executor.submit(
            self._prepare_structure_track_plan_async,
            presentation,
        )
        self._pending_structure_futures[future] = generation
        self._last_track_sync_reason = "structure-async-queued"

    def _cancel_pending_structure_sync(self, *, reason: str) -> None:
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                future.cancel()
                self._mark_generation_terminal_outcome(generation, "cancelled")
        self._pending_structure_futures.clear()
        _ = reason

    def _drain_completed_structure_outcomes_without_apply(self, *, outcome: str) -> None:
        if not self._pending_structure_futures:
            return
        for future, generation in list(self._pending_structure_futures.items()):
            if not future.done():
                continue
            self._pending_structure_futures.pop(future, None)
            self._mark_generation_terminal_outcome(generation, outcome)

    def _mark_generation_terminal_outcome(self, generation: int, outcome: str) -> None:
        if generation <= 0:
            return
        if generation in self._generation_terminal_outcomes:
            return
        self._generation_terminal_outcomes[generation] = outcome

    def _prepare_structure_track_plan_async(
        self,
        presentation: TimelinePresentation,
    ) -> PlaybackTrackPlan:
        track_plan = self._async_track_builder.build_track_plan(presentation)
        return track_plan

    def _current_engine_time_seconds(self) -> float:
        self._sync_preview_state()
        return float(self._engine.audible_time_seconds)

    def _replace_engine_tracks(
        self,
        track_plan: PlaybackTrackPlan,
        *,
        resume_seconds: float,
        resume_playing: bool,
        reason: str,
        resolve_audio: bool = True,
    ) -> None:
        started = time.perf_counter()
        engine_tracks = []
        for playback_track in track_plan.tracks:
            if resolve_audio:
                self._track_builder.resolve_audio(playback_track)
            engine_track = playback_track.to_audio_track(
                engine_track_id=self._engine_track_id(track_plan, playback_track),
                engine_sample_rate=self._engine.sample_rate,
            )
            engine_track.muted = bool(playback_track.muted)
            engine_tracks.append(engine_track)
        self._engine.replace_tracks(engine_tracks)
        self._loaded_track_signature = track_plan.signature
        self._loaded_uses_track_routing = track_plan.uses_track_routing
        if resume_seconds > 0.0:
            self._engine.seek_seconds(resume_seconds)
        if resume_playing and engine_tracks:
            self._engine.play()
        elapsed_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        self._structural_rebuild_count += 1
        self._last_structural_rebuild_ms = elapsed_ms
        if elapsed_ms > self._max_structural_rebuild_ms:
            self._max_structural_rebuild_ms = elapsed_ms
        self._last_track_sync_reason = reason

    def _apply_track_mix_state(self, mix_plan: PlaybackMixPlan) -> bool:
        if (
            mix_plan.signature != self._loaded_track_signature
            or mix_plan.uses_track_routing != self._loaded_uses_track_routing
        ):
            return True
        for playback_track in mix_plan.tracks:
            engine_track = self._engine.get_track(self._engine_track_id(mix_plan, playback_track))
            if engine_track is None:
                return True
            engine_track.muted = bool(playback_track.muted)
            engine_track.volume = _db_to_linear(playback_track.gain_db)
            engine_track.output_bus = playback_track.output_bus
        return False

    def _engine_track_id(
        self,
        track_plan: PlaybackTrackPlan | PlaybackMixPlan,
        playback_track: PlaybackTrack,
    ) -> str:
        if track_plan.uses_track_routing:
            return f"{self._ROUTED_TRACK_PREFIX}{playback_track.track_id}"
        return self._PRIMARY_TRACK_ID

    @staticmethod
    def _format_audio_status(status: object) -> str | None:
        if status is None:
            return None
        value = str(status).strip()
        return value or None

    @staticmethod
    def _format_output_device(output_device: object) -> str | None:
        if output_device is None:
            return "default"
        value = str(output_device).strip()
        return value or "default"


PresentationPlaybackRuntime = PlaybackController

__all__ = [
    "PlaybackController",
    "PresentationPlaybackRuntime",
]
