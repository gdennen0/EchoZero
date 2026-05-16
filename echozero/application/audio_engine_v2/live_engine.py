"""
Dev-gated live adapter for audio engine v2.
Exists to drive the current runtime-audio controller through the v2 RT graph path.
Connects v1-compatible playback control to PreparedGraph -> RtGraph callback rendering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from math import log10
import time
from typing import Any, Callable

import numpy as np

from echozero.application.audio_engine_v2.mapping import (
    build_prepared_graph_from_playback_plan,
)
from echozero.application.audio_engine_v2.graph import MixParameters
from echozero.application.audio_engine_v2.offline_render import (
    OfflineRenderMemory,
    OfflineRenderState,
    OfflineSourceBank,
    TransitionPolicy,
    render_offline_block,
)
from echozero.application.audio_engine_v2.rt_commands import (
    RtCommand,
    RtCommandBatch,
    RtRuntimeState,
)
from echozero.application.audio_engine_v2.rt_graph import prepare_rt_graph
from echozero.application.audio_engine_v2.transport import (
    TransportCommand,
    TransportPlayState,
    TransportState,
)
from echozero.audio.clock import Clock, ClockSubscriber
from echozero.audio.layer import AudioTrack, resample_buffer
from echozero.audio.output_backend import (
    DEFAULT_BUFFER_SIZE,
    AudioOutputBackend,
    AudioOutputConfig,
)
from echozero.audio.sounddevice_backend import SounddeviceBackend
from echozero.audio.transport import Transport
from echozero.output_routing import DEFAULT_STEREO_OUTPUT_BUS

_RUNTIME_EVENT_LIMIT = 32
_DECLICK_RAMP_SECONDS = 0.004


class _V2MixerFacade:
    """Small compatibility facade for controller/tests that inspect engine.mixer."""

    def __init__(self, engine: V2LiveAudioEngine) -> None:
        self._engine = engine
        self.master_output_bus: object = None

    @property
    def tracks(self) -> tuple[AudioTrack, ...]:
        return self._engine.tracks

    @property
    def duration_samples(self) -> int:
        return max((track.end_sample for track in self._engine.tracks), default=0)

    @property
    def ramp_samples_remaining(self) -> int:
        return 0

    def get_track(self, track_id: str) -> AudioTrack | None:
        return self._engine.get_track(track_id)

    def get_layer(self, layer_id: str) -> AudioTrack | None:
        return self.get_track(layer_id)


class V2LiveAudioEngine:
    """AudioEngine-compatible v2 live playback adapter behind a dev switch."""

    def __init__(
        self,
        sample_rate: int | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        channels: int | None = None,
        stream_factory: Callable[..., Any] | None = None,
        stream_blocksize: int | None = None,
        stream_latency: str | float | None = None,
        prime_output_buffers_using_stream_callback: bool = True,
        output_device: int | str | None = None,
        master_output_bus: object = None,
        backend: AudioOutputBackend | None = None,
    ) -> None:
        self._backend = backend or SounddeviceBackend(stream_factory=stream_factory)
        self._output_config = self._backend.resolve_output_config(
            sample_rate=sample_rate,
            channels=channels,
            buffer_size=buffer_size,
            output_device=output_device,
            stream_blocksize=stream_blocksize,
            stream_latency=stream_latency,
            prime_output_buffers_using_stream_callback=(
                prime_output_buffers_using_stream_callback
            ),
        )
        self._clock = Clock(sample_rate=self._output_config.sample_rate)
        self._transport = Transport(self._clock)
        self._mixer = _V2MixerFacade(self)
        self._mixer.master_output_bus = master_output_bus
        self._stream: Any = None
        self._buffer_size = int(buffer_size)
        self._channels = int(self._output_config.channels)
        self._stream_blocksize = int(self._output_config.blocksize)
        self._stream_latency = self._output_config.latency
        self._prime_output_buffers_using_stream_callback = (
            self._output_config.prime_output_buffers_using_stream_callback
        )
        self._output_device = self._output_config.output_device
        self._active = False
        self._tracks: tuple[AudioTrack, ...] = ()
        self._prepared_graph = self._build_prepared_graph(())
        self._source_bank = OfflineSourceBank({})
        self._memory = self._create_memory(block_frames=max(1, self._buffer_size))
        self._render_state = OfflineRenderState(
            runtime=RtRuntimeState(
                graph=prepare_rt_graph(self._prepared_graph),
                transport=TransportState(),
            )
        )
        self._policy = TransitionPolicy(
            ramp_frames=max(2, int(round(self.sample_rate * _DECLICK_RAMP_SECONDS)))
        )
        self._pending_commands: list[RtCommand] = []
        self._command_sequence = 0
        self._end_of_content = False
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds: float | None = None
        self._last_audible_monotonic_seconds: float | None = None
        self._glitch_count = 0
        self._last_status: Any = None
        self._last_discontinuity_reason: str | None = "v2-engine-startup"
        self._last_ramp_reason: str | None = None
        self._overlay_buffer: np.ndarray | None = None
        self._overlay_playback_buffer: np.ndarray | None = None
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(1.0)
        self._runtime_event_sequence = 0
        self._recent_runtime_events: deque[dict[str, object]] = deque(
            maxlen=_RUNTIME_EVENT_LIMIT
        )
        self._diagnostics_capture_active = False
        self._diagnostics_capture_include_audio = False
        self._diagnostics_capture_blocks: deque[dict[str, object]] = deque(maxlen=0)
        self._diagnostics_capture_max_blocks = 0
        self._diagnostics_capture_sequence = 0

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def transport(self) -> Transport:
        return self._transport

    @property
    def mixer(self) -> _V2MixerFacade:
        return self._mixer

    @property
    def tracks(self) -> tuple[AudioTrack, ...]:
        return self._tracks

    @property
    def layers(self) -> tuple[AudioTrack, ...]:
        return self.tracks

    @property
    def sample_rate(self) -> int:
        return int(self._clock.sample_rate)

    @property
    def output_channels(self) -> int:
        return self._channels

    @property
    def master_output_bus(self) -> str | None:
        value = self._mixer.master_output_bus
        return None if value is None else str(value)

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def reported_output_latency_seconds(self) -> float:
        return self._reported_output_latency_seconds

    @property
    def audible_time_seconds(self) -> float:
        clock_time = float(self._clock.position_seconds)
        if not self._transport.is_playing:
            return clock_time
        snapshot = self._last_audible_time_seconds
        snapshot_monotonic = self._last_audible_monotonic_seconds
        if snapshot is None or snapshot_monotonic is None:
            return max(0.0, clock_time - self._reported_output_latency_seconds)
        extrapolated = snapshot + max(0.0, time.monotonic() - snapshot_monotonic)
        return max(0.0, min(extrapolated, clock_time))

    @property
    def reached_end(self) -> bool:
        return self._end_of_content

    @property
    def glitch_count(self) -> int:
        return self._glitch_count

    @property
    def last_audio_status(self) -> Any:
        return self._last_status

    @property
    def backend_name(self) -> str:
        return "audio_engine_v2"

    @property
    def output_device(self) -> int | str | None:
        return self._output_device

    @property
    def output_config(self) -> AudioOutputConfig:
        return self._output_config

    @property
    def resolved_output_device(self) -> int | str | None:
        return self._output_config.resolved_output_device

    @property
    def resolved_output_device_name(self) -> str | None:
        return self._output_config.resolved_output_device_name

    @property
    def stream_latency(self) -> str | float | None:
        return self._stream_latency

    @property
    def stream_blocksize(self) -> int:
        return self._stream_blocksize

    @property
    def prime_output_buffers_using_stream_callback(self) -> bool:
        return bool(self._prime_output_buffers_using_stream_callback)

    @property
    def ramp_samples_remaining(self) -> int:
        return int(self._policy.ramp_frames)

    @property
    def last_discontinuity_reason(self) -> str | None:
        return self._last_discontinuity_reason

    @property
    def last_ramp_reason(self) -> str | None:
        return self._last_ramp_reason

    @property
    def recent_runtime_events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._recent_runtime_events)

    @property
    def overlay_active(self) -> bool:
        return self._overlay_playback_buffer is not None

    @property
    def rt_graph_identity_full_hash(self) -> str:
        return str(self._render_state.runtime.graph.identity_full_hash)

    def set_master_output_bus(self, output_bus: object) -> None:
        self._mixer.master_output_bus = output_bus
        self._commit_current_tracks(reason="master-output-bus-changed")

    def create_track(
        self,
        layer_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        return AudioTrack(
            layer_id=layer_id,
            name=name or layer_id,
            buffer=buffer,
            sample_rate=sample_rate,
            offset=offset,
            volume=volume,
            engine_sample_rate=self.sample_rate,
            output_bus=output_bus,
        )

    def set_track(self, track: AudioTrack) -> AudioTrack:
        self.replace_tracks([item for item in self._tracks if item.id != track.id] + [track])
        return track

    def load_track(
        self,
        track_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        *,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        track = self.create_track(
            track_id,
            buffer,
            sample_rate,
            name=name,
            offset=offset,
            volume=volume,
            output_bus=output_bus,
        )
        return self.set_track(track)

    def replace_tracks(self, tracks: list[AudioTrack]) -> None:
        self._tracks = tuple(tracks)
        self._commit_current_tracks(reason="tracks-replaced")

    def apply_track_mix_updates(
        self,
        updates: dict[str, tuple[bool, float, str | None]],
    ) -> bool:
        changed = False
        next_tracks: list[AudioTrack] = []
        for track in self._tracks:
            desired = updates.get(str(track.id))
            if desired is None:
                next_tracks.append(track)
                continue
            muted, volume, output_bus = desired
            if (
                bool(track.muted) == bool(muted)
                and abs(float(track.volume) - float(volume)) <= 1e-6
                and track.output_bus == output_bus
            ):
                next_tracks.append(track)
                continue
            cloned = AudioTrack(
                layer_id=str(track.id),
                name=str(track.name),
                buffer=track.buffer,
                sample_rate=int(track.sample_rate),
                offset=int(track.offset),
                volume=float(volume),
                engine_sample_rate=self.sample_rate,
                output_bus=output_bus,
            )
            cloned.muted = bool(muted)
            cloned.solo = bool(track.solo)
            next_tracks.append(cloned)
            changed = True
        if changed:
            route_changed = any(
                str(track.id) in updates and track.output_bus != updates[str(track.id)][2]
                for track in self._tracks
            )
            self._tracks = tuple(next_tracks)
            if route_changed:
                self._commit_current_tracks(reason="route-update")
            else:
                self._queue_track_mix_commands(tuple(next_tracks))
                self._last_discontinuity_reason = "mix-update"
                self._last_ramp_reason = "mix-update"
        return bool(changed)

    def clear_tracks(self) -> None:
        self._tracks = ()
        self._commit_current_tracks(reason="tracks-cleared")

    def add_layer(
        self,
        layer_id: str,
        buffer: np.ndarray,
        sample_rate: int,
        name: str | None = None,
        offset: int = 0,
        volume: float = 1.0,
        output_bus: str | None = None,
    ) -> AudioTrack:
        return self.load_track(
            layer_id,
            buffer,
            sample_rate,
            name=name,
            offset=offset,
            volume=volume,
            output_bus=output_bus,
        )

    def remove_track(self, track_id: str) -> AudioTrack | None:
        removed = next((track for track in self._tracks if track.id == track_id), None)
        if removed is not None:
            self._tracks = tuple(track for track in self._tracks if track.id != track_id)
            self._commit_current_tracks(reason="track-removed")
        return removed

    def remove_layer(self, layer_id: str) -> AudioTrack | None:
        return self.remove_track(layer_id)

    def get_track(self, track_id: str) -> AudioTrack | None:
        return next((track for track in self._tracks if str(track.id) == str(track_id)), None)

    def get_layer(self, layer_id: str) -> AudioTrack | None:
        return self.get_track(layer_id)

    def play(self) -> None:
        self._end_of_content = False
        if not self._active:
            self._open_stream()
        self._transport.play()
        self._queue_transport_command(TransportCommand.play(self._next_sequence()))
        self._last_discontinuity_reason = "play"

    def pause(self) -> None:
        self._transport.pause()
        self._last_audible_monotonic_seconds = None
        self._queue_transport_command(TransportCommand.pause(self._next_sequence()))
        self._last_discontinuity_reason = "pause"

    def stop(self) -> None:
        self._end_of_content = False
        self._transport.stop()
        self._render_state = replace(self._render_state, frame_position=0)
        self._last_audible_time_seconds = 0.0
        self._last_audible_monotonic_seconds = None
        self._queue_transport_command(TransportCommand.stop(self._next_sequence()))
        self._last_discontinuity_reason = "stop"

    def seek(self, position_samples: int) -> None:
        self._end_of_content = False
        self._transport.seek(position_samples)
        seconds = float(self._clock.position_seconds)
        self._render_state = replace(self._render_state, frame_position=max(0, position_samples))
        self._last_audible_time_seconds = seconds
        self._last_audible_monotonic_seconds = None
        self._queue_transport_command(
            TransportCommand.seek(self._next_sequence(), position_seconds=seconds)
        )
        self._last_discontinuity_reason = "seek"

    def seek_seconds(self, seconds: float) -> None:
        self.seek(int(seconds * self.sample_rate))

    def toggle_play_pause(self) -> None:
        if self._transport.is_playing:
            self.pause()
        else:
            self.play()

    def shutdown(self) -> None:
        self._transport.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._active = False
        self._reported_output_latency_seconds = 0.0
        self._last_audible_time_seconds = None
        self._last_audible_monotonic_seconds = None
        self._clear_overlay()
        self._last_discontinuity_reason = "shutdown"

    def request_declick(self) -> None:
        self._last_discontinuity_reason = "manual"

    def play_overlay(
        self,
        buffer: np.ndarray,
        sample_rate: int,
        *,
        volume: float = 1.0,
    ) -> bool:
        if buffer.size == 0 or sample_rate <= 0:
            self.stop_overlay()
            return False
        replacing_overlay = self._overlay_playback_buffer is not None
        source = np.array(buffer, dtype=np.float32, copy=True)
        playback = self._prepare_overlay_buffer(source, sample_rate=sample_rate)
        if playback.size == 0:
            self.stop_overlay()
            return False
        self._overlay_buffer = source
        self._overlay_playback_buffer = playback
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(max(0.0, float(volume)))
        self._record_runtime_event(
            "overlay-replace" if replacing_overlay else "overlay-start",
            reason="play-overlay",
            source_frames=int(source.shape[0]),
            playback_frames=int(playback.shape[0]),
            sample_rate=int(sample_rate),
            volume=float(self._overlay_volume),
        )
        self._last_ramp_reason = "overlay-start"
        if not self._active:
            self._open_stream()
        return True

    def stop_overlay(self) -> None:
        stopping_overlay = self._overlay_playback_buffer is not None
        self._clear_overlay()
        if stopping_overlay:
            self._record_runtime_event("overlay-stop", reason="stop-overlay")
        self._last_ramp_reason = "overlay-stop"

    def start_diagnostics_capture(
        self,
        *,
        include_audio_buffers: bool = True,
        max_audio_blocks: int = 64,
    ) -> dict[str, object]:
        max_blocks = max(0, min(256, int(max_audio_blocks)))
        self._diagnostics_capture_active = True
        self._diagnostics_capture_include_audio = bool(include_audio_buffers and max_blocks > 0)
        self._diagnostics_capture_max_blocks = max_blocks
        self._diagnostics_capture_blocks = deque(maxlen=max_blocks)
        self._diagnostics_capture_sequence = 0
        return self.diagnostics_capture_status()

    def stop_diagnostics_capture(self) -> dict[str, object]:
        blocks = tuple(dict(block) for block in self._diagnostics_capture_blocks)
        self._diagnostics_capture_active = False
        self._diagnostics_capture_include_audio = False
        self._diagnostics_capture_blocks = deque(maxlen=0)
        self._diagnostics_capture_max_blocks = 0
        return {
            "active": False,
            "audio_blocks": blocks,
            "audio_block_count": len(blocks),
        }

    def diagnostics_capture_status(self) -> dict[str, object]:
        return {
            "active": bool(self._diagnostics_capture_active),
            "include_audio_buffers": bool(self._diagnostics_capture_include_audio),
            "audio_block_count": int(len(self._diagnostics_capture_blocks)),
            "max_audio_blocks": int(self._diagnostics_capture_max_blocks),
        }

    def add_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.add_subscriber(sub)

    def remove_clock_subscriber(self, sub: ClockSubscriber) -> None:
        self._clock.remove_subscriber(sub)

    def _open_stream(self) -> None:
        if self._active:
            return
        self._stream = self._backend.open_output_stream(
            self._audio_callback,
            self._output_config,
        )
        self._stream.start()
        self._reported_output_latency_seconds = self._coerce_output_latency_seconds(
            getattr(self._stream, "latency", 0.0)
        )
        self._active = True

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        if status:
            self._glitch_count += 1
            self._last_status = status
        if frames <= 0:
            return
        if frames != self._memory.hardware_buffer.shape[0]:
            self._memory = self._create_memory(block_frames=frames)

        commands = self._drain_pending_command_batch()
        result = render_offline_block(
            self._render_state,
            sources=self._source_bank,
            memory=self._memory,
            policy=self._policy,
            commands=commands,
        )
        self._render_state = result.state
        self._clock.seek(int(self._render_state.frame_position))
        self._update_callback_timing_snapshot(time_info)

        block = result.block
        self._mix_overlay_into(block, frames)
        self._sanitize_output_samples(block)
        self._capture_output_callback_block(block, frames=frames)
        if self._channels == 1:
            outdata[:, 0] = block[:, 0]
        else:
            outdata[:, :] = block[:, : self._channels]
        if result.applied_sequences:
            self._record_runtime_event(
                "v2-command-batch-applied",
                applied_count=len(result.applied_sequences),
            )

    def _commit_current_tracks(self, *, reason: str) -> None:
        self._prepared_graph = self._build_prepared_graph(self._tracks)
        self._source_bank = OfflineSourceBank(
            {
                self._source_key_for_track(track): np.asarray(track.buffer, dtype=np.float32)
                for track in self._tracks
            }
        )
        self._queue_command(RtCommand.commit_graph(self._next_sequence(), self._prepared_graph))
        self._last_discontinuity_reason = reason
        self._last_ramp_reason = reason

    def _queue_track_mix_commands(self, tracks: tuple[AudioTrack, ...]) -> None:
        for track in tracks:
            self._queue_command(
                RtCommand.track_mix(
                    self._next_sequence(),
                    track_id=str(track.id),
                    mix=MixParameters(
                        gain_db=self._linear_to_db(float(track.volume)),
                        muted=bool(track.muted),
                        soloed=bool(track.solo),
                    ),
                )
            )

    def _build_prepared_graph(self, tracks: tuple[AudioTrack, ...]) -> Any:
        plan_tracks = tuple(
            _PlaybackTrackView(track, self._source_key_for_track(track)) for track in tracks
        )
        plan = _PlaybackPlanView(plan_tracks)
        return build_prepared_graph_from_playback_plan(
            plan,
            graph_id="v2-live",
            master_output_bus=(
                self._mixer.master_output_bus
                if self._mixer.master_output_bus is not None
                else DEFAULT_STEREO_OUTPUT_BUS
            ),
        )

    def _create_memory(self, *, block_frames: int) -> OfflineRenderMemory:
        return OfflineRenderMemory.create(
            bus_count=max(1, len(self._prepared_graph.buses)),
            block_frames=max(1, int(block_frames)),
            max_bus_channels=max(1, min(2, self._channels)),
            hardware_channels=max(1, self._channels),
        )

    def _queue_transport_command(self, command: TransportCommand) -> None:
        self._queue_command(RtCommand.transport(command))

    def _queue_command(self, command: RtCommand) -> None:
        self._pending_commands.append(command)
        if not self._transport.is_playing and command.kind.value != "transport":
            self._apply_pending_commands_without_render()

    def _apply_pending_commands_without_render(self) -> None:
        batch = self._drain_pending_command_batch()
        if batch is None:
            return
        from echozero.application.audio_engine_v2.rt_commands import apply_rt_command_batch

        result = apply_rt_command_batch(self._render_state.runtime, batch)
        self._render_state = replace(self._render_state, runtime=result.state)

    def _drain_pending_command_batch(self) -> RtCommandBatch | None:
        if not self._pending_commands:
            return None
        commands = tuple(self._pending_commands)
        self._pending_commands.clear()
        return RtCommandBatch(commands)

    def _next_sequence(self) -> int:
        self._command_sequence += 1
        return self._command_sequence

    @staticmethod
    def _source_key_for_track(track: AudioTrack) -> str:
        return f"engine:{track.id}"

    @staticmethod
    def _linear_to_db(volume: float) -> float:
        if volume <= 0.0:
            return -120.0
        return 20.0 * log10(float(volume))

    @staticmethod
    def _sanitize_output_samples(block: np.ndarray) -> None:
        np.nan_to_num(block, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(block, -1.0, 1.0, out=block)

    def _capture_output_callback_block(self, block: np.ndarray, *, frames: int) -> None:
        if not self._diagnostics_capture_active or not self._diagnostics_capture_include_audio:
            return
        if frames <= 0 or self._diagnostics_capture_max_blocks <= 0:
            return
        captured = np.array(block[:frames], dtype=np.float32, copy=True)
        self._diagnostics_capture_sequence += 1
        self._diagnostics_capture_blocks.append(
            {
                "seq": int(self._diagnostics_capture_sequence),
                "kind": "v2_output_callback_mixed",
                "monotonic_seconds": float(time.monotonic()),
                "clock_samples": int(self._clock.position),
                "clock_seconds": float(self._clock.position_seconds),
                "frames": int(frames),
                "channels": int(self._channels),
                "sample_rate": int(self.sample_rate),
                "is_playing": bool(self._transport.is_playing),
                "overlay_active": bool(self.overlay_active),
                "peak_abs": float(np.max(np.abs(captured))) if captured.size else 0.0,
                "rms": float(np.sqrt(np.mean(np.square(captured)))) if captured.size else 0.0,
                "samples": captured,
            }
        )

    def _update_callback_timing_snapshot(self, time_info: Any) -> None:
        output_latency_seconds = self._reported_output_latency_seconds
        measured_latency = self._extract_output_latency_seconds(time_info)
        if measured_latency is not None:
            output_latency_seconds = measured_latency
            self._reported_output_latency_seconds = measured_latency
        self._last_audible_time_seconds = max(
            0.0,
            float(self._clock.position_seconds) - output_latency_seconds,
        )
        self._last_audible_monotonic_seconds = time.monotonic()

    def _record_runtime_event(self, kind: str, **metrics: object) -> None:
        self._runtime_event_sequence += 1
        event: dict[str, object] = {
            "seq": int(self._runtime_event_sequence),
            "source": "audio_engine_v2",
            "kind": str(kind or "runtime-event"),
            "monotonic_seconds": float(time.monotonic()),
            "clock_samples": int(self._clock.position),
            "clock_seconds": float(self._clock.position_seconds),
            "is_playing": bool(self._transport.is_playing),
        }
        for key, value in metrics.items():
            if isinstance(value, (str, bool, int, float)):
                event[str(key)] = value
        self._recent_runtime_events.append(event)

    @staticmethod
    def _coerce_output_latency_seconds(latency: Any) -> float:
        if isinstance(latency, (tuple, list)):
            latency = latency[-1] if latency else 0.0
        try:
            return max(0.0, float(latency))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _extract_output_latency_seconds(time_info: Any) -> float | None:
        current_time = _coerce_callback_time_value(time_info, "currentTime")
        output_dac_time = _coerce_callback_time_value(time_info, "outputBufferDacTime")
        if current_time is None or output_dac_time is None:
            return None
        return max(0.0, output_dac_time - current_time)

    def _prepare_overlay_buffer(self, source: np.ndarray, *, sample_rate: int) -> np.ndarray:
        prepared: np.ndarray = source
        if int(sample_rate) != int(self.sample_rate):
            prepared = resample_buffer(prepared, int(sample_rate), int(self.sample_rate))
        prepared = np.asarray(prepared, dtype=np.float32)
        if self._channels <= 1:
            if prepared.ndim == 2:
                prepared = np.asarray(
                    np.mean(prepared, axis=1, dtype=np.float32),
                    dtype=np.float32,
                )
            return _copy_with_edge_fades(prepared, self._policy.ramp_frames)
        if prepared.ndim == 1:
            prepared = np.repeat(prepared[:, None], self._channels, axis=1)
        elif prepared.shape[1] < self._channels:
            pad_channels = self._channels - int(prepared.shape[1])
            pad = np.repeat(prepared[:, -1:], pad_channels, axis=1)
            prepared = np.concatenate((prepared, pad), axis=1)
        elif prepared.shape[1] > self._channels:
            prepared = np.asarray(prepared[:, : self._channels], dtype=np.float32)
        return _copy_with_edge_fades(prepared, self._policy.ramp_frames)

    def _mix_overlay_into(self, block: np.ndarray, frames: int) -> None:
        overlay = self._overlay_playback_buffer
        if overlay is None:
            return
        start = int(self._overlay_read_index)
        if start >= int(overlay.shape[0]):
            self._clear_overlay()
            return
        chunk_frames = min(int(frames), int(overlay.shape[0]) - start)
        if chunk_frames <= 0:
            self._clear_overlay()
            return
        overlay_view = overlay[start : start + chunk_frames]
        if block.ndim == 1:
            if overlay_view.ndim == 2:
                overlay_view = overlay_view[:, 0]
            block[:chunk_frames] += overlay_view * self._overlay_volume
        else:
            if overlay_view.ndim == 1:
                block[:chunk_frames, :] += overlay_view[:, None] * self._overlay_volume
            else:
                block[:chunk_frames, : overlay_view.shape[1]] += (
                    overlay_view * self._overlay_volume
                )
        self._overlay_read_index = start + chunk_frames
        if self._overlay_read_index >= int(overlay.shape[0]):
            self._last_ramp_reason = "overlay-end"
            self._clear_overlay()

    def _clear_overlay(self) -> None:
        self._overlay_buffer = None
        self._overlay_playback_buffer = None
        self._overlay_read_index = 0
        self._overlay_volume = np.float32(1.0)


class _PlaybackTrackView:
    def __init__(self, track: AudioTrack, source_key: str) -> None:
        self.track_id = str(track.id)
        self.name = str(track.name)
        self.source_key = source_key
        self.gain_db = V2LiveAudioEngine._linear_to_db(float(track.volume))
        self.muted = bool(track.muted)
        self.soloed = bool(track.solo)
        self.output_bus = track.output_bus
        self.buffer = track.buffer
        self.sample_rate = track.sample_rate


class _PlaybackPlanView:
    def __init__(self, tracks: tuple[_PlaybackTrackView, ...]) -> None:
        self.tracks = tracks


def _coerce_callback_time_value(time_info: Any, field: str) -> float | None:
    if time_info is None:
        return None
    if isinstance(time_info, dict):
        value = time_info.get(field)
    else:
        value = getattr(time_info, field, None)
        if value is None:
            try:
                value = time_info[field]
            except Exception:
                value = None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _copy_with_edge_fades(buffer: np.ndarray, ramp_frames: int) -> np.ndarray:
    faded = np.array(buffer, dtype=np.float32, copy=True)
    if faded.size == 0 or int(faded.shape[0]) <= 1:
        return faded
    frames = int(faded.shape[0])
    fade_frames = min(frames // 2, max(0, int(ramp_frames)))
    if fade_frames <= 1:
        return faded
    fade_in = np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
    if faded.ndim == 1:
        faded[:fade_frames] *= fade_in
        faded[frames - fade_frames : frames] *= fade_out
    else:
        faded[:fade_frames] *= fade_in[:, None]
        faded[frames - fade_frames : frames] *= fade_out[:, None]
    return faded


__all__ = ["V2LiveAudioEngine"]
