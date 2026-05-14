"""
Unavailable playback client for degraded app launch.
Exists so native audio-process startup failures do not prevent the app shell from opening.
Connects packaging/runtime smoke to a playback-disabled but operable timeline surface.
"""

from __future__ import annotations

from echozero.application.playback.models import PlaybackState, PlaybackTimingSnapshot
from echozero.application.presentation.models import TimelinePresentation


class UnavailablePlaybackClient:
    """No-op runtime-audio client used when the playback process cannot start."""

    def __init__(self, *, reason: str) -> None:
        self.reason = str(reason)
        self._seconds = 0.0

    def health(self) -> dict[str, object]:
        """Return degraded playback health."""
        return {
            "ok": False,
            "backend": "unavailable",
            "reason": self.reason,
        }

    def shutdown(self) -> None:
        """Release playback resources."""
        return None

    def sync_presentation(self, presentation: TimelinePresentation) -> None:
        self.sync_structure_state(presentation)

    def build_for_presentation(self, presentation: TimelinePresentation) -> None:
        self.sync_structure_state(presentation)

    def sync_structure_state(self, presentation: TimelinePresentation) -> None:
        return None

    def sync_mix_state(self, presentation: TimelinePresentation) -> None:
        return None

    def apply_mix_state(self, presentation: TimelinePresentation) -> None:
        return None

    def drain_pending_structure_sync(self) -> None:
        return None

    def record_coalesced_structural_edits(self, count: int = 1) -> None:
        return None

    def play(self) -> None:
        return None

    def pause(self) -> None:
        return None

    def stop(self) -> None:
        self._seconds = 0.0

    def seek(self, position_seconds: float) -> None:
        self._seconds = max(0.0, float(position_seconds))

    def preview_clip(
        self,
        source_ref: str,
        *,
        start_seconds: float,
        end_seconds: float,
        gain_db: float = 0.0,
    ) -> bool:
        return False

    def current_time_seconds(self) -> float:
        return self._seconds

    def is_playing(self) -> bool:
        return False

    def timing_snapshot(self) -> PlaybackTimingSnapshot:
        return PlaybackTimingSnapshot(
            audible_time_seconds=self._seconds,
            clock_time_seconds=self._seconds,
            snapshot_monotonic_seconds=0.0,
            is_playing=False,
            sample_position=0,
            frame_index=0,
            timecode_label="00:00:00:00",
            display_label="00:00:00:00",
        )

    def snapshot_state(self, presentation: TimelinePresentation) -> PlaybackState:
        state = PlaybackState(backend_name="unavailable")
        state.diagnostics.last_ipc_error = self.reason
        state.diagnostics.audio_process_connected = False
        return state

    def presentation_signature(
        self, presentation: TimelinePresentation
    ) -> tuple[tuple[str, str], ...]:
        return ()

    def record_local_sync_decision(
        self,
        change_kind: str,
        *,
        projection_build_ms: float = 0.0,
        classify_ms: float = 0.0,
    ) -> None:
        return None
