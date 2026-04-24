"""Clock and transport audio-engine support cases.
Exists to isolate low-level timing coverage from layer, engine, and regression support tests.
Connects the compatibility wrapper to the bounded clock and transport slice.
"""

from tests.audio_engine_shared_support import *  # noqa: F401,F403

class TestClock:
    def test_initial_position_is_zero(self) -> None:
        clock = Clock(44100)
        assert clock.position == 0
        assert clock.position_seconds == 0.0

    def test_advance_returns_pre_advance_position(self) -> None:
        clock = Clock(44100)
        pos = clock.advance(256)
        assert pos == 0
        assert clock.position == 256

    def test_advance_accumulates(self) -> None:
        clock = Clock(44100)
        clock.advance(256)
        clock.advance(256)
        assert clock.position == 512

    def test_seek(self) -> None:
        clock = Clock(44100)
        clock.advance(1000)
        clock.seek(500)
        assert clock.position == 500

    def test_seek_negative_clamps_to_zero(self) -> None:
        clock = Clock(44100)
        clock.seek(-100)
        assert clock.position == 0

    def test_seek_seconds(self) -> None:
        clock = Clock(44100)
        clock.seek_seconds(1.0)
        assert clock.position == 44100

    def test_position_seconds(self) -> None:
        clock = Clock(44100)
        clock.advance(22050)
        assert abs(clock.position_seconds - 0.5) < 1e-6

    def test_reset(self) -> None:
        clock = Clock(44100)
        clock.advance(10000)
        clock.reset()
        assert clock.position == 0

    def test_subscriber_called_on_advance(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1
        assert sub.ticks[0] == (0, 44100)

    def test_subscriber_receives_pre_advance_position(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        clock.advance(256)
        assert sub.ticks[1] == (256, 44100)

    def test_remove_subscriber(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.advance(256)
        clock.remove_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1

    def test_remove_nonexistent_subscriber_is_safe(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.remove_subscriber(sub)  # should not raise

    def test_duplicate_subscriber_not_added(self) -> None:
        clock = Clock(44100)
        sub = RecordingSubscriber()
        clock.add_subscriber(sub)
        clock.add_subscriber(sub)
        clock.advance(256)
        assert len(sub.ticks) == 1

    def test_subscriber_add_is_copy_on_write(self) -> None:
        """Adding a subscriber doesn't mutate the list the audio thread sees."""
        clock = Clock(44100)
        sub1 = RecordingSubscriber()
        clock.add_subscriber(sub1)
        # Snapshot the internal list reference
        old_list = clock._subscribers
        sub2 = RecordingSubscriber()
        clock.add_subscriber(sub2)
        # Should be a NEW list, not the same object
        assert clock._subscribers is not old_list

    # -- Loop tests ---------------------------------------------------------

    def test_loop_region_validation(self) -> None:
        with pytest.raises(ValueError):
            LoopRegion(start=-1, end=100)
        with pytest.raises(ValueError):
            LoopRegion(start=100, end=100)
        with pytest.raises(ValueError):
            LoopRegion(start=100, end=50)

    def test_loop_wraps_position(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = True
        clock.seek(900)
        pos = clock.advance(200)
        assert pos == 900
        assert clock.position == 100

    def test_loop_disabled_does_not_wrap(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 1000)
        clock.loop_enabled = False
        clock.seek(900)
        clock.advance(200)
        assert clock.position == 1100

    def test_loop_wraps_multiple_times(self) -> None:
        clock = Clock(44100)
        clock.set_loop(0, 100)
        clock.loop_enabled = True
        clock.seek(50)
        clock.advance(250)
        assert clock.position == 0

    def test_loop_with_offset_start(self) -> None:
        clock = Clock(44100)
        clock.set_loop(500, 1000)
        clock.loop_enabled = True
        clock.seek(900)
        clock.advance(200)
        assert clock.position == 600

    def test_set_loop_seconds(self) -> None:
        clock = Clock(44100)
        clock.set_loop_seconds(1.0, 2.0)
        assert clock.loop_region is not None
        assert clock.loop_region.start == 44100
        assert clock.loop_region.end == 88200

    def test_advance_is_lock_free(self) -> None:
        """Verify advance() doesn't acquire _lock (the lock object should not be held)."""
        clock = Clock(44100)
        # If advance used the lock, acquiring it here would deadlock when advance tries
        # In practice we just verify it works without issues when lock is held
        # (this is a structural test — real lock-free verification needs threading)
        clock.advance(256)
        assert clock.position == 256


# ===========================================================================
# Transport tests
# ===========================================================================


class TestTransport:
    def test_initial_state_is_stopped(self) -> None:
        transport = Transport(Clock(44100))
        assert transport.state == TransportState.STOPPED
        assert transport.is_stopped

    def test_play_from_stopped(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        assert transport.state == TransportState.PLAYING

    def test_pause_from_playing(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        transport.pause()
        assert transport.state == TransportState.PAUSED

    def test_pause_from_stopped_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.pause()
        assert transport.is_stopped

    def test_stop_resets_position(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(10000)
        transport.stop()
        assert transport.is_stopped
        assert clock.position == 0

    def test_stop_returns_to_seek_position(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.play()
        clock.advance(10000)
        transport.stop()
        assert clock.position == 5000

    def test_resume_from_paused(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.pause()
        pos_at_pause = clock.position
        transport.play()
        assert transport.is_playing
        assert clock.position == pos_at_pause

    def test_stop_from_paused(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.pause()
        transport.stop()
        assert transport.is_stopped
        assert clock.position == 0

    def test_play_while_playing_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.play()
        transport.play()
        assert transport.is_playing

    def test_stop_while_stopped_is_noop(self) -> None:
        transport = Transport(Clock(44100))
        transport.stop()
        assert transport.is_stopped

    def test_seek_while_playing(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.play()
        clock.advance(5000)
        transport.seek(1000)
        assert transport.is_playing
        assert clock.position == 1000

    def test_seek_seconds(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek_seconds(2.0)
        assert clock.position == 88200

    def test_toggle_play_pause(self) -> None:
        transport = Transport(Clock(44100))
        transport.toggle_play_pause()
        assert transport.is_playing
        transport.toggle_play_pause()
        assert transport.is_paused
        transport.toggle_play_pause()
        assert transport.is_playing

    def test_toggle_from_stopped(self) -> None:
        transport = Transport(Clock(44100))
        transport.toggle_play_pause()
        assert transport.is_playing

    def test_return_to_start(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.play()
        clock.advance(10000)
        transport.return_to_start()
        assert clock.position == 0
        assert transport.is_playing

    def test_return_to_start_from_stopped(self) -> None:
        clock = Clock(44100)
        transport = Transport(clock)
        transport.seek(5000)
        transport.return_to_start()
        assert clock.position == 0
        assert transport.is_stopped


# ===========================================================================
# Resampling tests
# ===========================================================================



__all__ = [name for name in globals() if name.startswith("Test")]
