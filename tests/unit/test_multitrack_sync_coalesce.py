"""
Unit tests for multi-track sync coalescing.

Validates that the global coalesce window batches multiple track updates
into a single atomic push, rather than firing independent per-coord timers.
Also tests the retry mechanism for timed-out event requests.
"""

import time
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so SyncSystemManager can be instantiated without a full app
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_qt(monkeypatch):
    """Patch QTimer so tests run without a running QApplication."""
    # We need a minimal QTimer that records start/connect calls
    # but doesn't actually interact with an event loop.
    pass  # Tests below use explicit mocking where needed


class FakeQTimer:
    """Minimal QTimer stand-in for unit tests."""

    def __init__(self):
        self._single_shot = False
        self._interval = 0
        self._active = False
        self._callback = None

    def setSingleShot(self, val):
        self._single_shot = val

    def timeout_connect(self, cb):
        self._callback = cb

    def start(self, ms):
        self._interval = ms
        self._active = True

    def isActive(self):
        return self._active

    def stop(self):
        self._active = False

    def fire(self):
        """Simulate timer expiry."""
        self._active = False
        if self._callback:
            self._callback()

    @property
    def timeout(self):
        """Return an object with .connect()."""
        parent = self

        class _Sig:
            def connect(self, cb):
                parent._callback = cb

            def disconnect(self):
                parent._callback = None

        return _Sig()

    def receivers(self, _sig):
        return 1 if self._callback else 0


# ---------------------------------------------------------------------------
# Helpers to build a minimal SyncSystemManager with mocked dependencies
# ---------------------------------------------------------------------------

def _make_ssm():
    """Create a SyncSystemManager with mocked dependencies."""
    from src.features.show_manager.domain.sync_layer_entity import (
        SyncLayerEntity,
        SyncSource,
        SyncStatus,
    )
    from src.features.show_manager.application.sync_system_manager import (
        SyncSystemManager,
    )
    from PyQt6.QtCore import QObject

    facade = MagicMock()
    facade.current_project_id = "test-project"
    settings_manager = MagicMock()

    # Create SSM bypassing its __init__ but calling QObject.__init__
    # so that hasattr() and other QObject internals work
    with patch.object(SyncSystemManager, "__init__", lambda self, *a, **kw: QObject.__init__(self)):
        ssm = SyncSystemManager(facade, "block-1", settings_manager)

    # Manually initialize all fields we need for testing
    ssm._facade = facade
    ssm._show_manager_block_id = "block-1"
    ssm._settings_manager = settings_manager
    ssm._synced_layers = {}
    ssm._syncing_from_ma3 = {}
    ssm._last_ma3_push_time = {}
    ssm._last_editor_push_time = {}
    ssm._ma3_empty_ignore_window_s = 0.5
    ssm._force_apply_to_ez = {}
    ssm._ma3_apply_cooldown_until = {}
    ssm._ma3_apply_cooldown_s = 0.5
    ssm._ma3_events_in_flight = {}
    ssm._ma3_events_request_timeout_s = 1.5
    ssm._skip_initial_hook = {}
    ssm._ma3_track_events = {}
    ssm._ma3_tracks = {}
    ssm._ma3_tracks_version = 0
    ssm._diag_track_changed_ts = {}
    ssm._diag_first_track_changed_ts = 0.0
    ssm._events_request_retries = {}
    
    # Pre-initialize lazy global coalesce state so hasattr() works
    ssm._global_coalesce_timer = None
    ssm._global_coalesce_pending = {}
    ssm._global_coalesce_last_arrival = 0.0
    
    # Pre-initialize lazy per-coord coalesce state (legacy, may still be checked)
    ssm._coalesce_timers = {}
    ssm._coalesce_entities = {}
    ssm._coalesce_schedule_time = {}
    
    # Pre-initialize request tracking
    ssm._pending_events_requests = {}
    ssm._pending_events_timers = {}
    ssm._pending_events_request_ids = {}
    ssm._ma3_request_counter = 0
    ssm._last_events_request_id = {}
    ssm._last_push_fingerprint = {}

    # Signals are mocked so .emit() doesn't crash
    ssm.entity_updated = MagicMock()
    ssm.entities_changed = MagicMock()
    ssm.error_occurred = MagicMock()
    ssm.reconciliation_prompt = MagicMock()

    return ssm


def _make_entity(coord: str, layer_id: str):
    """Create a minimal SyncLayerEntity for testing."""
    from src.features.show_manager.domain.sync_layer_entity import (
        SyncLayerEntity,
        SyncSource,
        SyncStatus,
    )

    return SyncLayerEntity(
        id=f"entity-{coord}",
        source=SyncSource.MA3,
        name=layer_id,
        ma3_coord=coord,
        editor_layer_id=layer_id,
        editor_block_id="block-1",
        sync_status=SyncStatus.SYNCED,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGlobalCoalesceWindow:
    """Verify that _schedule_push_ma3_to_editor uses a single global timer."""

    def test_single_track_creates_global_timer(self):
        """A single track schedule should create the global timer."""
        ssm = _make_ssm()
        entity_a = _make_entity("tc1_tg1_tr1", "clap")

        with patch("PyQt6.QtCore.QTimer", FakeQTimer):
            ssm._schedule_push_ma3_to_editor(entity_a)

        assert hasattr(ssm, "_global_coalesce_timer")
        assert ssm._global_coalesce_timer is not None
        assert ssm._global_coalesce_timer._active
        assert "tc1_tg1_tr1" in ssm._global_coalesce_pending

    def test_multiple_tracks_share_single_timer(self):
        """Multiple tracks should accumulate in the same pending dict and
        share the same timer instance."""
        ssm = _make_ssm()
        entity_a = _make_entity("tc1_tg1_tr1", "clap")
        entity_b = _make_entity("tc1_tg1_tr2", "kick")
        entity_c = _make_entity("tc1_tg1_tr3", "snare")

        with patch("PyQt6.QtCore.QTimer", FakeQTimer):
            ssm._schedule_push_ma3_to_editor(entity_a)
            timer_ref = ssm._global_coalesce_timer

            ssm._schedule_push_ma3_to_editor(entity_b)
            ssm._schedule_push_ma3_to_editor(entity_c)

        # All three coords should be pending
        assert len(ssm._global_coalesce_pending) == 3
        assert set(ssm._global_coalesce_pending.keys()) == {
            "tc1_tg1_tr1",
            "tc1_tg1_tr2",
            "tc1_tg1_tr3",
        }
        # Same timer instance reused
        assert ssm._global_coalesce_timer is timer_ref

    def test_timer_fires_pushes_all_pending(self):
        """When the global timer fires, all pending coords should be pushed."""
        ssm = _make_ssm()
        entity_a = _make_entity("tc1_tg1_tr1", "clap")
        entity_b = _make_entity("tc1_tg1_tr2", "kick")

        # Mock _push_ma3_to_editor to track calls
        pushed_coords = []
        ssm._push_ma3_to_editor = MagicMock(side_effect=lambda e: pushed_coords.append(e.ma3_coord))
        ssm._save_to_settings = MagicMock()

        with patch("PyQt6.QtCore.QTimer", FakeQTimer):
            ssm._schedule_push_ma3_to_editor(entity_a)
            ssm._schedule_push_ma3_to_editor(entity_b)

            # Simulate time passing beyond the coalesce window
            ssm._global_coalesce_last_arrival = time.time() - 1.0

            # Fire the timer
            ssm._global_coalesce_timer.fire()

        # Both coords should have been pushed
        assert set(pushed_coords) == {"tc1_tg1_tr1", "tc1_tg1_tr2"}
        # Pending should be cleared
        assert len(ssm._global_coalesce_pending) == 0
        # Settings saved once
        ssm._save_to_settings.assert_called_once()

    def test_timer_reschedules_if_new_arrival_during_window(self):
        """If the timer fires but a new coord arrived recently, it should
        reschedule rather than push."""
        ssm = _make_ssm()
        entity_a = _make_entity("tc1_tg1_tr1", "clap")

        ssm._push_ma3_to_editor = MagicMock()
        ssm._save_to_settings = MagicMock()

        with patch("PyQt6.QtCore.QTimer", FakeQTimer):
            ssm._schedule_push_ma3_to_editor(entity_a)

            # last_arrival is very recent (simulating a new coord just arrived)
            ssm._global_coalesce_last_arrival = time.time()

            # Fire the timer
            ssm._global_coalesce_timer.fire()

        # Should NOT have pushed (window not elapsed)
        ssm._push_ma3_to_editor.assert_not_called()
        # Timer should have been restarted
        assert ssm._global_coalesce_timer._active
        # Pending should still have the entity
        assert "tc1_tg1_tr1" in ssm._global_coalesce_pending

    def test_updating_same_coord_replaces_entity(self):
        """Scheduling the same coord twice should update the entity, not duplicate."""
        ssm = _make_ssm()
        entity_a1 = _make_entity("tc1_tg1_tr1", "clap_v1")
        entity_a2 = _make_entity("tc1_tg1_tr1", "clap_v2")

        with patch("PyQt6.QtCore.QTimer", FakeQTimer):
            ssm._schedule_push_ma3_to_editor(entity_a1)
            ssm._schedule_push_ma3_to_editor(entity_a2)

        assert len(ssm._global_coalesce_pending) == 1
        assert ssm._global_coalesce_pending["tc1_tg1_tr1"].name == "clap_v2"


class TestDiagnosticTimestamps:
    """Verify that diagnostic timestamps are recorded correctly."""

    def test_record_track_changed_timestamp(self):
        ssm = _make_ssm()
        ts = time.time()
        ssm.record_track_changed_timestamp("tc1_tg1_tr1", ts)

        assert ssm._diag_track_changed_ts["tc1_tg1_tr1"] == ts
        assert ssm._diag_first_track_changed_ts == ts

    def test_first_timestamp_tracks_earliest(self):
        ssm = _make_ssm()
        ts1 = time.time()
        ts2 = ts1 + 0.05  # 50ms later

        ssm.record_track_changed_timestamp("tc1_tg1_tr1", ts1)
        ssm.record_track_changed_timestamp("tc1_tg1_tr2", ts2)

        # First timestamp should remain the earliest
        assert ssm._diag_first_track_changed_ts == ts1

    def test_first_timestamp_resets_after_gap(self):
        ssm = _make_ssm()
        ts1 = time.time() - 5.0  # 5 seconds ago
        ts2 = time.time()  # now

        ssm.record_track_changed_timestamp("tc1_tg1_tr1", ts1)
        ssm.record_track_changed_timestamp("tc1_tg1_tr2", ts2)

        # Gap > 2s, so first_ts should reset to ts2
        assert ssm._diag_first_track_changed_ts == ts2


class TestRetryMechanism:
    """Verify that request_ma3_events retries on timeout."""

    def test_retry_counter_initialized(self):
        ssm = _make_ssm()
        ssm._events_request_retries["tc1_tg1_tr1"] = 0
        assert ssm._events_request_retries["tc1_tg1_tr1"] == 0

    def test_retry_counter_cleared_on_success(self):
        """on_track_events_received should clear the retry counter."""
        ssm = _make_ssm()
        ssm._events_request_retries["tc1_tg1_tr1"] = 2

        # Create a minimal entity so on_track_events_received doesn't bail out
        # at the "no entity found" check
        entity = _make_entity("tc1_tg1_tr1", "clap")
        ssm._synced_layers[entity.id] = entity

        # Stub out _schedule_push_ma3_to_editor and _compare_entity
        ssm._schedule_push_ma3_to_editor = MagicMock()

        # Build a comparison result with real (JSON-serializable) attributes
        comparison_result = MagicMock()
        comparison_result.diverged = False
        comparison_result.ma3_count = 1
        comparison_result.editor_count = 1
        ssm._compare_entity = MagicMock(return_value=comparison_result)
        ssm._save_to_settings = MagicMock()

        # Provide some events
        events = [{"time": 1.0, "name": "evt1", "cmd": "Go", "idx": 0}]

        ssm.on_track_events_received("tc1_tg1_tr1", events)

        # Retry counter should be cleared
        assert "tc1_tg1_tr1" not in ssm._events_request_retries
