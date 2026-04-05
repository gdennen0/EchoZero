from src.features.show_manager.application.sync_system_manager import SyncSystemManager


def test_coalesce_delay_decreases_with_pending_count():
    base = SyncSystemManager._compute_coalesce_delay_ms(pending_count=1, inter_arrival_ms=400.0)
    burst = SyncSystemManager._compute_coalesce_delay_ms(pending_count=6, inter_arrival_ms=400.0)

    assert base >= burst
    assert base == 300
    assert burst <= 110


def test_coalesce_delay_decreases_with_fast_interarrival():
    slow = SyncSystemManager._compute_coalesce_delay_ms(pending_count=1, inter_arrival_ms=240.0)
    fast = SyncSystemManager._compute_coalesce_delay_ms(pending_count=1, inter_arrival_ms=30.0)

    assert slow == 300
    assert fast == 120
