"""
Unit tests for ShowManagerPanel checksum utilities.
"""

from ui.qt_gui.block_panels.show_manager_panel import _compute_events_checksum, _merge_event_chunks


def test_compute_events_checksum_basic():
    events = [
        {"time": 1.0, "idx": 1},
        {"time": 2.5, "idx": 2},
    ]
    expected = int(round(1.0 * 1000.0)) + 1 + int(round(2.5 * 1000.0)) + 2
    assert _compute_events_checksum(events) == expected


def test_merge_event_chunks_orders_by_offset():
    entry = {
        "chunks": {
            2: {"offset": 3, "events": [{"id": "b"}]},
            1: {"offset": 1, "events": [{"id": "a"}]},
        }
    }
    merged = _merge_event_chunks(entry)
    assert [evt["id"] for evt in merged] == ["a", "b"]

