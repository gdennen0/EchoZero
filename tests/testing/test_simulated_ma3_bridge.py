from __future__ import annotations

from echozero.testing.ma3 import SimulatedMA3Bridge


def test_simulated_ma3_bridge_is_deterministic_and_records_events():
    bridge = SimulatedMA3Bridge()

    assert bridge.connected is False
    assert bridge.connect_calls == 0
    assert bridge.disconnect_calls == 0

    bridge.on_ma3_connected()
    bridge.on_ma3_connected()
    bridge.emit("osc", {"path": "/sync/start"})
    bridge.push_event("transport", {"state": "playing"})
    bridge.push_event("transport", {"state": "stopped"})
    first = bridge.pop_event()
    second = bridge.pop_event()
    third = bridge.pop_event()
    bridge.on_ma3_disconnected()

    assert bridge.connected is False
    assert bridge.connect_calls == 2
    assert bridge.disconnect_calls == 1
    assert bridge.emitted_events == [{"kind": "osc", "payload": {"path": "/sync/start"}}]
    assert first == {"kind": "transport", "payload": {"state": "playing"}}
    assert second == {"kind": "transport", "payload": {"state": "stopped"}}
    assert third is None
    assert bridge.pending_events() == []

