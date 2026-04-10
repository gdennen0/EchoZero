from __future__ import annotations

from echozero.application.shared.enums import SyncMode
from echozero.application.sync.adapters import InMemorySyncService, MA3SyncAdapter
from echozero.ui.qt.timeline.demo_app import build_demo_app


class _Bridge:
    def __init__(self):
        self.connected_calls = 0
        self.disconnected_calls = 0

    def on_ma3_connected(self) -> None:
        self.connected_calls += 1

    def on_ma3_disconnected(self) -> None:
        self.disconnected_calls += 1


def test_build_demo_app_defaults_to_in_memory_sync_service():
    app = build_demo_app()
    assert isinstance(app.sync_service, InMemorySyncService)


def test_build_demo_app_with_bridge_uses_ma3_sync_adapter():
    bridge = _Bridge()
    app = build_demo_app(sync_bridge=bridge)
    assert isinstance(app.sync_service, MA3SyncAdapter)

    state = app.enable_sync(SyncMode.MA3)
    assert state.connected is True
    assert bridge.connected_calls == 1

    state = app.disable_sync()
    assert state.connected is False
    assert state.mode == SyncMode.NONE
    assert bridge.disconnected_calls == 1
