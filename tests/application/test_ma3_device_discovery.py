"""MA3 device discovery command formatting tests.
Exists because discovery probes use the same grandMA3 `/cmd` path as live checks.
Connects bounded endpoint scans to the canonical EZ Lua command wrapper.
"""

from __future__ import annotations

from echozero.application.sync import ma3_device_discovery


def test_ma3_device_discovery_probe_wraps_commands_for_ma3_cmd_path(monkeypatch) -> None:
    sent_messages: list[tuple[str, str]] = []

    class CaptureClient:
        def __init__(self, _host: str, _port: int) -> None:
            return None

        def send_message(self, path: str, payload: str) -> None:
            sent_messages.append((path, payload))

    monkeypatch.setattr(ma3_device_discovery, "SimpleUDPClient", CaptureClient)

    ma3_device_discovery._send_probe("127.0.0.1", 9000, "127.0.0.1", 7100)

    assert sent_messages == [
        ("/cmd", "Lua \"EZ.SetTarget('127.0.0.1', 7100)\""),
        ("/cmd", 'Lua "EZ.Ping()"'),
    ]
