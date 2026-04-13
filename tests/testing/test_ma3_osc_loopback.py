from __future__ import annotations

from echozero.testing.ma3 import OSCLoopback


def test_osc_loopback_receives_sent_messages_deterministically():
    loopback = OSCLoopback().start()

    try:
        loopback.send("/ma3/transport", 1, "go")
        capture = loopback.wait_for("/ma3/transport", timeout=1.0)

        assert capture is not None
        assert capture.path == "/ma3/transport"
        assert capture.args == (1, "go")
        assert capture.timestamp > 0.0
        assert [message.path for message in loopback.captures()] == ["/ma3/transport"]
    finally:
        loopback.stop()


def test_osc_loopback_wait_for_and_clear_behaviour():
    loopback = OSCLoopback().start()

    try:
        assert loopback.wait_for("/missing", timeout=0.05) is None

        loopback.send("/ma3/state", "armed")
        assert loopback.wait_for("/ma3/state", timeout=1.0) is not None

        loopback.clear()

        assert loopback.captures() == []
        assert loopback.wait_for("/ma3/state", timeout=0.05) is None
    finally:
        loopback.stop()
