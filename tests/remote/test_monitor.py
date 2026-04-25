from __future__ import annotations

from echozero.remote.monitor import RemoteHealthMonitor, RemoteHealthState


def test_monitor_suppresses_initial_healthy_state():
    notifications: list[str] = []
    states = iter([RemoteHealthState(True, "bridge 127.0.0.1:43210")])
    monitor = RemoteHealthMonitor(
        probe=lambda: next(states),
        notify=notifications.append,
    )

    result = monitor.poll_once()

    assert result.is_healthy is True
    assert notifications == []


def test_monitor_notifies_initial_failure_and_recovery():
    notifications: list[str] = []
    states = iter(
        [
            RemoteHealthState(False, "Connection refused"),
            RemoteHealthState(False, "Connection refused"),
            RemoteHealthState(True, "bridge 127.0.0.1:43210"),
        ]
    )
    monitor = RemoteHealthMonitor(
        probe=lambda: next(states),
        notify=notifications.append,
        label="Phone remote",
    )

    first = monitor.poll_once()
    second = monitor.poll_once()
    third = monitor.poll_once()

    assert first.is_healthy is False
    assert second.is_healthy is False
    assert third.is_healthy is True
    assert notifications == [
        "Phone remote is down: Connection refused",
        "Phone remote recovered: bridge 127.0.0.1:43210",
    ]


def test_monitor_notifies_only_on_transitions():
    notifications: list[str] = []
    states = iter(
        [
            RemoteHealthState(True, "bridge 127.0.0.1:43210"),
            RemoteHealthState(False, "Connection refused"),
            RemoteHealthState(False, "Connection refused"),
            RemoteHealthState(True, "bridge 127.0.0.1:43210"),
        ]
    )
    monitor = RemoteHealthMonitor(
        probe=lambda: next(states),
        notify=notifications.append,
    )

    monitor.poll_once()
    monitor.poll_once()
    monitor.poll_once()
    monitor.poll_once()

    assert notifications == [
        "EchoZero remote is down: Connection refused",
        "EchoZero remote recovered: bridge 127.0.0.1:43210",
    ]
