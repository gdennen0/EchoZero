"""
Remote health monitor: Poll the private wrapper and notify only on state changes.
Exists because phone control needs one lightweight alert lane when the wrapper or bridge drops.
Connects the remote wrapper health endpoint to OpenClaw system events without widening the control surface.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from typing import Callable
from urllib import request


@dataclass(frozen=True, slots=True)
class RemoteHealthState:
    """One health snapshot for the private remote wrapper."""

    is_healthy: bool
    detail: str


class RemoteHealthMonitor:
    """Watch the remote wrapper and notify when health changes."""

    def __init__(
        self,
        *,
        probe: Callable[[], RemoteHealthState],
        notify: Callable[[str], None],
        label: str = "EchoZero remote",
        notify_initial_failure: bool = True,
    ) -> None:
        self._probe = probe
        self._notify = notify
        self._label = label
        self._notify_initial_failure = notify_initial_failure
        self._last_state: bool | None = None

    def poll_once(self) -> RemoteHealthState:
        """Probe the wrapper once and notify if health changed."""
        state = self._probe()
        previous = self._last_state
        self._last_state = state.is_healthy
        if previous is None:
            if not state.is_healthy and self._notify_initial_failure:
                self._notify(f"{self._label} is down: {state.detail}")
            return state
        if previous == state.is_healthy:
            return state
        if state.is_healthy:
            self._notify(f"{self._label} recovered: {state.detail}")
        else:
            self._notify(f"{self._label} is down: {state.detail}")
        return state

    def run_forever(self, *, interval_seconds: float) -> None:
        """Poll until interrupted, sleeping between each probe."""
        sleep_seconds = max(0.5, float(interval_seconds))
        while True:
            self.poll_once()
            time.sleep(sleep_seconds)


def build_health_probe(health_url: str) -> Callable[[], RemoteHealthState]:
    """Return one probe that reads the wrapper health endpoint."""
    resolved_url = str(health_url)

    def probe() -> RemoteHealthState:
        try:
            with request.urlopen(resolved_url, timeout=5.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                return RemoteHealthState(False, "health payload was not a JSON object")
            if payload.get("ok") is not True:
                return RemoteHealthState(False, "health endpoint reported not ok")
            bridge = payload.get("bridge")
            if not isinstance(bridge, dict):
                return RemoteHealthState(True, "wrapper healthy")
            address = bridge.get("address")
            if not isinstance(address, dict):
                return RemoteHealthState(True, "wrapper healthy")
            host = str(address.get("host") or "unknown")
            port = str(address.get("port") or "unknown")
            return RemoteHealthState(True, f"bridge {host}:{port}")
        except Exception as exc:
            return RemoteHealthState(False, str(exc))

    return probe


def notify_openclaw(text: str) -> None:
    """Emit one system event through the local OpenClaw CLI."""
    argv = ["openclaw", "system", "event", "--text", text, "--mode", "now"]
    try:
        subprocess.run(
            argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    """Run the remote health monitor against one wrapper health URL."""
    parser = argparse.ArgumentParser(description="Watch the EchoZero remote wrapper health endpoint.")
    parser.add_argument("--health-url", required=True, help="Wrapper health endpoint, for example http://100.x.y.z:8765/api/health.")
    parser.add_argument("--label", default="EchoZero remote", help="Label used in emitted alert messages.")
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=15.0,
        help="Polling interval between health checks.",
    )
    parser.add_argument(
        "--notify-openclaw",
        action="store_true",
        help="Emit state-change alerts through the local OpenClaw CLI.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Probe only once and print the resulting state.",
    )
    parsed = parser.parse_args(argv)

    monitor = RemoteHealthMonitor(
        probe=build_health_probe(str(parsed.health_url)),
        notify=notify_openclaw if parsed.notify_openclaw else print,
        label=str(parsed.label),
    )
    if parsed.once:
        state = monitor.poll_once()
        print(json.dumps({"ok": state.is_healthy, "detail": state.detail}))
        return 0 if state.is_healthy else 1
    monitor.run_forever(interval_seconds=float(parsed.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
