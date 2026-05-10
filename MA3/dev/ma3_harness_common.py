#!/usr/bin/env python3
"""Shared MA3 harness target/bridge helpers for EchoZero dev tools."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from echozero.application.settings import build_default_app_settings_service  # noqa: E402
from echozero.infrastructure.osc import OscUdpSendTransport  # noqa: E402
from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge  # noqa: E402


def resolve_target(
    *,
    ma3_host: str | None,
    ma3_port: int | None,
    command_path: str | None,
    settings_path: Path | None,
) -> tuple[str, int, str, Path]:
    service = build_default_app_settings_service(path=settings_path)
    runtime_config = service.resolve_ma3_osc_runtime_config()
    host = str(ma3_host or runtime_config.send.host or "").strip()
    if not host:
        raise SystemExit("MA3 host is not configured. Set app settings or pass --ma3-host.")
    port = ma3_port if ma3_port is not None else runtime_config.send.port
    if port is None or int(port) < 1:
        raise SystemExit(
            "MA3 command port is not configured. Set app settings or pass --ma3-port."
        )
    path = str(command_path or runtime_config.send.path or "/cmd").strip() or "/cmd"
    return host, int(port), path, service.store_path


def build_bridge(
    *,
    ma3_host: str | None,
    ma3_port: int | None,
    command_path: str | None,
    settings_path: Path | None,
    listen_host: str = "0.0.0.0",
    listen_port: int = 0,
    timeout: float = 2.0,
) -> tuple[MA3OSCBridge, dict[str, object]]:
    host, port, path, store_path = resolve_target(
        ma3_host=ma3_host,
        ma3_port=ma3_port,
        command_path=command_path,
        settings_path=settings_path,
    )
    transport = OscUdpSendTransport(host, port, path=path)
    bridge = MA3OSCBridge(
        listen_host=str(listen_host or "0.0.0.0"),
        listen_port=int(listen_port),
        response_timeout=max(0.05, float(timeout)),
        command_transport=transport,
    )
    return bridge, {
        "ma3_host": host,
        "ma3_port": port,
        "command_path": path,
        "settings_path": str(store_path),
    }
