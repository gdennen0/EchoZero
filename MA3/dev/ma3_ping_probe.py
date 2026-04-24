#!/usr/bin/env python3
"""Probe MA3 OSC ping round trips using EchoZero's production OSC paths."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from threading import Event


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from echozero.infrastructure.osc import (  # noqa: E402
    OscReceiveServer,
    OscReceiveServiceConfig,
    OscUdpSendTransport,
)
from echozero.infrastructure.sync.ma3_osc import format_ma3_lua_command  # noqa: E402


def _infer_listen_host(target_host: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((str(target_host), 1))
        host = sock.getsockname()[0]
        return str(host or "127.0.0.1")
    finally:
        sock.close()


def _parse_ports(value: str | None, *, default: list[int]) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    ports: list[int] = []
    for segment in text.split(","):
        part = segment.strip()
        if not part:
            continue
        ports.append(int(part))
    return ports or list(default)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send EZ.SetTarget/EZ.Ping to MA3 and wait for /ez/message replies.",
    )
    parser.add_argument("--ma3-host", required=True, help="Target MA3 host or IP address.")
    parser.add_argument(
        "--ma3-port",
        type=int,
        default=8000,
        help="Target MA3 OSC command port.",
    )
    parser.add_argument(
        "--probe-ports",
        default="",
        help="Optional comma-separated MA3 command ports to probe instead of only --ma3-port.",
    )
    parser.add_argument(
        "--listen-host",
        default="",
        help="Host/IP to bind for inbound /ez/message traffic. Defaults to the routed LAN IP.",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=0,
        help="Port to bind for inbound /ez/message traffic. Use 0 for an ephemeral port.",
    )
    parser.add_argument(
        "--command-path",
        default="/cmd",
        help="OSC path used for EchoZero -> MA3 commands.",
    )
    parser.add_argument(
        "--listen-path",
        default="/ez/message",
        help="OSC path used for MA3 -> EchoZero messages.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=4.0,
        help="Seconds to wait for a reply after sending commands.",
    )
    parser.add_argument(
        "--ping-count",
        type=int,
        default=2,
        help="How many EZ.Ping() commands to send per probed port.",
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Do not send OSC. Only print the manual MA3 console command and wait for replies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    listen_host = str(args.listen_host or "").strip() or _infer_listen_host(args.ma3_host)
    probe_ports = _parse_ports(args.probe_ports, default=[int(args.ma3_port)])
    message_event = Event()
    received: list[tuple[str, str]] = []

    def on_message(inbound) -> None:
        payload = inbound.first_text_arg()
        received.append((str(inbound.path), payload))
        print(f"RECV path={inbound.path} payload={payload}", flush=True)
        message_event.set()

    server = OscReceiveServer(
        OscReceiveServiceConfig(
            host=listen_host,
            port=int(args.listen_port),
            path=str(args.listen_path),
        ),
        on_message=on_message,
        thread_name="echozero-ma3-ping-probe",
    ).start()

    bound_host, bound_port = server.endpoint
    manual_command = f"Lua \"EZ.SetTarget('{bound_host}', {bound_port}); EZ.Ping()\""
    print(f"LISTEN {bound_host}:{bound_port} {args.listen_path}", flush=True)
    print(f"MANUAL {manual_command}", flush=True)

    try:
        if not args.manual_only:
            for port in probe_ports:
                print(f"PROBE {args.ma3_host}:{port} {args.command_path}", flush=True)
                transport = OscUdpSendTransport(
                    str(args.ma3_host),
                    int(port),
                    path=str(args.command_path),
                )
                try:
                    target_command = f"EZ.SetTarget({bound_host!r}, {bound_port})"
                    payload = format_ma3_lua_command(target_command)
                    print(f"  SEND {payload}", flush=True)
                    transport.send(payload)
                    time.sleep(0.15)
                    for _ in range(max(1, int(args.ping_count))):
                        payload = format_ma3_lua_command("EZ.Ping()")
                        print(f"  SEND {payload}", flush=True)
                        transport.send(payload)
                        time.sleep(0.15)
                finally:
                    transport.close()
                if message_event.is_set():
                    break

        if not message_event.is_set():
            message_event.wait(max(0.1, float(args.timeout)))
    finally:
        print(f"RECEIVED_ANY={message_event.is_set()}", flush=True)
        print(f"MESSAGE_COUNT={len(received)}", flush=True)
        server.stop()

    return 0 if message_event.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
