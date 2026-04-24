#!/usr/bin/env python3
"""Send the working MA3 plugin-reload console command over EchoZero's `/cmd` transport."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from echozero.application.settings import build_default_app_settings_service  # noqa: E402
from echozero.infrastructure.osc import OscUdpSendTransport  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send the MA3 RP (ReloadAllPlugins) console command over /cmd.",
    )
    parser.add_argument(
        "--ma3-host",
        default=None,
        help="Override the target MA3 host or IP address. Defaults to saved app settings.",
    )
    parser.add_argument(
        "--ma3-port",
        type=int,
        default=None,
        help="Override the target MA3 OSC command port. Defaults to saved app settings.",
    )
    parser.add_argument(
        "--command-path",
        default=None,
        help="Override the OSC path used for raw MA3 console commands. Defaults to saved app settings.",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Optional path to the local EchoZero app settings JSON.",
    )
    return parser


def _resolve_target(args: argparse.Namespace) -> tuple[str, int, str, Path]:
    service = build_default_app_settings_service(path=args.settings_path)
    runtime_config = service.resolve_ma3_osc_runtime_config()

    host = str(
        args.ma3_host
        or runtime_config.send.host
        or ""
    ).strip()
    if not host:
        raise SystemExit("MA3 host is not configured. Set app settings or pass --ma3-host.")

    port = args.ma3_port if args.ma3_port is not None else runtime_config.send.port
    if port is None or int(port) < 1:
        raise SystemExit("MA3 command port is not configured. Set app settings or pass --ma3-port.")

    command_path = str(
        args.command_path
        or runtime_config.send.path
        or "/cmd"
    ).strip() or "/cmd"
    return host, int(port), command_path, service.store_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    ma3_host, ma3_port, command_path, settings_path = _resolve_target(args)

    transport = OscUdpSendTransport(
        ma3_host,
        ma3_port,
        path=command_path,
    )
    try:
        print(
            f"SEND {ma3_host}:{ma3_port} {command_path} RP "
            f"(settings: {settings_path})",
            flush=True,
        )
        transport.send("RP")
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
