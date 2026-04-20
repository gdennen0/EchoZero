from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge, OSCCommandTransport
from echozero.ui.qt.automation_bridge import AutomationBridgeServer
from echozero.ui.qt.launcher_surface import (
    PROJECT_FILE_FILTER,
    LauncherController,
    build_launcher_surface,
)
from echozero.ui.qt.runtime_logging import install_runtime_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EchoZero Stage Zero shell.")
    parser.add_argument(
        "--smoke-exit-seconds",
        type=float,
        default=None,
        help="If set to a positive number, close the app after that many seconds.",
    )
    parser.add_argument(
        "--working-dir-root",
        type=Path,
        default=None,
        help="Override the project working-directory root used by the app shell runtime.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override for runtime log output directory.",
    )
    parser.add_argument(
        "--automation-port",
        type=int,
        default=None,
        help="If set, expose a localhost automation bridge for the running app on this port. Use 0 for an ephemeral port.",
    )
    parser.add_argument(
        "--automation-info-file",
        type=Path,
        default=None,
        help="Optional file to write automation bridge connection metadata into when --automation-port is enabled.",
    )
    parser.add_argument(
        "--ma3-osc-listen-host",
        type=str,
        default="127.0.0.1",
        help="Host for the EchoZero MA3 OSC listener.",
    )
    parser.add_argument(
        "--ma3-osc-listen-port",
        type=int,
        default=None,
        help="Enable the production MA3 OSC listener on this port. Use 0 for an ephemeral port.",
    )
    parser.add_argument(
        "--ma3-osc-command-host",
        type=str,
        default="127.0.0.1",
        help="Host for EchoZero -> MA3 OSC command traffic.",
    )
    parser.add_argument(
        "--ma3-osc-command-port",
        type=int,
        default=None,
        help="If set, send MA3 commands to this OSC port using the production bridge.",
    )
    parser.add_argument(
        "--ma3-osc-timecode",
        type=int,
        default=1,
        help="Default MA3 timecode number used by the production OSC bridge.",
    )
    parsed, qt_args = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    install_runtime_logging(parsed.log_dir)

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    working_dir_root = parsed.working_dir_root
    if (
        working_dir_root is None
        and parsed.smoke_exit_seconds is not None
        and parsed.smoke_exit_seconds > 0
    ):
        working_dir_root = Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working"
    sync_bridge = None
    if parsed.ma3_osc_listen_port is not None or parsed.ma3_osc_command_port is not None:
        command_transport = None
        if parsed.ma3_osc_command_port is not None:
            command_transport = OSCCommandTransport(
                parsed.ma3_osc_command_host,
                parsed.ma3_osc_command_port,
            )
        sync_bridge = MA3OSCBridge(
            listen_host=parsed.ma3_osc_listen_host,
            listen_port=0 if parsed.ma3_osc_listen_port is None else parsed.ma3_osc_listen_port,
            timecode_no=parsed.ma3_osc_timecode,
            command_transport=command_transport,
        )
    surface = build_launcher_surface(
        working_dir_root=working_dir_root,
        sync_bridge=sync_bridge,
    )
    widget = surface.widget
    widget.show()
    bridge = None
    if parsed.automation_port is not None:
        bridge = AutomationBridgeServer(
            runtime=surface.runtime,
            widget=surface.widget,
            launcher=surface.controller,
            app=app,
            port=parsed.automation_port,
        )
        bridge.start()
        host, port = bridge.address
        if parsed.automation_info_file is not None:
            parsed.automation_info_file.parent.mkdir(parents=True, exist_ok=True)
            parsed.automation_info_file.write_text(
                f"http://{host}:{port}\n",
                encoding="utf-8",
            )
        print(f"automation_bridge=http://{host}:{port}", flush=True)

    smoke_exit_seconds = parsed.smoke_exit_seconds
    if smoke_exit_seconds is not None and smoke_exit_seconds > 0:

        def _smoke_shutdown() -> None:
            app.quit()

        QTimer.singleShot(int(smoke_exit_seconds * 1000), _smoke_shutdown)

    try:
        return app.exec()
    finally:
        if bridge is not None:
            bridge.stop()
        surface.runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
