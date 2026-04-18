from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

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
    parsed, qt_args = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    install_runtime_logging(parsed.log_dir)

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    working_dir_root = parsed.working_dir_root
    if working_dir_root is None and parsed.smoke_exit_seconds is not None and parsed.smoke_exit_seconds > 0:
        working_dir_root = Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working"
    surface = build_launcher_surface(
        working_dir_root=working_dir_root,
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
