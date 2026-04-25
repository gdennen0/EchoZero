"""Canonical EchoZero desktop launcher entrypoint.
Exists to bootstrap the Qt app shell from CLI and packaged runtime environments.
Connects process startup and release launch flows to the Stage Zero shell.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from echozero.application.settings import (
    AppSettingsLaunchOverrides,
    build_default_app_settings_service,
)
from echozero.infrastructure.osc import OscUdpSendTransport
from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge
from echozero.ui.qt.automation_bridge import AutomationBridgeServer
from echozero.ui.qt.launcher_surface import (
    PROJECT_FILE_FILTER,
    LauncherController,
    build_launcher_surface,
)
from echozero.ui.qt.runtime_logging import install_runtime_logging
from echozero.ui.qt.window_geometry import fit_window_to_available_screen
from echozero.ui.style.qt import ensure_qt_theme_installed


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
        default=None,
        help="Override the saved host for the EchoZero MA3 OSC listener.",
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
        default=None,
        help="Override the saved host for EchoZero -> MA3 OSC command traffic.",
    )
    parser.add_argument(
        "--ma3-osc-command-port",
        type=int,
        default=None,
        help="If set, send MA3 commands to this OSC port using the production bridge.",
    )
    parsed, qt_args = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    install_runtime_logging(parsed.log_dir)
    app_settings_service = build_default_app_settings_service()
    audio_output_config = app_settings_service.resolve_audio_output_config()
    ma3_config = app_settings_service.resolve_ma3_osc_runtime_config(
        launch_overrides=AppSettingsLaunchOverrides(
            ma3_osc_listen_host=parsed.ma3_osc_listen_host,
            ma3_osc_listen_port=parsed.ma3_osc_listen_port,
            ma3_osc_command_host=parsed.ma3_osc_command_host,
            ma3_osc_command_port=parsed.ma3_osc_command_port,
        )
    )

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    ensure_qt_theme_installed(app)
    working_dir_root = parsed.working_dir_root
    if (
        working_dir_root is None
        and parsed.smoke_exit_seconds is not None
        and parsed.smoke_exit_seconds > 0
    ):
        working_dir_root = Path(tempfile.gettempdir()) / "EchoZero" / "smoke-working"
    sync_bridge = None
    if ma3_config.is_enabled:
        command_transport = None
        if ma3_config.send.enabled and ma3_config.send.port is not None:
            command_transport = OscUdpSendTransport(
                ma3_config.send.host,
                ma3_config.send.port,
                path=ma3_config.send.path,
            )
        sync_bridge = MA3OSCBridge(
            listen_host=ma3_config.receive.host,
            listen_port=ma3_config.receive.port,
            listen_path=ma3_config.receive.path,
            command_transport=command_transport,
        )
    surface = build_launcher_surface(
        working_dir_root=working_dir_root,
        sync_bridge=sync_bridge,
        app_settings_service=app_settings_service,
        audio_output_config=audio_output_config,
    )
    widget = surface.widget
    widget.show()
    fit_window_to_available_screen(widget)
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
