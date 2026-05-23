"""Canonical EchoZero desktop launcher entrypoint.
Exists to bootstrap the Qt app shell from CLI and packaged runtime environments.
Connects process startup and release launch flows to the Stage Zero shell.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

from echozero.application.settings import (
    AppSettingsLaunchOverrides,
    build_default_app_settings_service,
)
_QT_LAUNCH_SYMBOLS = {
    "QApplication",
    "QTimer",
    "OscUdpSendTransport",
    "MA3OSCBridge",
    "PROJECT_FILE_FILTER",
    "LauncherController",
    "build_launcher_surface",
    "install_runtime_logging",
    "fit_window_to_available_screen",
    "ensure_qt_theme_installed",
}


def _ensure_qt_launch_symbols() -> None:
    """Load Qt launch symbols lazily so frozen helper modes stay lightweight."""

    if "QApplication" in globals():
        return
    from PyQt6.QtCore import QTimer as _QTimer
    from PyQt6.QtWidgets import QApplication as _QApplication

    from echozero.infrastructure.osc import OscUdpSendTransport as _OscUdpSendTransport
    from echozero.infrastructure.sync.ma3_osc import MA3OSCBridge as _MA3OSCBridge
    from echozero.ui.qt.launcher_surface import (
        PROJECT_FILE_FILTER as _PROJECT_FILE_FILTER,
        LauncherController as _LauncherController,
        build_launcher_surface as _build_launcher_surface,
    )
    from echozero.ui.qt.runtime_logging import install_runtime_logging as _install_runtime_logging
    from echozero.ui.qt.window_geometry import (
        fit_window_to_available_screen as _fit_window_to_available_screen,
    )
    from echozero.ui.style.qt import ensure_qt_theme_installed as _ensure_qt_theme_installed

    globals().update(
        QApplication=_QApplication,
        QTimer=_QTimer,
        OscUdpSendTransport=_OscUdpSendTransport,
        MA3OSCBridge=_MA3OSCBridge,
        PROJECT_FILE_FILTER=_PROJECT_FILE_FILTER,
        LauncherController=_LauncherController,
        build_launcher_surface=_build_launcher_surface,
        install_runtime_logging=_install_runtime_logging,
        fit_window_to_available_screen=_fit_window_to_available_screen,
        ensure_qt_theme_installed=_ensure_qt_theme_installed,
    )


def __getattr__(name: str):
    if name in _QT_LAUNCH_SYMBOLS:
        _ensure_qt_launch_symbols()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_REPO_UI_AUTOMATION_SRC = Path(__file__).resolve().parent / "packages" / "ui_automation" / "src"
_QT_FFMPEG_DISABLE_DECODING_HW_BACKENDS = ","


def _configure_qt_ffmpeg_video_defaults() -> None:
    """Keep Qt Multimedia video decode predictable on macOS launch paths."""

    os.environ.setdefault(
        "QT_FFMPEG_DECODING_HW_DEVICE_TYPES",
        _QT_FFMPEG_DISABLE_DECODING_HW_BACKENDS,
    )


def _run_playback_service(argv: list[str]) -> int:
    """Run the playback service entrypoint inside a frozen app subprocess."""
    from echozero.application.playback.process_service_entry import main as service_main

    return service_main(argv)


def _ensure_repo_ui_automation_source_root(
    *,
    repo_ui_automation_src: Path = _REPO_UI_AUTOMATION_SRC,
    sys_path: list[str] | None = None,
) -> bool:
    """Add the repo ui_automation source root when launching from a checkout."""
    path_entries = sys.path if sys_path is None else sys_path
    resolved_src = repo_ui_automation_src.resolve()
    if not resolved_src.is_dir():
        return False
    resolved_text = str(resolved_src)
    if resolved_text in path_entries:
        return False
    path_entries.insert(0, resolved_text)
    return True


def _build_automation_bridge_server(*, runtime, widget, launcher, app, port: int):
    """Create the live automation bridge, repairing repo source roots when possible."""
    module_name = "echozero.ui.qt.automation_bridge"
    try:
        automation_bridge = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != "ui_automation" or not _ensure_repo_ui_automation_source_root():
            raise RuntimeError(
                "Automation bridge requires the ui_automation source root. "
                "Run `python3 scripts/dev_bootstrap.py` from the repo root or "
                'install the project with `pip install -e ".[dev]"`.'
            ) from exc
        sys.modules.pop(module_name, None)
        automation_bridge = importlib.import_module(module_name)
    automation_bridge_server = getattr(automation_bridge, "AutomationBridgeServer")
    return automation_bridge_server(
        runtime=runtime,
        widget=widget,
        launcher=launcher,
        app=app,
        port=port,
    )


def _shutdown_launcher_surface(surface) -> None:
    """Shut down the launcher surface while keeping a visible close progress dialog."""
    begin_shutdown_dialog = getattr(surface.controller, "begin_shutdown_dialog", None)
    finish_shutdown_dialog = getattr(surface.controller, "finish_shutdown_dialog", None)
    if callable(begin_shutdown_dialog):
        begin_shutdown_dialog()
    try:
        surface.runtime.shutdown()
    finally:
        if callable(finish_shutdown_dialog):
            finish_shutdown_dialog()


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "--playback-service":
        # PyTorch's distributed import path can call socket.getfqdn(), which may block
        # on reverse DNS in packaged macOS app helpers before the health server starts.
        # The playback service does not need FQDN resolution, so keep helper startup local
        # and deterministic.
        import socket

        socket.getfqdn = lambda name="": name or socket.gethostname()
        return _run_playback_service(raw_args[1:])

    _configure_qt_ffmpeg_video_defaults()
    _ensure_qt_launch_symbols()

    parser = argparse.ArgumentParser(description="Run the EchoZero Stage Zero shell.")
    parser.add_argument(
        "--playback-service",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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
    parsed, qt_args = parser.parse_known_args(raw_args)

    if parsed.playback_service:
        return _run_playback_service(qt_args)

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
    if sync_bridge is not None:
        try:
            surface.runtime.enable_sync()
        except Exception as exc:
            print(f"ma3_sync_enable_failed={exc}", flush=True)
    widget = surface.widget
    widget.show()
    fit_window_to_available_screen(widget)
    bridge = None
    if parsed.automation_port is not None:
        bridge = _build_automation_bridge_server(
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
        _shutdown_launcher_surface(surface)


if __name__ == "__main__":
    raise SystemExit(main())
