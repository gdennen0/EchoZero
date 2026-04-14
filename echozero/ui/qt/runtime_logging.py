from __future__ import annotations

import faulthandler
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler


class _TeeStream:
    def __init__(self, original, mirror) -> None:
        self._original = original
        self._mirror = mirror

    def write(self, data) -> int:
        text = str(data)
        if self._original is not None:
            try:
                self._original.write(text)
            except Exception:
                pass
        try:
            self._mirror.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass
        try:
            self._mirror.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        if self._original is not None and hasattr(self._original, "isatty"):
            try:
                return bool(self._original.isatty())
            except Exception:
                return False
        return False


_LOG_HANDLE = None
_LOG_PATH: Path | None = None
_INSTALLED = False
_PREVIOUS_QT_MESSAGE_HANDLER = None


def _default_log_dir(app_name: str) -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        root = Path(local_app_data)
    else:
        root = Path.home() / "AppData" / "Local"
    return root / app_name / "logs"


def _prune_old_logs(log_dir: Path, *, prefix: str, keep: int) -> None:
    if keep <= 0:
        return
    candidates = sorted(
        log_dir.glob(f"{prefix}*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for stale in candidates[keep:]:
        try:
            stale.unlink()
        except OSError:
            continue


def _qt_message_level(message_type: QtMsgType) -> str:
    if message_type == QtMsgType.QtDebugMsg:
        return "DEBUG"
    if message_type == QtMsgType.QtInfoMsg:
        return "INFO"
    if message_type == QtMsgType.QtWarningMsg:
        return "WARN"
    if message_type == QtMsgType.QtCriticalMsg:
        return "ERROR"
    if message_type == QtMsgType.QtFatalMsg:
        return "FATAL"
    return "UNKNOWN"


def install_runtime_logging(log_dir: Path | None = None, *, app_name: str = "EchoZero", keep_logs: int = 40) -> Path | None:
    global _LOG_HANDLE
    global _LOG_PATH
    global _INSTALLED
    global _PREVIOUS_QT_MESSAGE_HANDLER

    if _INSTALLED:
        return _LOG_PATH

    resolved_dir = Path(log_dir) if log_dir is not None else _default_log_dir(app_name)
    try:
        resolved_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_path = resolved_dir / f"{app_name.lower()}-session-{timestamp}.log"

    try:
        handle = session_path.open("a", encoding="utf-8", buffering=1)
    except OSError:
        return None

    _LOG_HANDLE = handle
    _LOG_PATH = session_path

    sys.stdout = _TeeStream(sys.stdout, handle)
    sys.stderr = _TeeStream(sys.stderr, handle)

    def _log_header() -> None:
        print(f"=== {app_name} runtime log start ===")
        print(f"timestamp={datetime.now().isoformat()}")
        print(f"python={sys.version.replace(chr(10), ' ')}")
        print(f"platform={platform.platform()}")
        print(f"pid={os.getpid()}")
        print(f"argv={sys.argv}")
        print(f"log_path={session_path}")

    def _main_excepthook(exc_type, exc, tb) -> None:
        print("\n=== UNHANDLED EXCEPTION (main thread) ===", file=sys.stderr)
        traceback.print_exception(exc_type, exc, tb)
        try:
            handle.flush()
        except Exception:
            pass

    def _thread_excepthook(args) -> None:
        print(f"\n=== UNHANDLED EXCEPTION (thread={args.thread.name}) ===", file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
        try:
            handle.flush()
        except Exception:
            pass

    def _unraisable_hook(unraisable) -> None:
        print("\n=== UNRAISABLE EXCEPTION ===", file=sys.stderr)
        if getattr(unraisable, "err_msg", None):
            print(f"err_msg={unraisable.err_msg}", file=sys.stderr)
        traceback.print_exception(
            unraisable.exc_type,
            unraisable.exc_value,
            unraisable.exc_traceback,
        )
        try:
            handle.flush()
        except Exception:
            pass

    def _qt_message_handler(message_type, context, message) -> None:
        location_parts: list[str] = []
        if context is not None:
            if getattr(context, "file", None):
                location_parts.append(str(context.file))
            if getattr(context, "line", None):
                location_parts.append(str(context.line))
        location = ":".join(location_parts)
        if location:
            print(f"[QT {_qt_message_level(message_type)}] {message} ({location})", file=sys.stderr)
        else:
            print(f"[QT {_qt_message_level(message_type)}] {message}", file=sys.stderr)

    _log_header()

    sys.excepthook = _main_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_excepthook
    if hasattr(sys, "unraisablehook"):
        sys.unraisablehook = _unraisable_hook

    try:
        faulthandler.enable(handle, all_threads=True)
    except Exception:
        print("faulthandler: enable failed", file=sys.stderr)

    try:
        _PREVIOUS_QT_MESSAGE_HANDLER = qInstallMessageHandler(_qt_message_handler)
    except Exception:
        print("qt message handler: install failed", file=sys.stderr)

    try:
        latest_hint = resolved_dir / "latest-log-path.txt"
        latest_hint.write_text(str(session_path), encoding="utf-8")
    except OSError:
        pass

    _prune_old_logs(resolved_dir, prefix=f"{app_name.lower()}-session-", keep=keep_logs)

    _INSTALLED = True
    print(f"[EchoZero] Runtime logging enabled: {session_path}", file=sys.stderr)
    return session_path
