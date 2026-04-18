"""
Automation bridge: localhost control server for a live EchoZero app process.
Exists so the canonical launcher can expose an attachable dev-only control path.
Connects run_echozero.py to the shared ui_automation EchoZero backend.
"""

from __future__ import annotations

import base64
import json
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from ui_automation.adapters.echozero.provider import create_backend_for_surface


class _BridgeInvocation:
    def __init__(self, callback):
        self.callback = callback
        self.event = threading.Event()
        self.result: Any = None
        self.error: Exception | None = None


class _MainThreadExecutor(QObject):
    run_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.run_requested.connect(self._run)

    def run_sync(self, callback):
        invocation = _BridgeInvocation(callback)
        self.run_requested.emit(invocation)
        if not invocation.event.wait(timeout=10.0):
            raise TimeoutError("automation bridge call timed out on the UI thread")
        if invocation.error is not None:
            raise invocation.error
        return invocation.result

    @pyqtSlot(object)
    def _run(self, invocation: _BridgeInvocation) -> None:
        try:
            invocation.result = invocation.callback()
        except Exception as exc:  # pragma: no cover - surfaced to HTTP caller
            invocation.error = exc
        finally:
            invocation.event.set()


class AutomationBridgeServer:
    """Local HTTP bridge for attaching to the live EchoZero launcher process."""

    def __init__(self, *, runtime, widget, launcher, app, host: str = "127.0.0.1", port: int = 0) -> None:
        self._backend = create_backend_for_surface(runtime=runtime, widget=widget, launcher=launcher, app=app)
        self._executor = _MainThreadExecutor()
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, name="echozero-automation-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _build_handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    if self.path == "/health":
                        self._send_json(200, {"ok": True, "address": {"host": bridge.address[0], "port": bridge.address[1]}})
                        return
                    if self.path == "/snapshot":
                        payload = bridge._executor.run_sync(lambda: _to_jsonable(bridge._backend.snapshot()))
                        self._send_json(200, payload)
                        return
                    self._send_json(404, {"error": f"unknown path: {self.path}"})
                except Exception as exc:  # pragma: no cover - surfaced to callers
                    self._send_json(500, {"error": str(exc), "error_type": type(exc).__name__})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    payload = json.loads(raw.decode("utf-8"))

                    if self.path == "/action":
                        result = bridge._executor.run_sync(lambda: bridge._dispatch_action(payload))
                        self._send_json(200, _to_jsonable(result))
                        return
                    if self.path == "/screenshot":
                        target_id = payload.get("target_id")
                        png_bytes = bridge._executor.run_sync(lambda: bridge._backend.screenshot(target_id=target_id))
                        self._send_json(
                            200,
                            {
                                "target_id": target_id,
                                "png_base64": base64.b64encode(png_bytes).decode("ascii"),
                            },
                        )
                        return
                    self._send_json(404, {"error": f"unknown path: {self.path}"})
                except Exception as exc:  # pragma: no cover - surfaced to callers
                    self._send_json(500, {"error": str(exc), "error_type": type(exc).__name__})

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return None

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def _dispatch_action(self, payload: dict[str, Any]):
        action = str(payload["action"])
        target_id = payload.get("target_id")
        args = payload.get("args")
        if action == "click":
            return self._backend.click(str(target_id), args=args)
        if action == "move_pointer":
            return self._backend.move_pointer(str(target_id), args=args)
        if action == "double_click":
            return self._backend.double_click(str(target_id), args=args)
        if action == "hover":
            return self._backend.hover(str(target_id), args=args)
        if action == "type_text":
            return self._backend.type_text(str(target_id), str(payload.get("text", "")), args=args)
        if action == "press_key":
            return self._backend.press_key(str(payload["key"]), args=args)
        if action == "drag":
            return self._backend.drag(str(target_id), payload["destination"], args=args)
        if action == "scroll":
            return self._backend.scroll(str(target_id), dx=int(payload.get("dx", 0)), dy=int(payload.get("dy", 0)), args=args)
        if action == "invoke":
            return self._backend.invoke(
                str(payload["action_id"]),
                target_id=None if target_id is None else str(target_id),
                params=payload.get("params"),
            )
        raise ValueError(f"unsupported automation action: {action}")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
