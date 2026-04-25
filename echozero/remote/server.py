"""
RemoteControlServer: Thin localhost wrapper around the EchoZero automation bridge.
Exists because private phone access needs session-scoped controls without exposing the raw bridge directly.
Connects browser and Tailscale-facing requests to the canonical ui_automation live client.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, urlparse

from ui_automation import EchoZeroAutomationProvider

from echozero.errors import InfrastructureError

from .http import (
    coerce_params,
    first_query_value,
    normalize_token,
    parse_bearer_token,
    parse_byte_range,
    to_jsonable,
)
from .mobile_web import build_mobile_web_page
from .session import (
    RemoteSession,
    RemoteSessionError,
    RemoteSessionStore,
    SessionExpiredError,
    SessionNotFoundError,
)


class RemoteBridgeError(InfrastructureError):
    """Raised when the wrapper cannot reach the canonical localhost automation bridge."""


class RemoteAutomationClient(Protocol):
    """Minimal backend protocol used by the remote wrapper server."""

    def close(self) -> None: ...

    def health(self) -> dict[str, Any]: ...

    def invoke(
        self,
        action_id: str,
        *,
        target_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any: ...

    def screenshot(self, *, target_id: str | None = None) -> bytes: ...

    def snapshot(self) -> Any: ...


class RemoteControlServer:
    """Serve one private phone-facing HTTP surface over the canonical live client."""

    def __init__(
        self,
        *,
        bridge_url: str,
        host: str = "127.0.0.1",
        port: int = 0,
        session_ttl_seconds: float = 900.0,
        backend_factory: Callable[[], RemoteAutomationClient] | None = None,
    ) -> None:
        self._bridge_url = bridge_url.rstrip("/")
        self._backend_factory = backend_factory or EchoZeroAutomationProvider(self._bridge_url).attach
        self._sessions = RemoteSessionStore(ttl_seconds=session_ttl_seconds)
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        """Return the host and bound port for the wrapper server."""
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Start the wrapper HTTP server in one background thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="echozero-remote-wrapper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the wrapper server and join its background thread."""
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _build_handler(self):
        remote = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    query_params = parse_qs(parsed.query)
                    if parsed.path in {"/", "/index.html"}:
                        self._send_html(HTTPStatus.OK, build_mobile_web_page())
                        return
                    if parsed.path == "/api/health":
                        self._send_json(HTTPStatus.OK, remote._build_health_payload())
                        return
                    if parsed.path == "/api/snapshot":
                        session = remote._require_authorized_session(
                            self.headers.get("Authorization"),
                            query_token=first_query_value(query_params, "session_token"),
                        )
                        payload = remote._build_snapshot_payload(session=session)
                        self._send_json(HTTPStatus.OK, payload)
                        return
                    if parsed.path == "/api/audio/current":
                        session = remote._require_authorized_session(
                            self.headers.get("Authorization"),
                            query_token=first_query_value(query_params, "session_token"),
                        )
                        source_path = remote._resolve_current_audio_path(session=session)
                        self._send_file(source_path)
                        return
                    if parsed.path.startswith("/api/transport/"):
                        session = remote._require_authorized_session(
                            self.headers.get("Authorization"),
                            query_token=first_query_value(query_params, "session_token"),
                        )
                        action_name = parsed.path.rsplit("/", 1)[-1]
                        action_id = f"transport.{action_name}"
                        response = remote._invoke_payload(
                            session=session,
                            action_id=action_id,
                            target_id=None,
                            params=None,
                        )
                        self._send_json(HTTPStatus.OK, response)
                        return
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": f"unknown path: {parsed.path}"},
                    )
                except SessionExpiredError as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": str(exc)})
                except SessionNotFoundError as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": str(exc)})
                except RemoteSessionError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except RemoteBridgeError as exc:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:  # pragma: no cover - defensive HTTP fallback
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    parsed = urlparse(self.path)
                    if parsed.path == "/api/session/start":
                        session = remote._sessions.create_session()
                        self._send_json(HTTPStatus.OK, session.as_payload())
                        return
                    if parsed.path == "/api/session/stop":
                        session = remote._require_authorized_session(self.headers.get("Authorization"))
                        remote._sessions.revoke_session(session.token)
                        self._send_json(HTTPStatus.OK, {"ok": True})
                        return
                    if parsed.path == "/api/screenshot":
                        session = remote._require_authorized_session(self.headers.get("Authorization"))
                        payload = self._read_json()
                        target_id = payload.get("target_id")
                        response = remote._build_screenshot_payload(
                            session=session,
                            target_id=None if target_id is None else str(target_id),
                        )
                        self._send_json(HTTPStatus.OK, response)
                        return
                    if parsed.path == "/api/action/invoke":
                        session = remote._require_authorized_session(self.headers.get("Authorization"))
                        payload = self._read_json()
                        response = remote._invoke_payload(
                            session=session,
                            action_id=str(payload["action_id"]),
                            target_id=None
                            if payload.get("target_id") is None
                            else str(payload["target_id"]),
                            params=coerce_params(payload.get("params")),
                        )
                        self._send_json(HTTPStatus.OK, response)
                        return
                    if parsed.path.startswith("/api/transport/"):
                        session = remote._require_authorized_session(self.headers.get("Authorization"))
                        action_name = parsed.path.rsplit("/", 1)[-1]
                        action_id = f"transport.{action_name}"
                        response = remote._invoke_payload(
                            session=session,
                            action_id=action_id,
                            target_id=None,
                            params=None,
                        )
                        self._send_json(HTTPStatus.OK, response)
                        return
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {"error": f"unknown path: {parsed.path}"},
                    )
                except KeyError as exc:
                    self._send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"error": f"missing required field: {exc.args[0]}"},
                    )
                except SessionExpiredError as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": str(exc)})
                except SessionNotFoundError as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, {"error": str(exc)})
                except RemoteSessionError as exc:
                    self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
                except RemoteBridgeError as exc:
                    self._send_json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                except Exception as exc:  # pragma: no cover - defensive HTTP fallback
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

            def log_message(self, format: str, *args: object) -> None:  # noqa: A003
                return None

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object body.")
                return payload

            def _send_html(self, status: HTTPStatus, body: str) -> None:
                encoded = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def _send_file(self, path: Path) -> None:
                total_size = int(path.stat().st_size)
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                try:
                    requested_range = parse_byte_range(
                        self.headers.get("Range"),
                        total_size=total_size,
                    )
                except ValueError:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{total_size}")
                    self.end_headers()
                    return

                status = HTTPStatus.PARTIAL_CONTENT if requested_range is not None else HTTPStatus.OK
                start, end = requested_range or (0, max(total_size - 1, 0))
                content_length = max(0, end - start + 1) if total_size > 0 else 0
                self.send_response(status)
                self.send_header("Content-Type", media_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(content_length))
                if requested_range is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
                self.end_headers()
                if content_length <= 0:
                    return
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = handle.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)

        return Handler

    def _build_health_payload(self) -> dict[str, Any]:
        bridge_health = self._with_client(lambda client: client.health())
        host, port = self.address
        return {
            "ok": True,
            "server": {"host": host, "port": port},
            "bridge": bridge_health,
        }

    def _build_snapshot_payload(self, *, session: RemoteSession) -> dict[str, Any]:
        snapshot = self._with_client(lambda client: client.snapshot())
        payload = to_jsonable(snapshot)
        if not isinstance(payload, dict):
            raise RemoteBridgeError("Snapshot payload was not a JSON object.")
        payload["remote_session"] = session.as_payload()
        return payload

    def _build_screenshot_payload(
        self,
        *,
        session: RemoteSession,
        target_id: str | None,
    ) -> dict[str, Any]:
        _ = session
        png_bytes = self._with_client(lambda client: client.screenshot(target_id=target_id))
        return {
            "target_id": target_id,
            "png_base64": base64.b64encode(png_bytes).decode("ascii"),
        }

    def _invoke_payload(
        self,
        *,
        session: RemoteSession,
        action_id: str,
        target_id: str | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not session.capabilities.can_invoke(action_id):
            raise RemoteSessionError(f"Action is not allowed for remote sessions: {action_id}")
        snapshot = self._with_client(
            lambda client: client.invoke(action_id, target_id=target_id, params=params)
        )
        payload = to_jsonable(snapshot)
        if not isinstance(payload, dict):
            raise RemoteBridgeError("Invoke payload was not a JSON object.")
        payload["remote_session"] = session.as_payload()
        return payload

    def _resolve_current_audio_path(self, *, session: RemoteSession) -> Path:
        snapshot = self._build_snapshot_payload(session=session)
        playback = snapshot.get("artifacts", {}).get("playback", {})
        if not isinstance(playback, dict):
            raise RemoteBridgeError("Snapshot playback payload was not an object.")
        active_sources = playback.get("active_sources")
        if not isinstance(active_sources, list):
            raise RemoteSessionError("No remote audio source is available for the current playback target.")
        source_ref = next(
            (
                str(candidate.get("source_ref", "")).strip()
                for candidate in active_sources
                if isinstance(candidate, dict) and str(candidate.get("source_ref", "")).strip()
            ),
            "",
        )
        if not source_ref:
            raise RemoteSessionError("No remote audio source is available for the current playback target.")
        source_path = Path(source_ref)
        if not source_path.exists() or not source_path.is_file():
            raise RemoteBridgeError(f"Remote audio source was not found on disk: {source_path}")
        return source_path

    def _require_authorized_session(
        self,
        authorization_header: str | None,
        *,
        query_token: str | None = None,
    ) -> RemoteSession:
        token = parse_bearer_token(authorization_header) or normalize_token(query_token)
        if token is None:
            raise SessionNotFoundError("Missing bearer token.")
        return self._sessions.require_session(token)

    def _with_client(self, callback: Callable[[RemoteAutomationClient], Any]) -> Any:
        client = self._backend_factory()
        try:
            return callback(client)
        except InfrastructureError:
            raise
        except Exception as exc:
            raise RemoteBridgeError(str(exc)) from exc
        finally:
            client.close()

def main(argv: list[str] | None = None) -> int:
    """Run the remote wrapper server against one existing localhost automation bridge."""
    parser = argparse.ArgumentParser(description="Run the EchoZero private remote wrapper.")
    parser.add_argument("--bridge-url", required=True, help="Base URL for the canonical localhost automation bridge.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind the wrapper server to.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the wrapper server to.")
    parser.add_argument(
        "--session-ttl-seconds",
        type=float,
        default=900.0,
        help="Lifetime for issued remote browser sessions.",
    )
    parsed = parser.parse_args(argv)

    server = RemoteControlServer(
        bridge_url=str(parsed.bridge_url),
        host=str(parsed.host),
        port=int(parsed.port),
        session_ttl_seconds=float(parsed.session_ttl_seconds),
    )
    server.start()
    host, port = server.address
    print(f"remote_wrapper=http://{host}:{port}", flush=True)
    try:
        server._thread.join()
    except KeyboardInterrupt:
        return 0
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
