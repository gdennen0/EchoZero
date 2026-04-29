"""ReviewServer exposes a minimal phone-first HTTP surface for review sessions.
Exists because fast verification work often happens away from the desktop UI.
Connects persisted review sessions to a tiny stdlib web app and audio streamer.
"""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from echozero.foundry.domain import ReviewOutcome
from echozero.foundry.services import ReviewSessionService
from echozero.foundry.review_web import build_review_page


class _ReviewHTTPServer(ThreadingHTTPServer):
    """HTTP server bound to one review session and its service façade."""

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        service: ReviewSessionService,
        session_id: str | None,
        application_session: dict[str, object] | None,
    ):
        super().__init__(address, handler)
        self.service = service
        self.default_session_id = str(session_id or "").strip()
        self.current_application_session = (
            dict(application_session) if application_session is not None else None
        )
        self.state_revision = 0


def create_review_http_server(
    root: Path,
    session_id: str | None,
    *,
    host: str,
    port: int,
    application_session: dict[str, object] | None = None,
) -> _ReviewHTTPServer:
    """Build an HTTP server bound to one persisted review session."""
    handler = type("ReviewRequestHandler", (_ReviewRequestHandler,), {})
    return _ReviewHTTPServer(
        (host, port),
        handler,
        service=ReviewSessionService(root),
        session_id=session_id,
        application_session=application_session,
    )


def serve_review_session(root: Path, session_id: str, *, host: str = "127.0.0.1", port: int = 8421) -> int:
    """Run the review server until interrupted."""
    server = create_review_http_server(root, session_id, host=host, port=port)
    try:
        print(json.dumps({"session_id": session_id, "host": host, "port": server.server_port}, indent=2))
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


class _ReviewRequestHandler(BaseHTTPRequestHandler):
    server: _ReviewHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_html(build_review_page())
            return
        if parsed.path == "/api/sessions":
            payload = self.server.service.build_session_index(
                default_session_id=self.server.default_session_id or None,
                application_session=self.server.current_application_session,
            )
            payload["stateRevision"] = self.server.state_revision
            self._write_json(payload)
            return
        if parsed.path == "/api/session":
            try:
                payload = self._snapshot_from_query(parsed.query)
                payload["stateRevision"] = self.server.state_revision
                self._write_json(payload)
            except ValueError as exc:
                self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if parsed.path.startswith("/audio/"):
            self._serve_audio(parsed.path.rsplit("/", 1)[-1], query=parsed.query)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/review":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            return
        payload = self._read_json_body()
        try:
            outcome = ReviewOutcome(str(payload.get("outcome", "")).strip())
            item_id = str(payload.get("itemId", "")).strip()
            if not item_id:
                raise ValueError("itemId is required")
            self.server.service.set_item_review(
                self._resolve_session_id(parsed.query),
                item_id,
                outcome=outcome,
                corrected_label=_optional_payload_text(payload, "correctedLabel"),
                review_note=_optional_payload_text(payload, "reviewNote"),
                decision_kind=_optional_payload_text(payload, "decisionKind"),
                original_start_ms=_optional_payload_float(payload, "originalStartMs"),
                original_end_ms=_optional_payload_float(payload, "originalEndMs"),
                corrected_start_ms=_optional_payload_float(payload, "correctedStartMs"),
                corrected_end_ms=_optional_payload_float(payload, "correctedEndMs"),
                created_event_ref=_optional_payload_text(payload, "createdEventRef"),
                surface=_optional_payload_text(payload, "surface") or "phone_review",
                workflow=_optional_payload_text(payload, "workflow") or "manual_review",
                operator_action=_optional_payload_text(payload, "operatorAction"),
            )
            snapshot = self._snapshot_from_query(parsed.query)
        except (ValueError, KeyError) as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._write_json(snapshot)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _snapshot_from_query(self, query: str) -> dict[str, object]:
        params = parse_qs(query)
        session_id = self._resolve_session_id(query)
        payload = self.server.service.build_snapshot(  # type: ignore[attr-defined]
            session_id,
            outcome=_first_param(params, "outcome", default="pending"),
            polarity=_first_param(params, "polarity"),
            target_class=_first_param(params, "targetClass"),
            song_ref=_first_param(params, "songRef"),
            layer_ref=_first_param(params, "layerRef"),
            cursor=_int_param(params, "cursor", default=0),
            item_id=_optional_param(params, "itemId"),
        )
        current_item = payload.get("currentItem")
        if isinstance(current_item, dict):
            current_item["audioUrl"] = f"/audio/{current_item['itemId']}?sessionId={session_id}"
        return payload

    def _resolve_session_id(self, query: str) -> str:
        params = parse_qs(query)
        session_id = _first_param(params, "sessionId", default=self.server.default_session_id)
        if not session_id:
            raise ValueError("sessionId is required")
        return session_id

    def _serve_audio(self, item_id: str, *, query: str) -> None:
        session = self.server.service.get_session(self._resolve_session_id(query))
        if session is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Review session not found")
            return
        item = next((candidate for candidate in session.items if candidate.item_id == item_id), None)
        if item is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Review item not found")
            return
        path = Path(item.audio_path)
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
            return
        total_size = int(path.stat().st_size)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            requested_range = _parse_byte_range(
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

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _write_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _write_json(self, payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)


def _first_param(params: dict[str, list[str]], key: str, *, default: str = "all") -> str:
    values = params.get(key)
    return values[0] if values else default


def _optional_payload_text(payload: dict, key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_param(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    if not values:
        return None
    text = values[0].strip()
    return text or None


def _optional_payload_float(payload: dict, key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_param(params: dict[str, list[str]], key: str, *, default: int) -> int:
    values = params.get(key)
    if not values:
        return default
    return int(values[0])


def _parse_byte_range(value: str | None, *, total_size: int) -> tuple[int, int] | None:
    if value is None or total_size <= 0:
        return None
    prefix = "bytes="
    if not value.startswith(prefix):
        raise ValueError("Malformed Range header.")
    first_range = value[len(prefix) :].split(",", 1)[0].strip()
    if "-" not in first_range:
        raise ValueError("Malformed Range header.")
    start_text, end_text = first_range.split("-", 1)
    if not start_text and not end_text:
        raise ValueError("Malformed Range header.")
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError("Malformed Range header.")
        start = max(total_size - length, 0)
        end = total_size - 1
        return start, end
    start = int(start_text)
    end = total_size - 1 if not end_text else int(end_text)
    if start < 0 or start >= total_size or end < start:
        raise ValueError("Malformed Range header.")
    return start, min(end, total_size - 1)
