from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib import request
from urllib.error import HTTPError

from echozero.remote.server import RemoteControlServer


@dataclass(slots=True)
class _FakeSnapshot:
    app: str = "EchoZero"
    selection: tuple[str, ...] = ("timeline.layer:source_audio",)
    focused_target_id: str | None = "shell.timeline"
    focused_object_id: str | None = None
    sync: dict[str, object] = None  # type: ignore[assignment]
    targets: tuple[dict[str, object], ...] = ()
    actions: tuple[dict[str, object], ...] = ()
    objects: tuple[dict[str, object], ...] = ()
    hit_targets: tuple[dict[str, object], ...] = ()
    artifacts: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sync is None:
            self.sync = {}
        if self.artifacts is None:
            self.artifacts = {"project_title": "Remote Test"}


class _FakeBackend:
    def __init__(self, *, artifacts: dict[str, object] | None = None) -> None:
        self.invocations: list[str] = []
        self._artifacts = {
            "project_title": "Remote Test",
            "transport": {
                "is_playing": False,
                "playhead_seconds": 0.0,
                "current_time_label": "00:00.00",
            },
            "playback": {
                "status": "stopped",
                "active_layer_id": None,
                "active_take_id": None,
                "active_sources": [],
                "latency_ms": 0.0,
                "backend_name": "sounddevice",
            },
        }
        if artifacts is not None:
            self._artifacts.update(artifacts)

    def close(self) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {"ok": True, "address": {"host": "127.0.0.1", "port": 43210}}

    def invoke(self, action_id: str, *, target_id: str | None = None, params=None) -> _FakeSnapshot:
        _ = target_id, params
        self.invocations.append(action_id)
        return _FakeSnapshot(
            actions=({"action_id": action_id, "label": action_id},),
            artifacts=dict(self._artifacts),
        )

    def screenshot(self, *, target_id: str | None = None) -> bytes:
        _ = target_id
        return b"png-bytes"

    def snapshot(self) -> _FakeSnapshot:
        return _FakeSnapshot(artifacts=dict(self._artifacts))


def test_remote_server_serves_mobile_page():
    server = _start_server()
    try:
        body = _request_json(server, "GET", "/")
        assert "EchoZero Remote" in body["raw_text"]
        assert "/api/session/start" in body["raw_text"]
    finally:
        server.stop()


def test_remote_server_proxies_snapshot_for_authorized_session():
    server = _start_server()
    try:
        session = _request_json(server, "POST", "/api/session/start")
        snapshot = _request_json(
            server,
            "GET",
            "/api/snapshot",
            token=str(session["token"]),
        )
        assert snapshot["app"] == "EchoZero"
        assert snapshot["artifacts"]["project_title"] == "Remote Test"
    finally:
        server.stop()


def test_remote_server_rejects_unauthorized_snapshot_requests():
    server = _start_server()
    try:
        error = _request_error(server, "GET", "/api/snapshot")
        assert error["status"] == 401
        assert "Missing bearer token" in error["payload"]["error"]
    finally:
        server.stop()


def test_remote_server_allows_safe_transport_actions_only():
    server = _start_server()
    try:
        session = _request_json(server, "POST", "/api/session/start")
        allowed = _request_json(
            server,
            "POST",
            "/api/transport/play",
            token=str(session["token"]),
        )
        blocked = _request_error(
            server,
            "POST",
            "/api/action/invoke",
            payload={"action_id": "app.open"},
            token=str(session["token"]),
        )
        assert allowed["actions"][0]["action_id"] == "transport.play"
        assert blocked["status"] == 403
    finally:
        server.stop()


def test_remote_server_allows_get_transport_shortcuts_with_query_token():
    server = _start_server()
    try:
        session = _request_json(server, "POST", "/api/session/start")
        allowed = _request_json(
            server,
            "GET",
            f"/api/transport/pause?session_token={session['token']}",
        )
        assert allowed["actions"][0]["action_id"] == "transport.pause"
    finally:
        server.stop()


def test_remote_server_serves_current_audio_for_authorized_session(tmp_path: Path):
    audio_path = tmp_path / "monitor.wav"
    audio_bytes = b"RIFFmonitor-audio"
    audio_path.write_bytes(audio_bytes)
    server = _start_server(
        artifacts={
            "playback": {
                "status": "playing",
                "active_layer_id": "layer-1",
                "active_take_id": None,
                "active_sources": [
                    {
                        "layer_id": "layer-1",
                        "take_id": None,
                        "source_ref": str(audio_path),
                        "mode": "continuous_audio",
                    }
                ],
                "latency_ms": 12.5,
                "backend_name": "sounddevice",
            }
        }
    )
    try:
        session = _request_json(server, "POST", "/api/session/start")
        response = _request_bytes(
            server,
            "GET",
            f"/api/audio/current?session_token={session['token']}",
            headers={"Range": "bytes=0-3"},
        )
        assert response["status"] == 206
        assert response["headers"]["Content-Range"] == f"bytes 0-3/{len(audio_bytes)}"
        assert response["body"] == audio_bytes[:4]
    finally:
        server.stop()


def _start_server(*, artifacts: dict[str, object] | None = None) -> RemoteControlServer:
    server = RemoteControlServer(
        bridge_url="http://127.0.0.1:43210",
        backend_factory=lambda: _FakeBackend(artifacts=artifacts),
    )
    server.start()
    return server


def _request_json(
    server: RemoteControlServer,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    token: str | None = None,
) -> dict[str, object]:
    host, port = server.address
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(f"http://{host}:{port}{path}", data=body, method=method, headers=headers)
    with request.urlopen(req, timeout=5.0) as response:
        if response.headers.get_content_type() == "text/html":
            return {"raw_text": response.read().decode("utf-8")}
        return json.loads(response.read().decode("utf-8"))


def _request_error(
    server: RemoteControlServer,
    method: str,
    path: str,
    *,
    payload: dict[str, object] | None = None,
    token: str | None = None,
) -> dict[str, object]:
    try:
        _request_json(server, method, path, payload=payload, token=token)
    except HTTPError as exc:
        return {
            "status": exc.code,
            "payload": json.loads(exc.read().decode("utf-8")),
        }
    raise AssertionError("Expected HTTPError")


def _request_bytes(
    server: RemoteControlServer,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, object]:
    host, port = server.address
    req = request.Request(
        f"http://{host}:{port}{path}",
        method=method,
        headers=headers or {},
    )
    with request.urlopen(req, timeout=5.0) as response:
        return {
            "status": response.status,
            "headers": dict(response.headers.items()),
            "body": response.read(),
        }
