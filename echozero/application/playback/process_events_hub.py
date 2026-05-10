"""
Playback process events hub: threaded websocket broadcaster for runtime telemetry.
Exists because process-isolated playback needs a push channel for non-request state transitions.
Connects service-side telemetry events to app-process websocket subscribers.
"""

from __future__ import annotations

import json
import queue
import threading

from echozero.application.playback.process_shared import PLAYBACK_IPC_WS_PATH

try:
    from websockets.sync.server import ServerConnection, serve
except ImportError as exc:  # pragma: no cover - environment contract
    raise RuntimeError("websockets package is required for playback process IPC") from exc


class PlaybackEventsHub:
    """Threaded websocket event hub for playback runtime telemetry."""

    def __init__(self, *, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._event_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=2048)
        self._clients: set[ServerConnection] = set()
        self._clients_lock = threading.Lock()
        self._ws_server = None
        self._server_thread: threading.Thread | None = None
        self._broadcast_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def ws_url(self) -> str:
        return f"ws://{self._host}:{self._port}{PLAYBACK_IPC_WS_PATH}"

    def start(self) -> None:
        if self._server_thread is not None:
            return
        self._server_thread = threading.Thread(
            target=self._run_server,
            name="ez-playback-events-ws",
            daemon=True,
        )
        self._server_thread.start()
        self._broadcast_thread = threading.Thread(
            target=self._run_broadcaster,
            name="ez-playback-events-broadcast",
            daemon=True,
        )
        self._broadcast_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._ws_server is not None:
            self._ws_server.shutdown()
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.5)
            self._server_thread = None
        if self._broadcast_thread is not None:
            self._broadcast_thread.join(timeout=1.5)
            self._broadcast_thread = None
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for connection in clients:
            try:
                connection.close(code=1001, reason="service shutdown")
            except Exception:
                continue

    def publish(self, event: dict[str, object]) -> None:
        if self._stop_event.is_set():
            return
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            try:
                _ = self._event_queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._event_queue.put_nowait(event)
            except queue.Full:
                return

    def _run_server(self) -> None:
        with serve(
            self._handle_connection,
            host=self._host,
            port=self._port,
            open_timeout=2,
            ping_interval=10,
            ping_timeout=10,
            close_timeout=1,
        ) as ws_server:
            self._ws_server = ws_server
            ws_server.serve_forever()

    def _handle_connection(self, websocket: ServerConnection) -> None:
        with self._clients_lock:
            self._clients.add(websocket)
        try:
            for _ in websocket:
                continue
        finally:
            with self._clients_lock:
                self._clients.discard(websocket)

    def _run_broadcaster(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            encoded = json.dumps(event, separators=(",", ":"))
            with self._clients_lock:
                clients = list(self._clients)
            stale: list[ServerConnection] = []
            for connection in clients:
                try:
                    connection.send(encoded)
                except Exception:
                    stale.append(connection)
            if stale:
                with self._clients_lock:
                    for connection in stale:
                        self._clients.discard(connection)


__all__ = ["PlaybackEventsHub"]
