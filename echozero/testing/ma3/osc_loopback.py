from __future__ import annotations

from dataclasses import dataclass
from threading import Condition, Lock, Thread
from time import monotonic

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


@dataclass(frozen=True, slots=True)
class OSCMessageCapture:
    path: str
    args: tuple[object, ...]
    timestamp: float


class OSCLoopback:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._host = host
        self._requested_port = port
        self._captures: list[OSCMessageCapture] = []
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._dispatcher: Dispatcher | None = None
        self._server: ThreadingOSCUDPServer | None = None
        self._client: SimpleUDPClient | None = None
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def endpoint(self) -> tuple[str, int]:
        if self._server is None:
            raise RuntimeError("OSC loopback server is not running")
        host, port = self._server.server_address
        return str(host), int(port)

    @property
    def thread(self) -> Thread | None:
        return self._thread

    def start(self) -> "OSCLoopback":
        if self._server is not None:
            return self

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._capture_message)
        server = ThreadingOSCUDPServer((self._host, self._requested_port), dispatcher)
        host, port = server.server_address
        client = SimpleUDPClient(str(host), int(port))
        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
            name="echozero-ma3-osc-loopback",
        )
        thread.start()

        self._dispatcher = dispatcher
        self._server = server
        self._client = client
        self._thread = thread
        return self

    def stop(self) -> None:
        server = self._server
        thread = self._thread

        self._dispatcher = None
        self._server = None
        self._client = None
        self._thread = None

        if server is None:
            return

        try:
            server.shutdown()
        finally:
            server.server_close()
            if thread is not None:
                thread.join(timeout=1.0)

    def send(self, path: str, *args: object) -> None:
        if self._client is None:
            raise RuntimeError("OSC loopback server is not running")
        if len(args) == 0:
            payload: object = []
        elif len(args) == 1:
            payload = args[0]
        else:
            payload = list(args)
        self._client.send_message(path, payload)

    def clear(self) -> None:
        with self._condition:
            self._captures.clear()

    def captures(self) -> list[OSCMessageCapture]:
        with self._lock:
            return list(self._captures)

    def wait_for(self, path: str, timeout: float = 1.0) -> OSCMessageCapture | None:
        deadline = monotonic() + timeout
        with self._condition:
            while True:
                for capture in self._captures:
                    if capture.path == path:
                        return capture
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def _capture_message(self, path: str, *args: object) -> None:
        capture = OSCMessageCapture(path=path, args=tuple(args), timestamp=monotonic())
        with self._condition:
            self._captures.append(capture)
            self._condition.notify_all()
