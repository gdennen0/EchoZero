"""Generic OSC transport services for EchoZero integrations.
Exists to keep OSC send/receive lifecycle separate from MA3-specific protocol logic.
Connects local endpoint configuration to reusable UDP transport and listener services.
"""

from __future__ import annotations

import errno
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Protocol

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


@dataclass(frozen=True, slots=True)
class OscInboundMessage:
    """One inbound OSC packet delivered to a reusable listener callback."""

    path: str
    args: tuple[object, ...]

    def first_text_arg(self) -> str:
        """Return the first string OSC argument or an empty string when none exists."""

        for arg in self.args:
            if isinstance(arg, str):
                return arg
        return ""


@dataclass(frozen=True, slots=True)
class OscReceiveServiceConfig:
    """Network binding and path configuration for one OSC receive service."""

    host: str = "127.0.0.1"
    port: int = 0
    path: str = "/"


@dataclass(frozen=True, slots=True)
class OscSendServiceConfig:
    """Network destination and path configuration for one OSC send service."""

    host: str = "127.0.0.1"
    port: int = 0
    path: str = "/"


class OscSendTransport(Protocol):
    """Outbound OSC transport that sends one payload to a configured path."""

    def send(self, payload: object) -> None: ...

    def close(self) -> None: ...


class OscUdpSendTransport:
    """UDP-backed OSC sender bound to one target address and OSC path."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        path: str = "/",
    ) -> None:
        self._config = OscSendServiceConfig(
            host=str(host),
            port=int(port),
            path=str(path),
        )
        self._client = SimpleUDPClient(self._config.host, self._config.port)

    @classmethod
    def from_config(cls, config: OscSendServiceConfig) -> "OscUdpSendTransport":
        """Build a sender transport from one typed OSC send config."""

        return cls(config.host, config.port, path=config.path)

    @property
    def endpoint(self) -> tuple[str, int]:
        """Return the configured remote host and port."""

        return self._config.host, self._config.port

    def send(self, payload: object) -> None:
        self._client.send_message(self._config.path, payload)

    def close(self) -> None:
        return None


class OscReceiveServer:
    """Threaded UDP OSC listener that forwards inbound packets to one callback."""

    def __init__(
        self,
        config: OscReceiveServiceConfig,
        *,
        on_message: Callable[[OscInboundMessage], None],
        thread_name: str = "echozero-osc-server",
    ) -> None:
        self._config = config
        self._on_message = on_message
        self._thread_name = str(thread_name)
        self._dispatcher: Dispatcher | None = None
        self._server: ThreadingOSCUDPServer | None = None
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        """True when the OSC listener thread is alive."""

        return self._server is not None and self._thread is not None and self._thread.is_alive()

    @property
    def endpoint(self) -> tuple[str, int]:
        """Return the bound listener endpoint once started."""

        if self._server is None:
            raise RuntimeError("OSC receive server is not running")
        host, port = self._server.server_address
        return str(host), int(port)

    def start(self) -> "OscReceiveServer":
        """Start the OSC listener if it is not already running."""

        if self._server is not None:
            return self

        dispatcher = Dispatcher()
        dispatcher.map(self._config.path, self._handle_message)
        requested_host = str(self._config.host).strip() or "127.0.0.1"
        candidate_hosts: list[str] = [requested_host]
        if requested_host not in {"127.0.0.1", "localhost"}:
            candidate_hosts.extend(["127.0.0.1", "localhost"])

        last_bind_error: OSError | None = None
        server = None
        for host in dict.fromkeys(candidate_hosts):
            try:
                server = ThreadingOSCUDPServer((host, self._config.port), dispatcher)
                break
            except OSError as exc:
                if getattr(exc, "errno", None) != errno.EADDRNOTAVAIL:
                    raise
                last_bind_error = exc
        if server is None:
            if last_bind_error is None:
                raise RuntimeError("Unable to start MA3 OSC receive server.")
            raise last_bind_error

        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
            name=self._thread_name,
        )
        thread.start()

        self._dispatcher = dispatcher
        self._server = server
        self._thread = thread
        return self

    def stop(self) -> None:
        """Stop the OSC listener and release its bound socket."""

        server = self._server
        thread = self._thread
        self._dispatcher = None
        self._server = None
        self._thread = None

        if server is None:
            return

        try:
            server.shutdown()
        finally:
            server.server_close()
            if thread is not None:
                thread.join(timeout=1.0)

    def _handle_message(self, address: str, *args: object) -> None:
        self._on_message(
            OscInboundMessage(
                path=str(address),
                args=tuple(args),
            )
        )
