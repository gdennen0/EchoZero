"""
ShowManager Listener Service

Delegates inbound OSC listening to MA3CommunicationService.
Keeps per-block listening state for compatibility with existing UI logic.
"""
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from src.utils.message import Log
from src.application.events.event_bus import EventBus


@dataclass
class ListenerState:
    """State for a single ShowManager listener."""
    block_id: str
    listen_port: int
    listen_address: str = "127.0.0.1"
    is_listening: bool = False
    message_handler: Optional[Callable[[str, list, tuple, bytes], None]] = None


class ShowManagerListenerService:
    """
    Service for managing OSC listeners for ShowManager blocks.

    Inbound OSC is handled by MA3CommunicationService only.
    This service tracks per-block listening intent/state.
    """

    def __init__(self, event_bus: EventBus, ma3_comm: Optional[Any] = None):
        self._event_bus = event_bus
        self._ma3_comm = ma3_comm
        self._listeners: Dict[str, ListenerState] = {}
        self._lock = threading.Lock()

        # Subscribe to block removal events to clean up listeners
        event_bus.subscribe("BlockRemoved", self._on_block_removed)
        event_bus.subscribe("project.loaded", self._on_project_loaded)

        Log.info("ShowManagerListenerService: Initialized")

    def start_listener(
        self,
        block_id: str,
        listen_port: int,
        listen_address: Optional[str] = None,
        message_handler: Optional[Callable[[str, list, tuple, bytes], None]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Start OSC listener for a ShowManager block.
        Delegates to MA3CommunicationService.
        """
        with self._lock:
            if block_id not in self._listeners:
                self._listeners[block_id] = ListenerState(
                    block_id=block_id,
                    listen_port=listen_port,
                    listen_address=listen_address or "127.0.0.1",
                    message_handler=message_handler
                )
            else:
                listener = self._listeners[block_id]
                listener.listen_port = listen_port
                if listen_address:
                    listener.listen_address = listen_address
                listener.message_handler = message_handler

            if not self._ma3_comm:
                Log.warning("ShowManagerListenerService: MA3 communication service not available")
                return False, "MA3 communication service not available"

            try:
                self._ma3_comm.set_listen_port(listen_port)
                if listen_address:
                    self._ma3_comm.set_listen_address(listen_address)
                if self._ma3_comm.is_listening():
                    current_addr = getattr(self._ma3_comm, "listen_address", None)
                    current_port = getattr(self._ma3_comm, "listen_port", None)
                    requested_addr = listen_address or current_addr
                    if requested_addr == current_addr and listen_port == current_port:
                        self._listeners[block_id].is_listening = True
                        return True, None
                    self._ma3_comm.stop_listening()
                success = self._ma3_comm.start_listening()
                self._listeners[block_id].is_listening = bool(success)
                return bool(success), None if success else "Failed to start MA3 listener"
            except Exception as e:
                self._listeners[block_id].is_listening = False
                return False, str(e)

    def stop_listener(self, block_id: str) -> None:
        """Stop OSC listener for a ShowManager block."""
        with self._lock:
            if self._ma3_comm:
                self._ma3_comm.stop_listening()
            if block_id in self._listeners:
                self._listeners[block_id].is_listening = False

    def is_listening(self, block_id: str) -> bool:
        """Check if listener is running (delegated)."""
        if not self._ma3_comm:
            return False
        if block_id not in self._listeners:
            return False
        return bool(self._ma3_comm.is_listening())

    def get_message_queue(self, block_id: str):
        """Inbound messages are handled by MA3CommunicationService."""
        return None

    def set_message_handler(
        self,
        block_id: str,
        handler: Optional[Callable[[str, list, tuple, bytes], None]]
    ) -> None:
        with self._lock:
            if block_id in self._listeners:
                self._listeners[block_id].message_handler = handler

    def cleanup_all(self) -> None:
        """Stop listening and clear state for all blocks."""
        with self._lock:
            if self._ma3_comm:
                self._ma3_comm.stop_listening()
            for listener in self._listeners.values():
                listener.is_listening = False
            self._listeners.clear()
        Log.info("ShowManagerListenerService: Cleaned up all listeners")

    def _on_block_removed(self, event) -> None:
        """Clean up listener state when a ShowManager block is removed."""
        if not hasattr(event, 'data'):
            return
        block_id = event.data.get('id')
        if not block_id:
            return
        with self._lock:
            if block_id in self._listeners:
                self._listeners[block_id].is_listening = False
                del self._listeners[block_id]

    def _on_project_loaded(self, event) -> None:
        """Clean up listeners when a new project is loaded."""
        self.cleanup_all()
