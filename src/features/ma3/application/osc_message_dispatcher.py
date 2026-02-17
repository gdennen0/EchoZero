"""
OSC Message Dispatcher

Routes parsed OSC messages to registered handlers.
Provides a clean interface for handling MA3 messages.
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.features.ma3.domain.osc_message import OSCMessage, MessageType, ChangeType
from src.features.ma3.infrastructure.osc_parser import OSCParser, get_osc_parser
from src.utils.message import Log


# Handler function signature: (message: OSCMessage) -> None
MessageHandler = Callable[[OSCMessage], None]


@dataclass
class HandlerRegistration:
    """Registration info for a message handler."""
    handler: MessageHandler
    message_type: Optional[MessageType] = None
    change_type: Optional[ChangeType] = None
    priority: int = 0  # Higher priority handlers run first


class OSCMessageDispatcher:
    """
    Dispatcher for routing OSC messages to handlers.
    
    Supports:
    - Registration by type key (e.g., "trackgroups.list")
    - Registration by message type only (e.g., all "trackgroups" messages)
    - Wildcard handlers (receive all messages)
    - Priority-based ordering
    
    Usage:
        dispatcher = OSCMessageDispatcher()
        
        # Register handlers
        dispatcher.register("trackgroups.list", handle_trackgroups_list)
        dispatcher.register_type(MessageType.EVENTS, handle_all_events)
        dispatcher.register_wildcard(log_all_messages)
        
        # Dispatch message
        dispatcher.dispatch(message)
    """
    
    def __init__(self, parser: Optional[OSCParser] = None):
        self._parser = parser or get_osc_parser()
        
        # Handlers by type key (e.g., "trackgroups.list")
        self._handlers: Dict[str, List[HandlerRegistration]] = {}
        
        # Handlers by message type only
        self._type_handlers: Dict[MessageType, List[HandlerRegistration]] = {}
        
        # Wildcard handlers (receive all messages)
        self._wildcard_handlers: List[HandlerRegistration] = []
    
    @property
    def parser(self) -> OSCParser:
        """Get the parser instance."""
        return self._parser
    
    def register(
        self, 
        type_key: str, 
        handler: MessageHandler,
        priority: int = 0
    ) -> None:
        """
        Register a handler for a specific type key.
        
        Args:
            type_key: Message type key (e.g., "trackgroups.list")
            handler: Handler function
            priority: Handler priority (higher runs first)
        """
        if type_key not in self._handlers:
            self._handlers[type_key] = []
        
        registration = HandlerRegistration(
            handler=handler,
            priority=priority
        )
        self._handlers[type_key].append(registration)
        self._handlers[type_key].sort(key=lambda r: -r.priority)
        
        Log.debug(f"OSCMessageDispatcher: Registered handler for '{type_key}'")
    
    def register_type(
        self, 
        message_type: MessageType, 
        handler: MessageHandler,
        priority: int = 0
    ) -> None:
        """
        Register a handler for all messages of a type.
        
        Args:
            message_type: Message type to handle
            handler: Handler function
            priority: Handler priority (higher runs first)
        """
        if message_type not in self._type_handlers:
            self._type_handlers[message_type] = []
        
        registration = HandlerRegistration(
            handler=handler,
            message_type=message_type,
            priority=priority
        )
        self._type_handlers[message_type].append(registration)
        self._type_handlers[message_type].sort(key=lambda r: -r.priority)
        
        Log.debug(f"OSCMessageDispatcher: Registered type handler for '{message_type.value}'")
    
    def register_wildcard(
        self, 
        handler: MessageHandler,
        priority: int = 0
    ) -> None:
        """
        Register a handler that receives all messages.
        
        Args:
            handler: Handler function
            priority: Handler priority (higher runs first)
        """
        registration = HandlerRegistration(
            handler=handler,
            priority=priority
        )
        self._wildcard_handlers.append(registration)
        self._wildcard_handlers.sort(key=lambda r: -r.priority)
        
        Log.debug("OSCMessageDispatcher: Registered wildcard handler")
    
    def unregister(self, handler: MessageHandler) -> bool:
        """
        Unregister a handler from all registrations.
        
        Args:
            handler: Handler function to remove
            
        Returns:
            True if any registrations were removed
        """
        removed = False
        
        # Remove from type key handlers
        for handlers in self._handlers.values():
            before = len(handlers)
            handlers[:] = [r for r in handlers if r.handler != handler]
            if len(handlers) < before:
                removed = True
        
        # Remove from type handlers
        for handlers in self._type_handlers.values():
            before = len(handlers)
            handlers[:] = [r for r in handlers if r.handler != handler]
            if len(handlers) < before:
                removed = True
        
        # Remove from wildcard handlers
        before = len(self._wildcard_handlers)
        self._wildcard_handlers[:] = [r for r in self._wildcard_handlers if r.handler != handler]
        if len(self._wildcard_handlers) < before:
            removed = True
        
        return removed
    
    def dispatch(self, message: OSCMessage) -> int:
        """
        Dispatch a message to all registered handlers.
        
        Args:
            message: Parsed OSC message
            
        Returns:
            Number of handlers that processed the message
        """
        count = 0
        
        # Call wildcard handlers first
        for registration in self._wildcard_handlers:
            try:
                registration.handler(message)
                count += 1
            except Exception as e:
                Log.error(f"OSCMessageDispatcher: Wildcard handler error: {e}")
        
        # Call type handlers
        type_handlers = self._type_handlers.get(message.message_type, [])
        for registration in type_handlers:
            try:
                registration.handler(message)
                count += 1
            except Exception as e:
                Log.error(f"OSCMessageDispatcher: Type handler error for {message.message_type.value}: {e}")
        
        # Call specific handlers
        type_key = message.type_key
        specific_handlers = self._handlers.get(type_key, [])
        for registration in specific_handlers:
            try:
                registration.handler(message)
                count += 1
            except Exception as e:
                Log.error(f"OSCMessageDispatcher: Handler error for {type_key}: {e}")
        
        return count
    
    def dispatch_raw(self, data: bytes) -> int:
        """
        Parse raw OSC packet and dispatch to handlers.
        
        Args:
            data: Raw UDP packet bytes
            
        Returns:
            Number of handlers that processed the message
        """
        address, args = self._parser.parse_raw_packet(data)
        
        if address == "/ez/message" and args and isinstance(args[0], str):
            message = self._parser.parse_message(args[0], address)
            if message:
                return self.dispatch(message)
        
        return 0
    
    def dispatch_string(self, message_str: str, address: str = "/ez/message") -> int:
        """
        Parse message string and dispatch to handlers.
        
        Args:
            message_str: Pipe-delimited message string
            address: OSC address
            
        Returns:
            Number of handlers that processed the message
        """
        message = self._parser.parse_message(message_str, address)
        if message:
            return self.dispatch(message)
        return 0
    
    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._type_handlers.clear()
        self._wildcard_handlers.clear()


# Singleton instance for global dispatcher
_dispatcher_instance: Optional[OSCMessageDispatcher] = None


def get_osc_dispatcher() -> OSCMessageDispatcher:
    """Get the singleton OSC message dispatcher instance."""
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = OSCMessageDispatcher()
    return _dispatcher_instance
