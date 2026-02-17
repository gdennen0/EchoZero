"""
Application layer for MA3 (grandMA3) feature.

Contains:
- MA3CommunicationService - OSC communication with grandMA3
- MA3SyncService - synchronization with grandMA3
- ShowManagerListenerService - event listener
- ShowManagerStateService - state management
- MA3 Commands - track, layer, event, sync commands

Note: Layer sync orchestration is now handled by SyncSystemManager.
Import from: src.features.show_manager.application.sync_system_manager

Usage:
    from src.features.ma3.application import MA3CommunicationService
"""
# Services
from src.features.ma3.application.ma3_communication_service import MA3CommunicationService
from src.features.ma3.application.ma3_sync_service import MA3SyncService
from src.features.ma3.application.ma3_layer_mapping_service import MA3LayerMappingService
from src.features.ma3.application.ma3_routing_service import MA3RoutingService
from src.features.show_manager.application.show_manager_listener_service import ShowManagerListenerService
from src.features.show_manager.application.show_manager_state_service import ShowManagerStateService
from src.features.ma3.application.osc_message_dispatcher import (
    OSCMessageDispatcher,
    get_osc_dispatcher,
)

__all__ = [
    # Services
    'MA3CommunicationService',
    'MA3SyncService',
    'MA3LayerMappingService',
    'MA3RoutingService',
    'ShowManagerListenerService',
    'ShowManagerStateService',
    'OSCMessageDispatcher',
    'get_osc_dispatcher',
]
