"""
MA3 Sync Service

Orchestrates bidirectional synchronization between MA3 and EchoZero.
Handles conflict detection, resolution, and event routing.
"""

from typing import List, Dict, Any, Optional, Tuple
import hashlib
from datetime import datetime

from src.features.ma3.domain.ma3_event import MA3Event
from src.features.ma3.domain.ma3_sync_state import (
    MA3SyncState, EventSyncState, ConflictRecord, 
    SyncDirection, ConflictResolution, ChangeType
)
from src.features.ma3.application.ma3_routing_service import MA3RoutingService, RoutingConfig
from src.features.blocks.domain import Block
from src.utils.message import Log


class MA3SyncService:
    """
    Service for synchronizing MA3 timecode events with EchoZero Editor blocks.
    
    Responsibilities:
    - Convert MA3 events to EZ events (via routing service)
    - Detect conflicts between MA3 and EZ versions
    - Apply conflict resolution strategies
    - Track sync state for all events
    """
    
    def __init__(self, routing_service: Optional[MA3RoutingService] = None):
        """
        Initialize sync service.
        
        Args:
            routing_service: Optional routing service (creates default if None)
        """
        self._routing_service = routing_service or MA3RoutingService()
        self._sync_states: Dict[str, MA3SyncState] = {}  # Key: "tc{tc}_editor{block_id}"
    
    def get_or_create_sync_state(
        self, 
        timecode_no: int, 
        editor_block_id: str,
        sync_direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    ) -> MA3SyncState:
        """Get or create sync state for a timecode/editor pair."""
        key = f"tc{timecode_no}_editor{editor_block_id}"
        if key not in self._sync_states:
            self._sync_states[key] = MA3SyncState(
                timecode_no=timecode_no,
                editor_block_id=editor_block_id,
                sync_direction=sync_direction,
            )
        return self._sync_states[key]
    
    def sync_ma3_to_editor(
        self,
        ma3_events: List[MA3Event],
        editor_block: Block,
        routing_config: Optional[RoutingConfig] = None,
        detect_conflicts: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[ConflictRecord]]:
        """
        Sync MA3 events to Editor block.
        
        Args:
            ma3_events: List of MA3Event objects
            editor_block: Target Editor block
            routing_config: Optional routing configuration
            detect_conflicts: Whether to detect conflicts
            
        Returns:
            Tuple of (ez_events_to_add, conflicts)
        """
        if not ma3_events:
            return [], []
        
        # Set routing config if provided
        if routing_config:
            self._routing_service.set_config(routing_config)
        
        # Get sync state
        timecode_no = ma3_events[0].timecode_no
        sync_state = self.get_or_create_sync_state(timecode_no, editor_block.id)
        
        # Convert MA3 events to EZ format
        ez_events = []
        for ma3_event in ma3_events:
            # Apply routing
            classification, layer_id = self._routing_service.route_ma3_event(ma3_event)
            
            # Convert to TimelineEvent format
            ez_event = ma3_event.to_timeline_event(
                classification=classification,
                layer_id=layer_id
            )
            ez_events.append(ez_event)
        
        # Detect conflicts if requested
        conflicts = []
        if detect_conflicts:
            # Get current events from editor
            editor_events = self._get_editor_events(editor_block)
            conflicts = self.detect_conflicts(ma3_events, editor_events, sync_state)
        
        # Update sync state
        for ma3_event in ma3_events:
            checksum = self._calculate_event_checksum(ma3_event.to_dict())
            event_state = sync_state.get_or_create_event_state(ma3_event.ma3_id)
            event_state.mark_synced(SyncDirection.MA3_TO_EZ, checksum)
        
        sync_state.mark_full_sync()
        
        Log.info(f"Synced {len(ez_events)} events from MA3 to Editor {editor_block.name}")
        if conflicts:
            Log.warning(f"Detected {len(conflicts)} conflicts")
        
        return ez_events, conflicts
    
    def sync_editor_to_ma3(
        self,
        editor_events: List[Dict[str, Any]],
        timecode_no: int,
        track_group: int,
        track: int,
        detect_conflicts: bool = True
    ) -> Tuple[List[MA3Event], List[ConflictRecord]]:
        """
        Sync Editor events to MA3.
        
        Args:
            editor_events: List of TimelineEvent dicts
            timecode_no: Target timecode number
            track_group: Target track group
            track: Target track
            detect_conflicts: Whether to detect conflicts
            
        Returns:
            Tuple of (ma3_events_to_add, conflicts)
        """
        if not editor_events:
            return [], []
        
        # Convert EZ events to MA3 format
        ma3_events = []
        for ez_event in editor_events:
            ma3_event = MA3Event.from_timeline_event(
                ez_event,
                timecode_no=timecode_no,
                track_group=track_group,
                track=track
            )
            if ma3_event:
                ma3_events.append(ma3_event)
        
        # Detect conflicts if requested
        conflicts = []
        if detect_conflicts:
            # Would need to get current MA3 events here
            # For now, skip conflict detection in this direction
            pass
        
        Log.info(f"Prepared {len(ma3_events)} events for sync to MA3 timecode {timecode_no}")
        
        return ma3_events, conflicts
    
    def detect_conflicts(
        self,
        ma3_events: List[MA3Event],
        editor_events: List[Dict[str, Any]],
        sync_state: MA3SyncState
    ) -> List[ConflictRecord]:
        """
        Detect conflicts between MA3 and Editor events.
        
        A conflict occurs when:
        - Same event exists in both but with different properties
        - Event was modified on both sides since last sync
        
        Args:
            ma3_events: List of MA3Event objects
            editor_events: List of TimelineEvent dicts
            sync_state: Current sync state
            
        Returns:
            List of ConflictRecord objects
        """
        conflicts = []
        
        # Build lookup maps
        ma3_by_id = {e.ma3_id: e for e in ma3_events}
        ez_by_id = {e['id']: e for e in editor_events if e.get('user_data', {}).get('ma3_source')}
        
        # Check for conflicts
        for event_id in set(ma3_by_id.keys()) & set(ez_by_id.keys()):
            ma3_event = ma3_by_id[event_id]
            ez_event = ez_by_id[event_id]
            
            # Get event sync state
            event_state = sync_state.get_or_create_event_state(event_id)
            
            # Calculate checksums
            ma3_checksum = self._calculate_event_checksum(ma3_event.to_dict())
            ez_checksum = self._calculate_event_checksum(ez_event)
            
            # Check if both sides changed
            ma3_changed = event_state.detect_change("ma3", ma3_checksum)
            ez_changed = event_state.detect_change("ez", ez_checksum)
            
            if ma3_changed and ez_changed:
                # Conflict: both sides modified
                conflict = ConflictRecord(
                    event_id=event_id,
                    ma3_version=ma3_event.to_dict(),
                    ez_version=ez_event,
                    conflict_type="modification",
                )
                conflicts.append(conflict)
                sync_state.add_conflict(conflict)
                
                Log.warning(f"Conflict detected on event {event_id}")
        
        return conflicts
    
    def resolve_conflict(
        self,
        conflict: ConflictRecord,
        resolution: ConflictResolution,
        sync_state: MA3SyncState
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve a conflict.
        
        Args:
            conflict: ConflictRecord to resolve
            resolution: How to resolve it
            sync_state: Current sync state
            
        Returns:
            Resolved event data (dict) or None if skipped
        """
        if resolution == ConflictResolution.USE_MA3:
            # MA3 version wins
            result = conflict.ma3_version
        elif resolution == ConflictResolution.USE_EZ:
            # EZ version wins
            result = conflict.ez_version
        elif resolution == ConflictResolution.MERGE:
            # Attempt to merge (take MA3 time, EZ classification)
            result = conflict.ez_version.copy()
            if conflict.ma3_version:
                result['time'] = conflict.ma3_version.get('time', result['time'])
        elif resolution == ConflictResolution.SKIP:
            # Skip this event
            result = None
        else:
            # Unknown resolution
            result = None
        
        # Mark as resolved
        sync_state.resolve_conflict(conflict.event_id, resolution)
        
        Log.info(f"Resolved conflict on {conflict.event_id}: {resolution.name}")
        
        return result
    
    def _calculate_event_checksum(self, event_dict: Dict[str, Any]) -> str:
        """Calculate checksum for event data."""
        # Use time and key properties for checksum
        key_props = {
            'time': event_dict.get('time', 0),
            'classification': event_dict.get('classification', event_dict.get('name', '')),
            'type': event_dict.get('event_type', event_dict.get('type', '')),
        }
        data_str = str(sorted(key_props.items()))
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_editor_events(self, editor_block: Block) -> List[Dict[str, Any]]:
        """
        Get current events from Editor block.
        
        TODO: Implement actual retrieval from Editor block data.
        For now, returns empty list.
        """
        # This would query the Editor block's event data
        # For now, return empty to avoid conflicts
        return []
    
    @property
    def routing_service(self) -> MA3RoutingService:
        """Get the routing service."""
        return self._routing_service
