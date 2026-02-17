"""
MA3 Sync Commands

Commands for synchronizing MA3 timecode events with EchoZero Editor blocks.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import CommandResult
from src.features.ma3.domain.ma3_event import MA3Event
from src.features.ma3.domain.ma3_sync_state import ConflictResolution
from src.features.ma3.application.ma3_sync_service import MA3SyncService
from src.features.ma3.application.ma3_routing_service import RoutingConfig
from src.utils.message import Log


@dataclass
class SyncMA3ToEditorCommand:
    """
    Sync MA3 timecode events to an Editor block.
    
    Converts MA3 events to EchoZero format and sends them to the Editor
    via the manipulator port.
    """
    
    editor_block_id: str
    ma3_events: Optional[List[MA3Event]] = None
    routing_config: Optional[RoutingConfig] = None
    detect_conflicts: bool = True
    events: Optional[List[Dict[str, Any]]] = None
    
    def execute(self, facade) -> CommandResult:
        """Execute the sync command."""
        try:
            if self.events is not None:
                if not hasattr(facade, "data_item_repo") or not facade.data_item_repo:
                    return CommandResult.fail("Data item repository not available")
                if hasattr(facade, "describe_block"):
                    block_result = facade.describe_block(self.editor_block_id)
                    if not block_result.success or not block_result.data:
                        return CommandResult.fail(f"Editor block not found: {self.editor_block_id}")
                    editor_block = block_result.data
                elif hasattr(facade, "project_repo"):
                    editor_block = facade.project_repo.get_block(self.editor_block_id)
                    if not editor_block:
                        return CommandResult.fail(f"Editor block not found: {self.editor_block_id}")
                else:
                    return CommandResult.fail("Facade does not support block lookup")

                from src.shared.domain.entities import EventDataItem
                existing_items = facade.data_item_repo.list_by_block(self.editor_block_id)
                event_item = None
                for item in existing_items:
                    if isinstance(item, EventDataItem) and item.metadata.get("source") == "ma3":
                        event_item = item
                        break

                if event_item is None:
                    event_item = EventDataItem(
                        block_id=self.editor_block_id,
                        name=f"{editor_block.name}_ma3_events",
                        metadata={"source": "ma3", "output_port": "events"},
                    )
                    facade.data_item_repo.create(event_item)
                else:
                    event_item.clear_events()

                for event_dict in self.events:
                    event_item.add_event(
                        time=event_dict.get("time", 0.0),
                        classification=event_dict.get("classification", "event"),
                        duration=event_dict.get("duration", 0.0),
                        metadata=event_dict.get("metadata", {}),
                    )

                facade.data_item_repo.update(event_item)
                return CommandResult.ok({
                    "event_count": event_item.event_count,
                    "data_item_id": event_item.id,
                })

            # Get Editor block
            editor_block = facade.project_repo.get_block(self.editor_block_id)
            if not editor_block:
                return CommandResult.fail(f"Editor block not found: {self.editor_block_id}")
            
            # Get sync service
            sync_service = MA3SyncService()
            
            ma3_events = self.ma3_events or []
            # Perform sync
            ez_events, conflicts = sync_service.sync_ma3_to_editor(
                ma3_events,
                editor_block,
                self.routing_config,
                self.detect_conflicts
            )
            
            if conflicts:
                Log.warning(f"Sync completed with {len(conflicts)} conflicts")
                return CommandResult.ok({
                    'ez_events': ez_events,
                    'conflicts': [c.to_dict() for c in conflicts],
                })
            
            # Send events to Editor block via manipulator port
            # TODO: Implement actual sending via manipulator port
            # For now, just log the events
            Log.info(f"Would send {len(ez_events)} events to Editor {editor_block.name}")
            
            return CommandResult.ok({'ez_events': ez_events})
            
        except Exception as e:
            Log.error(f"Failed to sync MA3 to Editor: {e}")
            return CommandResult.fail(f"Sync failed: {str(e)}")


@dataclass
class SyncEditorToMA3Command:
    """
    Sync Editor block events to MA3 timecode.
    
    Converts EchoZero events to MA3 format and sends OSC commands to create
    them in MA3.
    """
    
    editor_block_id: str
    timecode_no: int
    track_group: int
    track: int
    detect_conflicts: bool = True
    
    def execute(self, facade) -> CommandResult:
        """Execute the sync command."""
        try:
            # Get Editor block
            editor_block = facade.project_repo.get_block(self.editor_block_id)
            if not editor_block:
                return CommandResult.fail(f"Editor block not found: {self.editor_block_id}")
            
            # Get events from Editor
            # TODO: Implement actual event retrieval from Editor block
            editor_events = []
            
            if not editor_events:
                return CommandResult.ok({'ma3_events': []})
            
            # Get sync service
            sync_service = MA3SyncService()
            
            # Perform sync
            ma3_events, conflicts = sync_service.sync_editor_to_ma3(
                editor_events,
                self.timecode_no,
                self.track_group,
                self.track,
                self.detect_conflicts
            )
            
            if conflicts:
                Log.warning(f"Sync completed with {len(conflicts)} conflicts")
                return CommandResult.ok({
                    'ma3_events': [e.to_dict() for e in ma3_events],
                    'conflicts': [c.to_dict() for c in conflicts],
                })
            
            # Send OSC commands to MA3
            # TODO: Implement actual OSC sending
            Log.info(f"Would send {len(ma3_events)} events to MA3 timecode {self.timecode_no}")
            
            return CommandResult.ok({'ma3_events': [e.to_dict() for e in ma3_events]})
            
        except Exception as e:
            Log.error(f"Failed to sync Editor to MA3: {e}")
            return CommandResult.fail(f"Sync failed: {str(e)}")


@dataclass
class ResolveConflictCommand:
    """
    Resolve a sync conflict.
    
    Applies the user's chosen resolution strategy to a conflicting event.
    """
    
    conflict_id: str  # Event ID
    resolution: ConflictResolution
    timecode_no: int
    editor_block_id: str
    
    def execute(self, facade) -> CommandResult:
        """Execute the resolution command."""
        try:
            # Get sync service
            sync_service = MA3SyncService()
            
            # Get sync state
            sync_state = sync_service.get_or_create_sync_state(
                self.timecode_no,
                self.editor_block_id
            )
            
            # Find conflict
            conflict = None
            for c in sync_state.conflicts:
                if c.event_id == self.conflict_id and not c.is_resolved:
                    conflict = c
                    break
            
            if not conflict:
                return CommandResult.fail(f"Conflict not found: {self.conflict_id}")
            
            # Resolve conflict
            resolved_event = sync_service.resolve_conflict(
                conflict,
                self.resolution,
                sync_state
            )
            
            if resolved_event is None and self.resolution != ConflictResolution.SKIP:
                return CommandResult.fail("Failed to resolve conflict")
            
            return CommandResult.ok({'resolved_event': resolved_event})
            
        except Exception as e:
            Log.error(f"Failed to resolve conflict: {e}")
            return CommandResult.fail(f"Resolution failed: {str(e)}")


@dataclass
class RequestEditorEventsCommand:
    """
    Request current events from Editor block.
    
    Queries the Editor block for its current event list.
    """
    
    editor_block_id: str
    
    def execute(self, facade) -> CommandResult:
        """Execute the command."""
        try:
            # Get Editor block
            editor_block = facade.project_repo.get_block(self.editor_block_id)
            if not editor_block:
                return CommandResult.fail(f"Editor block not found: {self.editor_block_id}")
            
            # Get EventDataItems owned by the Editor block
            from src.shared.domain.entities import EventDataItem
            
            data_item_repo = facade.data_item_repo
            existing_items = data_item_repo.list_by_block(editor_block.id)
            
            # Collect all events from all EventDataItems
            events = []
            for item in existing_items:
                if isinstance(item, EventDataItem):
                    for event in item.get_events():
                        # Convert TimelineEvent to dict
                        events.append({
                            "time": event.time,
                            "duration": event.duration,
                            "classification": event.classification,
                            "metadata": event.metadata if event.metadata else {}
                        })
            
            Log.info(f"Retrieved {len(events)} events from Editor {editor_block.name}")
            
            return CommandResult.ok({'events': events})
            
        except Exception as e:
            Log.error(f"Failed to request events from Editor: {e}")
            import traceback
            traceback.print_exc()
            return CommandResult.fail(f"Request failed: {str(e)}")
