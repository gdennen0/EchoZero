"""
Integration Tests for ShowManager Event Creation

Tests that MA3 events are properly created as EventDataItems with correct
time values, classifications (layer names), and metadata.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from src.application.commands.ma3.sync_commands import SyncMA3ToEditorCommand as SendEventsToEditorCommand
from src.shared.domain.entities import EventDataItem
from src.features.blocks.domain import Block


class MockDataItemRepository:
    """Mock data item repository for testing."""
    def __init__(self):
        self._items = {}
    
    def list_by_block(self, block_id: str):
        """List all data items for a block."""
        return [item for item in self._items.values() if item.block_id == block_id]
    
    def get(self, item_id: str):
        """Get data item by ID."""
        return self._items.get(item_id)
    
    def create(self, item: EventDataItem) -> EventDataItem:
        """Create a new data item."""
        self._items[item.id] = item
        return item
    
    def update(self, item: EventDataItem) -> None:
        """Update an existing data item."""
        # For new items (created in memory but not yet in repo), treat update as create
        if item.id not in self._items:
            # Item doesn't exist yet - create it (this happens when EventDataItem generates ID in __init__)
            self._items[item.id] = item
        else:
            # Item exists - update it
            self._items[item.id] = item


class MockResult:
    """Mock result for facade methods."""
    def __init__(self, success=True, data=None):
        self.success = success
        self.data = data


class MockFacade:
    """Mock facade for testing."""
    def __init__(self):
        self.data_item_repo = MockDataItemRepository()
        self._blocks = {}
    
    def describe_block(self, block_id: str):
        """Describe a block."""
        block = self._blocks.get(block_id)
        if not block:
            return MockResult(success=False, data=None)
        return MockResult(success=True, data=block)
    
    def add_block(self, block: Block):
        """Add a block to the mock."""
        self._blocks[block.id] = block


class TestSendEventsToEditorCommand:
    """Test SendEventsToEditorCommand event creation."""
    
    def test_create_event_data_item_with_events(self):
        """Test that SendEventsToEditorCommand creates EventDataItem with events."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        # Create events data
        events_data = [
            {
                "time": 10.5,
                "duration": 0.0,
                "classification": "layer_kicks",
                "metadata": {
                    "name": "Kick Event 1",
                    "cmd": "Go+",
                    "ma3_timecode_no": 101,
                    "ma3_track_group": 1,
                    "ma3_track": 1,
                }
            },
            {
                "time": 20.75,
                "duration": 0.0,
                "classification": "layer_snares",
                "metadata": {
                    "name": "Snare Event 1",
                    "cmd": "Go+",
                    "ma3_timecode_no": 101,
                    "ma3_track_group": 1,
                    "ma3_track": 2,
                }
            },
            {
                "time": 30.0,
                "duration": 0.0,
                "classification": "layer_hats",
                "metadata": {
                    "name": "Hat Event 1",
                    "cmd": "Go+",
                    "ma3_timecode_no": 101,
                    "ma3_track_group": 1,
                    "ma3_track": 3,
                }
            },
        ]
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=events_data
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify EventDataItem was created
        items = facade.data_item_repo.list_by_block(editor_block.id)
        assert len(items) == 1, f"Expected 1 EventDataItem, got {len(items)}"
        
        event_item = items[0]
        assert isinstance(event_item, EventDataItem), "Should be EventDataItem"
        assert event_item.block_id == editor_block.id, "Block ID should match"
        assert event_item.metadata.get("source") == "ma3", "Source should be 'ma3'"
        assert event_item.event_count == 3, f"Expected 3 events, got {event_item.event_count}"
        
        # Verify events have correct properties
        events = event_item.get_events()
        assert len(events) == 3, f"Expected 3 events, got {len(events)}"
        
        # Check first event
        event1 = events[0]
        assert event1.time == 10.5, f"Event 1 time should be 10.5, got {event1.time}"
        assert event1.classification == "layer_kicks", f"Event 1 classification should be 'layer_kicks', got {event1.classification}"
        assert event1.duration == 0.0, f"Event 1 duration should be 0.0, got {event1.duration}"
        assert event1.metadata.get("name") == "Kick Event 1", "Event 1 name should match"
        assert event1.metadata.get("ma3_timecode_no") == 101, "Event 1 timecode should match"
        
        # Check second event
        event2 = events[1]
        assert event2.time == 20.75, f"Event 2 time should be 20.75, got {event2.time}"
        assert event2.classification == "layer_snares", f"Event 2 classification should be 'layer_snares', got {event2.classification}"
        assert event2.metadata.get("name") == "Snare Event 1", "Event 2 name should match"
        
        # Check third event
        event3 = events[2]
        assert event3.time == 30.0, f"Event 3 time should be 30.0, got {event3.time}"
        assert event3.classification == "layer_hats", f"Event 3 classification should be 'layer_hats', got {event3.classification}"
    
    def test_update_existing_event_data_item(self):
        """Test that SendEventsToEditorCommand updates existing EventDataItem."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        # Create existing EventDataItem
        existing_item = EventDataItem(
            id="existing-item-id",
            block_id=editor_block.id,
            name="Editor1_ma3_events",
            type="Event",
            metadata={
                "source": "ma3",
                "output_port": "events",
            }
        )
        existing_item.add_event(
            time=5.0,
            classification="old_layer",
            duration=0.0,
            metadata={"name": "Old Event"}
        )
        facade.data_item_repo.create(existing_item)
        
        # New events data
        events_data = [
            {
                "time": 10.5,
                "duration": 0.0,
                "classification": "layer_kicks",
                "metadata": {
                    "name": "New Event 1",
                }
            },
        ]
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=events_data
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify EventDataItem was updated (not duplicated)
        items = facade.data_item_repo.list_by_block(editor_block.id)
        assert len(items) == 1, f"Expected 1 EventDataItem, got {len(items)}"
        
        event_item = items[0]
        assert event_item.id == existing_item.id, "Should update existing item, not create new one"
        assert event_item.event_count == 1, f"Should have 1 event after update, got {event_item.event_count}"
        
        # Verify old event was cleared
        events = event_item.get_events()
        assert len(events) == 1, "Should have 1 event"
        assert events[0].time == 10.5, "Should have new event time"
        assert events[0].classification == "layer_kicks", "Should have new event classification"
    
    def test_events_with_layer_mappings(self):
        """Test that events use correct classifications based on layer mappings."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        # Events with mapped layer names
        events_data = [
            {
                "time": 10.0,
                "duration": 0.0,
                "classification": "layer_kicks",  # Mapped layer name
                "metadata": {
                    "name": "Kick",
                    "ma3_coord": "tc101_tg1_tr1",
                }
            },
            {
                "time": 20.0,
                "duration": 0.0,
                "classification": "layer_snares",  # Mapped layer name
                "metadata": {
                    "name": "Snare",
                    "ma3_coord": "tc101_tg1_tr2",
                }
            },
        ]
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=events_data
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify events have correct classifications (layer names)
        items = facade.data_item_repo.list_by_block(editor_block.id)
        event_item = items[0]
        events = event_item.get_events()
        
        assert events[0].classification == "layer_kicks", "First event should have 'layer_kicks' classification"
        assert events[1].classification == "layer_snares", "Second event should have 'layer_snares' classification"
    
    def test_events_with_correct_time_values(self):
        """Test that events preserve correct time values from MA3."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        # Events with various time values
        events_data = [
            {
                "time": 0.0,
                "duration": 0.0,
                "classification": "layer_test",
                "metadata": {}
            },
            {
                "time": 1.234,
                "duration": 0.0,
                "classification": "layer_test",
                "metadata": {}
            },
            {
                "time": 100.567,
                "duration": 0.0,
                "classification": "layer_test",
                "metadata": {}
            },
        ]
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=events_data
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify events have correct time values
        items = facade.data_item_repo.list_by_block(editor_block.id)
        event_item = items[0]
        events = event_item.get_events()
        
        assert events[0].time == 0.0, f"Event 1 time should be 0.0, got {events[0].time}"
        assert events[1].time == 1.234, f"Event 2 time should be 1.234, got {events[1].time}"
        assert events[2].time == 100.567, f"Event 3 time should be 100.567, got {events[2].time}"
    
    def test_events_with_metadata_preserved(self):
        """Test that event metadata is properly preserved."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        events_data = [
            {
                "time": 10.0,
                "duration": 0.0,
                "classification": "layer_test",
                "metadata": {
                    "name": "Test Event",
                    "cmd": "Go+",
                    "fade": 2.5,
                    "delay": 0.1,
                    "ma3_timecode_no": 101,
                    "ma3_track_group": 1,
                    "ma3_track": 1,
                    "ma3_event_layer": 1,
                    "ma3_event_index": 0,
                    "ma3_coord": "tc101_tg1_tr1",
                }
            },
        ]
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=events_data
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify metadata is preserved
        items = facade.data_item_repo.list_by_block(editor_block.id)
        event_item = items[0]
        events = event_item.get_events()
        
        event = events[0]
        assert event.metadata.get("name") == "Test Event", "Name should be preserved"
        assert event.metadata.get("cmd") == "Go+", "Cmd should be preserved"
        assert event.metadata.get("fade") == 2.5, "Fade should be preserved"
        assert event.metadata.get("delay") == 0.1, "Delay should be preserved"
        assert event.metadata.get("ma3_timecode_no") == 101, "Timecode should be preserved"
        assert event.metadata.get("ma3_coord") == "tc101_tg1_tr1", "Coord should be preserved"
    
    def test_empty_events_list(self):
        """Test that empty events list creates EventDataItem with no events."""
        # Arrange
        facade = MockFacade()
        editor_block = Block(
            id="editor-1",
            project_id="test-project",
            name="Editor1",
            type="Editor",
            metadata={}
        )
        facade.add_block(editor_block)
        
        cmd = SendEventsToEditorCommand(
            editor_block_id=editor_block.id,
            events=[]
        )
        
        # Act
        result = cmd.execute(facade)
        
        # Assert
        assert result.success, f"Command should succeed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
        
        # Verify EventDataItem was created with no events
        items = facade.data_item_repo.list_by_block(editor_block.id)
        assert len(items) == 1, "Should create EventDataItem even with no events"
        
        event_item = items[0]
        assert event_item.event_count == 0, "Should have 0 events"
        assert len(event_item.get_events()) == 0, "Events list should be empty"
