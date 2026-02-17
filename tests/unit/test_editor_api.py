"""
Unit Tests for EditorAPI

Tests the unified Editor block API with signal integration.
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from src.features.blocks.application.editor_api import (
    EditorAPI,
    EditorAPIError,
    LayerInfo,
    EventInfo,
    create_editor_api,
)
from src.features.show_manager.application.sync_subscription_service import (
    SyncSubscriptionService,
    SourceType,
    ChangeType,
)


class MockResult:
    """Mock result object."""
    def __init__(self, success=True, data=None, message=""):
        self.success = success
        self.data = data
        self.message = message


class MockBlock:
    """Mock block for testing."""
    def __init__(self, id="test_block", name="Test Block", type="Editor"):
        self.id = id
        self.name = name
        self.type = type
        self.x = 0
        self.y = 0


class MockCommandBus:
    """Mock command bus that executes commands."""
    def __init__(self, facade):
        self._facade = facade

    def execute(self, cmd):
        # Simulate command execution
        try:
            from src.application.commands.editor_commands import EditorGetLayersCommand
            if isinstance(cmd, EditorGetLayersCommand):
                cmd.layers = self._facade._layer_state.get('layers', [])
                return True
        except Exception:
            pass
        if hasattr(cmd, 'redo'):
            cmd.redo()
        return True
    
    def begin_macro(self, name):
        pass
    
    def end_macro(self):
        pass


@pytest.fixture
def mock_facade():
    """Create a mock ApplicationFacade."""
    facade = MagicMock()
    facade.command_bus = MockCommandBus(facade)
    facade.current_project_id = "test_project"
    
    # Default layer state
    facade._layer_state = {'layers': []}
    
    def get_ui_state(state_type, entity_id):
        if state_type == 'editor_layers':
            return MockResult(success=True, data=facade._layer_state.copy())
        return MockResult(success=True, data={})
    
    def set_ui_state(state_type, entity_id, data):
        if state_type == 'editor_layers':
            facade._layer_state = data.copy()
        return MockResult(success=True)
    
    facade.get_ui_state = MagicMock(side_effect=get_ui_state)
    facade.set_ui_state = MagicMock(side_effect=set_ui_state)
    facade.describe_block = MagicMock(return_value=MockResult(
        success=True, data=MockBlock()
    ))
    facade.ui_state_repo = MagicMock()
    facade.data_item_repo = MagicMock()
    facade.data_item_repo.list_by_block = MagicMock(return_value=[])
    facade.preferences_repo = None
    
    return facade


@pytest.fixture
def sync_service():
    """Create a SyncSubscriptionService instance."""
    return SyncSubscriptionService()


@pytest.fixture
def editor_api(mock_facade, sync_service):
    """Create an EditorAPI instance."""
    return EditorAPI(mock_facade, "test_block", sync_service)


class TestEditorAPI:
    """Tests for EditorAPI."""
    
    def test_init(self, editor_api):
        """Test API initialization."""
        assert editor_api is not None
        assert editor_api.block_id == "test_block"
    
    def test_block_id_property(self, editor_api):
        """Test block_id property."""
        assert editor_api.block_id == "test_block"

    def test_apply_layer_snapshot_replaces_events(self, editor_api, mock_facade):
        """Apply snapshot should replace layer events atomically."""
        from src.shared.domain.entities import EventDataItem
        from src.shared.domain.entities.event_layer import EventLayer
        from src.shared.domain.entities.event_data_item import Event
        
        data_item = EventDataItem(
            id="item_1",
            block_id="test_block",
            name="Editor_events",
            metadata={"source": "onset"}
        )
        layer = EventLayer(name="LayerA")
        layer.add_event(Event(time=1.0, classification="LayerA", duration=0.0, metadata={}, id="old_1"))
        data_item.add_layer(layer)
        
        mock_facade.data_item_repo.get = MagicMock(return_value=data_item)
        mock_facade.data_item_repo.list_by_block = MagicMock(return_value=[data_item])
        mock_facade.data_item_repo.update = MagicMock()
        mock_facade._layer_state = {'layers': [{'name': 'LayerA'}]}
        
        events = [{
            "id": "new_1",
            "time": 2.0,
            "duration": 0.0,
            "classification": "LayerA",
            "metadata": {}
        }]
        
        applied = editor_api.apply_layer_snapshot("LayerA", events, source="onset", update_source="ma3_sync")
        assert applied == 1
        
        updated_layer = data_item.get_layer_by_name("LayerA")
        assert updated_layer is not None
        updated_events = updated_layer.get_events()
        assert len(updated_events) == 1
        assert updated_events[0].id == "new_1"


class TestLayerInfo:
    """Tests for LayerInfo dataclass."""
    
    def test_create_layer_info(self):
        """Test creating LayerInfo."""
        layer = LayerInfo(name="test_layer")
        assert layer.name == "test_layer"
        assert layer.height == 40
        assert layer.visible == True
        assert layer.locked == False
        assert layer.is_synced == False
    
    def test_layer_info_to_dict(self):
        """Test LayerInfo.to_dict()."""
        layer = LayerInfo(
            name="kicks",
            height=60,
            color="#ff0000",
            is_synced=True,
            ma3_track_coord="tc101_tg1_tr1"
        )
        d = layer.to_dict()
        assert d['name'] == "kicks"
        assert d['height'] == 60
        assert d['color'] == "#ff0000"
        assert d['is_synced'] == True
        assert d['ma3_track_coord'] == "tc101_tg1_tr1"
    
    def test_layer_info_from_dict(self):
        """Test LayerInfo.from_dict()."""
        d = {
            'name': 'snares',
            'height': 80,
            'is_synced': True,
            'show_manager_block_id': 'sm1'
        }
        layer = LayerInfo.from_dict(d)
        assert layer.name == 'snares'
        assert layer.height == 80
        assert layer.is_synced == True
        assert layer.show_manager_block_id == 'sm1'


class TestEventInfo:
    """Tests for EventInfo dataclass."""
    
    def test_create_event_info(self):
        """Test creating EventInfo."""
        event = EventInfo(id="evt1", time=1.5)
        assert event.id == "evt1"
        assert event.time == 1.5
        assert event.duration == 0.0
        assert event.classification == "event"
    
    def test_event_info_to_dict(self):
        """Test EventInfo.to_dict()."""
        event = EventInfo(
            id="evt1",
            time=2.5,
            duration=0.5,
            classification="kicks",
            metadata={"source": "ma3"}
        )
        d = event.to_dict()
        assert d['id'] == "evt1"
        assert d['time'] == 2.5
        assert d['duration'] == 0.5
        assert d['classification'] == "kicks"
        assert d['metadata']['source'] == "ma3"


class TestEditorAPILayerOperations:
    """Tests for EditorAPI layer operations."""
    
    def test_get_layers_empty(self, editor_api, mock_facade):
        """Test getting layers when none exist."""
        layers = editor_api.get_layers()
        assert layers == []
    
    def test_get_layers_with_data(self, editor_api, mock_facade):
        """Test getting layers with data."""
        mock_facade._layer_state = {
            'layers': [
                {'name': 'layer1', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'},
                {'name': 'layer2', 'height': 60, 'group_id': 'group-1', 'group_name': 'Group 1'}
            ]
        }
        
        layers = editor_api.get_layers()
        assert len(layers) == 2
        assert layers[0].name == 'layer1'
        assert layers[1].name == 'layer2'
    
    def test_get_layer_found(self, editor_api, mock_facade):
        """Test getting a specific layer."""
        mock_facade._layer_state = {
            'layers': [{'name': 'kicks', 'height': 60, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        layer = editor_api.get_layer('kicks')
        assert layer is not None
        assert layer.name == 'kicks'
    
    def test_get_layer_not_found(self, editor_api, mock_facade):
        """Test getting a non-existent layer."""
        layer = editor_api.get_layer('nonexistent')
        assert layer is None
    
    def test_layer_exists_true(self, editor_api, mock_facade):
        """Test layer_exists returns True for existing layer."""
        mock_facade._layer_state = {
            'layers': [{'name': 'kicks', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        assert editor_api.layer_exists('kicks') == True
    
    def test_layer_exists_false(self, editor_api, mock_facade):
        """Test layer_exists returns False for non-existing layer."""
        assert editor_api.layer_exists('nonexistent') == False
    
    def test_get_synced_layers(self, editor_api, mock_facade):
        """Test getting only synced layers."""
        mock_facade._layer_state = {
            'layers': [
                {'name': 'kicks', 'is_synced': True, 'group_id': 'group-1', 'group_name': 'Group 1'},
                {'name': 'snares', 'is_synced': False, 'group_id': 'group-1', 'group_name': 'Group 1'},
                {'name': 'hats', 'is_synced': True, 'group_id': 'group-1', 'group_name': 'Group 1'}
            ]
        }
        
        synced = editor_api.get_synced_layers()
        assert len(synced) == 2
        assert all(layer.is_synced for layer in synced)


class TestEditorAPISignalEmission:
    """Tests for EditorAPI signal emission."""
    
    def test_layer_created_emits_local_signal(self, mock_facade, sync_service):
        """Test that create_layer emits local signal."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        received = []
        api.layer_created.connect(lambda name: received.append(name))
        
        # Set up layer state that will be returned after creation
        layer_state = {'layers': [{'name': 'test_layer', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]}
        
        # Mock command execution - handles both Create and GetLayers commands
        def execute_handler(cmd):
            from src.application.commands.editor_commands import EditorCreateLayerCommand, EditorGetLayersCommand
            if isinstance(cmd, EditorCreateLayerCommand):
                cmd._created_layer_name = "test_layer"
                mock_facade._layer_state = layer_state
            elif isinstance(cmd, EditorGetLayersCommand):
                cmd.layers = mock_facade._layer_state.get('layers', [])
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        api.create_layer("test_layer")
        
        assert len(received) == 1
        assert received[0] == "test_layer"
    
    def test_layer_created_emits_sync_event(self, mock_facade, sync_service):
        """Test that create_layer emits SyncSubscriptionService event."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        received = []
        sync_service.layer_changed.connect(lambda e: received.append(e))
        
        # Set up layer state
        layer_state = {'layers': [{'name': 'kicks', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]}
        
        # Mock command execution - handles both Create and GetLayers commands
        def execute_handler(cmd):
            from src.application.commands.editor_commands import EditorCreateLayerCommand, EditorGetLayersCommand
            if isinstance(cmd, EditorCreateLayerCommand):
                cmd._created_layer_name = "kicks"
                mock_facade._layer_state = layer_state
            elif isinstance(cmd, EditorGetLayersCommand):
                cmd.layers = mock_facade._layer_state.get('layers', [])
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        api.create_layer("kicks")
        
        assert len(received) == 1
        assert received[0].source == SourceType.EDITOR
        assert received[0].change_type == ChangeType.ADDED
        assert received[0].layer_id == "kicks"


class TestEditorAPIEventOperations:
    """Tests for EditorAPI event operations."""
    
    def test_add_event_single(self, mock_facade, sync_service):
        """Test add_event for single event (core method)."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = MagicMock()
        
        result = api.add_event(
            time=1.5,
            classification="kicks",
            duration=0.1,
            metadata={"velocity": 127}
        )
        
        assert result is True
        mock_facade.command_bus.execute.assert_called_once()
    
    def test_add_events_returns_count(self, editor_api, mock_facade):
        """Test add_events returns correct count."""
        events = [
            {"time": 1.0, "classification": "kicks"},
            {"time": 2.0, "classification": "kicks"},
            {"time": 3.0, "classification": "snares"},
        ]
        
        count = editor_api.add_events(events)
        assert count == 3
    
    def test_add_events_empty_list(self, editor_api):
        """Test add_events with empty list."""
        count = editor_api.add_events([])
        assert count == 0
    
    def test_add_events_emits_signal_per_layer(self, mock_facade, sync_service):
        """Test add_events emits signal for each layer."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        received = []
        api.events_added.connect(lambda layer, count: received.append((layer, count)))
        
        events = [
            {"time": 1.0, "classification": "kicks"},
            {"time": 2.0, "classification": "kicks"},
            {"time": 3.0, "classification": "snares"},
        ]
        
        api.add_events(events)
        
        # Should emit two signals: one for kicks (2 events), one for snares (1 event)
        assert len(received) == 2
        # Find kicks and snares in received (order may vary)
        kicks_entry = next((r for r in received if r[0] == "kicks"), None)
        snares_entry = next((r for r in received if r[0] == "snares"), None)
        assert kicks_entry == ("kicks", 2)
        assert snares_entry == ("snares", 1)


class TestEditorAPIError:
    """Tests for EditorAPIError."""
    
    def test_error_message(self):
        """Test error message."""
        error = EditorAPIError("Layer not found")
        assert str(error) == "Layer not found"
    
    def test_delete_layer_raises_on_not_found(self, editor_api, mock_facade):
        """Test delete_layer raises error when layer not found."""
        with pytest.raises(EditorAPIError, match="not found"):
            editor_api.delete_layer("nonexistent")


class TestCreateEditorApiFactory:
    """Tests for create_editor_api factory function."""
    
    def test_factory_creates_api(self, mock_facade):
        """Test factory function creates EditorAPI."""
        api = create_editor_api(mock_facade, "test_block")
        assert isinstance(api, EditorAPI)
        assert api.block_id == "test_block"
    
    def test_factory_with_sync_service(self, mock_facade, sync_service):
        """Test factory function with sync service."""
        api = create_editor_api(mock_facade, "test_block", sync_service)
        assert api._sync_service == sync_service


class TestEditorAPISetSyncService:
    """Tests for set_sync_service method."""
    
    def test_set_sync_service(self, mock_facade):
        """Test setting sync service after creation."""
        api = EditorAPI(mock_facade, "test_block", None)
        assert api._sync_service is None
        
        sync_service = SyncSubscriptionService()
        api.set_sync_service(sync_service)
        
        assert api._sync_service == sync_service


class TestEditorAPIGetBlockInfo:
    """Tests for get_block_info method."""
    
    def test_get_block_info(self, editor_api, mock_facade):
        """Test getting block info."""
        info = editor_api.get_block_info()
        
        assert info['id'] == "test_block"
        assert info['name'] == "Test Block"
        assert info['type'] == "Editor"


class TestEditorAPILayerConvenienceMethods:
    """Tests for layer convenience methods."""
    
    def test_rename_layer(self, mock_facade, sync_service):
        """Test rename_layer delegates to update_layer."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        # Set up state
        mock_facade._layer_state = {
            'layers': [{'name': 'old_name', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        # Mock command execution
        def execute_handler(cmd):
            from src.application.commands.editor_commands import EditorUpdateLayerCommand, EditorGetLayersCommand
            if isinstance(cmd, EditorUpdateLayerCommand):
                # Simulate rename
                mock_facade._layer_state = {
                    'layers': [{'name': 'new_name', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]
                }
            elif isinstance(cmd, EditorGetLayersCommand):
                cmd.layers = mock_facade._layer_state.get('layers', [])
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        result = api.rename_layer('old_name', 'new_name')
        assert result.name == 'new_name'
    
    def test_set_layer_visibility(self, editor_api, mock_facade):
        """Test set_layer_visibility."""
        mock_facade._layer_state = {
            'layers': [{'name': 'layer1', 'visible': True, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        result = editor_api.set_layer_visibility('layer1', False)
        # The method calls update_layer which should update state
        # In real usage, the command would update visible to False
    
    def test_set_layer_locked(self, editor_api, mock_facade):
        """Test set_layer_locked."""
        mock_facade._layer_state = {
            'layers': [{'name': 'layer1', 'locked': False, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        # Just verify the method exists and can be called
        result = editor_api.set_layer_locked('layer1', True)
    
    def test_set_layer_color(self, editor_api, mock_facade):
        """Test set_layer_color."""
        mock_facade._layer_state = {
            'layers': [{'name': 'layer1', 'color': None, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        result = editor_api.set_layer_color('layer1', '#ff0000')
    
    def test_set_layer_height(self, editor_api, mock_facade):
        """Test set_layer_height."""
        mock_facade._layer_state = {
            'layers': [{'name': 'layer1', 'height': 40, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        result = editor_api.set_layer_height('layer1', 80)
    
    def test_mark_layer_synced(self, editor_api, mock_facade):
        """Test mark_layer_synced."""
        mock_facade._layer_state = {
            'layers': [{'name': 'layer1', 'is_synced': False, 'group_id': 'group-1', 'group_name': 'Group 1'}]
        }
        
        result = editor_api.mark_layer_synced('layer1', 'sm1', 'tc101_tg1_tr1')
    
    def test_unmark_layer_synced(self, editor_api, mock_facade):
        """Test unmark_layer_synced."""
        mock_facade._layer_state = {
            'layers': [
                {
                    'name': 'layer1',
                    'is_synced': True,
                    'show_manager_block_id': 'sm1',
                    'group_id': 'group-1',
                    'group_name': 'Group 1',
                }
            ]
        }
        
        result = editor_api.unmark_layer_synced('layer1')
    
    def test_get_layer_names(self, editor_api, mock_facade):
        """Test get_layer_names."""
        mock_facade._layer_state = {
            'layers': [
                {'name': 'kicks', 'group_id': 'group-1', 'group_name': 'Group 1'},
                {'name': 'snares', 'group_id': 'group-1', 'group_name': 'Group 1'},
                {'name': 'hats', 'group_id': 'group-1', 'group_name': 'Group 1'},
            ]
        }
        
        names = editor_api.get_layer_names()
        assert names == ['kicks', 'snares', 'hats']


class TestEditorAPIEventConvenienceMethods:
    """Tests for event convenience methods."""
    
    def test_get_events_in_layer(self, editor_api, mock_facade):
        """Test get_events_in_layer."""
        # This just wraps get_events with layer_name filter
        events = editor_api.get_events_in_layer('kicks')
        assert isinstance(events, list)
    
    def test_get_data_item_ids_empty(self, editor_api, mock_facade):
        """Test get_data_item_ids with no items."""
        mock_facade.data_item_repo.list_by_block = MagicMock(return_value=[])
        
        ids = editor_api.get_data_item_ids()
        assert ids == []
    
    def test_delete_events_empty_list(self, editor_api):
        """Test delete_events with empty list."""
        count = editor_api.delete_events([], 'data_item_1')
        assert count == 0
    
    def test_delete_events_by_index(self, mock_facade, sync_service):
        """Test delete_events_by_index with index tuples."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        # Mock the command execution
        def execute_handler(cmd):
            pass  # Simulate successful deletion
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        deletions = [('data_item_1', 0), ('data_item_1', 2)]
        result = api.delete_events_by_index(deletions, 'kicks')
        assert result == 2
    
    def test_delete_events_by_index_empty(self, editor_api):
        """Test delete_events_by_index with empty list."""
        result = editor_api.delete_events_by_index([])
        assert result == 0
    
    def test_get_event_found(self, mock_facade, sync_service):
        """Test get_event when event exists."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        # Mock get_events to return test data
        mock_facade._events = [
            {'id': 'evt1', 'time': 1.0, 'classification': 'kicks'},
            {'id': 'evt2', 'time': 2.0, 'classification': 'snares'}
        ]
        
        def execute_handler(cmd):
            from src.application.commands.editor_commands import EditorGetEventsCommand
            if isinstance(cmd, EditorGetEventsCommand):
                cmd.events = mock_facade._events
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        event = api.get_event('evt1')
        assert event is not None
        assert event.id == 'evt1'
        assert event.time == 1.0
    
    def test_get_event_not_found(self, mock_facade, sync_service):
        """Test get_event when event doesn't exist."""
        api = EditorAPI(mock_facade, "test_block", sync_service)
        
        mock_facade._events = []
        
        def execute_handler(cmd):
            from src.application.commands.editor_commands import EditorGetEventsCommand
            if isinstance(cmd, EditorGetEventsCommand):
                cmd.events = mock_facade._events
        
        mock_facade.command_bus = MagicMock()
        mock_facade.command_bus.execute = execute_handler
        
        event = api.get_event('nonexistent')
        assert event is None
    
    def test_move_events_empty_list(self, editor_api):
        """Test move_events with empty list."""
        count = editor_api.move_events([], 'data_item_1', 1.0)
        assert count == 0
