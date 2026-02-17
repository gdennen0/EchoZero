"""
Unit Tests for SyncSubscriptionService

Tests the signal-based subscription service for layer/event changes.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.features.show_manager.application.sync_subscription_service import (
    SyncSubscriptionService,
    ChangeType,
    SourceType,
    LayerChangeEvent,
    EventChangeEvent,
    Subscription,
    editor_layer_added,
    editor_layer_modified,
    editor_layer_deleted,
    editor_events_added,
    ma3_track_changed,
)


class TestSyncSubscriptionService:
    """Tests for SyncSubscriptionService."""
    
    @pytest.fixture
    def service(self):
        """Create a SyncSubscriptionService instance."""
        return SyncSubscriptionService()
    
    def test_init(self, service):
        """Test service initialization."""
        assert service is not None
        assert service._subscriptions == {}
        assert service._watchers == {}
    
    def test_subscribe_creates_subscription(self, service):
        """Test subscribing creates a subscription."""
        sub = service.subscribe(
            subscriber_id="test_subscriber",
            source=SourceType.EDITOR,
            layer_id="101_kicks"
        )
        
        assert sub is not None
        assert sub.subscriber_id == "test_subscriber"
        assert sub.source == SourceType.EDITOR
        assert sub.layer_id == "101_kicks"
    
    def test_subscribe_stores_in_dict(self, service):
        """Test subscription is stored in internal dict."""
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.EDITOR,
            layer_id="layer1"
        )
        
        assert "sub1" in service._subscriptions
        assert len(service._subscriptions["sub1"]) == 1
    
    def test_subscribe_multiple(self, service):
        """Test multiple subscriptions for same subscriber."""
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.EDITOR,
            layer_id="layer1"
        )
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.MA3,
            layer_id="tc101_tg1_tr1"
        )
        
        assert len(service._subscriptions["sub1"]) == 2
    
    def test_unsubscribe_removes_subscription(self, service):
        """Test unsubscribing removes the subscription."""
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.EDITOR,
            layer_id="layer1"
        )
        
        removed = service.unsubscribe("sub1")
        
        assert removed == 1
        assert "sub1" not in service._subscriptions
    
    def test_unsubscribe_by_source(self, service):
        """Test unsubscribing by source only."""
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.EDITOR,
            layer_id="layer1"
        )
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.MA3,
            layer_id="tc101_tg1_tr1"
        )
        
        removed = service.unsubscribe("sub1", source=SourceType.EDITOR)
        
        assert removed == 1
        assert len(service._subscriptions["sub1"]) == 1
        assert service._subscriptions["sub1"][0].source == SourceType.MA3
    
    def test_unsubscribe_all(self, service):
        """Test unsubscribe_all removes all subscriptions."""
        service.subscribe("sub1", SourceType.EDITOR, "layer1")
        service.subscribe("sub1", SourceType.MA3, "track1")
        service.subscribe("sub1", SourceType.EDITOR, "layer2")
        
        removed = service.unsubscribe_all("sub1")
        
        assert removed == 3
        assert "sub1" not in service._subscriptions
    
    def test_get_subscriptions(self, service):
        """Test getting subscriptions for a subscriber."""
        service.subscribe("sub1", SourceType.EDITOR, "layer1")
        service.subscribe("sub1", SourceType.MA3, "track1")
        
        subs = service.get_subscriptions("sub1")
        
        assert len(subs) == 2
    
    def test_get_subscriptions_empty(self, service):
        """Test getting subscriptions for unknown subscriber."""
        subs = service.get_subscriptions("unknown")
        
        assert subs == []
    
    def test_get_subscribers_for_layer_exact(self, service):
        """Test getting subscribers for a specific layer."""
        service.subscribe("sub1", SourceType.EDITOR, "layer1", "block1")
        service.subscribe("sub2", SourceType.EDITOR, "layer1", "block1")
        service.subscribe("sub3", SourceType.EDITOR, "layer2", "block1")
        
        subscribers = service.get_subscribers_for_layer(
            source=SourceType.EDITOR,
            layer_id="layer1",
            block_id="block1"
        )
        
        assert subscribers == {"sub1", "sub2"}
    
    def test_get_subscribers_with_wildcard(self, service):
        """Test subscribers with wildcard (None) layer_id."""
        # sub1 subscribes to all layers
        service.subscribe("sub1", SourceType.EDITOR, None, "block1")
        # sub2 subscribes to specific layer
        service.subscribe("sub2", SourceType.EDITOR, "layer1", "block1")
        
        subscribers = service.get_subscribers_for_layer(
            source=SourceType.EDITOR,
            layer_id="layer1",
            block_id="block1"
        )
        
        assert "sub1" in subscribers
        assert "sub2" in subscribers
    
    def test_emit_layer_change_emits_signal(self, service):
        """Test layer change emission."""
        received = []
        service.layer_changed.connect(lambda e: received.append(e))
        
        event = LayerChangeEvent(
            source=SourceType.EDITOR,
            change_type=ChangeType.ADDED,
            layer_id="layer1",
            block_id="block1"
        )
        service.emit_layer_change(event)
        
        assert len(received) == 1
        assert received[0].layer_id == "layer1"
    
    def test_emit_layer_change_calls_callback(self, service):
        """Test layer change calls subscriber callback."""
        callback = MagicMock()
        service.subscribe(
            subscriber_id="sub1",
            source=SourceType.EDITOR,
            layer_id="layer1",
            callback=callback
        )
        
        event = LayerChangeEvent(
            source=SourceType.EDITOR,
            change_type=ChangeType.ADDED,
            layer_id="layer1",
            block_id="block1"
        )
        service.emit_layer_change(event)
        
        callback.assert_called_once_with(event)
    
    def test_emit_events_change(self, service):
        """Test events change emission."""
        received = []
        service.events_changed.connect(lambda e: received.append(e))
        
        event = EventChangeEvent(
            source=SourceType.EDITOR,
            change_type=ChangeType.ADDED,
            layer_id="layer1",
            block_id="block1",
            event_ids=["ev1", "ev2"],
            events=[{"id": "ev1"}, {"id": "ev2"}]
        )
        service.emit_events_change(event)
        
        assert len(received) == 1
        assert len(received[0].event_ids) == 2
    
    def test_request_sync_emits_signal(self, service):
        """Test sync request emission."""
        received = []
        service.sync_requested.connect(lambda s, t, l: received.append((s, t, l)))
        
        service.request_sync("sub1", SourceType.EDITOR, "layer1")
        
        assert len(received) == 1
        assert received[0] == ("sub1", "editor", "layer1")
    
    def test_complete_sync_emits_signal(self, service):
        """Test sync completion emission."""
        received = []
        service.sync_completed.connect(lambda s, t, l, r: received.append((s, t, l, r)))
        
        service.complete_sync("sub1", SourceType.EDITOR, "layer1", True)
        
        assert len(received) == 1
        assert received[0] == ("sub1", "editor", "layer1", True)
    
    def test_connection_status_changed(self, service):
        """Test connection status change emission."""
        received = []
        service.connection_status_changed.connect(lambda b, c: received.append((b, c)))
        
        service.update_connection_status("block1", True)
        
        assert len(received) == 1
        assert received[0] == ("block1", True)
    
    def test_cleanup(self, service):
        """Test cleanup removes all subscriptions."""
        service.subscribe("sub1", SourceType.EDITOR, "layer1")
        service.subscribe("sub2", SourceType.MA3, "track1")
        
        service.cleanup()
        
        assert service._subscriptions == {}
        assert service._watchers == {}


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_editor_layer_added(self):
        """Test editor_layer_added creates correct event."""
        event = editor_layer_added("layer1", "block1", height=60)
        
        assert event.source == SourceType.EDITOR
        assert event.change_type == ChangeType.ADDED
        assert event.layer_id == "layer1"
        assert event.block_id == "block1"
        assert event.data["height"] == 60
    
    def test_editor_layer_modified(self):
        """Test editor_layer_modified creates correct event."""
        event = editor_layer_modified("layer1", "block1", name="New Name")
        
        assert event.source == SourceType.EDITOR
        assert event.change_type == ChangeType.MODIFIED
        assert event.data["name"] == "New Name"
    
    def test_editor_layer_deleted(self):
        """Test editor_layer_deleted creates correct event."""
        event = editor_layer_deleted("layer1", "block1")
        
        assert event.source == SourceType.EDITOR
        assert event.change_type == ChangeType.DELETED
        assert event.data == {}
    
    def test_editor_events_added(self):
        """Test editor_events_added creates correct event."""
        event = editor_events_added(
            "layer1", "block1",
            ["ev1", "ev2"],
            [{"id": "ev1"}, {"id": "ev2"}]
        )
        
        assert event.source == SourceType.EDITOR
        assert event.change_type == ChangeType.ADDED
        assert len(event.event_ids) == 2
        assert len(event.events) == 2
    
    def test_ma3_track_changed(self):
        """Test ma3_track_changed creates correct event."""
        event = ma3_track_changed("tc101_tg1_tr1", "showmanager1", event_count=15)
        
        assert event.source == SourceType.MA3
        assert event.change_type == ChangeType.MODIFIED
        assert event.layer_id == "tc101_tg1_tr1"
        assert event.data["event_count"] == 15


class TestLayerChangeEvent:
    """Tests for LayerChangeEvent dataclass."""
    
    def test_create_event(self):
        """Test creating a LayerChangeEvent."""
        event = LayerChangeEvent(
            source=SourceType.EDITOR,
            change_type=ChangeType.ADDED,
            layer_id="101_kicks",
            block_id="editor_block_1"
        )
        
        assert event.source == SourceType.EDITOR
        assert event.change_type == ChangeType.ADDED
        assert event.layer_id == "101_kicks"
        assert event.block_id == "editor_block_1"
        assert event.data == {}
    
    def test_event_with_data(self):
        """Test event with additional data."""
        event = LayerChangeEvent(
            source=SourceType.MA3,
            change_type=ChangeType.MODIFIED,
            layer_id="tc101_tg1_tr1",
            block_id="showmanager_1",
            data={"event_count": 10, "name": "Kicks"}
        )
        
        assert event.data["event_count"] == 10
        assert event.data["name"] == "Kicks"


class TestEventChangeEvent:
    """Tests for EventChangeEvent dataclass."""
    
    def test_create_event(self):
        """Test creating an EventChangeEvent."""
        event = EventChangeEvent(
            source=SourceType.EDITOR,
            change_type=ChangeType.ADDED,
            layer_id="101_kicks",
            block_id="editor_block_1",
            event_ids=["evt_001", "evt_002"],
            events=[
                {"id": "evt_001", "time": 1.5},
                {"id": "evt_002", "time": 3.0}
            ]
        )
        
        assert len(event.event_ids) == 2
        assert len(event.events) == 2
        assert event.events[0]["time"] == 1.5


class TestChangeType:
    """Tests for ChangeType enum."""
    
    def test_change_types(self):
        """Test all change types exist."""
        assert ChangeType.ADDED
        assert ChangeType.MODIFIED
        assert ChangeType.DELETED
        assert ChangeType.MOVED
        assert ChangeType.BATCH


class TestSourceType:
    """Tests for SourceType enum."""
    
    def test_source_types(self):
        """Test source types have correct values."""
        assert SourceType.EDITOR.value == "editor"
        assert SourceType.MA3.value == "ma3"
