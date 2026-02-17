"""
Tests for the StatusPublisher pattern.

Tests status publishing, subscriptions, and change detection.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.shared.application.status.status_publisher import (
    StatusPublisher,
    Status,
    StatusLevel,
    StatusSubscriber,
)


# =============================================================================
# StatusLevel Tests
# =============================================================================

class TestStatusLevel:
    """Tests for StatusLevel enum."""
    
    def test_all_levels_have_colors(self):
        """Test that all levels have colors defined."""
        for level in StatusLevel:
            assert level.color.startswith("#")
            assert len(level.color) == 7  # #RRGGBB format
    
    def test_terminal_status(self):
        """Test is_terminal property."""
        assert StatusLevel.SUCCESS.is_terminal is True
        assert StatusLevel.ERROR.is_terminal is True
        assert StatusLevel.PROCESSING.is_terminal is False
        assert StatusLevel.PENDING.is_terminal is False
    
    def test_active_status(self):
        """Test is_active property."""
        assert StatusLevel.PROCESSING.is_active is True
        assert StatusLevel.SUCCESS.is_active is False
        assert StatusLevel.ERROR.is_active is False


# =============================================================================
# Status Tests
# =============================================================================

class TestStatus:
    """Tests for Status dataclass."""
    
    def test_default_values(self):
        """Test Status has sensible defaults."""
        status = Status()
        assert status.level == StatusLevel.UNKNOWN
        assert status.message == ""
        assert status.details is None
        assert status.source == ""
        assert status.metadata == {}
        assert isinstance(status.timestamp, datetime)
    
    def test_equality_ignores_timestamp(self):
        """Test that equality ignores timestamp."""
        status1 = Status(level=StatusLevel.SUCCESS, message="Done")
        status2 = Status(level=StatusLevel.SUCCESS, message="Done")
        # Even with different timestamps, should be equal
        assert status1 == status2
    
    def test_inequality_on_level(self):
        """Test inequality when levels differ."""
        status1 = Status(level=StatusLevel.SUCCESS, message="Done")
        status2 = Status(level=StatusLevel.ERROR, message="Done")
        assert status1 != status2
    
    def test_inequality_on_message(self):
        """Test inequality when messages differ."""
        status1 = Status(level=StatusLevel.SUCCESS, message="Done")
        status2 = Status(level=StatusLevel.SUCCESS, message="Finished")
        assert status1 != status2
    
    def test_to_dict(self):
        """Test serialization to dict."""
        status = Status(
            level=StatusLevel.ERROR,
            message="Failed",
            details="Connection timeout",
            source="MyService",
            metadata={"retry_count": 3}
        )
        result = status.to_dict()
        
        assert result["level"] == "ERROR"
        assert result["message"] == "Failed"
        assert result["details"] == "Connection timeout"
        assert result["source"] == "MyService"
        assert result["metadata"]["retry_count"] == 3
        assert "timestamp" in result
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "level": "SUCCESS",
            "message": "Completed",
            "details": None,
            "source": "Worker",
            "timestamp": "2026-01-15T12:00:00",
            "metadata": {"count": 42}
        }
        status = Status.from_dict(data)
        
        assert status.level == StatusLevel.SUCCESS
        assert status.message == "Completed"
        assert status.source == "Worker"
        assert status.metadata["count"] == 42
    
    def test_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        original = Status(
            level=StatusLevel.WARNING,
            message="Slow response",
            details="Timeout exceeded",
            source="API",
            metadata={"latency_ms": 5000}
        )
        
        restored = Status.from_dict(original.to_dict())
        
        assert restored.level == original.level
        assert restored.message == original.message
        assert restored.details == original.details
        assert restored.source == original.source
        assert restored.metadata == original.metadata


# =============================================================================
# StatusPublisher Tests
# =============================================================================

class TestStatusPublisher:
    """Tests for StatusPublisher class."""
    
    def test_initial_status(self):
        """Test publisher starts with UNKNOWN status."""
        publisher = StatusPublisher(source_name="Test")
        assert publisher.status.level == StatusLevel.UNKNOWN
        assert publisher.status.source == "Test"
    
    def test_set_status(self):
        """Test setting status."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_status(StatusLevel.SUCCESS, "Done!")
        
        assert publisher.status.level == StatusLevel.SUCCESS
        assert publisher.status.message == "Done!"
    
    def test_set_status_with_details(self):
        """Test setting status with details."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_status(
            StatusLevel.ERROR,
            "Failed",
            details="Connection refused"
        )
        
        assert publisher.status.details == "Connection refused"
    
    def test_set_status_with_metadata(self):
        """Test setting status with metadata."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_status(
            StatusLevel.PROCESSING,
            "Working",
            metadata={"progress": 50}
        )
        
        assert publisher.status.metadata["progress"] == 50
    
    def test_previous_status(self):
        """Test previous status tracking."""
        publisher = StatusPublisher(source_name="Test")
        
        # Initially no previous status
        assert publisher.previous_status is None
        
        # Set first status
        publisher.set_status(StatusLevel.PROCESSING, "Working")
        assert publisher.previous_status is not None
        assert publisher.previous_status.level == StatusLevel.UNKNOWN
        
        # Set second status
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        assert publisher.previous_status.level == StatusLevel.PROCESSING
    
    def test_subscribe_and_notify(self):
        """Test subscription and notification."""
        publisher = StatusPublisher(source_name="Test")
        handler = MagicMock()
        
        publisher.subscribe(handler)
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        handler.assert_called_once()
        status_arg = handler.call_args[0][0]
        assert status_arg.level == StatusLevel.SUCCESS
        assert status_arg.message == "Done"
    
    def test_no_notify_on_same_status(self):
        """Test that same status doesn't trigger notification."""
        publisher = StatusPublisher(source_name="Test")
        handler = MagicMock()
        
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        publisher.subscribe(handler)
        
        # Set same status again
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        # Handler should NOT be called (status didn't change)
        handler.assert_not_called()
    
    def test_force_notify(self):
        """Test force_notify bypasses change detection."""
        publisher = StatusPublisher(source_name="Test")
        handler = MagicMock()
        
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        publisher.subscribe(handler)
        
        # Force notify even though status is same
        publisher.set_status(StatusLevel.SUCCESS, "Done", force_notify=True)
        
        handler.assert_called_once()
    
    def test_unsubscribe(self):
        """Test unsubscription."""
        publisher = StatusPublisher(source_name="Test")
        handler = MagicMock()
        
        publisher.subscribe(handler)
        publisher.unsubscribe(handler)
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        handler.assert_not_called()
    
    def test_clear_subscriptions(self):
        """Test clearing all subscriptions."""
        publisher = StatusPublisher(source_name="Test")
        handler1 = MagicMock()
        handler2 = MagicMock()
        
        publisher.subscribe(handler1)
        publisher.subscribe(handler2)
        publisher.clear_subscriptions()
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        handler1.assert_not_called()
        handler2.assert_not_called()
    
    def test_reset_status(self):
        """Test status reset."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        publisher.reset_status()
        
        assert publisher.status.level == StatusLevel.UNKNOWN
        assert publisher.status.message == ""
    
    def test_history_tracking(self):
        """Test status history when enabled."""
        publisher = StatusPublisher(source_name="Test", track_history=True)
        
        publisher.set_status(StatusLevel.PENDING, "Step 1")
        publisher.set_status(StatusLevel.PROCESSING, "Step 2")
        publisher.set_status(StatusLevel.SUCCESS, "Step 3")
        
        history = publisher.history
        assert len(history) == 3
        assert history[0].message == "Step 1"
        assert history[1].message == "Step 2"
        assert history[2].message == "Step 3"
    
    def test_history_limit(self):
        """Test history respects max_history limit."""
        publisher = StatusPublisher(source_name="Test", track_history=True, max_history=3)
        
        for i in range(5):
            publisher.set_status(StatusLevel.PROCESSING, f"Step {i}")
        
        history = publisher.history
        assert len(history) == 3
        # Should have kept the last 3
        assert history[0].message == "Step 2"
        assert history[2].message == "Step 4"
    
    def test_no_history_by_default(self):
        """Test history not tracked by default."""
        publisher = StatusPublisher(source_name="Test")
        
        publisher.set_status(StatusLevel.PROCESSING, "Step 1")
        publisher.set_status(StatusLevel.SUCCESS, "Step 2")
        
        assert publisher.history == []


class TestConvenienceMethods:
    """Tests for convenience status methods."""
    
    def test_set_idle(self):
        """Test set_idle convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_idle("Waiting")
        assert publisher.status.level == StatusLevel.IDLE
        assert publisher.status.message == "Waiting"
    
    def test_set_pending(self):
        """Test set_pending convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_pending("Need input")
        assert publisher.status.level == StatusLevel.PENDING
    
    def test_set_processing(self):
        """Test set_processing convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_processing("Working...")
        assert publisher.status.level == StatusLevel.PROCESSING
    
    def test_set_success(self):
        """Test set_success convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_success("Completed")
        assert publisher.status.level == StatusLevel.SUCCESS
    
    def test_set_warning(self):
        """Test set_warning convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_warning("Slow", details="Timeout exceeded")
        assert publisher.status.level == StatusLevel.WARNING
        assert publisher.status.details == "Timeout exceeded"
    
    def test_set_error(self):
        """Test set_error convenience method."""
        publisher = StatusPublisher(source_name="Test")
        publisher.set_error("Failed", details="Connection refused")
        assert publisher.status.level == StatusLevel.ERROR
        assert publisher.status.details == "Connection refused"


# =============================================================================
# StatusSubscriber Tests
# =============================================================================

class TestStatusSubscriber:
    """Tests for StatusSubscriber helper class."""
    
    def test_subscribe_to_multiple(self):
        """Test subscribing to multiple publishers."""
        pub1 = StatusPublisher(source_name="Pub1")
        pub2 = StatusPublisher(source_name="Pub2")
        
        handler = MagicMock()
        subscriber = StatusSubscriber()
        
        subscriber.subscribe_to(pub1, handler)
        subscriber.subscribe_to(pub2, handler)
        
        pub1.set_status(StatusLevel.SUCCESS, "From pub1")
        pub2.set_status(StatusLevel.SUCCESS, "From pub2")
        
        assert handler.call_count == 2
    
    def test_cleanup(self):
        """Test cleanup unsubscribes from all."""
        pub1 = StatusPublisher(source_name="Pub1")
        pub2 = StatusPublisher(source_name="Pub2")
        
        handler = MagicMock()
        subscriber = StatusSubscriber()
        
        subscriber.subscribe_to(pub1, handler)
        subscriber.subscribe_to(pub2, handler)
        subscriber.cleanup()
        
        # After cleanup, handlers should not be called
        pub1.set_status(StatusLevel.SUCCESS, "From pub1")
        pub2.set_status(StatusLevel.SUCCESS, "From pub2")
        
        handler.assert_not_called()
    
    def test_context_manager(self):
        """Test StatusSubscriber as context manager."""
        pub = StatusPublisher(source_name="Pub")
        handler = MagicMock()
        
        with StatusSubscriber() as subscriber:
            subscriber.subscribe_to(pub, handler)
            pub.set_status(StatusLevel.SUCCESS, "Inside context")
        
        # After context, handler should be unsubscribed
        pub.set_status(StatusLevel.ERROR, "Outside context")
        
        # Only called once (inside context)
        handler.assert_called_once()
    
    def test_unsubscribe_from_specific(self):
        """Test unsubscribing from a specific publisher."""
        pub1 = StatusPublisher(source_name="Pub1")
        pub2 = StatusPublisher(source_name="Pub2")
        
        handler = MagicMock()
        subscriber = StatusSubscriber()
        
        subscriber.subscribe_to(pub1, handler)
        subscriber.subscribe_to(pub2, handler)
        subscriber.unsubscribe_from(pub1, handler)
        
        pub1.set_status(StatusLevel.SUCCESS, "From pub1")  # Should NOT notify
        pub2.set_status(StatusLevel.SUCCESS, "From pub2")  # Should notify
        
        assert handler.call_count == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for StatusPublisher pattern."""
    
    def test_multiple_handlers(self):
        """Test multiple handlers receive notifications."""
        publisher = StatusPublisher(source_name="Test")
        
        results = []
        handler1 = lambda s: results.append(f"h1:{s.message}")
        handler2 = lambda s: results.append(f"h2:{s.message}")
        
        publisher.subscribe(handler1)
        publisher.subscribe(handler2)
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        assert "h1:Done" in results
        assert "h2:Done" in results
    
    def test_handler_error_doesnt_break_others(self):
        """Test that one handler's error doesn't affect others."""
        publisher = StatusPublisher(source_name="Test")
        
        results = []
        def bad_handler(s):
            raise ValueError("Oops!")
        
        def good_handler(s):
            results.append(s.message)
        
        publisher.subscribe(bad_handler)
        publisher.subscribe(good_handler)
        
        # Should not raise, and good_handler should still be called
        publisher.set_status(StatusLevel.SUCCESS, "Done")
        
        assert "Done" in results
    
    def test_subclass_usage(self):
        """Test using StatusPublisher as base class."""
        class MyProcessor(StatusPublisher):
            def __init__(self):
                super().__init__(source_name="MyProcessor")
            
            def process(self, data):
                self.set_processing(f"Processing {data}...")
                # Simulate work
                result = data.upper()
                self.set_success(f"Processed: {result}")
                return result
        
        processor = MyProcessor()
        statuses = []
        processor.subscribe(lambda s: statuses.append(s.level))
        
        processor.process("test")
        
        assert StatusLevel.PROCESSING in statuses
        assert StatusLevel.SUCCESS in statuses
