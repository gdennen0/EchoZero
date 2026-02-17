"""
Tests for the Progress Tracking Pattern

Tests the centralized progress tracking system:
- ProgressContext (context manager API)
- ProgressEventStore (backend)
- ProgressState/ProgressLevel (models)
"""
import pytest
from datetime import datetime
import time

from src.application.services import (
    ProgressContext,
    ProgressEventStore,
    ProgressState,
    ProgressLevel,
    ProgressStatus,
    get_progress_context,
    get_progress_store,
    reset_progress_store,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the global progress store before each test"""
    reset_progress_store()
    yield
    reset_progress_store()


class TestProgressModels:
    """Tests for ProgressState and ProgressLevel models"""
    
    def test_progress_level_creation(self):
        """Test creating a progress level"""
        level = ProgressLevel(
            level_id="test_1",
            level_type="item",
            name="Test Item"
        )
        
        assert level.level_id == "test_1"
        assert level.level_type == "item"
        assert level.name == "Test Item"
        assert level.status == ProgressStatus.PENDING
        assert level.current == 0
        assert level.total == 0
        assert level.percentage == 0.0
    
    def test_progress_level_update(self):
        """Test updating progress level"""
        level = ProgressLevel(
            level_id="test_1",
            level_type="item",
            name="Test Item"
        )
        
        level.update(current=5, total=10, message="Halfway")
        
        assert level.current == 5
        assert level.total == 10
        assert level.percentage == 50.0
        assert level.message == "Halfway"
    
    def test_progress_level_increment(self):
        """Test incrementing progress"""
        level = ProgressLevel(
            level_id="test_1",
            level_type="item",
            name="Test Item",
            total=10
        )
        
        level.update(increment=1)
        assert level.current == 1
        
        level.update(increment=2)
        assert level.current == 3
    
    def test_progress_level_start_complete(self):
        """Test start and complete lifecycle"""
        level = ProgressLevel(
            level_id="test_1",
            level_type="item",
            name="Test Item",
            total=10
        )
        
        # Start
        level.start("Starting...")
        assert level.status == ProgressStatus.RUNNING
        assert level.started_at is not None
        assert level.message == "Starting..."
        
        # Complete
        level.complete("Done!")
        assert level.status == ProgressStatus.COMPLETED
        assert level.completed_at is not None
        assert level.percentage == 100.0
        assert level.message == "Done!"
    
    def test_progress_level_fail(self):
        """Test failure handling"""
        level = ProgressLevel(
            level_id="test_1",
            level_type="item",
            name="Test Item"
        )
        
        level.start()
        level.fail("Something went wrong", {"code": 500})
        
        assert level.status == ProgressStatus.FAILED
        assert level.error == "Something went wrong"
        assert level.error_details == {"code": 500}
        assert level.completed_at is not None
    
    def test_progress_state_add_levels(self):
        """Test adding levels to progress state"""
        state = ProgressState(
            operation_id="op_1",
            operation_type="test"
        )
        
        # Add root level
        level1 = state.add_level("level_1", "song", "Song 1")
        assert level1 is not None
        assert "level_1" in state.levels
        assert "level_1" in state.root_level_ids
        
        # Add child level
        level2 = state.add_level("level_2", "action", "Action 1", parent_id="level_1")
        assert level2 is not None
        assert "level_2" in state.levels
        assert "level_2" not in state.root_level_ids
        assert "level_2" in state.levels["level_1"].children
    
    def test_progress_state_overall_progress(self):
        """Test overall progress calculation"""
        state = ProgressState(
            operation_id="op_1",
            operation_type="test"
        )
        
        # Add 3 root levels
        for i in range(3):
            state.add_level(f"level_{i}", "item", f"Item {i}")
        
        # Complete 2
        state.levels["level_0"].status = ProgressStatus.COMPLETED
        state.levels["level_1"].status = ProgressStatus.COMPLETED
        
        overall = state.get_overall_progress()
        assert overall["total"] == 3
        assert overall["completed"] == 2
        assert overall["pending"] == 1
        assert overall["percentage"] == pytest.approx(66.67, abs=0.01)


class TestProgressEventStore:
    """Tests for ProgressEventStore"""
    
    def test_start_operation(self):
        """Test starting an operation"""
        store = ProgressEventStore()
        
        state = store.start_operation("test_op", "Test Operation")
        
        assert state.operation_id.startswith("test_op_")
        assert state.operation_type == "test_op"
        assert state.name == "Test Operation"
        assert state.status == ProgressStatus.RUNNING
    
    def test_get_state(self):
        """Test getting operation state"""
        store = ProgressEventStore()
        
        state = store.start_operation("test", "Test")
        retrieved = store.get_state(state.operation_id)
        
        assert retrieved is not None
        assert retrieved.operation_id == state.operation_id
    
    def test_add_and_update_levels(self):
        """Test adding and updating levels"""
        store = ProgressEventStore()
        
        state = store.start_operation("test", "Test")
        op_id = state.operation_id
        
        # Add level
        store.add_level(op_id, "level_1", "item", "Item 1", total=10)
        
        # Start level
        store.start_level(op_id, "level_1")
        
        # Update level
        store.update_level(op_id, "level_1", current=5, message="Halfway")
        
        # Verify
        state = store.get_state(op_id)
        level = state.get_level("level_1")
        assert level.status == ProgressStatus.RUNNING
        assert level.current == 5
        assert level.message == "Halfway"
    
    def test_complete_operation(self):
        """Test completing an operation"""
        store = ProgressEventStore()
        
        state = store.start_operation("test", "Test")
        op_id = state.operation_id
        
        store.complete_operation(op_id)
        
        # Should be removed from active
        assert store.get_state(op_id) is None
        
        # Should be in history
        history = store.get_history()
        assert len(history) == 1
        assert history[0].status == ProgressStatus.COMPLETED
    
    def test_fail_operation(self):
        """Test failing an operation"""
        store = ProgressEventStore()
        
        state = store.start_operation("test", "Test")
        op_id = state.operation_id
        
        store.complete_operation(op_id, error="Something failed")
        
        history = store.get_history()
        assert history[0].status == ProgressStatus.FAILED
        assert history[0].error == "Something failed"
    
    def test_callbacks(self):
        """Test progress callbacks"""
        store = ProgressEventStore()
        events = []
        
        def callback(event_type, state):
            events.append((event_type, state.name))
        
        store.add_callback(callback)
        
        state = store.start_operation("test", "Test Op")
        store.complete_operation(state.operation_id)
        
        assert ("started", "Test Op") in events
        assert ("completed", "Test Op") in events


class TestProgressContext:
    """Tests for ProgressContext (context manager API)"""
    
    def test_basic_operation(self):
        """Test basic operation tracking"""
        progress = ProgressContext()
        
        with progress.operation("test", "Test Operation") as op:
            assert op is not None
        
        history = progress.get_history()
        assert len(history) == 1
        assert history[0].status == ProgressStatus.COMPLETED
    
    def test_nested_levels(self):
        """Test nested level tracking"""
        progress = ProgressContext()
        
        with progress.operation("test", "Test") as op:
            with op.level("item", "1", "Item 1") as item:
                item.update(message="Processing")
        
        history = progress.get_history()
        assert len(history) == 1
        
        state = history[0]
        assert len(state.levels) == 1
    
    def test_exception_handling(self):
        """Test that exceptions mark operations as failed"""
        progress = ProgressContext()
        
        try:
            with progress.operation("test", "Test") as op:
                with op.level("item", "1", "Item") as item:
                    raise ValueError("Test error")
        except ValueError:
            pass
        
        history = progress.get_history()
        assert len(history) == 1
        assert history[0].status == ProgressStatus.FAILED
        assert "Test error" in history[0].error
    
    def test_convenience_methods(self):
        """Test convenience methods (song, action, block, subprocess)"""
        progress = ProgressContext()
        
        with progress.setlist_processing("setlist_1", "My Setlist") as op:
            with op.song("song_1", "Track 1.wav") as song:
                with song.action("0", "LoadAudio") as action:
                    with action.block("block_1", "LoadAudio1", "LoadAudio") as block:
                        with block.subprocess("load", "Loading file") as sub:
                            sub.update(current=50, total=100)
        
        history = progress.get_history()
        assert history[0].status == ProgressStatus.COMPLETED
    
    def test_update_and_increment(self):
        """Test update and increment methods"""
        progress = ProgressContext()
        
        with progress.operation("test", "Test") as op:
            with op.level("item", "1", "Item", total=5) as ctx:
                ctx.increment()
                ctx.increment(message="Step 2")
                ctx.update(current=5, message="Done")
        
        # Operation should complete successfully
        history = progress.get_history()
        assert history[0].status == ProgressStatus.COMPLETED


class TestGlobalProgressContext:
    """Tests for global progress context singleton"""
    
    def test_get_progress_context(self):
        """Test getting the global progress context"""
        ctx1 = get_progress_context()
        ctx2 = get_progress_context()
        
        # Should share the same store
        assert ctx1.store is ctx2.store
    
    def test_get_progress_store(self):
        """Test getting the global progress store"""
        store1 = get_progress_store()
        store2 = get_progress_store()
        
        assert store1 is store2

