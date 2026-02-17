"""
Timeline Data Types
====================

Public data contracts for the TimelineWidget.

These types define the input/output interface for the widget.
Consumers use these types to communicate with the timeline -
they don't need to know about internal representations.

All types are dataclasses for easy serialization and comparison.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto

# Minimum event duration - all events must have a duration >= this value
# This ensures all events have valid start and end times
MIN_EVENT_DURATION = 0.001  # 1ms minimum

# =============================================================================
# Input Types (Data you pass to the widget)
# =============================================================================

@dataclass
class TimelineEvent:
    """
    Input/output data structure for timeline events.
    
    This is the primary format for loading and retrieving events.
    
    ALL events have start_time and end_time. There is no distinction
    between "one-shot" and "clip" events at the data level - only
    visual rendering differs.
    
    Attributes:
        id: Unique identifier for the event
        time: Start time in seconds
        duration: Duration in seconds (must be >= MIN_EVENT_DURATION)
        end_time: End time in seconds (calculated from time + duration, or explicitly set)
        classification: Semantic type/category (e.g., "kick", "snare", "note")
                       This is metadata - it does NOT determine which layer
                       the event appears on.
        layer_id: Which layer to place the event on. If None, the widget
                  will auto-assign based on classification.
        audio_id: Optional audio source ID (for events with waveforms)
        audio_name: Optional audio source name (fallback for waveform lookup)
        render_as_marker: Visual property - if True, renders as marker/diamond instead of clip/range
        user_data: Custom user/application metadata (pass-through, not used by widget)
    
    Example:
        event = TimelineEvent(
            id="event_001",
            time=1.5,
            duration=0.25,
            classification="kick",
            audio_id="abc-123",  # For waveform display
            render_as_marker=False,  # Render as clip/range
            user_data={"velocity": 100}
        )
    """
    id: str
    time: float
    duration: float = MIN_EVENT_DURATION
    classification: str = "Event"
    layer_id: Optional[str] = None
    
    # Visual rendering property (purely visual, not data-level distinction)
    render_as_marker: bool = False
    
    # Audio source (for events with waveforms)
    audio_id: Optional[str] = None
    audio_name: Optional[str] = None
    
    # User/application metadata (pass-through)
    user_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event data."""
        if self.time < 0:
            raise ValueError(f"Event time cannot be negative: {self.time}")
        if self.duration < MIN_EVENT_DURATION:
            raise ValueError(
                f"Event duration must be >= {MIN_EVENT_DURATION}s (got {self.duration}s). "
                f"All events must have valid start and end times."
            )
    
    @property
    def end_time(self) -> float:
        """End time of the event (start + duration). Always valid."""
        return self.time + self.duration
    
    @property
    def is_marker(self) -> bool:
        """
        DEPRECATED: Use render_as_marker instead.
        This property is kept for backward compatibility but should not be used.
        All events have duration >= MIN_EVENT_DURATION.
        """
        return self.render_as_marker
    
    @property
    def is_clip_event(self) -> bool:
        """Whether this event has audio source (for waveform display)."""
        return bool(self.audio_id or self.audio_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'id': self.id,
            'time': self.time,
            'duration': self.duration,
            'classification': self.classification,
            'layer_id': self.layer_id,
            'render_as_marker': self.render_as_marker,
            'user_data': self.user_data.copy(),
        }
        # Only include audio fields if set (backward compatibility)
        if self.audio_id:
            result['audio_id'] = self.audio_id
        if self.audio_name:
            result['audio_name'] = self.audio_name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimelineEvent':
        """Create from dictionary."""
        # Handle legacy 'metadata' field as 'user_data'
        user_data = data.get('user_data', data.get('metadata', {}))
        
        # Handle legacy duration=0 by converting to minimum duration and setting render_as_marker
        duration = data.get('duration', MIN_EVENT_DURATION)
        render_as_marker = data.get('render_as_marker', False)
        
        # Normalize duration to minimum for valid TimelineEvent
        if duration < MIN_EVENT_DURATION:
            duration = MIN_EVENT_DURATION
            # Do NOT override render_as_marker -- respect the stored value.
        
        return cls(
            id=data['id'],
            time=data['time'],
            duration=duration,
            classification=data.get('classification', 'Event'),
            layer_id=data.get('layer_id'),
            render_as_marker=render_as_marker,
            audio_id=data.get('audio_id'),
            audio_name=data.get('audio_name'),
            user_data=user_data,
        )
    
    @classmethod
    def from_event(
        cls,
        event: 'Event',
        layer_id: Optional[str] = None,
        audio_id: Optional[str] = None,
        audio_name: Optional[str] = None,
        user_data: Optional[Dict[str, Any]] = None
    ) -> 'TimelineEvent':
        """
        Convert domain Event to TimelineEvent with proper normalization.
        
        SINGLE SOURCE OF TRUTH: This method is the authoritative conversion
        from domain Event objects (from database) to UI TimelineEvent objects.
        All Event -> TimelineEvent conversions should use this method.
        
        Args:
            event: Domain Event object (from EventDataItem/EventLayer)
            layer_id: Optional layer ID for timeline placement
            audio_id: Optional audio ID (extracted from metadata if not provided)
            audio_name: Optional audio name (extracted from metadata if not provided)
            user_data: Optional additional user data (merged with event.metadata)
            
        Returns:
            TimelineEvent with normalized duration and proper field mapping
        """
        # Import Event type for type hints (avoid circular import)
        from src.shared.domain.entities.event_data_item import Event
        
        # Extract audio fields from metadata if not provided
        if audio_id is None and event.metadata:
            audio_id = event.metadata.get('audio_id')
        if audio_name is None and event.metadata:
            audio_name = event.metadata.get('audio_name')
        
        # SINGLE SOURCE OF TRUTH: Start with event.metadata from database
        # Database metadata is authoritative - user_data only adds UI tracking fields
        merged_user_data = dict(event.metadata) if event.metadata else {}
        if user_data:
            # Merge user_data (UI tracking fields) into database metadata
            # Database fields take precedence - user_data only adds missing UI fields
            merged_user_data.update(user_data)
        
        # Handle legacy duration=0 by converting to minimum duration
        # TimelineEvent requires duration >= MIN_EVENT_DURATION
        duration = event.duration
        # SINGLE SOURCE OF TRUTH: render_as_marker comes from database metadata
        render_as_marker = merged_user_data.get('render_as_marker', False)
        
        
        if duration < MIN_EVENT_DURATION:
            # Normalize duration to minimum for valid TimelineEvent
            duration = MIN_EVENT_DURATION
            # Do NOT override render_as_marker here -- the database value
            # (read from event.metadata above) is the single source of truth.
            # New MA3 events get render_as_marker=True at creation time;
            # user overrides are respected on reload.
        
        return cls(
            id=event.id,
            time=event.time,
            duration=duration,
            classification=event.classification,
            layer_id=layer_id,
            render_as_marker=render_as_marker,
            audio_id=audio_id,
            audio_name=audio_name,
            user_data=merged_user_data,
        )

@dataclass
class TimelineLayer:
    """
    Layer configuration for the timeline.
    
    Layers are independent containers that can hold any events.
    Event classification does NOT determine layer - layers are
    explicit user-controlled constructs.
    
    Attributes:
        id: Unique identifier for the layer
        name: Display name (shown in layer label)
        index: Visual order (0 = top of timeline)
        height: Layer height in pixels
        color: Layer color (hex string like "#4682B4" or None for auto)
        locked: Whether events on this layer can be edited
        visible: Whether the layer is visible
        collapsed: Whether the layer is collapsed (minimized height)
        group_id: Optional EventDataItem source ID for visual grouping
        group_name: Optional EventDataItem name for group header display
        group_index: Optional position within the group (0 = first in group)
        is_synced: Whether this layer is synced with a ShowManager/MA3 track
        show_manager_block_id: Optional ShowManager block ID if synced
        ma3_track_coord: Optional MA3 track coordinate (e.g., "tc101_tg1_tr1") if synced
        derived_from_ma3: Whether this layer was previously synced to MA3 and then detached
    
    Example:
        layer = TimelineLayer(
            id="layer_drums",
            name="Drums",
            color="#E74C3C",
            height=50.0
        )
    """
    id: str
    name: str
    index: int = 0
    height: float = 40.0
    color: Optional[str] = None
    locked: bool = False
    visible: bool = True
    collapsed: bool = False
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    group_index: Optional[int] = None
    is_synced: bool = False
    show_manager_block_id: Optional[str] = None
    ma3_track_coord: Optional[str] = None
    derived_from_ma3: bool = False
    sync_connection_state: str = "none"  # "active", "disconnected", "derived", "none" -- set by SSM, read by UI
    
    def __post_init__(self):
        """Validate layer data."""
        if self.height < 20:
            self.height = 20.0
        if self.height > 200:
            self.height = 200.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'id': self.id,
            'name': self.name,
            'index': self.index,
            'height': self.height,
            'color': self.color,
            'locked': self.locked,
            'visible': self.visible,
            'collapsed': self.collapsed,
        }
        # Only include grouping fields if they're set (backward compatibility)
        if self.group_id is not None:
            result['group_id'] = self.group_id
        if self.group_name is not None:
            result['group_name'] = self.group_name
        if self.group_index is not None:
            result['group_index'] = self.group_index
        # Include sync fields if synced
        if self.is_synced:
            result['is_synced'] = True
            if self.show_manager_block_id:
                result['show_manager_block_id'] = self.show_manager_block_id
            if self.ma3_track_coord:
                result['ma3_track_coord'] = self.ma3_track_coord
        # Include derived_from_ma3 if set (backward compatible)
        if self.derived_from_ma3:
            result['derived_from_ma3'] = True
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimelineLayer':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            index=data.get('index', 0),
            height=data.get('height', 40.0),
            color=data.get('color'),
            locked=data.get('locked', False),
            visible=data.get('visible', True),
            collapsed=data.get('collapsed', False),
            group_id=data.get('group_id'),
            group_name=data.get('group_name'),
            group_index=data.get('group_index'),
            is_synced=data.get('is_synced', False),
            show_manager_block_id=data.get('show_manager_block_id'),
            ma3_track_coord=data.get('ma3_track_coord'),
            derived_from_ma3=data.get('derived_from_ma3', False),
        )

# =============================================================================
# Output Types (Data the widget emits via signals)
# =============================================================================

@dataclass
class EventMoveResult:
    """
    Result data emitted when events are moved.
    
    Emitted via TimelineWidget.events_moved signal.
    Contains before/after state for undo support.
    
    Note: Classification does NOT change when moving between layers.
    """
    event_id: str
    old_time: float
    new_time: float
    old_layer_id: str
    new_layer_id: str
    
    @property
    def time_changed(self) -> bool:
        """Whether the time position changed."""
        return self.old_time != self.new_time
    
    @property
    def layer_changed(self) -> bool:
        """Whether the layer changed."""
        return self.old_layer_id != self.new_layer_id
    
    @property
    def changed(self) -> bool:
        """Whether anything changed."""
        return self.time_changed or self.layer_changed

@dataclass
class EventResizeResult:
    """
    Result data emitted when events are resized.
    
    Emitted via TimelineWidget.events_resized signal.
    Resize can change both start time (left edge) and duration (right edge).
    """
    event_id: str
    old_time: float
    new_time: float
    old_duration: float
    new_duration: float
    
    @property
    def start_changed(self) -> bool:
        """Whether the start time changed."""
        return self.old_time != self.new_time
    
    @property
    def duration_changed(self) -> bool:
        """Whether the duration changed."""
        return self.old_duration != self.new_duration

@dataclass
class EventCreateResult:
    """
    Result data emitted when a new event is created.
    
    Emitted via TimelineWidget.event_created signal.
    The widget does NOT auto-generate IDs - the consumer must
    create the event with an ID and add it back.
    """
    time: float
    duration: float
    layer_id: str
    classification: str = "Event"

@dataclass
class EventDeleteResult:
    """
    Result data emitted when events are deleted.
    
    Emitted via TimelineWidget.event_deleted signal.
    Contains the event data for undo support.
    """
    event_id: str
    event_data: Optional[TimelineEvent] = None

@dataclass
class EventSliceResult:
    """
    Result data emitted when an event is sliced/split.
    
    Emitted via TimelineWidget.event_sliced signal.
    Contains the original event and the two new events that result from the slice.
    """
    original_event_id: str
    original_event_data: TimelineEvent
    slice_time: float
    first_event_data: TimelineEvent  # From original start to slice_time
    second_event_data: TimelineEvent  # From slice_time to original end

# =============================================================================
# State Types (Internal state representations)
# =============================================================================

class DragState(Enum):
    """State machine for drag operations."""
    IDLE = auto()
    PENDING = auto()      # Mouse down, waiting for movement threshold
    DRAGGING = auto()     # Active drag in progress
    COMMITTING = auto()   # Releasing, about to commit changes

class EditHandle(Enum):
    """Which part of an event is being edited."""
    NONE = auto()
    MOVE = auto()
    RESIZE_LEFT = auto()
    RESIZE_RIGHT = auto()

@dataclass
class DragContext:
    """
    Context for an active drag operation.
    
    Captures the initial state of all items being dragged.
    Used for calculating relative movements and for undo.
    """
    primary_event_id: str
    primary_start_time: float
    primary_start_layer_id: str
    start_mouse_pos: tuple  # (x, y) in scene coordinates
    
    # All items being dragged with their relative offsets
    # Format: {event_id: (time_offset, layer_offset)}
    items: Dict[str, tuple] = field(default_factory=dict)
    
    # Original states for all items (for undo/cancel)
    # Format: {event_id: (time, layer_id)}
    original_states: Dict[str, tuple] = field(default_factory=dict)

# =============================================================================
# Selection Types
# =============================================================================

@dataclass
class SelectionState:
    """
    Current selection state.
    
    Tracks which events are selected and the anchor for range selection.
    """
    selected_ids: List[str] = field(default_factory=list)
    anchor_id: Optional[str] = None  # For shift+click range selection
    
    def is_selected(self, event_id: str) -> bool:
        """Check if an event is selected."""
        return event_id in self.selected_ids
    
    def select(self, event_id: str, clear_others: bool = True):
        """Select an event."""
        if clear_others:
            self.selected_ids = [event_id]
        elif event_id not in self.selected_ids:
            self.selected_ids.append(event_id)
        self.anchor_id = event_id
    
    def deselect(self, event_id: str):
        """Deselect an event."""
        if event_id in self.selected_ids:
            self.selected_ids.remove(event_id)
    
    def clear(self):
        """Clear all selection."""
        self.selected_ids.clear()
        self.anchor_id = None
    
    def select_all(self, event_ids: List[str]):
        """Select all given events."""
        self.selected_ids = list(event_ids)
        self.anchor_id = event_ids[0] if event_ids else None

