"""
Domain Events

Events that represent significant occurrences in the domain.
Used for loose coupling between components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional


@dataclass
class DomainEvent:
    """Base class for all domain events"""
    name: ClassVar[str] = "DomainEvent"
    project_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


# Project Events
@dataclass
class ProjectInitialized(DomainEvent):
    name: ClassVar[str] = "ProjectInitialized"


@dataclass
class ProjectCreated(DomainEvent):
    name: ClassVar[str] = "ProjectCreated"


@dataclass
class ProjectLoaded(DomainEvent):
    name: ClassVar[str] = "project.loaded"


@dataclass
class ProjectUpdated(DomainEvent):
    name: ClassVar[str] = "ProjectUpdated"


@dataclass
class ProjectDeleted(DomainEvent):
    name: ClassVar[str] = "ProjectDeleted"


# Block Events
@dataclass
class BlockAdded(DomainEvent):
    name: ClassVar[str] = "BlockAdded"


@dataclass
class BlockUpdated(DomainEvent):
    name: ClassVar[str] = "BlockUpdated"


@dataclass
class EventVisualUpdate(DomainEvent):
    """
    Direct visual update for a single event (bypasses full reload).
    
    Published when MA3 sync moves an event. EditorPanel should handle this
    by directly updating the TimelineWidget visual, NOT by reloading all events.
    
    This is MUCH faster than BlockUpdated for move operations because:
    - No full EventDataItem reload
    - No layer recreation
    - Just one TimelineWidget.update_event() call
    
    Data fields:
        - block_id: Editor block ID
        - event_id: Event ID to update
        - new_time: New time in seconds
    """
    name: ClassVar[str] = "EventVisualUpdate"


@dataclass
class BlockRemoved(DomainEvent):
    name: ClassVar[str] = "BlockRemoved"


@dataclass
class BlockChanged(DomainEvent):
    """
    Unified event raised when a block changes in any way.
    
    Used to notify UI components and services that a block has changed.
    All block-changing operations should publish this event.
    
    Data fields:
        - block_id: ID of the block that changed
        - change_type: Type of change ("metadata", "data", "connection", "execution", etc.)
        - data: Additional context about the change
    """
    name: ClassVar[str] = "BlockChanged"


@dataclass
class StatusChanged(DomainEvent):
    """
    Event raised when a block's status actually changes.
    
    Published by BlockStatusService when status calculation results in a different
    status than previously cached. This ensures all status indicators (panels, nodes, etc.)
    update automatically when status changes.
    
    Data fields:
        - block_id: ID of the block whose status changed
        - status: The new BlockStatus object (serialized as dict)
        - previous_status: The previous BlockStatus (for comparison, optional)
    """
    name: ClassVar[str] = "StatusChanged"


@dataclass
class ShowManagerCheckRehookResolveRequested(DomainEvent):
    """
    Request ShowManager to check hooks, rehook missing, and reconcile divergences.

    Data fields:
        - show_manager_id: ShowManager block ID
    """
    name: ClassVar[str] = "ShowManagerCheckRehookResolveRequested"


@dataclass
class MA3OscOutbound(DomainEvent):
    """
    Outbound OSC/Lua command sent to MA3.
    
    Data fields:
        - ip: Target IP address
        - port: Target port
        - lua_code: Lua code string (without Lua wrapper)
        - osc_len: OSC packet byte length
        - success: Whether send succeeded
        - error: Optional error string on failure
    """
    name: ClassVar[str] = "MA3OscOutbound"


# Connection Events
@dataclass
class ConnectionsChanged(DomainEvent):
    name: ClassVar[str] = "ConnectionsChanged"


@dataclass
class ConnectionCreated(DomainEvent):
    name: ClassVar[str] = "ConnectionCreated"


@dataclass
class ConnectionRemoved(DomainEvent):
    name: ClassVar[str] = "ConnectionRemoved"


# Execution Events
@dataclass
class ExecutionStarted(DomainEvent):
    """Event raised when block execution starts"""
    name: ClassVar[str] = "ExecutionStarted"


@dataclass
class ExecutionProgress(DomainEvent):
    """Event raised during block execution to report progress"""
    name: ClassVar[str] = "ExecutionProgress"


@dataclass
class BlockExecuted(DomainEvent):
    """Event raised when a block executes successfully"""
    name: ClassVar[str] = "BlockExecuted"


@dataclass
class BlockExecutionFailed(DomainEvent):
    """Event raised when a block execution fails"""
    name: ClassVar[str] = "BlockExecutionFailed"


@dataclass
class ExecutionCompleted(DomainEvent):
    """Event raised when block execution completes"""
    name: ClassVar[str] = "ExecutionCompleted"


@dataclass
class SubprocessProgress(DomainEvent):
    """
    Event raised for subprocess progress (e.g., demucs, ffmpeg).
    
    Used for fine-grained progress reporting within a single block execution.
    """
    name: ClassVar[str] = "SubprocessProgress"


# Run Events
@dataclass
class RunProgress(DomainEvent):
    name: ClassVar[str] = "RunProgress"


@dataclass
class RunCompleted(DomainEvent):
    name: ClassVar[str] = "RunCompleted"


# Error Events
@dataclass
class ErrorOccurred(DomainEvent):
    name: ClassVar[str] = "ErrorOccurred"


# UI Events
@dataclass
class GraphPositionsUpdated(DomainEvent):
    name: ClassVar[str] = "GraphPositionsUpdated"


@dataclass
class UIStateChanged(DomainEvent):
    name: ClassVar[str] = "UIStateChanged"


# MA3 Communication Events
@dataclass
class MA3MessageReceived(DomainEvent):
    """
    Event raised when a message is received from grandMA3 console.
    
    Data fields:
        - object_type: Type of MA3 object (e.g., "sequence", "cue")
        - object_name: Name of the object
        - change_type: Type of change (e.g., "changed", "created", "deleted")
        - timestamp: Timestamp when change occurred
        - ma3_data: Additional data from MA3
    """
    name: ClassVar[str] = "MA3MessageReceived"


# Setlist Events
@dataclass
class SetlistProcessingStarted(DomainEvent):
    """
    Event raised when setlist processing begins.
    
    Data fields:
        - setlist_id: Setlist identifier
        - song_count: Total number of songs to process
        - setlist_name: Display name of the setlist
    """
    name: ClassVar[str] = "SetlistProcessingStarted"


@dataclass
class SetlistSongProcessing(DomainEvent):
    """
    Event raised when a song starts processing.
    
    Data fields:
        - setlist_id: Setlist identifier
        - song_id: Song identifier
        - song_index: 0-based index of the song
        - song_name: Display name of the song
        - audio_path: Full path to the audio file
    """
    name: ClassVar[str] = "SetlistSongProcessing"


@dataclass
class SetlistSongCompleted(DomainEvent):
    """
    Event raised when a song finishes processing (success or failure).
    
    Data fields:
        - setlist_id: Setlist identifier
        - song_id: Song identifier
        - song_index: 0-based index of the song
        - success: Whether processing succeeded
        - error_message: Error message if failed
    """
    name: ClassVar[str] = "SetlistSongCompleted"


@dataclass
class SetlistProcessingCompleted(DomainEvent):
    """
    Event raised when all setlist processing finishes.
    
    Data fields:
        - setlist_id: Setlist identifier
        - results: Dict mapping song_id -> success (True/False)
        - total_songs: Total number of songs
        - successful_songs: Count of successfully processed songs
        - failed_songs: Count of failed songs
    """
    name: ClassVar[str] = "SetlistProcessingCompleted"


@dataclass
class MA3OscInbound(DomainEvent):
    """
    Event raised when a raw OSC packet is received from MA3.
    
    Data fields:
        - address: OSC address
        - args: Parsed OSC arguments
        - addr: (ip, port) tuple of sender
        - raw_data: Raw OSC bytes
    """
    name: ClassVar[str] = "MA3OscInbound"

