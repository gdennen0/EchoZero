"""
Sync Layer Entity

Unified entity representing a synced layer between EchoZero and MA3.
The 'source' field indicates where the layer originated, preventing duplicates by design.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class SyncSource(Enum):
    """Where the sync layer originated."""
    MA3 = "ma3"      # Layer originated from MA3 track
    EDITOR = "editor"  # Layer originated from EchoZero Editor


class SyncStatus(Enum):
    """Current sync status."""
    UNMAPPED = "unmapped"      # Not yet synced
    SYNCED = "synced"          # In sync
    DIVERGED = "diverged"      # Changes detected on both sides
    ERROR = "error"            # Sync error occurred
    PENDING = "pending"        # Reconnected, waiting for divergence check
    DISCONNECTED = "disconnected"  # Editor layer was deleted; needs user attention
    AWAITING_CONNECTION = "awaiting_connection"  # No MA3 connection; waiting for MA3


class SyncDirection(Enum):
    """Direction of synchronization."""
    MA3_TO_EZ = "ma3_to_ez"
    EZ_TO_MA3 = "ez_to_ma3"
    BIDIRECTIONAL = "bidirectional"


class ConflictStrategy(Enum):
    """How to resolve conflicts."""
    MA3_WINS = "ma3_wins"
    EZ_WINS = "ez_wins"
    MERGE = "merge"
    PROMPT_USER = "prompt_user"


@dataclass
class SyncLayerSettings:
    """Settings for a sync layer."""
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    conflict_strategy: ConflictStrategy = ConflictStrategy.PROMPT_USER
    auto_apply: bool = False
    apply_updates_enabled: bool = True
    sync_on_change: bool = True
    sequence_no: int = 1  # MA3 sequence assignment
    track_group_no: int = 1  # MA3 track group number (1-99999)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "direction": self.direction.value,
            "conflict_strategy": self.conflict_strategy.value,
            "auto_apply": self.auto_apply,
            "apply_updates_enabled": self.apply_updates_enabled,
            "sync_on_change": self.sync_on_change,
            "sequence_no": self.sequence_no,
            "track_group_no": self.track_group_no,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncLayerSettings":
        """Create from dictionary."""
        direction = SyncDirection.BIDIRECTIONAL
        if data.get("direction"):
            try:
                direction = SyncDirection(data["direction"])
            except ValueError:
                # Handle legacy values
                legacy_map = {
                    "MA3_TO_EZ": SyncDirection.MA3_TO_EZ,
                    "EZ_TO_MA3": SyncDirection.EZ_TO_MA3,
                    "BIDIRECTIONAL": SyncDirection.BIDIRECTIONAL,
                    "MA3_TO_EDITOR": SyncDirection.MA3_TO_EZ,
                    "EDITOR_TO_MA3": SyncDirection.EZ_TO_MA3,
                }
                direction = legacy_map.get(str(data["direction"]).upper(), SyncDirection.BIDIRECTIONAL)
        
        conflict_strategy = ConflictStrategy.PROMPT_USER
        if data.get("conflict_strategy"):
            try:
                conflict_strategy = ConflictStrategy(data["conflict_strategy"])
            except ValueError:
                legacy_map = {
                    "USE_MA3": ConflictStrategy.MA3_WINS,
                    "USE_EZ": ConflictStrategy.EZ_WINS,
                    "MERGE": ConflictStrategy.MERGE,
                    "PROMPT_USER": ConflictStrategy.PROMPT_USER,
                }
                conflict_strategy = legacy_map.get(
                    str(data["conflict_strategy"]).upper(), 
                    ConflictStrategy.PROMPT_USER
                )
        
        return cls(
            direction=direction,
            conflict_strategy=conflict_strategy,
            auto_apply=bool(data.get("auto_apply", False)),
            apply_updates_enabled=bool(data.get("apply_updates_enabled", True)),
            sync_on_change=bool(data.get("sync_on_change", True)),
            sequence_no=int(data.get("sequence_no", 1) or 1),
            track_group_no=int(data.get("track_group_no", 1) or 1),
        )


@dataclass
class SyncLayerEntity:
    """
    Unified entity representing a synced layer between EchoZero and MA3.
    
    The 'source' field indicates where this layer originated:
    - "ma3": Layer was created from an MA3 track (MA3 is authoritative)
    - "editor": Layer was created from an Editor layer (Editor is authoritative)
    
    This design prevents duplicates by ensuring each sync layer appears
    exactly once in the UI, in the section corresponding to its source.
    
    Attributes:
        id: Unique identifier (UUID)
        source: Where this layer originated ("ma3" or "editor")
        name: Display name (prefixed with ma3_ or ez_)
        
        # MA3-side identity (set when source="ma3" or when synced to MA3)
        ma3_coord: MA3 track coordinate (e.g., "tc1_tg1_tr1")
        ma3_timecode_no: MA3 timecode number
        ma3_track_group: MA3 track group number
        ma3_track: MA3 track number
        
        # Editor-side identity (set when source="editor" or when synced to Editor)
        editor_layer_id: Editor layer ID
        editor_block_id: Editor block ID
        
        # State
        sync_status: Current sync status
        event_count: Number of events
        last_sync_time: Last successful sync timestamp
        error_message: Error message if status is ERROR
        
        # Settings
        settings: Sync settings (direction, conflict strategy, etc.)
        
        # Grouping (mirrors MA3 track group structure)
        group_name: Group name for organizing layers
    """
    
    # Identity
    id: str
    source: SyncSource
    name: str
    
    # MA3-side identity
    ma3_coord: Optional[str] = None
    ma3_timecode_no: Optional[int] = None
    ma3_track_group: Optional[int] = None
    ma3_track: Optional[int] = None
    
    # Persistent EZ identity (stored in MA3 track .note property)
    # Format: "ez:{editor_layer_name}" e.g. "ez:Drums"
    # Used for reliable reconnection across sessions even when track order changes
    ez_track_id: Optional[str] = None
    
    # Editor-side identity
    editor_layer_id: Optional[str] = None
    editor_block_id: Optional[str] = None
    editor_data_item_id: Optional[str] = None  # The EventDataItem holding MA3 sync events
    
    # State
    sync_status: SyncStatus = SyncStatus.UNMAPPED
    event_count: int = 0
    last_sync_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Settings
    settings: SyncLayerSettings = field(default_factory=SyncLayerSettings)
    
    # Grouping
    group_name: Optional[str] = None
    
    def __post_init__(self):
        """Validate entity state."""
        if self.source == SyncSource.MA3 and not self.ma3_coord:
            raise ValueError("MA3-sourced entity requires ma3_coord")
        # EDITOR-sourced entities require editor_layer_id unless they are
        # DISCONNECTED (Editor layer was deleted; entity kept for user attention).
        if (self.source == SyncSource.EDITOR
                and not self.editor_layer_id
                and self.sync_status != SyncStatus.DISCONNECTED):
            raise ValueError("Editor-sourced entity requires editor_layer_id")
    
    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        if self.source == SyncSource.MA3:
            name_part = f" ({self.name})" if self.name and not self.name.startswith("ma3_") else ""
            return f"TC {self.ma3_timecode_no} / TG {self.ma3_track_group} / Track {self.ma3_track}{name_part}"
        return self.name
    
    @property
    def is_synced(self) -> bool:
        """Check if layer is currently synced."""
        return self.sync_status in (SyncStatus.SYNCED, SyncStatus.DIVERGED)
    
    @property
    def has_ma3_side(self) -> bool:
        """Check if MA3 side is configured."""
        return self.ma3_coord is not None
    
    @property
    def has_editor_side(self) -> bool:
        """Check if Editor side is configured."""
        return self.editor_layer_id is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "source": self.source.value,
            "name": self.name,
            "ma3_coord": self.ma3_coord,
            "ma3_timecode_no": self.ma3_timecode_no,
            "ma3_track_group": self.ma3_track_group,
            "ma3_track": self.ma3_track,
            "editor_layer_id": self.editor_layer_id,
            "editor_block_id": self.editor_block_id,
            "editor_data_item_id": self.editor_data_item_id,
            "sync_status": self.sync_status.value,
            "event_count": self.event_count,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "error_message": self.error_message,
            "settings": self.settings.to_dict(),
            "group_name": self.group_name,
        }
        if self.ez_track_id:
            result["ez_track_id"] = self.ez_track_id
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncLayerEntity":
        """Create from dictionary."""
        # Parse source
        source = SyncSource.MA3
        if data.get("source"):
            try:
                source = SyncSource(data["source"])
            except ValueError:
                # Infer from available data
                if data.get("editor_layer_id") and not data.get("ma3_coord"):
                    source = SyncSource.EDITOR
        
        # Parse sync status
        sync_status = SyncStatus.UNMAPPED
        if data.get("sync_status"):
            try:
                sync_status = SyncStatus(data["sync_status"])
            except ValueError:
                pass
        
        # Parse last sync time
        last_sync_time = None
        if data.get("last_sync_time"):
            try:
                last_sync_time = datetime.fromisoformat(data["last_sync_time"])
            except (ValueError, TypeError):
                pass
        
        # Parse settings
        settings = SyncLayerSettings.from_dict(data.get("settings", {}))
        
        return cls(
            id=data["id"],
            source=source,
            name=data.get("name", ""),
            ma3_coord=data.get("ma3_coord"),
            ma3_timecode_no=data.get("ma3_timecode_no"),
            ma3_track_group=data.get("ma3_track_group"),
            ma3_track=data.get("ma3_track"),
            ez_track_id=data.get("ez_track_id"),
            editor_layer_id=data.get("editor_layer_id"),
            editor_block_id=data.get("editor_block_id"),
            editor_data_item_id=data.get("editor_data_item_id"),
            sync_status=sync_status,
            event_count=data.get("event_count", 0),
            last_sync_time=last_sync_time,
            error_message=data.get("error_message"),
            settings=settings,
            group_name=data.get("group_name"),
        )
    
    @classmethod
    def from_ma3_track(
        cls,
        id: str,
        coord: str,
        timecode_no: int,
        track_group: int,
        track: int,
        name: str = "",
        group_name: Optional[str] = None,
        event_count: int = 0,
        sequence_no: Optional[int] = None,
        ez_track_id: Optional[str] = None,
    ) -> "SyncLayerEntity":
        """
        Create entity from MA3 track info.
        
        Args:
            id: Unique identifier
            coord: MA3 track coordinate
            timecode_no: Timecode number
            track_group: Track group number
            track: Track number
            name: Track name (will be prefixed with ma3_)
            group_name: Track group name
            event_count: Number of events
            sequence_no: Sequence number assigned to track (if any)
            ez_track_id: Persistent EZ identity from MA3 .note (e.g. "ez:Drums")
        """
        # Add ma3_ prefix if not present
        prefixed_name = name
        if name and not name.startswith("ma3_"):
            prefixed_name = f"ma3_{name}"
        elif not name:
            prefixed_name = f"ma3_tc{timecode_no}_tg{track_group}_tr{track}"
        
        # Create settings with sequence number if provided
        # Default to 1 if None (MA3 tracks without sequence assignment)
        settings = SyncLayerSettings()
        settings.sequence_no = sequence_no if sequence_no is not None else 1
        settings.track_group_no = track_group
        
        return cls(
            id=id,
            source=SyncSource.MA3,
            name=prefixed_name,
            ma3_coord=coord,
            ma3_timecode_no=timecode_no,
            ma3_track_group=track_group,
            ma3_track=track,
            ez_track_id=ez_track_id,
            group_name=group_name,
            event_count=event_count,
            settings=settings,
        )
    
    @classmethod
    def from_editor_layer(
        cls,
        id: str,
        layer_id: str,
        block_id: str,
        name: str = "",
        group_name: Optional[str] = None,
        event_count: int = 0,
    ) -> "SyncLayerEntity":
        """
        Create entity from Editor layer info.
        
        Args:
            id: Unique identifier
            layer_id: Editor layer ID
            block_id: Editor block ID
            name: Layer name (will be prefixed with ez_)
            group_name: Layer group name
            event_count: Number of events
        """
        # Add ez_ prefix if not present
        prefixed_name = name
        if name and not name.startswith("ez_"):
            prefixed_name = f"ez_{name}"
        elif not name:
            prefixed_name = f"ez_{layer_id}"
        
        return cls(
            id=id,
            source=SyncSource.EDITOR,
            name=prefixed_name,
            editor_layer_id=layer_id,
            editor_block_id=block_id,
            group_name=group_name,
            event_count=event_count,
        )
    
    def link_to_ma3(
        self,
        coord: str,
        timecode_no: int,
        track_group: int,
        track: int,
    ) -> None:
        """
        Link this entity to an MA3 track.
        
        Used when syncing an Editor-sourced layer to MA3.
        """
        self.ma3_coord = coord
        self.ma3_timecode_no = timecode_no
        self.ma3_track_group = track_group
        self.ma3_track = track
    
    def link_to_editor(
        self,
        layer_id: str,
        block_id: str,
    ) -> None:
        """
        Link this entity to an Editor layer.
        
        Used when syncing an MA3-sourced layer to Editor.
        """
        self.editor_layer_id = layer_id
        self.editor_block_id = block_id
    
    def mark_synced(self) -> None:
        """Mark entity as successfully synced."""
        self.sync_status = SyncStatus.SYNCED
        self.last_sync_time = datetime.now()
        self.error_message = None
    
    def mark_diverged(self) -> None:
        """Mark entity as diverged (changes on both sides)."""
        self.sync_status = SyncStatus.DIVERGED
    
    def mark_error(self, message: str) -> None:
        """Mark entity as having an error."""
        self.sync_status = SyncStatus.ERROR
        self.error_message = message

    def mark_disconnected(self, message: str = "") -> None:
        """Mark entity as disconnected (Editor layer was deleted).
        
        The sync entity is kept so the user can see it in ShowManager
        and decide whether to reconnect to another Editor layer or remove it.
        """
        self.sync_status = SyncStatus.DISCONNECTED
        self.error_message = message or "Editor layer was deleted"

    def unlink_editor(self) -> None:
        """Clear Editor-side identity (e.g. after Editor layer deletion).
        
        Keeps the entity around for user attention but removes the stale
        Editor references so no sync operations target a missing layer.
        """
        self.editor_layer_id = None
        self.editor_block_id = None
        self.editor_data_item_id = None


__all__ = [
    "SyncLayerEntity",
    "SyncLayerSettings",
    "SyncSource",
    "SyncStatus",
    "SyncDirection",
    "ConflictStrategy",
]
