"""
Progress Tracking Models

Core data models for centralized progress tracking.
Provides hierarchical progress state with automatic timing and metrics.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum


class ProgressStatus(str, Enum):
    """Status of a progress operation or level"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressLevel:
    """
    Progress at a specific level (song, action, block, subprocess, etc.)
    
    Supports hierarchical nesting for detailed progress tracking.
    Automatically tracks timing and calculates percentages.
    
    Example hierarchy:
        setlist (overall)
          song 1
            action 1 (LoadAudio)
            action 2 (DetectOnsets)
          song 2
            action 1 (LoadAudio)
            ...
    """
    level_id: str
    level_type: str  # "setlist", "song", "action", "block", "subprocess"
    name: str
    status: ProgressStatus = ProgressStatus.PENDING
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    message: str = ""
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error context
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    children: Dict[str, 'ProgressLevel'] = field(default_factory=dict)
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self, message: Optional[str] = None) -> None:
        """Mark level as started"""
        self.status = ProgressStatus.RUNNING
        self.started_at = datetime.now()
        if message:
            self.message = message
    
    def update(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        increment: int = 0,
        **metadata
    ) -> None:
        """
        Update progress level.
        
        Args:
            current: Set current progress to this value
            total: Update total (if changed)
            message: Update progress message
            increment: Increment current by this amount (ignored if current is set)
            **metadata: Additional metadata to merge
        """
        if current is not None:
            self.current = current
        elif increment > 0:
            self.current += increment
        
        if total is not None:
            self.total = total
        
        if message is not None:
            self.message = message
        
        # Calculate percentage
        if self.total > 0:
            self.percentage = min(100.0, max(0.0, (self.current / self.total) * 100.0))
        
        # Merge metadata
        if metadata:
            self.metadata.update(metadata)
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark level as completed"""
        self.status = ProgressStatus.COMPLETED
        self.completed_at = datetime.now()
        self.current = self.total if self.total > 0 else self.current
        self.percentage = 100.0
        if message:
            self.message = message
    
    def fail(self, error: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark level as failed"""
        self.status = ProgressStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error
        self.error_details = error_details
    
    def cancel(self, message: Optional[str] = None) -> None:
        """Mark level as cancelled"""
        self.status = ProgressStatus.CANCELLED
        self.completed_at = datetime.now()
        if message:
            self.message = message
    
    def get_elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds"""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def get_duration_str(self) -> str:
        """Get human-readable duration string"""
        elapsed = self.get_elapsed_seconds()
        if elapsed is None:
            return ""
        
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/events"""
        return {
            "level_id": self.level_id,
            "level_type": self.level_type,
            "name": self.name,
            "status": self.status.value,
            "current": self.current,
            "total": self.total,
            "percentage": self.percentage,
            "message": self.message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "duration": self.get_duration_str(),
            "error": self.error,
            "error_details": self.error_details,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "children": {k: v.to_dict() for k, v in self.children.items()}
        }


@dataclass
class ProgressState:
    """
    Complete progress state for a processing operation.
    
    Maintains hierarchical progress levels and provides query-able state.
    Used by ProgressEventStore to track operations.
    """
    operation_id: str
    operation_type: str  # "setlist_processing", "block_execution", etc.
    name: str = ""
    status: ProgressStatus = ProgressStatus.PENDING
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error context
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Hierarchical progress levels
    levels: Dict[str, ProgressLevel] = field(default_factory=dict)
    
    # Root level IDs (top-level items like songs in a setlist)
    root_level_ids: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self, name: Optional[str] = None) -> None:
        """Mark operation as started"""
        self.status = ProgressStatus.RUNNING
        self.started_at = datetime.now()
        if name:
            self.name = name
    
    def complete(self, error: Optional[str] = None) -> None:
        """Mark operation as complete"""
        if error:
            self.status = ProgressStatus.FAILED
            self.error = error
        else:
            self.status = ProgressStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def get_level(self, level_id: str) -> Optional[ProgressLevel]:
        """Get progress level by ID"""
        return self.levels.get(level_id)
    
    def add_level(
        self,
        level_id: str,
        level_type: str,
        name: str,
        parent_id: Optional[str] = None,
        total: int = 0,
        **metadata
    ) -> ProgressLevel:
        """
        Add a new progress level.
        
        Args:
            level_id: Unique identifier for this level
            level_type: Type of level (song, action, block, etc.)
            name: Display name
            parent_id: Parent level ID (None for root levels)
            total: Total items at this level
            **metadata: Additional metadata
            
        Returns:
            Created ProgressLevel
        """
        level = ProgressLevel(
            level_id=level_id,
            level_type=level_type,
            name=name,
            parent_id=parent_id,
            total=total,
            metadata=metadata
        )
        self.levels[level_id] = level
        
        # Track root levels
        if parent_id is None:
            if level_id not in self.root_level_ids:
                self.root_level_ids.append(level_id)
        else:
            # Add to parent's children
            parent = self.levels.get(parent_id)
            if parent:
                parent.children[level_id] = level
        
        return level
    
    def get_or_create_level(
        self,
        level_id: str,
        level_type: str,
        name: str,
        parent_id: Optional[str] = None,
        total: int = 0,
        **metadata
    ) -> ProgressLevel:
        """Get existing level or create new one"""
        if level_id in self.levels:
            return self.levels[level_id]
        return self.add_level(level_id, level_type, name, parent_id, total, **metadata)
    
    def get_elapsed_seconds(self) -> Optional[float]:
        """Get elapsed time in seconds"""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress summary"""
        total_levels = len(self.root_level_ids)
        completed_levels = sum(
            1 for lid in self.root_level_ids
            if lid in self.levels and self.levels[lid].status == ProgressStatus.COMPLETED
        )
        failed_levels = sum(
            1 for lid in self.root_level_ids
            if lid in self.levels and self.levels[lid].status == ProgressStatus.FAILED
        )
        
        percentage = (completed_levels / total_levels * 100) if total_levels > 0 else 0
        
        return {
            "total": total_levels,
            "completed": completed_levels,
            "failed": failed_levels,
            "pending": total_levels - completed_levels - failed_levels,
            "percentage": percentage,
            "elapsed_seconds": self.get_elapsed_seconds()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "error": self.error,
            "error_details": self.error_details,
            "overall": self.get_overall_progress(),
            "levels": {k: v.to_dict() for k, v in self.levels.items()},
            "root_level_ids": self.root_level_ids,
            "metadata": self.metadata
        }

