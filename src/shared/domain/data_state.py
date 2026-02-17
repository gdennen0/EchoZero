"""
Data State Value Object

Represents the state of block data (fresh, stale, or no data).
Used for tracking data freshness across the block graph.
"""
from enum import Enum
from typing import Optional


class DataState(Enum):
    """
    Data state enumeration.
    
    States are ordered by severity (worst to best):
    - NO_DATA: Block has no data items or no local state pulled
    - STALE: Block has data but source blocks have newer data
    - FRESH: Block has data and it's up-to-date with source blocks
    """
    NO_DATA = "no_data"
    STALE = "stale"
    FRESH = "fresh"
    
    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    def __repr__(self) -> str:
        """Developer representation"""
        return f"DataState.{self.name}"
    
    @property
    def color(self) -> str:
        """
        Get color representation for UI.
        
        Returns:
            Color name/hex for this state
        """
        colors = {
            DataState.NO_DATA: "#ff6b6b",  # Red
            DataState.STALE: "#ffa94d",     # Orange
            DataState.FRESH: "#51cf66",     # Green
        }
        return colors.get(self, "#999999")
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name"""
        names = {
            DataState.NO_DATA: "No Data",
            DataState.STALE: "Stale",
            DataState.FRESH: "Fresh",
        }
        return names.get(self, "Unknown")
    
    @classmethod
    def worst(cls, states: list['DataState']) -> 'DataState':
        """
        Get the worst state from a list of states.
        
        Order: NO_DATA > STALE > FRESH
        
        Args:
            states: List of DataState values
            
        Returns:
            Worst state (NO_DATA if any, else STALE if any, else FRESH)
        """
        if not states:
            return cls.NO_DATA
        
        # Check for NO_DATA (worst)
        if any(s == cls.NO_DATA for s in states):
            return cls.NO_DATA
        
        # Check for STALE
        if any(s == cls.STALE for s in states):
            return cls.STALE
        
        # Otherwise all are FRESH
        return cls.FRESH


