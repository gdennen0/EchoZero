"""
MA3 Layer Mapping Service

Handles bidirectional layer mapping between MA3 tracks and EchoZero Editor layers.
Includes built-in validation (Council recommendation: one service, not two).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from difflib import SequenceMatcher

from src.utils.message import Log


class MappingStatus(Enum):
    """Status of a layer mapping."""
    MAPPED = "mapped"              # Both sides exist and mapped
    UNMAPPED_MA3 = "unmapped_ma3"  # MA3 track has no EZ layer
    UNMAPPED_EZ = "unmapped_ez"    # EZ layer has no MA3 track
    CONFLICT = "conflict"          # Multiple MA3 tracks → same EZ layer
    INVALID_MA3 = "invalid_ma3"    # MA3 track doesn't exist (stale)
    INVALID_EZ = "invalid_ez"      # EZ layer doesn't exist (stale)
    EXCLUDED_MA3 = "excluded_ma3"  # MA3 track excluded from EZ
    EXCLUDED_EZ = "excluded_ez"    # EZ layer excluded from MA3


class ValidationSeverity(Enum):
    """Severity of validation error."""
    ERROR = "error"      # Must be fixed
    WARNING = "warning"  # Should be reviewed
    INFO = "info"        # Informational only


@dataclass
class MA3TrackInfo:
    """
    Information about an MA3 track.
    
    Represents a track in the MA3 timecode hierarchy.
    """
    timecode_no: int
    track_group: int
    track: int
    name: str = ""
    event_count: int = 0
    
    @property
    def coord(self) -> str:
        """
        Get coordinate string for this track.
        
        Format: tc{tc}_tg{tg}_tr{tr}
        Example: tc101_tg1_tr1
        """
        return f"tc{self.timecode_no}_tg{self.track_group}_tr{self.track}"
    
    @property
    def display_name(self) -> str:
        """Get human-readable display name."""
        name_part = f" ({self.name})" if self.name else ""
        return f"TC {self.timecode_no} / TG {self.track_group} / Track {self.track}{name_part}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timecode_no': self.timecode_no,
            'track_group': self.track_group,
            'track': self.track,
            'name': self.name,
            'event_count': self.event_count,
            'coord': self.coord,
            'display_name': self.display_name,
        }


@dataclass
class LayerMapping:
    """
    A single layer mapping between MA3 and EZ.
    """
    ma3_track_coord: str
    ez_layer_id: Optional[str] = None
    status: MappingStatus = MappingStatus.UNMAPPED_MA3
    ma3_track_info: Optional[MA3TrackInfo] = None
    
    @property
    def is_valid(self) -> bool:
        """Whether this mapping is valid (both sides exist)."""
        return self.status == MappingStatus.MAPPED
    
    @property
    def is_unmapped(self) -> bool:
        """Whether this mapping is unmapped."""
        return self.status in (MappingStatus.UNMAPPED_MA3, MappingStatus.UNMAPPED_EZ)
    
    @property
    def is_conflict(self) -> bool:
        """Whether this mapping has a conflict."""
        return self.status == MappingStatus.CONFLICT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ma3_track_coord': self.ma3_track_coord,
            'ez_layer_id': self.ez_layer_id,
            'status': self.status.value,
            'is_valid': self.is_valid,
            'is_unmapped': self.is_unmapped,
            'is_conflict': self.is_conflict,
            'ma3_track_info': self.ma3_track_info.to_dict() if self.ma3_track_info else None,
        }


@dataclass
class ValidationError:
    """
    A validation error or warning.
    """
    type: str  # "unmapped_ma3", "unmapped_ez", "conflict", "missing_track", etc.
    severity: ValidationSeverity
    message: str
    ma3_track: Optional[str] = None
    ez_layer: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': self.type,
            'severity': self.severity.value,
            'message': self.message,
            'ma3_track': self.ma3_track,
            'ez_layer': self.ez_layer,
            'details': self.details,
        }


class MA3LayerMappingService:
    """
    Service for managing bidirectional layer mappings between MA3 and EZ.
    
    Includes built-in validation (no separate validator class).
    Provides graceful degradation when MA3/Editor are offline.
    """
    
    def __init__(self):
        """Initialize the mapping service."""
        self._ma3_tracks_cache: Dict[str, MA3TrackInfo] = {}
        self._ez_layers_cache: List[str] = []
        self._ma3_available = False
        self._ez_available = False
    
    # =========================================================================
    # Track/Layer Discovery
    # =========================================================================
    
    def parse_ma3_structure(self, structure_data: Dict[str, Any]) -> List[MA3TrackInfo]:
        """
        Parse MA3 structure data into track info list.
        
        Args:
            structure_data: Structure data from EZ.GetStructure()
            
        Returns:
            List of MA3TrackInfo objects
        """
        tracks = []
        timecode_no = structure_data.get('timecode_no', 0)
        
        for tg_data in structure_data.get('track_groups', []):
            tg_idx = tg_data.get('index', 0)
            
            for track_data in tg_data.get('tracks', []):
                track_idx = track_data.get('index', 0)
                track_name = track_data.get('name', '')
                
                track_info = MA3TrackInfo(
                    timecode_no=timecode_no,
                    track_group=tg_idx,
                    track=track_idx,
                    name=track_name,
                    event_count=0  # Could be populated if available
                )
                tracks.append(track_info)
                
                # Cache for quick lookup
                self._ma3_tracks_cache[track_info.coord] = track_info
        
        self._ma3_available = True
        Log.info(f"Parsed {len(tracks)} MA3 tracks from structure")
        return tracks
    
    def set_ez_layers(self, layer_ids: List[str]):
        """
        Set available EZ layers.
        
        Args:
            layer_ids: List of EZ layer IDs
        """
        self._ez_layers_cache = layer_ids.copy()
        self._ez_available = True
        Log.info(f"Set {len(layer_ids)} EZ layers")
    
    def get_ma3_track_info(self, ma3_track_coord: str) -> Optional[MA3TrackInfo]:
        """
        Get MA3 track info by coordinate.
        
        Args:
            ma3_track_coord: MA3 track coordinate
            
        Returns:
            MA3TrackInfo if found, None otherwise
        """
        return self._ma3_tracks_cache.get(ma3_track_coord)
    
    def get_cached_ma3_tracks(self) -> List[MA3TrackInfo]:
        """Get all cached MA3 tracks."""
        return list(self._ma3_tracks_cache.values())
    
    def get_cached_ez_layers(self) -> List[str]:
        """Get all cached EZ layers."""
        return self._ez_layers_cache.copy()
    
    @property
    def is_ma3_available(self) -> bool:
        """Whether MA3 data is available."""
        return self._ma3_available
    
    @property
    def is_ez_available(self) -> bool:
        """Whether EZ data is available."""
        return self._ez_available
    
    # =========================================================================
    # Mapping Operations
    # =========================================================================
    
    def build_mappings(
        self,
        mapping_dict: Dict[str, str],
        ma3_tracks: Optional[List[MA3TrackInfo]] = None,
        ez_layers: Optional[List[str]] = None,
        reverse_mappings: Optional[Dict[str, str]] = None,
        excluded_ma3: Optional[List[str]] = None,
        excluded_ez: Optional[List[str]] = None
    ) -> List[LayerMapping]:
        """
        Build LayerMapping objects from mapping dictionary.
        
        Args:
            mapping_dict: Dict mapping MA3 coord -> EZ layer ID
            ma3_tracks: Optional list of MA3 tracks (uses cache if None)
            ez_layers: Optional list of EZ layers (uses cache if None)
            reverse_mappings: Optional dict mapping EZ layer ID -> MA3 coord (for bidirectional)
            excluded_ma3: Optional list of excluded MA3 track coords
            excluded_ez: Optional list of excluded EZ layer IDs
            
        Returns:
            List of LayerMapping objects with status
        """
        if ma3_tracks is None:
            ma3_tracks = self.get_cached_ma3_tracks()
        if ez_layers is None:
            ez_layers = self.get_cached_ez_layers()
        if excluded_ma3 is None:
            excluded_ma3 = []
        if excluded_ez is None:
            excluded_ez = []
        
        # Build sets for quick lookup
        ez_layer_set = set(ez_layers)
        ma3_coord_set = {track.coord for track in ma3_tracks}
        ma3_track_map = {track.coord: track for track in ma3_tracks}
        excluded_ma3_set = set(excluded_ma3)
        excluded_ez_set = set(excluded_ez)
        
        # Track which EZ layers are used
        ez_layer_usage: Dict[str, List[str]] = {}
        
        mappings = []
        
        # Process existing mappings
        for ma3_coord, ez_layer in mapping_dict.items():
            # Track EZ layer usage
            if ez_layer not in ez_layer_usage:
                ez_layer_usage[ez_layer] = []
            ez_layer_usage[ez_layer].append(ma3_coord)
            
            # Check exclusions first (excluded items take precedence)
            if ma3_coord in excluded_ma3_set:
                status = MappingStatus.EXCLUDED_MA3
            elif ez_layer in excluded_ez_set:
                status = MappingStatus.EXCLUDED_EZ
            else:
                # Determine status based on existence
                ma3_exists = ma3_coord in ma3_coord_set
                ez_exists = ez_layer in ez_layer_set
                
                if not ma3_exists:
                    status = MappingStatus.INVALID_MA3
                elif not ez_exists:
                    status = MappingStatus.INVALID_EZ
                else:
                    # Will check for conflicts after processing all
                    status = MappingStatus.MAPPED
            
            mapping = LayerMapping(
                ma3_track_coord=ma3_coord,
                ez_layer_id=ez_layer,
                status=status,
                ma3_track_info=ma3_track_map.get(ma3_coord)
            )
            mappings.append(mapping)
        
        # Mark conflicts (multiple MA3 tracks → same EZ layer)
        # Only mark conflicts for non-excluded mappings
        for ez_layer, ma3_coords in ez_layer_usage.items():
            if len(ma3_coords) > 1:
                # Check if any of these mappings are excluded (conflicts don't apply to excluded)
                non_excluded_coords = [
                    coord for coord in ma3_coords
                    if coord not in excluded_ma3_set and ez_layer not in excluded_ez_set
                ]
                if len(non_excluded_coords) > 1:
                    # This is a conflict
                    for mapping in mappings:
                        if (mapping.ez_layer_id == ez_layer and 
                            mapping.status == MappingStatus.MAPPED and
                            mapping.ma3_track_coord in non_excluded_coords):
                            mapping.status = MappingStatus.CONFLICT
        
        # Add unmapped MA3 tracks (including excluded ones)
        for track in ma3_tracks:
            if track.coord not in mapping_dict:
                # Check if excluded
                if track.coord in excluded_ma3_set:
                    status = MappingStatus.EXCLUDED_MA3
                else:
                    status = MappingStatus.UNMAPPED_MA3
                
                mapping = LayerMapping(
                    ma3_track_coord=track.coord,
                    ez_layer_id=None,
                    status=status,
                    ma3_track_info=track
                )
                mappings.append(mapping)
        
        return mappings
    
    def apply_mapping(
        self,
        ma3_track_coord: str,
        mappings: Dict[str, str]
    ) -> Optional[str]:
        """
        Apply mapping to get EZ layer ID for MA3 track.
        
        Args:
            ma3_track_coord: MA3 track coordinate
            mappings: Mapping dictionary
            
        Returns:
            EZ layer ID if mapped, None otherwise
        """
        return mappings.get(ma3_track_coord)
    
    def reverse_lookup(
        self,
        ez_layer_id: str,
        mappings: Dict[str, str]
    ) -> Optional[str]:
        """
        Reverse lookup: EZ layer → MA3 track.
        
        Args:
            ez_layer_id: EZ layer ID
            mappings: Mapping dictionary
            
        Returns:
            MA3 track coordinate if found, None otherwise
        """
        for ma3_coord, ez_layer in mappings.items():
            if ez_layer == ez_layer_id:
                return ma3_coord
        return None
    
    # =========================================================================
    # Validation (Built-in, not separate class)
    # =========================================================================
    
    def validate_mappings(
        self,
        mappings: Dict[str, str],
        ma3_tracks: Optional[List[MA3TrackInfo]] = None,
        ez_layers: Optional[List[str]] = None
    ) -> List[ValidationError]:
        """
        Validate mappings and return errors/warnings.
        
        Args:
            mappings: Mapping dictionary
            ma3_tracks: Optional list of MA3 tracks (uses cache if None)
            ez_layers: Optional list of EZ layers (uses cache if None)
            
        Returns:
            List of ValidationError objects
        """
        if ma3_tracks is None:
            ma3_tracks = self.get_cached_ma3_tracks()
        if ez_layers is None:
            ez_layers = self.get_cached_ez_layers()
        
        errors = []
        
        # Check for unmapped MA3 tracks
        ma3_coords = {track.coord for track in ma3_tracks}
        unmapped_ma3 = ma3_coords - set(mappings.keys())
        if unmapped_ma3:
            errors.append(ValidationError(
                type="unmapped_ma3",
                severity=ValidationSeverity.WARNING,
                message=f"{len(unmapped_ma3)} MA3 tracks have no mapping",
                details={'unmapped_tracks': list(unmapped_ma3)}
            ))
        
        # Check for conflicts (multiple MA3 tracks → same EZ layer)
        ez_layer_usage: Dict[str, List[str]] = {}
        for ma3_coord, ez_layer in mappings.items():
            if ez_layer not in ez_layer_usage:
                ez_layer_usage[ez_layer] = []
            ez_layer_usage[ez_layer].append(ma3_coord)
        
        for ez_layer, conflicting_tracks in ez_layer_usage.items():
            if len(conflicting_tracks) > 1:
                errors.append(ValidationError(
                    type="conflict",
                    severity=ValidationSeverity.ERROR,
                    message=f"Multiple MA3 tracks mapped to same EZ layer: {ez_layer}",
                    ez_layer=ez_layer,
                    details={'ma3_tracks': conflicting_tracks}
                ))
        
        # Check for missing MA3 tracks (mapping references non-existent track)
        for ma3_coord in mappings.keys():
            if ma3_coord not in ma3_coords:
                errors.append(ValidationError(
                    type="missing_ma3_track",
                    severity=ValidationSeverity.WARNING,
                    message=f"Mapping references non-existent MA3 track: {ma3_coord}",
                    ma3_track=ma3_coord
                ))
        
        # Check for missing EZ layers
        ez_layer_set = set(ez_layers)
        for ma3_coord, ez_layer in mappings.items():
            if ez_layer not in ez_layer_set:
                errors.append(ValidationError(
                    type="missing_ez_layer",
                    severity=ValidationSeverity.WARNING,
                    message=f"Mapping references non-existent EZ layer: {ez_layer}",
                    ma3_track=ma3_coord,
                    ez_layer=ez_layer
                ))
        
        return errors
    
    # =========================================================================
    # Auto-Detection (Name Matching)
    # =========================================================================
    
    def suggest_mappings(
        self,
        ma3_tracks: List[MA3TrackInfo],
        ez_layers: List[str],
        threshold: float = 0.6
    ) -> Dict[str, Tuple[str, float]]:
        """
        Suggest mappings based on name similarity.
        
        Args:
            ma3_tracks: List of MA3 tracks
            ez_layers: List of EZ layer IDs
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Dict mapping MA3 coord -> (EZ layer ID, similarity score)
        """
        suggestions = {}
        
        for track in ma3_tracks:
            if not track.name:
                continue  # Skip tracks with no name
            
            best_match = None
            best_score = 0.0
            
            for ez_layer in ez_layers:
                # Calculate similarity
                score = self._calculate_similarity(track.name, ez_layer)
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = ez_layer
            
            if best_match:
                suggestions[track.coord] = (best_match, best_score)
        
        Log.info(f"Generated {len(suggestions)} mapping suggestions")
        return suggestions
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings.
        
        Uses SequenceMatcher for fuzzy matching.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize strings
        s1 = str1.lower().strip()
        s2 = str2.lower().strip()
        
        # Exact match
        if s1 == s2:
            return 1.0
        
        # Contains match
        if s1 in s2 or s2 in s1:
            return 0.9
        
        # Fuzzy match
        return SequenceMatcher(None, s1, s2).ratio()
    
    # =========================================================================
    # Graceful Degradation
    # =========================================================================
    
    def get_status_message(self) -> str:
        """
        Get status message for UI display.
        
        Provides graceful degradation when components are offline.
        """
        if not self._ma3_available and not self._ez_available:
            return "⚠️ MA3 and Editor offline - mappings in read-only mode"
        elif not self._ma3_available:
            return "⚠️ MA3 offline - cannot refresh MA3 tracks"
        elif not self._ez_available:
            return "⚠️ Editor offline - cannot refresh EZ layers"
        else:
            ma3_count = len(self._ma3_tracks_cache)
            ez_count = len(self._ez_layers_cache)
            return f"✓ {ma3_count} MA3 tracks, {ez_count} EZ layers available"
