"""
Simple Waveform Cache

ONE job: Cache base waveforms, slice for clips.
No async. No signals. No threads. No complexity.

Audio lookup: Uses data_item_repo.get(audio_id) directly.
No callback. No fallback. No searching across blocks.
"""

import numpy as np
from typing import Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.shared.domain.entities import AudioDataItem

# Try imports
try:
    from src.shared.application.services.waveform_service import get_waveform_service
    HAS_WAVEFORM_SERVICE = True
except ImportError:
    HAS_WAVEFORM_SERVICE = False

from ..logging import TimelineLog as Log


# =============================================================================
# SIMPLE GLOBAL CACHE
# =============================================================================

# Base waveform cache: {audio_id: np.ndarray}
# One entry per audio file. Shared across all clips from same source.
_base_waveform_cache: Dict[str, np.ndarray] = {}

# Duration cache: {audio_id: float}
_duration_cache: Dict[str, float] = {}


def clear_cache():
    """Clear all cached waveforms."""
    global _base_waveform_cache, _duration_cache
    _base_waveform_cache.clear()
    _duration_cache.clear()
    Log.info("WaveformSimple: Cache cleared")


def get_base_waveform(audio_item: 'AudioDataItem') -> Tuple[Optional[np.ndarray], float]:
    """
    Get base waveform for audio item (cached).
    
    Returns:
        (waveform_array, duration_seconds) or (None, 0.0) if unavailable
    """
    if not HAS_WAVEFORM_SERVICE:
        Log.warning("WaveformSimple: No waveform service available!")
        return None, 0.0
    
    audio_id = audio_item.id
    
    # Check cache first
    if audio_id in _base_waveform_cache:
        return _base_waveform_cache[audio_id], _duration_cache.get(audio_id, 0.0)
    
    # Load from service
    try:
        waveform_service = get_waveform_service()
        
        # Get duration
        duration = audio_item.length_ms / 1000.0 if audio_item.length_ms else 0.0
        if duration <= 0:
            Log.warning(f"WaveformSimple: Audio {audio_item.name} has no duration")
            return None, 0.0
        
        # Check if waveform exists
        if not waveform_service.has_waveform(audio_item):
            Log.warning(f"WaveformSimple: No waveform file for {audio_item.name} - needs generation")
            return None, 0.0
        
        # Load base waveform (uses mmap internally)
        waveform_data = waveform_service.load_waveform_mmap(audio_item)
        if waveform_data is None:
            Log.warning(f"WaveformSimple: Failed to load waveform mmap for {audio_item.name}")
            return None, 0.0
        
        # Convert to regular array (copy from mmap) - prevents memory issues
        waveform_array = np.array(waveform_data, dtype=np.float32)
        
        # Cache it
        _base_waveform_cache[audio_id] = waveform_array
        _duration_cache[audio_id] = duration
        
        Log.info(f"WaveformSimple: Cached base waveform for {audio_item.name} ({len(waveform_array)} points)")
        return waveform_array, duration
        
    except Exception as e:
        Log.warning(f"WaveformSimple: Failed to load waveform for {audio_item.name}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def _get_resolution_setting() -> int:
    """Get waveform resolution from timeline settings."""
    try:
        from ..settings.storage import get_timeline_settings_manager
        settings_mgr = get_timeline_settings_manager()
        if settings_mgr:
            return settings_mgr.waveform_resolution
    except Exception:
        pass
    return 50  # Default


def get_waveform_slice(
    audio_item: 'AudioDataItem',
    clip_start: float,
    clip_end: float,
    max_points: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Get waveform slice for a clip (from cached base waveform).
    
    Args:
        audio_item: Source audio
        clip_start: Clip start time in seconds
        clip_end: Clip end time in seconds  
        max_points: Maximum points to return (None = use resolution setting * clip duration)
    
    Returns:
        Waveform slice as numpy array, or None if unavailable
    """
    # Get base waveform (cached)
    base_waveform, duration = get_base_waveform(audio_item)
    if base_waveform is None or duration <= 0:
        return None
    
    # Calculate target points based on resolution setting and clip duration
    if max_points is None:
        clip_duration = clip_end - clip_start
        resolution = _get_resolution_setting()
        max_points = max(10, int(clip_duration * resolution))
    
    # Calculate indices
    waveform_len = len(base_waveform)
    start_idx = int((clip_start / duration) * waveform_len)
    end_idx = int((clip_end / duration) * waveform_len)
    
    # Clamp to valid range
    start_idx = max(0, min(start_idx, waveform_len - 1))
    end_idx = max(start_idx + 1, min(end_idx, waveform_len))
    
    # Extract slice
    slice_data = base_waveform[start_idx:end_idx]
    
    # Downsample if too large
    if len(slice_data) > max_points:
        indices = np.linspace(0, len(slice_data) - 1, max_points, dtype=int)
        slice_data = slice_data[indices]
    
    return slice_data


# =============================================================================
# DIRECT REPO LOOKUP (no callback, no fallback)
# =============================================================================

# Data item repository reference (set by scene)
_data_item_repo = None


def set_data_item_repo(repo):
    """Set data_item_repo for direct audio item lookup by ID."""
    global _data_item_repo
    _data_item_repo = repo




_debug_logged = set()  # Track what we've logged to avoid spam

def get_waveform_for_event(
    audio_id: Optional[str],
    audio_name: Optional[str],
    clip_start: float,
    clip_end: float,
    max_points: Optional[int] = None
) -> Tuple[Optional[np.ndarray], float]:
    """
    Get waveform slice for an event (main API).
    
    Uses direct data_item_repo.get(audio_id) -- no callback, no fallback.
    Events must reference valid Editor-owned audio_id values.
    
    Args:
        audio_id: Audio item ID (direct DB lookup)
        audio_name: Audio item name (for logging only, not used for lookup)
        clip_start: Clip start time in seconds
        clip_end: Clip end time in seconds
        max_points: Maximum points (None = use resolution setting)
    
    Returns:
        (waveform_slice, clip_duration) or (None, 0.0) if unavailable
    """
    if not audio_id:
        return None, 0.0
    
    if not _data_item_repo:
        if "no_repo" not in _debug_logged:
            Log.warning("WaveformSimple: No data_item_repo set!")
            _debug_logged.add("no_repo")
        return None, 0.0
    
    # Direct DB lookup -- one call, no fallback
    audio_item = _data_item_repo.get(audio_id)
    
    if not audio_item:
        debug_key = f"missing:{audio_id}"
        if debug_key not in _debug_logged:
            Log.warning(f"WaveformSimple: Audio not found by ID: {audio_id} (name hint: {audio_name})")
            _debug_logged.add(debug_key)
        return None, 0.0
    
    # Get slice (uses resolution setting if max_points is None)
    slice_data = get_waveform_slice(audio_item, clip_start, clip_end, max_points)
    if slice_data is None:
        debug_key = f"no_waveform:{audio_id}"
        if debug_key not in _debug_logged:
            Log.warning(f"WaveformSimple: No waveform data for {audio_item.name}")
            _debug_logged.add(debug_key)
        return None, 0.0
    
    duration = clip_end - clip_start
    return slice_data, duration

