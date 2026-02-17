"""
MA3 Track Resolver

Utilities for matching MA3 tracks by name.
"""
from typing import Iterable, Optional, Any


def normalize_track_name(name: Optional[str]) -> str:
    """Normalize a track name for comparison.
    
    Strips common EZ/MA3 prefixes (ez_, ma3_) so that names like
    "ez_Drums", "ma3_Drums", and "Drums" all normalize to "drums".
    Also normalizes whitespace, underscores, and hyphens.
    """
    if not name:
        return ""
    normalized = name.strip().lower()
    # Strip common prefixes used by EchoZero and MA3 sync
    for prefix in ("ez_", "ez ", "ez-", "ma3_", "ma3 ", "ma3-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return normalized


def find_track_by_name(tracks: Iterable[Any], name: str) -> Optional[Any]:
    """Find a track object by normalized name."""
    target = normalize_track_name(name)
    if not target:
        return None
    for track in tracks:
        track_name = None
        if isinstance(track, dict):
            track_name = track.get("name")
        else:
            track_name = getattr(track, "name", None)
        if normalize_track_name(track_name) == target:
            return track
    return None


__all__ = ["normalize_track_name", "find_track_by_name"]
