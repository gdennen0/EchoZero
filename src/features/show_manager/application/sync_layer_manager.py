"""
Sync Layer Manager

Provides comparison and reconciliation helpers for MA3 <-> Editor sync layers.

Divergence checking uses tolerance-based matching (not exact fingerprints) to
handle minor floating-point drift between Editor and MA3 event times.  MA3
uses frame-based timing internally, so round-tripping times can introduce
differences of up to ~20 ms depending on the timecode frame rate.
"""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.features.show_manager.domain.sync_state import compute_fingerprint


# Tolerance for considering two event times "equal" during divergence checks.
# 50 ms comfortably covers frame-level quantisation at 24/25/30 fps and minor
# floating-point serialisation drift between Python, Lua, and MA3.
TIME_TOLERANCE: float = 0.05

# Tolerance for duration comparison (same rationale as time).
DURATION_TOLERANCE: float = 0.05

@dataclass
class SyncLayerComparison:
    """Comparison result for a synced layer."""
    ma3_count: int
    editor_count: int
    matched_count: int
    diverged: bool

class SyncLayerManager:
    """
    Helper for comparing MA3 events with Editor events.

    This class is UI-agnostic and does not perform any network operations.
    """

    @staticmethod
    def normalize_editor_events(events: Iterable[Any]) -> List[Dict[str, Any]]:
        """Convert Editor Event objects into MA3-like dicts for comparison."""
        normalized: List[Dict[str, Any]] = []
        for event in events:
            metadata = getattr(event, "metadata", {}) or {}
            norm_evt = {
                "time": getattr(event, "time", 0.0),
                "name": metadata.get("ma3_name") or metadata.get("name") or getattr(event, "classification", ""),
                "cmd": metadata.get("ma3_cmd") or metadata.get("cmd", ""),
                "duration": getattr(event, "duration", 0.0),
            }
            normalized.append(norm_evt)
        return normalized

    @staticmethod
    def normalize_ma3_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize MA3 event dicts for comparison."""
        normalized: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            norm_evt = {
                "time": float(event.get("time", 0.0)),
                "name": event.get("name", ""),
                "cmd": event.get("cmd", ""),
                "duration": float(event.get("duration", 0.0)),
                "idx": event.get("idx") or event.get("no") or event.get("index"),
            }
            normalized.append(norm_evt)
        return normalized

    @staticmethod
    def compare_events(
        editor_events: Iterable[Any],
        ma3_events: Iterable[Dict[str, Any]],
        time_tolerance: float = TIME_TOLERANCE,
        duration_tolerance: float = DURATION_TOLERANCE,
    ) -> SyncLayerComparison:
        """Compare Editor events with MA3 events using tolerance-based matching.

        Instead of exact fingerprint comparison, this uses numeric tolerance
        to handle minor floating-point drift introduced by MA3's frame-based
        timing and serialisation round-trips.

        Two events are considered a match when both their ``time`` and
        ``duration`` values are within the respective tolerances.

        Args:
            editor_events: Editor Event objects.
            ma3_events: MA3 event dicts.
            time_tolerance: Max allowed difference in ``time`` (seconds).
            duration_tolerance: Max allowed difference in ``duration`` (seconds).

        Returns:
            SyncLayerComparison with match counts and diverged flag.
        """
        normalized_editor = SyncLayerManager.normalize_editor_events(editor_events)
        normalized_ma3 = SyncLayerManager.normalize_ma3_events(ma3_events)

        ed_count = len(normalized_editor)
        ma3_count = len(normalized_ma3)

        # Fast path: different counts means definitely diverged.
        if ed_count != ma3_count:
            return SyncLayerComparison(
                ma3_count=ma3_count,
                editor_count=ed_count,
                matched_count=0,
                diverged=True,
            )

        # Fast path: both empty means synced.
        if ed_count == 0:
            return SyncLayerComparison(
                ma3_count=0,
                editor_count=0,
                matched_count=0,
                diverged=False,
            )

        # Sort both lists by time for an O(n+m) two-pointer matching pass.
        sorted_editor = sorted(normalized_editor, key=lambda e: e["time"])
        sorted_ma3 = sorted(normalized_ma3, key=lambda e: e["time"])

        matched = 0
        j = 0  # MA3 pointer -- only advances forward

        for ed_evt in sorted_editor:
            ed_time = ed_evt["time"]
            ed_dur = ed_evt["duration"]
            # Advance MA3 pointer past events that are too early to match.
            while j < ma3_count and sorted_ma3[j]["time"] < ed_time - time_tolerance:
                j += 1
            if j < ma3_count:
                ma3_evt = sorted_ma3[j]
                if (abs(ed_time - ma3_evt["time"]) <= time_tolerance
                        and abs(ed_dur - ma3_evt["duration"]) <= duration_tolerance):
                    matched += 1
                    j += 1  # consume this MA3 event

        diverged = matched != ed_count

        return SyncLayerComparison(
            ma3_count=ma3_count,
            editor_count=ed_count,
            matched_count=matched,
            diverged=diverged,
        )

    @staticmethod
    def merge_events(
        editor_events: Iterable[Any],
        ma3_events: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge editor and MA3 events into a unified MA3-like list.

        Uses fingerprint matching to de-duplicate.
        """
        normalized_editor = SyncLayerManager.normalize_editor_events(editor_events)
        normalized_ma3 = SyncLayerManager.normalize_ma3_events(ma3_events)

        merged_by_fp: Dict[str, Dict[str, Any]] = {}
        for evt in normalized_ma3:
            merged_by_fp[compute_fingerprint(evt)] = evt
        for evt in normalized_editor:
            fp = compute_fingerprint(evt)
            if fp not in merged_by_fp:
                merged_by_fp[fp] = evt

        return list(merged_by_fp.values())
