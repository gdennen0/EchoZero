"""
Structured result types for setlist processing.

Replaces the previous bool/dict returns with typed objects that carry
step context, action name, and error messages through the entire chain
without lossy string conversions at each layer.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SongProcessingResult:
    """Result of processing a single song."""
    success: bool
    song_id: str
    failed_step: Optional[str] = None
    failed_action: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class SetlistProcessingResult:
    """Result of processing an entire setlist."""
    song_results: List[SongProcessingResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.song_results if r.success)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.song_results if not r.success)

    @property
    def total_count(self) -> int:
        return len(self.song_results)

    @property
    def errors(self) -> List[SongProcessingResult]:
        return [r for r in self.song_results if not r.success]

    @property
    def all_succeeded(self) -> bool:
        return all(r.success for r in self.song_results)
