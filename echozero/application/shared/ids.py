"""Stable typed ID aliases for EchoZero application concepts."""

from typing import NewType

ProjectId = NewType("ProjectId", str)
SessionId = NewType("SessionId", str)
SongId = NewType("SongId", str)
SongVersionId = NewType("SongVersionId", str)
TimelineId = NewType("TimelineId", str)
LayerId = NewType("LayerId", str)
TakeId = NewType("TakeId", str)
EventId = NewType("EventId", str)
RegionId = NewType("RegionId", str)
SectionCueId = NewType("SectionCueId", str)
