"""Stable typed ID aliases for EchoZero application concepts."""

from typing import NewType

ProjectId = NewType("ProjectId", str)
SessionId = NewType("SessionId", str)
SongId = NewType("SongId", str)
SongVersionId = NewType("SongVersionId", str)
TimelineId = NewType("TimelineId", str)
TimelineObjectId = NewType("TimelineObjectId", str)
ObjectContentId = NewType("ObjectContentId", str)
ObjectCandidateId = NewType("ObjectCandidateId", str)
ObjectRevisionId = NewType("ObjectRevisionId", str)
LayerId = NewType("LayerId", str)
TakeId = NewType("TakeId", str)
EventId = NewType("EventId", str)
RegionId = NewType("RegionId", str)
SectionCueId = NewType("SectionCueId", str)
