"""Application progress contracts for long-running operations.
Exists to provide one typed, reusable progress standard across app workflows.
Connects producer callbacks and app-visible status surfaces through canonical models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OperationProgressStatus = Literal[
    "queued",
    "resolving",
    "running",
    "persisting",
    "completed",
    "failed",
    "cancelled",
]

ACTIVE_OPERATION_PROGRESS_STATUSES = frozenset({"queued", "resolving", "running", "persisting"})
FINAL_OPERATION_PROGRESS_STATUSES = frozenset({"completed", "failed", "cancelled"})

OperationProgressStage = Literal[
    "loading_configuration",
    "preparing_pipeline",
    "executing_pipeline",
    "persisting_results",
    "complete",
]


@dataclass(slots=True, frozen=True)
class OperationProgressUpdate:
    """One typed progress update emitted by an operation producer."""

    stage: OperationProgressStage
    message: str
    fraction_complete: float | None

    def __post_init__(self) -> None:
        stage = str(self.stage or "").strip().lower().replace(" ", "_")
        if stage not in {
            "loading_configuration",
            "preparing_pipeline",
            "executing_pipeline",
            "persisting_results",
            "complete",
        }:
            raise ValueError(f"Unsupported operation progress stage: {self.stage!r}")
        object.__setattr__(self, "stage", stage)

        message = str(self.message or "").strip()
        object.__setattr__(self, "message", message)

        if self.fraction_complete is None:
            return
        clamped = max(0.0, min(1.0, float(self.fraction_complete)))
        object.__setattr__(self, "fraction_complete", clamped)


@dataclass(slots=True, frozen=True)
class OperationProgressContext:
    """Context key used to scope operation visibility in app surfaces."""

    song_id: str | None = None
    song_version_id: str | None = None

    def scope_key(self) -> str:
        if self.song_version_id:
            return f"version:{self.song_version_id}"
        if self.song_id:
            return f"song:{self.song_id}"
        return "global"
