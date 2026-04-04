"""
Domain events: Frozen dataclasses representing state changes in the pipeline model.
Exists because blocks must stay isolated (FP1) — events decouple producers from consumers.
Published breadth-first by EventBus after Unit of Work commits; discarded on rollback.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from echozero.domain.enums import BlockState


def create_event_id() -> str:
    """Generate a unique identifier for a domain event."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class DomainEvent:
    """Base type for all domain events — carries identity and correlation metadata."""

    event_id: str
    timestamp: float
    correlation_id: str


# ---------------------------------------------------------------------------
# Block lifecycle events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockAddedEvent(DomainEvent):
    """A new block was added to the graph."""

    block_id: str
    block_type: str


@dataclass(frozen=True)
class BlockRemovedEvent(DomainEvent):
    """A block was removed from the graph."""

    block_id: str


@dataclass(frozen=True)
class ConnectionAddedEvent(DomainEvent):
    """A connection was created between two blocks."""

    source_block_id: str
    target_block_id: str


@dataclass(frozen=True)
class ConnectionRemovedEvent(DomainEvent):
    """A connection was removed between two blocks."""

    source_block_id: str
    target_block_id: str


# ---------------------------------------------------------------------------
# Block state and execution events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockStateChangedEvent(DomainEvent):
    """A block transitioned from one execution state to another."""

    block_id: str
    old_state: BlockState
    new_state: BlockState


@dataclass(frozen=True)
class SettingsChangedEvent(DomainEvent):
    """A block's configuration was modified. Carries before/after values for undo and stale reasons."""

    block_id: str
    setting_key: str
    old_value: Any
    new_value: Any


# ---------------------------------------------------------------------------
# ProjectRecord lifecycle events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectLoadedEvent(DomainEvent):
    """A project was loaded from disk."""

    project_id: str


@dataclass(frozen=True)
class ProjectSavedEvent(DomainEvent):
    """A project was saved to disk."""

    project_id: str


# ---------------------------------------------------------------------------
# Foundry lifecycle events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoundryRunCreatedEvent(DomainEvent):
    """A Foundry train run was created."""

    run_id: str
    dataset_version_id: str
    status: str


@dataclass(frozen=True)
class FoundryRunStartedEvent(DomainEvent):
    """A Foundry train run transitioned to running."""

    run_id: str
    status: str


@dataclass(frozen=True)
class FoundryArtifactFinalizedEvent(DomainEvent):
    """A Foundry artifact manifest was finalized."""

    artifact_id: str
    run_id: str


@dataclass(frozen=True)
class FoundryArtifactValidatedEvent(DomainEvent):
    """A Foundry artifact was validated for a consumer contract."""

    artifact_id: str
    consumer: str
    ok: bool
    error_count: int
    warning_count: int
