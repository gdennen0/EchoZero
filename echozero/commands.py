"""
Commands: Frozen dataclasses representing intentions to mutate the pipeline model.
Exists because all mutations flow through pipeline.dispatch() — no direct graph manipulation.
Two categories: editable (undoable structural edits) and operational (execute, save, load).
"""

from __future__ import annotations

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from echozero.domain.enums import BlockCategory


def _create_command_id() -> str:
    """Generate a unique identifier for a command."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Base command
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """Base type for all pipeline commands — carries identity and correlation metadata."""

    command_id: str = field(default_factory=_create_command_id)
    correlation_id: str = field(default_factory=_create_command_id)

    @property
    @abstractmethod
    def is_undoable(self) -> bool:
        """Whether this command can be placed on the undo stack."""


# ---------------------------------------------------------------------------
# Editable commands (undoable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AddBlockCommand(Command):
    """Add a new processing block to the graph."""

    block_id: str = ""
    name: str = ""
    block_type: str = ""
    category: BlockCategory = BlockCategory.PROCESSOR
    input_ports: tuple[tuple[str, str, str], ...] = ()
    output_ports: tuple[tuple[str, str, str], ...] = ()

    @property
    def is_undoable(self) -> bool:
        return True


@dataclass(frozen=True)
class RemoveBlockCommand(Command):
    """Remove a block and its connections from the graph."""

    block_id: str = ""

    @property
    def is_undoable(self) -> bool:
        return True


@dataclass(frozen=True)
class AddConnectionCommand(Command):
    """Create a connection between two block ports."""

    source_block_id: str = ""
    source_output_name: str = ""
    target_block_id: str = ""
    target_input_name: str = ""

    @property
    def is_undoable(self) -> bool:
        return True


@dataclass(frozen=True)
class RemoveConnectionCommand(Command):
    """Remove an existing connection between two block ports."""

    source_block_id: str = ""
    source_output_name: str = ""
    target_block_id: str = ""
    target_input_name: str = ""

    @property
    def is_undoable(self) -> bool:
        return True


@dataclass(frozen=True)
class ChangeBlockSettingsCommand(Command):
    """Modify a configuration value on a block instance."""

    block_id: str = ""
    setting_key: str = ""
    new_value: Any = None

    @property
    def is_undoable(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Operational commands (not undoable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecuteBlockCommand(Command):
    """Trigger execution of a single block in the pipeline."""

    block_id: str = ""

    @property
    def is_undoable(self) -> bool:
        return False
