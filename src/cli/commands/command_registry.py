"""
Command Registry

Declarative store that keeps command metadata (names, aliases, help, handler).
UIs and the CLI can query the registry to discover available commands and execute them
without scattering `if/elif` chains throughout the parser.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.application.api.result_types import CommandResult


@dataclass
class CommandDefinition:
    """Describe a single command entry."""
    name: str
    handler: Callable[[List[str], Dict[str, str]], CommandResult]
    usage: str
    description: str
    aliases: List[str] = field(default_factory=list)


class CommandRegistry:
    """Registry of command definitions and lookup helpers."""

    def __init__(self):
        self._definitions: Dict[str, CommandDefinition] = {}
        self._lookup: Dict[str, CommandDefinition] = {}

    def register(self, definition: CommandDefinition) -> None:
        """Register a command definition and its aliases."""
        key = definition.name.lower()
        self._definitions[key] = definition
        self._lookup[key] = definition

        for alias in definition.aliases:
            self._lookup[alias.lower()] = definition

    def get(self, command_name: str) -> Optional[CommandDefinition]:
        """Lookup a command definition by name or alias."""
        if not command_name:
            return None
        return self._lookup.get(command_name.lower())

    def list_commands(self) -> List[CommandDefinition]:
        """Return all registered definitions (one per canonical name)."""
        return sorted(self._definitions.values(), key=lambda d: d.name.lower())

