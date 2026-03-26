"""Pipeline template registry. Templates are Python builder functions that return Graphs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

from echozero.domain.graph import Graph
from echozero.domain.types import BlockSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromotedParam:
    """A user-visible parameter promoted from a block setting."""

    key: str           # programmatic key used in bindings
    name: str          # display name
    type: type         # python type (str, float, int, bool)
    required: bool = False
    default: Any = None
    description: str = ''
    maps_to_block: str = ''    # block_id this param maps to
    maps_to_setting: str = ''  # setting key within that block


def apply_bindings(
    graph: Graph,
    bindings: dict[str, Any],
    promoted_params: tuple[PromotedParam, ...],
) -> Graph:
    """Apply user-supplied bindings to the graph by updating block settings.

    For each binding key/value, finds the promoted param's maps_to_block and
    maps_to_setting, then updates that block's settings with the bound value.
    """
    for key, value in bindings.items():
        for param in promoted_params:
            if param.key == key and param.maps_to_block and param.maps_to_setting:
                block = graph.blocks.get(param.maps_to_block)
                if block is not None:
                    new_entries = dict(block.settings.entries)
                    new_entries[param.maps_to_setting] = value
                    new_block = replace(
                        block, settings=BlockSettings(entries=new_entries)
                    )
                    # Replace in-place to preserve connections
                    graph.blocks[block.id] = new_block
                break
    return graph


@dataclass(frozen=True)
class PipelineTemplate:
    """A registered pipeline template with metadata."""

    id: str
    name: str
    description: str
    promoted_params: tuple[PromotedParam, ...]
    builder: Callable[[], Graph]

    def build(self, bindings: dict[str, Any] | None = None) -> Graph:
        """Build the pipeline graph, optionally applying user-supplied bindings.

        Args:
            bindings: Optional dict mapping promoted param keys to values.
                      If provided, block settings are updated after building.

        Returns:
            A Graph instance with bindings applied.
        """
        graph = self.builder()
        if bindings:
            graph = apply_bindings(graph, bindings, self.promoted_params)
        return graph

    def validate_bindings(self, bindings: dict[str, Any]) -> list[str]:
        """Return list of validation errors for the given bindings.

        Checks for missing required params, wrong types (with int→float coercion),
        and warns about unknown keys not in promoted_params.
        """
        errors = []
        known_keys = {p.key for p in self.promoted_params}

        # Warn about unknown keys
        for key in bindings:
            if key not in known_keys:
                errors.append(f'Unknown parameter: {key}')

        for param in self.promoted_params:
            if param.required and param.key not in bindings:
                errors.append(f'Missing required parameter: {param.name} ({param.key})')
            if param.key in bindings:
                value = bindings[param.key]
                # Allow int where float is expected (int→float coercion)
                if param.type is float and isinstance(value, int) and not isinstance(value, bool):
                    continue
                if not isinstance(value, param.type):
                    errors.append(
                        f'Parameter {param.key}: expected {param.type.__name__}, '
                        f'got {type(value).__name__}'
                    )
        return errors


class PipelineRegistry:
    """Global registry of pipeline templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PipelineTemplate] = {}

    def register(self, template: PipelineTemplate) -> None:
        """Register a pipeline template. Overwrites any existing template with the same ID."""
        self._templates[template.id] = template

    def get(self, template_id: str) -> PipelineTemplate | None:
        """Return a template by ID, or None if not found."""
        return self._templates.get(template_id)

    def list(self) -> list[PipelineTemplate]:
        """Return all registered templates."""
        return list(self._templates.values())

    def ids(self) -> list[str]:
        """Return all registered template IDs."""
        return list(self._templates.keys())


# Global registry instance
_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
    """Return the global pipeline registry singleton."""
    return _registry


def pipeline_template(
    id: str,
    name: str,
    description: str = '',
    promoted_params: list[PromotedParam] | None = None,
) -> Callable:
    """Decorator that registers a graph builder function as a pipeline template."""
    def decorator(fn: Callable[[], Graph]) -> Callable[[], Graph]:
        template = PipelineTemplate(
            id=id,
            name=name,
            description=description,
            promoted_params=tuple(promoted_params or []),
            builder=fn,
        )
        _registry.register(template)
        return fn
    return decorator
