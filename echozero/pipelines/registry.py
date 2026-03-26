"""Pipeline template registry. Templates are Python builder functions that return Graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from echozero.domain.graph import Graph


@dataclass(frozen=True)
class PromotedParam:
    """A user-visible parameter promoted from a block setting."""

    key: str           # programmatic key used in bindings
    name: str          # display name
    type: type         # python type (str, float, int, bool)
    required: bool = False
    default: Any = None
    description: str = ''


@dataclass(frozen=True)
class PipelineTemplate:
    """A registered pipeline template with metadata."""

    id: str
    name: str
    description: str
    promoted_params: tuple[PromotedParam, ...]
    builder: Callable[[], Graph]

    def build(self) -> Graph:
        return self.builder()

    def validate_bindings(self, bindings: dict[str, Any]) -> list[str]:
        """Return list of validation errors for the given bindings."""
        errors = []
        for param in self.promoted_params:
            if param.required and param.key not in bindings:
                errors.append(f'Missing required parameter: {param.name} ({param.key})')
            if param.key in bindings:
                value = bindings[param.key]
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
        self._templates[template.id] = template

    def get(self, template_id: str) -> PipelineTemplate | None:
        return self._templates.get(template_id)

    def list(self) -> list[PipelineTemplate]:
        return list(self._templates.values())

    def ids(self) -> list[str]:
        return list(self._templates.keys())


# Global registry instance
_registry = PipelineRegistry()


def get_registry() -> PipelineRegistry:
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
