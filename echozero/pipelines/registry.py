"""Pipeline template registry. Templates are Python builder functions that return Pipelines."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from echozero.domain.graph import Graph
from echozero.pipelines.params import Knob, extract_knobs, validate_bindings as _validate_knob_bindings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineTemplate:
    """A registered pipeline template with metadata."""

    id: str
    name: str
    description: str
    knobs: dict[str, Knob] = field(default_factory=dict)
    builder: Callable[..., Any] = None  # Returns Pipeline (from pipelines.pipeline)

    def build(self, bindings: dict[str, Any] | None = None) -> Graph:
        """Build the pipeline, passing knob values to the builder function.

        Returns the Graph from the built Pipeline for backward compatibility.
        """
        from echozero.pipelines.pipeline import Pipeline as EnginePipeline

        sig = inspect.signature(self.builder)
        params = sig.parameters

        # Build kwargs from knob defaults + bindings
        kwargs: dict[str, Any] = {}
        for pname, param in params.items():
            if pname in self.knobs:
                if bindings and pname in bindings:
                    kwargs[pname] = bindings[pname]
                else:
                    kwargs[pname] = self.knobs[pname].default
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default

        result = self.builder(**kwargs)

        # Support both Pipeline objects and raw Graph returns
        if isinstance(result, EnginePipeline):
            return result.graph
        elif isinstance(result, Graph):
            return result
        else:
            raise TypeError(
                f"Pipeline builder '{self.id}' returned {type(result).__name__}, "
                f"expected Pipeline or Graph."
            )

    def build_pipeline(self, bindings: dict[str, Any] | None = None) -> Any:
        """Build and return the full Pipeline object (not just Graph).

        Use this when you need access to pipeline outputs.
        """
        from echozero.pipelines.pipeline import Pipeline as EnginePipeline

        sig = inspect.signature(self.builder)
        params = sig.parameters

        kwargs: dict[str, Any] = {}
        for pname, param in params.items():
            if pname in self.knobs:
                if bindings and pname in bindings:
                    kwargs[pname] = bindings[pname]
                else:
                    kwargs[pname] = self.knobs[pname].default
            elif param.default is not inspect.Parameter.empty:
                kwargs[pname] = param.default

        result = self.builder(**kwargs)

        if isinstance(result, EnginePipeline):
            return result
        elif isinstance(result, Graph):
            # Wrap legacy Graph in a Pipeline
            p = EnginePipeline(id=self.id, name=self.name)
            p._graph = result
            return p
        else:
            raise TypeError(
                f"Pipeline builder '{self.id}' returned {type(result).__name__}, "
                f"expected Pipeline or Graph."
            )

    def validate_bindings(self, bindings: dict[str, Any]) -> list[str]:
        """Validate user-supplied bindings against this template's knobs."""
        if not self.knobs:
            return []
        return _validate_knob_bindings(self.knobs, bindings)


class PipelineRegistry:
    """Global registry of pipeline templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PipelineTemplate] = {}

    def register(self, template: PipelineTemplate) -> None:
        """Register a pipeline template. Warns if overwriting an existing template."""
        if template.id in self._templates:
            logger.warning(
                "Overwriting existing pipeline template '%s'. "
                "This may indicate a duplicate registration.",
                template.id,
            )
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
    knobs: dict[str, Knob] | None = None,
) -> Callable:
    """Decorator that registers a builder function as a pipeline template.

    The builder function should return a Pipeline (or Graph for backward compatibility).
    Knobs can be specified explicitly or extracted from function signature.

    Example::

        @pipeline_template(id="onset_detection", name="Detect Onsets")
        def onset_detection(
            threshold=knob(0.3, label="Sensitivity", min_value=0.0, max_value=1.0),
        ):
            p = Pipeline("onset_detection", name="Detect Onsets")
            load = p.add(LoadAudio())
            onsets = p.add(DetectOnsets(threshold=threshold), audio_in=load.audio_out)
            p.output("onsets", onsets.events_out)
            return p
    """
    def decorator(fn: Callable) -> Callable:
        # Extract knobs from function signature if not provided explicitly
        resolved_knobs = knobs if knobs is not None else extract_knobs(fn)

        template = PipelineTemplate(
            id=id,
            name=name,
            description=description or (fn.__doc__ or "").strip(),
            knobs=resolved_knobs,
            builder=fn,
        )
        _registry.register(template)
        return fn
    return decorator
