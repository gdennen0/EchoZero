"""
Pipeline: Engine-level pipeline construction via add() / output() API.

Exists because manually constructing Block/Port/Connection objects is tedious.
This module provides a clean, fluent API for building pipeline graphs with
named outputs. Engine-level concept — knows nothing about DB, layers, takes, or UI.
"""

from __future__ import annotations

from typing import Any

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
from echozero.errors import ValidationError


# ---------------------------------------------------------------------------
# PortRef / BlockHandle — references returned by Pipeline.add()
# ---------------------------------------------------------------------------


class PortRef:
    """Reference to a specific output port of a block."""

    def __init__(self, block_id: str, port_name: str) -> None:
        self.block_id = block_id
        self.port_name = port_name

    def __repr__(self) -> str:
        return f"<PortRef: {self.block_id}.{self.port_name}>"


class BlockHandle:
    """Reference to a block in the pipeline. Attribute access returns port references."""

    def __init__(self, block_id: str, output_ports: dict[str, str]) -> None:
        self._block_id = block_id
        self._output_ports = output_ports  # name -> port_type

    def __getattr__(self, name: str) -> PortRef:
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._output_ports:
            return PortRef(self._block_id, name)
        available = ", ".join(self._output_ports.keys())
        raise AttributeError(
            f"Block '{self._block_id}' has no output port '{name}'. "
            f"Available: {available}"
        )

    def __repr__(self) -> str:
        return f"<BlockHandle: {self._block_id}>"


# ---------------------------------------------------------------------------
# PipelineOutput — declared named output
# ---------------------------------------------------------------------------


class PipelineOutput:
    """Declared output of a pipeline — name + port reference."""

    def __init__(self, name: str, port_ref: PortRef) -> None:
        self.name = name
        self.port_ref = port_ref

    def __repr__(self) -> str:
        return f"<PipelineOutput: {self.name} -> {self.port_ref}>"


# ---------------------------------------------------------------------------
# Pipeline — the centerpiece
# ---------------------------------------------------------------------------


class Pipeline:
    """A processing pipeline: graph + named outputs.

    Engine-level concept. Knows nothing about DB, layers, takes, or UI.

    Usage::

        p = Pipeline("onset_detection", name="Detect Onsets")
        load = p.add(LoadAudio(file_path="song.wav"))
        onsets = p.add(DetectOnsets(threshold=0.3), audio_in=load.audio_out)
        p.output("onsets", onsets.events_out)
    """

    def __init__(self, id: str, name: str = "", description: str = "") -> None:
        self.id = id
        self.name = name or id
        self.description = description
        self._graph = Graph()
        self._outputs: list[PipelineOutput] = []
        self._block_counter: dict[str, int] = {}

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def outputs(self) -> list[PipelineOutput]:
        return list(self._outputs)

    def add(self, block_spec: "BlockSpec", *, id: str | None = None, **input_connections: Any) -> BlockHandle:
        """Add a block to the pipeline. Returns a handle for referencing outputs.

        Args:
            block_spec: A BlockSpec describing the block's type, ports, and settings.
            id: Optional explicit block ID. Auto-generated if not provided.
            **input_connections: keyword args mapping input port names to PortRef objects
                                e.g., audio_in=load.audio_out

        Returns:
            BlockHandle for referencing this block's outputs.
        """
        from echozero.pipelines.block_specs import BlockSpec as _BS

        block_type = block_spec.block_type

        # Auto-generate ID if not provided
        if id is None:
            count = self._block_counter.get(block_type, 0) + 1
            self._block_counter[block_type] = count
            block_id = f"{block_type}_{count}"
        else:
            block_id = id

        # Build Port tuples from PortSpec tuples
        input_ports = tuple(
            Port(
                name=ps.name,
                port_type=ps.port_type,
                direction=ps.direction,
            )
            for ps in block_spec.input_ports
        )
        output_ports = tuple(
            Port(
                name=ps.name,
                port_type=ps.port_type,
                direction=ps.direction,
            )
            for ps in block_spec.output_ports
        )

        # Resolve any Knob defaults in settings
        from echozero.pipelines.params import Knob

        resolved_settings: dict[str, Any] = {}
        for k, v in block_spec.settings.items():
            if isinstance(v, Knob):
                resolved_settings[k] = v.default
            else:
                resolved_settings[k] = v

        block = Block(
            id=block_id,
            name=block_spec.name or block_type,
            block_type=block_type,
            category=block_spec.category,
            input_ports=input_ports,
            output_ports=output_ports,
            settings=BlockSettings(resolved_settings),
        )
        self._graph.add_block(block)

        # Create connections from input_connections kwargs
        for input_port_name, port_ref in input_connections.items():
            if isinstance(port_ref, PortRef):
                connection = Connection(
                    source_block_id=port_ref.block_id,
                    source_output_name=port_ref.port_name,
                    target_block_id=block_id,
                    target_input_name=input_port_name,
                )
                self._graph.add_connection(connection)
            else:
                raise ValidationError(
                    f"Input '{input_port_name}' on block '{block_id}' expected a PortRef "
                    f"(output of another block), got {type(port_ref).__name__}."
                )

        # Build output port dict for the handle
        handle_ports = {ps.name: ps.port_type for ps in block_spec.output_ports}
        return BlockHandle(block_id, handle_ports)

    def output(self, name: str, port_ref_or_handle: PortRef | BlockHandle) -> None:
        """Declare a named output of this pipeline.

        Args:
            name: Output name (e.g., "drums_onsets"). Must be unique.
            port_ref_or_handle: Either a PortRef (specific port) or BlockHandle (default output)

        Raises:
            ValidationError: If an output with this name already exists.
        """
        existing_names = {o.name for o in self._outputs}
        if name in existing_names:
            raise ValidationError(
                f"Duplicate pipeline output name '{name}'. "
                f"Each output must have a unique name."
            )
        if isinstance(port_ref_or_handle, BlockHandle):
            # Use first output port as default
            port_ref = PortRef(
                port_ref_or_handle._block_id,
                list(port_ref_or_handle._output_ports.keys())[0],
            )
        elif isinstance(port_ref_or_handle, PortRef):
            port_ref = port_ref_or_handle
        else:
            raise ValidationError(
                f"output() expects a PortRef or BlockHandle, got {type(port_ref_or_handle).__name__}"
            )
        self._outputs.append(PipelineOutput(name, port_ref))

