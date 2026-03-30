"""
Serialization: Save and load durable state containers (Graph + TakeLayers) to JSON.
Exists because project persistence requires a stable serialization format for the pipeline model.
Used by the application layer to save/load projects; cache is discardable and not serialized.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from echozero.domain.enums import BlockCategory, BlockState, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import (
    AudioData,
    Block,
    BlockSettings,
    Connection,
    Event,
    EventData,
    Layer,
    Port,
)
from echozero.pipelines.pipeline import Pipeline, PipelineOutput, PortRef
from echozero.takes import Take, TakeLayer, TakeSource


# ---------------------------------------------------------------------------
# Pipeline serialization (Graph + named outputs)
# ---------------------------------------------------------------------------


def serialize_pipeline(pipeline: Pipeline) -> dict[str, Any]:
    """Convert a Pipeline (graph + outputs) to a JSON-serializable dict."""
    return {
        "id": pipeline.id,
        "name": pipeline.name,
        "description": pipeline.description,
        "graph": serialize_graph(pipeline.graph),
        "outputs": [
            {
                "name": out.name,
                "block_id": out.port_ref.block_id,
                "port_name": out.port_ref.port_name,
            }
            for out in pipeline.outputs
        ],
    }


def deserialize_pipeline(data: dict[str, Any]) -> Pipeline:
    """Reconstruct a Pipeline from a serialized dict.

    Uses Pipeline's public constructor (graph + outputs params) instead of
    reaching into private attributes — avoids bypassing validation (S2).
    Duplicate output names are caught explicitly here.
    """
    from echozero.errors import ValidationError

    graph = deserialize_graph(data["graph"])

    outputs: list[PipelineOutput] = []
    seen_names: set[str] = set()
    for out_data in data.get("outputs", []):
        name = out_data["name"]
        if name in seen_names:
            raise ValidationError(f"Duplicate pipeline output name: {name!r}")
        seen_names.add(name)
        port_ref = PortRef(out_data["block_id"], out_data["port_name"])
        outputs.append(PipelineOutput(name, port_ref))

    return Pipeline(
        id=data["id"],
        name=data.get("name", data["id"]),
        description=data.get("description", ""),
        graph=graph,
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Graph serialization
# ---------------------------------------------------------------------------


def serialize_graph(graph: Graph) -> dict[str, Any]:
    """Convert a Graph to a JSON-serializable dict."""
    blocks = []
    for block in graph.blocks.values():
        blocks.append({
            "id": block.id,
            "name": block.name,
            "block_type": block.block_type,
            "category": block.category.name,
            "state": block.state.name,
            "input_ports": [
                {"name": p.name, "port_type": p.port_type.name, "direction": p.direction.name}
                for p in block.input_ports
            ],
            "output_ports": [
                {"name": p.name, "port_type": p.port_type.name, "direction": p.direction.name}
                for p in block.output_ports
            ],
            "control_ports": [
                {"name": p.name, "port_type": p.port_type.name, "direction": p.direction.name}
                for p in block.control_ports
            ],
            "settings": dict(block.settings),
        })

    connections = []
    for conn in graph.connections:
        connections.append({
            "source_block_id": conn.source_block_id,
            "source_output_name": conn.source_output_name,
            "target_block_id": conn.target_block_id,
            "target_input_name": conn.target_input_name,
        })

    return {"blocks": blocks, "connections": connections}


def deserialize_graph(data: dict[str, Any]) -> Graph:
    """Reconstruct a Graph from a serialized dict. All blocks loaded as STALE."""
    graph = Graph()

    for block_data in data["blocks"]:
        block = Block(
            id=block_data["id"],
            name=block_data["name"],
            block_type=block_data["block_type"],
            category=BlockCategory[block_data["category"]],
            input_ports=tuple(
                Port(
                    name=p["name"],
                    port_type=PortType[p["port_type"]],
                    direction=Direction[p["direction"]],
                )
                for p in block_data["input_ports"]
            ),
            output_ports=tuple(
                Port(
                    name=p["name"],
                    port_type=PortType[p["port_type"]],
                    direction=Direction[p["direction"]],
                )
                for p in block_data["output_ports"]
            ),
            control_ports=tuple(
                Port(
                    name=p["name"],
                    port_type=PortType[p["port_type"]],
                    direction=Direction[p["direction"]],
                )
                for p in block_data.get("control_ports", [])
            ),
            settings=BlockSettings(block_data.get("settings", {})),
            state=BlockState[block_data.get("state", "FRESH")],
        )
        graph.add_block(block)

    for conn_data in data["connections"]:
        graph.add_connection(
            Connection(
                source_block_id=conn_data["source_block_id"],
                source_output_name=conn_data["source_output_name"],
                target_block_id=conn_data["target_block_id"],
                target_input_name=conn_data["target_input_name"],
            )
        )

    return graph


# ---------------------------------------------------------------------------
# Event / data serialization helpers
# ---------------------------------------------------------------------------


def _serialize_event(event: Event) -> dict[str, Any]:
    """Serialize an Event to a dict."""
    return {
        "id": event.id,
        "time": event.time,
        "duration": event.duration,
        "classifications": event.classifications,
        "metadata": event.metadata,
        "origin": event.origin,
    }


def _deserialize_event(data: dict[str, Any]) -> Event:
    """Reconstruct an Event from a dict."""
    return Event(
        id=data["id"],
        time=data["time"],
        duration=data["duration"],
        classifications=data["classifications"],
        metadata=data["metadata"],
        origin=data["origin"],
    )


def _serialize_layer(layer: Layer) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "events": [_serialize_event(e) for e in layer.events],
    }


def _deserialize_layer(data: dict[str, Any]) -> Layer:
    return Layer(
        id=data["id"],
        name=data["name"],
        events=tuple(_deserialize_event(e) for e in data["events"]),
    )


def _serialize_event_data(ed: EventData) -> dict[str, Any]:
    return {
        "type": "EventData",
        "layers": [_serialize_layer(l) for l in ed.layers],
    }


def _deserialize_event_data(data: dict[str, Any]) -> EventData:
    return EventData(
        layers=tuple(_deserialize_layer(l) for l in data["layers"]),
    )


def _serialize_audio_data(ad: AudioData) -> dict[str, Any]:
    return {
        "type": "AudioData",
        "sample_rate": ad.sample_rate,
        "duration": ad.duration,
        "file_path": ad.file_path,
        "channel_count": ad.channel_count,
    }


def _deserialize_audio_data(data: dict[str, Any]) -> AudioData:
    return AudioData(
        sample_rate=data["sample_rate"],
        duration=data["duration"],
        file_path=data["file_path"],
        channel_count=data.get("channel_count", 1),
    )


def serialize_take_data(data: EventData | AudioData) -> dict[str, Any]:
    """Serialize EventData or AudioData to a JSON-compatible dict."""
    if isinstance(data, EventData):
        return _serialize_event_data(data)
    elif isinstance(data, AudioData):
        return _serialize_audio_data(data)
    raise TypeError(f"Unknown take data type: {type(data)}")


def deserialize_take_data(data: dict[str, Any]) -> EventData | AudioData:
    """Reconstruct EventData or AudioData from a serialized dict."""
    dtype = data.get("type")
    if dtype == "EventData":
        return _deserialize_event_data(data)
    elif dtype == "AudioData":
        return _deserialize_audio_data(data)
    raise ValueError(f"Unknown take data type: {dtype}")


# ---------------------------------------------------------------------------
# Take serialization
# ---------------------------------------------------------------------------


def serialize_take(take: Take) -> dict[str, Any]:
    """Convert a Take to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "id": take.id,
        "label": take.label,
        "origin": take.origin,
        "created_at": take.created_at.isoformat(),
        "is_main": take.is_main,
        "notes": take.notes,
        "data": serialize_take_data(take.data),
    }
    if take.source is not None:
        result["source"] = {
            "block_id": take.source.block_id,
            "block_type": take.source.block_type,
            "settings_snapshot": take.source.settings_snapshot,
            "run_id": take.source.run_id,
        }
    else:
        result["source"] = None
    return result


def deserialize_take(data: dict[str, Any]) -> Take:
    """Reconstruct a Take from a serialized dict."""
    source = None
    if data.get("source") is not None:
        s = data["source"]
        source = TakeSource(
            block_id=s["block_id"],
            block_type=s["block_type"],
            settings_snapshot=s["settings_snapshot"],
            run_id=s["run_id"],
        )
    return Take(
        id=data["id"],
        label=data["label"],
        data=deserialize_take_data(data["data"]),
        origin=data["origin"],
        source=source,
        created_at=datetime.fromisoformat(data["created_at"]),
        is_main=data["is_main"],
        notes=data.get("notes", ""),
    )


def serialize_take_layer(layer: TakeLayer) -> dict[str, Any]:
    """Convert a TakeLayer to a JSON-serializable dict."""
    return {
        "layer_id": layer.layer_id,
        "takes": [serialize_take(t) for t in layer.takes],
    }


def deserialize_take_layer(data: dict[str, Any]) -> TakeLayer:
    """Reconstruct a TakeLayer from a serialized dict."""
    takes = [deserialize_take(t) for t in data["takes"]]
    return TakeLayer(layer_id=data["layer_id"], takes=takes)


# ---------------------------------------------------------------------------
# Project save/load
# ---------------------------------------------------------------------------


def save_project(
    path: str,
    graph: Graph,
    take_layers: list[TakeLayer] | None = None,
) -> None:
    """Write the project (Graph + TakeLayers) to a JSON file."""
    data: dict[str, Any] = {
        "version": "2.1.0",
        "graph": serialize_graph(graph),
    }
    if take_layers:
        data["take_layers"] = [serialize_take_layer(tl) for tl in take_layers]
    else:
        data["take_layers"] = []
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_project(path: str) -> tuple[Graph, list[TakeLayer]]:
    """Read a project from a JSON file. All blocks loaded as STALE."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = deserialize_graph(data["graph"])
    take_layers = [
        deserialize_take_layer(tl) for tl in data.get("take_layers", [])
    ]
    return graph, take_layers

