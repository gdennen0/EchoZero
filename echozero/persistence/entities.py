"""
Persistence entities: Frozen dataclasses for EchoZero's project storage model.
Exists because the domain layer's types are engine-facing (runtime pipeline data),
while persistence needs additional UI/project state (color, order, visibility).
These DTOs map 1:1 to SQLite rows; repositories translate between them and domain types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

LayerType = Literal["analysis", "structure", "manual"]


# ---------------------------------------------------------------------------
# ProjectRecord settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectSettingsRecord:
    """Global project configuration — sample rate, tempo, timecode."""

    sample_rate: int = 44100
    bpm: float | None = None
    bpm_confidence: float | None = None
    timecode_fps: float | None = None


# ---------------------------------------------------------------------------
# ProjectRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectRecord:
    """Top-level container for a live-production project file."""

    id: str
    name: str
    settings: ProjectSettingsRecord
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# SongRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SongRecord:
    """A song in the setlist — owns one or more SongVersions."""

    id: str
    project_id: str
    title: str
    artist: str
    order: int
    active_version_id: str | None = None


# ---------------------------------------------------------------------------
# SongRecord version
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SongVersionRecord:
    """A specific mix or arrangement of a song, tied to one audio file."""

    id: str
    song_id: str
    label: str
    audio_file: str
    duration_seconds: float
    original_sample_rate: int
    audio_hash: str
    created_at: datetime
    rebuild_plan: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer record (persistence DTO — NOT the domain Layer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRecord:
    """
    Persistence-layer entity for a timeline layer. NOT the domain Layer.
    Carries UI state (color, order, visible, locked) that the pipeline engine ignores.

    Provenance/freshness notes:
    - source_pipeline stores generation provenance (pipeline_id, output_name, block_id, data_type)
    - state_flags stores editor/application state that is not engine concern
      Examples: stale, manually_modified, source_main_changed, derived
    - provenance stores richer source identity hooks for future UI inspection/remap work
      Examples: source_layer_id, source_song_version_id, source_run_id, source_output_name
    """

    id: str
    song_version_id: str
    name: str
    layer_type: LayerType
    color: str | None
    order: int
    visible: bool
    locked: bool
    parent_layer_id: str | None
    source_pipeline: dict[str, Any] | None
    created_at: datetime
    state_flags: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline config (persistent pipeline configuration per song)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfigRecord:
    """Persistent pipeline configuration for a song version.

    This is the first-class entity for pipeline settings. When a user adds a
    pipeline to a song, a template creates the initial config. After that,
    the config lives in the DB and the user owns it.

    - template_id: which template factory created this (for migration/reset)
    - graph_json: full serialized graph (blocks + connections + block settings)
    - outputs_json: pipeline output declarations (which ports → layers/takes)
    - knob_values: the user's current knob values (UI-facing settings panel)

    Knob values and graph_json are always in sync — updating a knob also updates
    the corresponding BlockSettings in the graph.
    """

    id: str
    song_version_id: str
    template_id: str
    name: str
    graph_json: str
    outputs_json: str
    knob_values: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    block_overrides: dict[str, list[str]] = field(default_factory=dict)
    """Tracks per-block setting overrides. {block_id: [setting_key, ...]}
    
    When a user edits a block's setting directly (via with_block_setting),
    that (block_id, key) pair is recorded here. Global knob updates skip
    overridden settings, preserving the user's per-block customizations.
    
    Use clear_block_override() to re-link a setting back to the global knob.
    """

    def with_knob_value(
        self,
        key: str,
        value: Any,
        knob_metadata: dict[str, Any] | None = None,
    ) -> PipelineConfigRecord:
        """Return a new config with an updated knob value.

        Updates knob_values and the corresponding BlockSettings in graph_json.

        If knob_metadata is provided and the knob has maps_to_block set,
        only that specific block is updated. Otherwise all blocks with
        the matching setting key are updated (global behavior).

        Args:
            key: Knob name (e.g. "threshold")
            value: New value
            knob_metadata: Optional dict of {knob_name: Knob} from the template.
                           Used to determine maps_to_block targeting.
        """
        import json
        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings
        from echozero.serialization import deserialize_graph, serialize_graph

        new_knob_values = dict(self.knob_values)
        new_knob_values[key] = value

        graph_data = json.loads(self.graph_json)
        graph = deserialize_graph(graph_data)

        # Determine target block(s)
        target_block_id = None
        if knob_metadata and key in knob_metadata:
            target_block_id = getattr(knob_metadata[key], "maps_to_block", None)

        for block_id, block in graph.blocks.items():
            if key not in block.settings:
                continue
            if target_block_id is not None and block_id != target_block_id:
                continue
            # Skip blocks where this setting has been overridden per-block
            if block_id in self.block_overrides and key in self.block_overrides[block_id]:
                continue
            new_settings = dict(block.settings)
            new_settings[key] = value
            graph.replace_block(_replace(block, settings=BlockSettings(new_settings)))

        return replace(
            self,
            knob_values=new_knob_values,
            graph_json=json.dumps(serialize_graph(graph)),
            updated_at=datetime.now(timezone.utc),
        )

    def with_knob_values(
        self,
        updates: dict[str, Any],
        knob_metadata: dict[str, Any] | None = None,
    ) -> PipelineConfigRecord:
        """Return a new config with multiple updated knob values. Batch version."""
        import json
        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings
        from echozero.serialization import deserialize_graph, serialize_graph

        new_knob_values = {**self.knob_values, **updates}

        graph_data = json.loads(self.graph_json)
        graph = deserialize_graph(graph_data)

        # Build per-block change sets respecting maps_to_block and overrides
        for block_id, block in graph.blocks.items():
            changed = {}
            overridden_keys = self.block_overrides.get(block_id, [])
            for key, value in updates.items():
                if key not in block.settings:
                    continue
                target = None
                if knob_metadata and key in knob_metadata:
                    target = getattr(knob_metadata[key], "maps_to_block", None)
                if target is not None and block_id != target:
                    continue
                # Skip overridden settings
                if key in overridden_keys:
                    continue
                changed[key] = value
            if changed:
                new_settings = {**dict(block.settings), **changed}
                graph.replace_block(_replace(block, settings=BlockSettings(new_settings)))

        return replace(
            self,
            knob_values=new_knob_values,
            graph_json=json.dumps(serialize_graph(graph)),
            updated_at=datetime.now(timezone.utc),
        )

    def with_block_setting(
        self,
        block_id: str,
        key: str,
        value: Any,
    ) -> PipelineConfigRecord:
        """Return a new config with a single block's setting updated.

        This is the per-block "inspector" level edit. Does NOT update
        knob_values — this is an independent override on a specific block.
        Marks the (block_id, key) pair as overridden so global knob updates
        won't clobber it.

        Raises KeyError if the block doesn't exist in the graph.
        """
        import json
        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings
        from echozero.serialization import deserialize_graph, serialize_graph

        graph_data = json.loads(self.graph_json)
        graph = deserialize_graph(graph_data)

        block = graph.blocks.get(block_id)
        if block is None:
            raise KeyError(f"Block '{block_id}' not found in pipeline graph")

        new_settings = dict(block.settings)
        new_settings[key] = value
        graph.replace_block(_replace(block, settings=BlockSettings(new_settings)))

        # Record override
        new_overrides = {k: list(v) for k, v in self.block_overrides.items()}
        if block_id not in new_overrides:
            new_overrides[block_id] = []
        if key not in new_overrides[block_id]:
            new_overrides[block_id].append(key)

        return replace(
            self,
            graph_json=json.dumps(serialize_graph(graph)),
            block_overrides=new_overrides,
            updated_at=datetime.now(timezone.utc),
        )

    def with_block_settings(
        self,
        block_id: str,
        updates: dict[str, Any],
    ) -> PipelineConfigRecord:
        """Return a new config with multiple settings updated on a single block."""
        import json
        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings
        from echozero.serialization import deserialize_graph, serialize_graph

        graph_data = json.loads(self.graph_json)
        graph = deserialize_graph(graph_data)

        block = graph.blocks.get(block_id)
        if block is None:
            raise KeyError(f"Block '{block_id}' not found in pipeline graph")

        new_settings = {**dict(block.settings), **updates}
        graph.replace_block(_replace(block, settings=BlockSettings(new_settings)))

        # Record overrides
        new_overrides = {k: list(v) for k, v in self.block_overrides.items()}
        if block_id not in new_overrides:
            new_overrides[block_id] = []
        for key in updates:
            if key not in new_overrides[block_id]:
                new_overrides[block_id].append(key)

        return replace(
            self,
            graph_json=json.dumps(serialize_graph(graph)),
            block_overrides=new_overrides,
            updated_at=datetime.now(timezone.utc),
        )

    def clear_block_override(
        self,
        block_id: str,
        key: str,
    ) -> PipelineConfigRecord:
        """Remove a per-block override, re-linking the setting to the global knob.

        The setting value is updated to match the current knob value.
        """
        import json
        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings
        from echozero.serialization import deserialize_graph, serialize_graph

        new_overrides = {k: list(v) for k, v in self.block_overrides.items()}
        if block_id in new_overrides and key in new_overrides[block_id]:
            new_overrides[block_id].remove(key)
            if not new_overrides[block_id]:
                del new_overrides[block_id]

        # Restore block setting to current knob value
        graph_data = json.loads(self.graph_json)
        graph = deserialize_graph(graph_data)

        block = graph.blocks.get(block_id)
        if block is not None and key in self.knob_values:
            new_settings = dict(block.settings)
            new_settings[key] = self.knob_values[key]
            graph.replace_block(_replace(block, settings=BlockSettings(new_settings)))

        return replace(
            self,
            graph_json=json.dumps(serialize_graph(graph)),
            block_overrides=new_overrides,
            updated_at=datetime.now(timezone.utc),
        )

    def to_pipeline(self):
        """Deserialize into a runnable Pipeline object."""
        import json

        from echozero.serialization import deserialize_pipeline

        data = json.loads(self.graph_json)
        outputs = json.loads(self.outputs_json)
        # Merge graph + outputs into pipeline format
        pipeline_data = {
            "id": self.template_id,
            "name": self.name,
            "graph": data,
            "outputs": outputs,
        }
        return deserialize_pipeline(pipeline_data)

    @classmethod
    def from_pipeline(
        cls,
        pipeline,
        template_id: str,
        song_version_id: str,
        knob_values: dict[str, Any],
        config_id: str | None = None,
        name: str | None = None,
    ) -> PipelineConfigRecord:
        """Create a PipelineConfigRecord from a freshly-built Pipeline."""
        import json
        import uuid

        from echozero.serialization import serialize_graph

        now = datetime.now(timezone.utc)

        graph_json = json.dumps(serialize_graph(pipeline.graph))
        outputs_json = json.dumps(
            [
                {
                    "name": out.name,
                    "block_id": out.port_ref.block_id,
                    "port_name": out.port_ref.port_name,
                }
                for out in pipeline.outputs
            ]
        )

        return cls(
            id=config_id or uuid.uuid4().hex,
            song_version_id=song_version_id,
            template_id=template_id,
            name=name or pipeline.name,
            graph_json=graph_json,
            outputs_json=outputs_json,
            knob_values=dict(knob_values),
            created_at=now,
            updated_at=now,
        )


@dataclass(frozen=True)
class SongDefaultPipelineConfigRecord:
    """Persistent song-level default pipeline configuration."""

    id: str
    song_id: str
    template_id: str
    name: str
    graph_json: str
    outputs_json: str
    knob_values: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    block_overrides: dict[str, list[str]] = field(default_factory=dict)

    def with_knob_value(
        self,
        key: str,
        value: Any,
        knob_metadata: dict[str, Any] | None = None,
    ) -> SongDefaultPipelineConfigRecord:
        updated = PipelineConfigRecord(
            id=self.id,
            song_version_id="",
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        ).with_knob_value(key, value, knob_metadata=knob_metadata)
        return self.from_version_config(updated, song_id=self.song_id)

    def with_knob_values(
        self,
        updates: dict[str, Any],
        knob_metadata: dict[str, Any] | None = None,
    ) -> SongDefaultPipelineConfigRecord:
        updated = PipelineConfigRecord(
            id=self.id,
            song_version_id="",
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        ).with_knob_values(updates, knob_metadata=knob_metadata)
        return self.from_version_config(updated, song_id=self.song_id)

    def with_block_setting(
        self, block_id: str, key: str, value: Any
    ) -> SongDefaultPipelineConfigRecord:
        updated = PipelineConfigRecord(
            id=self.id,
            song_version_id="",
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        ).with_block_setting(block_id, key, value)
        return self.from_version_config(updated, song_id=self.song_id)

    def with_block_settings(
        self, block_id: str, updates: dict[str, Any]
    ) -> SongDefaultPipelineConfigRecord:
        updated = PipelineConfigRecord(
            id=self.id,
            song_version_id="",
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        ).with_block_settings(block_id, updates)
        return self.from_version_config(updated, song_id=self.song_id)

    def clear_block_override(self, block_id: str, key: str) -> SongDefaultPipelineConfigRecord:
        updated = PipelineConfigRecord(
            id=self.id,
            song_version_id="",
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        ).clear_block_override(block_id, key)
        return self.from_version_config(updated, song_id=self.song_id)

    def to_version_config(self, *, song_version_id: str) -> PipelineConfigRecord:
        return PipelineConfigRecord(
            id=self.id,
            song_version_id=song_version_id,
            template_id=self.template_id,
            name=self.name,
            graph_json=self.graph_json,
            outputs_json=self.outputs_json,
            knob_values=self.knob_values,
            created_at=self.created_at,
            updated_at=self.updated_at,
            block_overrides=self.block_overrides,
        )

    @classmethod
    def from_version_config(
        cls,
        config: PipelineConfigRecord,
        *,
        song_id: str,
        config_id: str | None = None,
    ) -> SongDefaultPipelineConfigRecord:
        return cls(
            id=config_id or config.id,
            song_id=song_id,
            template_id=config.template_id,
            name=config.name,
            graph_json=config.graph_json,
            outputs_json=config.outputs_json,
            knob_values=dict(config.knob_values),
            created_at=config.created_at,
            updated_at=config.updated_at,
            block_overrides={k: list(v) for k, v in config.block_overrides.items()},
        )
