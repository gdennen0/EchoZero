"""
Orchestrator: Runs pipelines against songs and persists results.

The bridge between engine (computes) and persistence (stores).
Flow: load audio path → build pipeline → execute → map outputs → persist.

Persistence mapping uses PIPELINE OUTPUTS, not internal block wiring.
The pipeline declares p.output("onsets", ...) — the Orchestrator maps
"onsets" to a Layer+Take. It never reaches into block IDs or port names.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from echozero.domain.types import AudioData, EventData
from echozero.errors import ExecutionError, ValidationError
from echozero.execution import BlockExecutor, ExecutionEngine, GraphPlanner
from echozero.persistence.entities import LayerRecord, PipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.pipeline import Pipeline, PipelineOutput
from echozero.pipelines.registry import PipelineRegistry
from echozero.progress import RuntimeBus
from echozero.result import Err, Result, err, is_err, ok, unwrap
from echozero.takes import Take, TakeSource


# ---------------------------------------------------------------------------
# OutputMapping — how a named pipeline output gets persisted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputMapping:
    """Declares how a named pipeline output should be persisted.

    This is purely an application-level concept.

    Attributes:
        output_name: Pipeline output name (matches p.output("name", ...))
        target: Persistence target — "layer_take" or "song_version"
        label: Human-readable label for the persisted artifact
        params: Extra handler params (e.g., layer_type override)
    """

    output_name: str
    target: str = "auto"  # "auto", "layer_take", "song_version"
    label: str = ""       # empty = derive from output_name
    params: dict[str, Any] = field(default_factory=dict)



@dataclass(frozen=True)
class AnalysisResult:
    """Result of running analysis on a song version."""

    song_version_id: str
    pipeline_id: str
    layer_ids: list[str]
    take_ids: list[str]
    duration_ms: float


class Orchestrator:
    """Runs pipelines against songs and persists results.

    Pipeline outputs are the contract. The Orchestrator reads
    pipeline.outputs after execution and maps each to persistence.
    It never reaches into block IDs or port names.

    Auto-mapping:
        - EventData outputs → Layer + Take (layer_take)
        - AudioData outputs → SongVersionRecord (song_version)
        - Other types → skipped with warning

    Custom OutputMappings can override labels and target types.
    """

    PersistenceHandler = Callable[..., tuple[list[str], list[str]]]

    DEFAULT_MAX_TAKES_PER_LAYER = 20

    def __init__(
        self,
        registry: PipelineRegistry,
        executors: dict[str, BlockExecutor],
        output_mappings: dict[str, list[OutputMapping]] | None = None,
        max_takes_per_layer: int = DEFAULT_MAX_TAKES_PER_LAYER,
    ) -> None:
        self._registry = registry
        self._executors = executors
        self._output_mappings: dict[str, list[OutputMapping]] = output_mappings or {}
        self._max_takes_per_layer = max_takes_per_layer

        self._persistence_handlers: dict[str, Orchestrator.PersistenceHandler] = {
            "layer_take": self._handle_persist_as_layer_take,
            "song_version": self._handle_persist_as_song_version,
        }

    def create_config(
        self,
        session: ProjectStorage,
        song_version_id: str,
        template_id: str,
        knob_overrides: dict[str, Any] | None = None,
    ) -> Result[PipelineConfigRecord]:
        """Create a new PipelineConfigRecord from a template with default knob values.

        This is how a pipeline gets added to a song. The template factory builds
        the initial pipeline, and we persist the full config to the DB.
        """
        template = self._registry.get(template_id)
        if template is None:
            return err(ValidationError(f"Pipeline template not found: {template_id}"))

        song_version = session.song_versions.get(song_version_id)
        if song_version is None:
            return err(ValidationError(f"SongVersionRecord not found: {song_version_id}"))

        # Build default knob values + any overrides
        knob_values = {k: v.default for k, v in template.knobs.items()}
        if knob_overrides:
            errors = template.validate_bindings(knob_overrides)
            if errors:
                return err(ValidationError("; ".join(errors)))
            knob_values.update(knob_overrides)

        # Build the pipeline from template
        auto_bindings = {"audio_file": song_version.audio_file}
        merged = {**auto_bindings, **knob_values}
        pipeline = template.build_pipeline(merged)

        # Create persistent config
        config = PipelineConfigRecord.from_pipeline(
            pipeline=pipeline,
            template_id=template_id,
            song_version_id=song_version_id,
            knob_values=knob_values,
            name=template.name,
        )

        session.pipeline_configs.create(config)
        session.commit()

        return ok(config)

    def set_output_mappings(self, pipeline_id: str, mappings: list[OutputMapping]) -> None:
        """Configure custom output mappings for a pipeline."""
        self._output_mappings[pipeline_id] = mappings

    # -- Main entry point (config-based) --

    def execute(
        self,
        session: ProjectStorage,
        config_id: str,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> Result[AnalysisResult]:
        """Run analysis from a persisted PipelineConfigRecord.

        The config already contains the full graph and settings.
        No template rebuild — we deserialize and execute directly.
        """
        start_time = time.monotonic()

        if on_progress:
            on_progress("Loading configuration", 0.0)

        # 1. Load config from DB
        config = session.pipeline_configs.get(config_id)
        if config is None:
            return err(ValidationError(f"PipelineConfigRecord not found: {config_id}"))

        # 2. Load SongVersionRecord (for audio path)
        song_version = session.song_versions.get(config.song_version_id)
        if song_version is None:
            return err(ValidationError(
                f"SongVersionRecord not found: {config.song_version_id}"
            ))

        if on_progress:
            on_progress("Preparing pipeline", 0.1)

        # 3. Deserialize the persisted pipeline
        pipeline = config.to_pipeline()

        # 4. Inject audio_file into LoadAudio block settings
        #    (audio_file is always the song version's path, not a knob)
        from echozero.domain.types import BlockSettings
        from dataclasses import replace as _replace
        # Collect IDs first, then mutate — avoids dict-mutation-during-iteration (O1)
        load_audio_ids = [
            bid for bid, b in pipeline.graph.blocks.items()
            if b.block_type == "LoadAudio"
        ]
        for block_id in load_audio_ids:
            block = pipeline.graph.blocks[block_id]
            new_settings = {**dict(block.settings), "file_path": song_version.audio_file}
            updated = _replace(block, settings=BlockSettings(new_settings))
            pipeline.graph.replace_block(updated)

        if on_progress:
            on_progress("Executing pipeline", 0.2)

        # 5. Execute
        runtime_bus = RuntimeBus()
        engine = ExecutionEngine(pipeline.graph, runtime_bus)
        for block_type, executor in self._executors.items():
            engine.register_executor(block_type, executor)

        planner = GraphPlanner()
        plan = planner.plan(pipeline.graph)
        result = engine.run(plan)

        if is_err(result):
            assert isinstance(result, Err)
            return err(result.error)

        raw_outputs = unwrap(result)

        if on_progress:
            on_progress("Persisting results", 0.8)

        # 6. Map pipeline outputs → persistence
        layer_ids, take_ids = self._persist_outputs(
            pipeline=pipeline,
            raw_outputs=raw_outputs,
            session=session,
            song_version_id=config.song_version_id,
            pipeline_id=config.template_id,
            execution_id=plan.execution_id,
        )

        # 7. Commit
        session.commit()

        if on_progress:
            on_progress("Complete", 1.0)

        duration_ms = (time.monotonic() - start_time) * 1000

        return ok(AnalysisResult(
            song_version_id=config.song_version_id,
            pipeline_id=config.template_id,
            layer_ids=layer_ids,
            take_ids=take_ids,
            duration_ms=duration_ms,
        ))

    # -- Legacy entry point (template + bindings) --

    def analyze(
        self,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        bindings: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> Result[AnalysisResult]:
        """Run a pipeline against a song version and persist results.

        Legacy entry point — creates a config from the template, then delegates
        to execute() so the two code paths share a single implementation.
        """
        # Create a config from the template (validates template + song_version_id)
        config_result = self.create_config(
            session,
            song_version_id,
            pipeline_id,
            knob_overrides=bindings,
        )
        if is_err(config_result):
            return config_result  # propagate validation error

        config = unwrap(config_result)
        return self.execute(session, config.id, on_progress=on_progress)

    # -- Output → Persistence routing --

    def _persist_outputs(
        self,
        pipeline: Pipeline,
        raw_outputs: dict[str, Any],
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        execution_id: str,
    ) -> tuple[list[str], list[str]]:
        """Map pipeline outputs to persistence via handlers."""
        all_layer_ids: list[str] = []
        all_take_ids: list[str] = []

        # Build mapping lookup: output_name → OutputMapping
        custom_mappings = {
            m.output_name: m
            for m in self._output_mappings.get(pipeline_id, [])
        }

        for pipeline_output in pipeline.outputs:
            name = pipeline_output.name
            port_ref = pipeline_output.port_ref

            # Resolve the actual data from raw engine outputs
            data = self._resolve_output(port_ref, raw_outputs)
            if data is None:
                logger.warning(
                    "Pipeline output '%s' resolved to None (block=%s, port=%s)",
                    name, port_ref.block_id, port_ref.port_name,
                )
                continue

            # Get mapping (custom or auto)
            mapping = custom_mappings.get(name)
            target = self._resolve_target(data, mapping)

            if target is None:
                logger.warning(
                    "Pipeline output '%s' has type %s — no persistence handler, skipping",
                    name, type(data).__name__,
                )
                continue

            label = (mapping.label if mapping and mapping.label else
                     self._label_from_name(name))
            extra_params = mapping.params if mapping else {}

            # Look up the block for provenance
            block = pipeline.graph.blocks.get(port_ref.block_id)
            block_type = block.block_type if block else ""

            handler = self._persistence_handlers.get(target)
            if handler is None:
                logger.warning("No handler for target '%s'", target)
                continue

            layer_ids, take_ids = handler(
                data,
                session=session,
                song_version_id=song_version_id,
                pipeline_id=pipeline_id,
                execution_id=execution_id,
                block_id=port_ref.block_id,
                block_type=block_type,
                label=label,
                output_name=name,
                **extra_params,
            )
            all_layer_ids.extend(layer_ids)
            all_take_ids.extend(take_ids)

        return all_layer_ids, all_take_ids

    @staticmethod
    def _resolve_output(port_ref: Any, raw_outputs: dict[str, Any]) -> Any:
        """Resolve a PipelineOutput's port_ref to actual data from engine results.

        Engine outputs are normalized to {block_id: {port_name: value}}.
        Resolution is always: outputs[block_id][port_name].
        """
        block_output = raw_outputs.get(port_ref.block_id)
        if block_output is None:
            return None

        if isinstance(block_output, dict):
            return block_output.get(port_ref.port_name)
        else:
            # Fallback for any non-normalized output (shouldn't happen with new engine)
            return block_output

    @staticmethod
    def _resolve_target(data: Any, mapping: OutputMapping | None) -> str | None:
        """Determine persistence target from mapping or auto-detect from data type."""
        if mapping and mapping.target != "auto":
            return mapping.target

        # Auto-detect from type
        if isinstance(data, EventData):
            return "layer_take"
        elif isinstance(data, AudioData):
            return "song_version"
        else:
            return None

    @staticmethod
    def _label_from_name(name: str) -> str:
        """Derive a human label from a pipeline output name.

        "drums_onsets" → "Drums Onsets"
        "onsets" → "Onsets"
        """
        return name.replace("_", " ").title()

    # -- Persistence handlers --

    def _handle_persist_as_layer_take(
        self,
        event_data: EventData,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        execution_id: str,
        block_id: str,
        block_type: str,
        label: str = "",
        output_name: str = "",
        **_,
    ) -> tuple[list[str], list[str]]:
        """Persist EventData → Layer + Take.

        Layer naming: uses output_name as a namespace prefix when the pipeline
        has multiple outputs (e.g., "drums_onsets" → layer name "drums_onsets"
        rather than just the domain layer's generic "onsets"). If the pipeline
        has only one output or the domain layer already has a unique name,
        the domain layer name is used as-is.
        """
        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = datetime.now(timezone.utc)

        for domain_layer in event_data.layers:
            # Use output_name as layer name when provided — ensures distinct
            # layers for multi-output pipelines (e.g., drums_onsets vs bass_onsets)
            layer_name = output_name if output_name else domain_layer.name

            # Find existing layer by name for this song version
            existing_layers = session.layers.list_by_version(song_version_id)
            existing = next(
                (lr for lr in existing_layers if lr.name == layer_name),
                None,
            )

            if existing is not None:
                # Layer exists — add new non-main take (re-analysis)
                layer_record_id = existing.id
                is_main = False
            else:
                # New layer — create with main take
                layer_record_id = uuid.uuid4().hex
                order = len(existing_layers)
                layer_record = LayerRecord(
                    id=layer_record_id,
                    song_version_id=song_version_id,
                    name=layer_name,
                    layer_type="analysis",
                    color=None,
                    order=order,
                    visible=True,
                    locked=False,
                    parent_layer_id=None,
                    source_pipeline={
                        "pipeline_id": pipeline_id,
                        "block_id": block_id,
                    },
                    created_at=now,
                )
                session.layers.create(layer_record)
                is_main = True

            layer_ids.append(layer_record_id)

            # Build per-layer EventData for the take
            layer_event_data = EventData(layers=(domain_layer,))
            take_label = label or f"{domain_layer.name} — {pipeline_id}"

            take = Take.create(
                data=layer_event_data,
                label=take_label,
                origin="pipeline",
                source=TakeSource(
                    block_id=block_id,
                    block_type=block_type,
                    settings_snapshot={},
                    run_id=execution_id,
                ),
                is_main=is_main,
            )
            session.takes.create(layer_record_id, take)
            take_ids.append(take.id)

            # Enforce take limit: archive oldest non-main takes
            if self._max_takes_per_layer > 0:
                all_takes = session.takes.list_by_layer(layer_record_id)
                non_main = [t for t in all_takes if not t.is_main and not t.is_archived]
                if len(non_main) > self._max_takes_per_layer - 1:
                    # Archive oldest (list is ordered by created_at)
                    to_archive = non_main[:len(non_main) - (self._max_takes_per_layer - 1)]
                    for old_take in to_archive:
                        from dataclasses import replace as _replace
                        archived = _replace(old_take, is_archived=True)
                        session.takes.update(archived)
                        logger.info(
                            "Archived take '%s' (layer=%s) — exceeded limit of %d",
                            old_take.id, layer_record_id, self._max_takes_per_layer,
                        )

        return layer_ids, take_ids

    def _handle_persist_as_song_version(
        self,
        audio_data: Any,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        execution_id: str,
        block_id: str,
        block_type: str,
        label: str = "",
        **_,
    ) -> tuple[list[str], list[str]]:
        """Persist AudioData → SongVersionRecord (stems, filtered audio, etc).

        Not yet implemented — stems are cached on disk, not DB entities (V1).
        """
        logger.info(
            "Audio output '%s' from block '%s' — disk cache only (V1), "
            "no DB SongVersionRecord created.",
            label, block_id,
        )
        return [], []


# Backward-compat alias
AnalysisService = Orchestrator
