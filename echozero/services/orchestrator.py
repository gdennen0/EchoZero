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
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from echozero.domain.types import AudioData, EventData
from echozero.errors import ValidationError
from echozero.execution import BlockExecutor, ExecutionEngine, GraphPlanner
from echozero.persistence.entities import LayerRecord, PipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import PipelineRegistry
from echozero.progress import RuntimeBus
from echozero.result import Err, Result, err, is_err, ok, unwrap
from echozero.services.provenance import initialize_generated_layer_state
from echozero.takes import Take, TakeSource


@dataclass(frozen=True)
class OutputMapping:
    """Declares how a named pipeline output should be persisted."""

    output_name: str
    target: str = "auto"  # "auto", "layer_take", "song_version"
    label: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    song_version_id: str
    pipeline_id: str
    layer_ids: list[str]
    take_ids: list[str]
    duration_ms: float


class Orchestrator:
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
        template = self._registry.get(template_id)
        if template is None:
            return err(ValidationError(f"Pipeline template not found: {template_id}"))

        song_version = session.song_versions.get(song_version_id)
        if song_version is None:
            return err(ValidationError(f"SongVersionRecord not found: {song_version_id}"))

        knob_values = {k: v.default for k, v in template.knobs.items()}
        if knob_overrides:
            errors = template.validate_bindings(knob_overrides)
            if errors:
                return err(ValidationError("; ".join(errors)))
            knob_values.update(knob_overrides)

        auto_bindings = {"audio_file": song_version.audio_file}
        merged = {**auto_bindings, **knob_values}
        pipeline = template.build_pipeline(merged)

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
        self._output_mappings[pipeline_id] = mappings

    def execute(
        self,
        session: ProjectStorage,
        config_id: str,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> Result[AnalysisResult]:
        start_time = time.monotonic()

        if on_progress:
            on_progress("Loading configuration", 0.0)

        config = session.pipeline_configs.get(config_id)
        if config is None:
            return err(ValidationError(f"PipelineConfigRecord not found: {config_id}"))

        song_version = session.song_versions.get(config.song_version_id)
        if song_version is None:
            return err(ValidationError(f"SongVersionRecord not found: {config.song_version_id}"))

        if on_progress:
            on_progress("Preparing pipeline", 0.1)

        pipeline = config.to_pipeline()

        from echozero.domain.types import BlockSettings
        from dataclasses import replace as _replace
        resolved_audio_path = self._resolve_audio_path(session, song_version.audio_file)
        load_audio_ids = [
            bid for bid, b in pipeline.graph.blocks.items()
            if b.block_type == "LoadAudio"
        ]
        for block_id in load_audio_ids:
            block = pipeline.graph.blocks[block_id]
            new_settings = {**dict(block.settings), "file_path": resolved_audio_path}
            updated = _replace(block, settings=BlockSettings(new_settings))
            pipeline.graph.replace_block(updated)

        if on_progress:
            on_progress("Executing pipeline", 0.2)

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

        layer_ids, take_ids = self._persist_outputs(
            pipeline=pipeline,
            raw_outputs=raw_outputs,
            session=session,
            song_version_id=config.song_version_id,
            pipeline_id=config.template_id,
            execution_id=plan.execution_id,
        )

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

    def analyze(
        self,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        bindings: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> Result[AnalysisResult]:
        config_result = self.create_config(
            session,
            song_version_id,
            pipeline_id,
            knob_overrides=bindings,
        )
        if is_err(config_result):
            return config_result

        config = unwrap(config_result)
        return self.execute(session, config.id, on_progress=on_progress)

    def _persist_outputs(
        self,
        pipeline: Pipeline,
        raw_outputs: dict[str, Any],
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        execution_id: str,
    ) -> tuple[list[str], list[str]]:
        all_layer_ids: list[str] = []
        all_take_ids: list[str] = []

        custom_mappings = {
            m.output_name: m
            for m in self._output_mappings.get(pipeline_id, [])
        }

        for pipeline_output in pipeline.outputs:
            name = pipeline_output.name
            port_ref = pipeline_output.port_ref
            data = self._resolve_output(port_ref, raw_outputs)
            if data is None:
                logger.warning(
                    "Pipeline output '%s' resolved to None (block=%s, port=%s)",
                    name, port_ref.block_id, port_ref.port_name,
                )
                continue

            mapping = custom_mappings.get(name)
            target = self._resolve_target(data, mapping)
            if target is None:
                logger.warning(
                    "Pipeline output '%s' has type %s — no persistence handler, skipping",
                    name, type(data).__name__,
                )
                continue

            label = (mapping.label if mapping and mapping.label else self._label_from_name(name))
            extra_params = mapping.params if mapping else {}
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
    def _resolve_audio_path(session: ProjectStorage, audio_file: str) -> str:
        raw = Path(audio_file)
        if raw.is_absolute():
            return str(raw)
        return str((session.working_dir / raw).resolve())

    @staticmethod
    def _resolve_output(port_ref: Any, raw_outputs: dict[str, Any]) -> Any:
        block_output = raw_outputs.get(port_ref.block_id)
        if block_output is None:
            return None
        if isinstance(block_output, dict):
            return block_output.get(port_ref.port_name)
        return block_output

    @staticmethod
    def _resolve_target(data: Any, mapping: OutputMapping | None) -> str | None:
        if mapping and mapping.target != "auto":
            return mapping.target
        if isinstance(data, EventData):
            return "layer_take"
        elif isinstance(data, AudioData):
            return "song_version"
        return None

    @staticmethod
    def _label_from_name(name: str) -> str:
        return name.replace("_", " ").title()

    def _make_generated_layer(
        self,
        *,
        layer_record_id: str,
        song_version_id: str,
        layer_name: str,
        order: int,
        pipeline_id: str,
        block_id: str,
        output_name: str,
        data_type: str,
        execution_id: str,
        now,
    ) -> LayerRecord:
        base = LayerRecord(
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
                "output_name": output_name,
                "data_type": data_type,
            },
            created_at=now,
        )
        return initialize_generated_layer_state(
            base,
            pipeline_id=pipeline_id,
            output_name=output_name,
            block_id=block_id,
            data_type=data_type,
            source_song_version_id=song_version_id,
            source_run_id=execution_id,
        )

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
        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = datetime.now(timezone.utc)

        for domain_layer in event_data.layers:
            layer_name = domain_layer.name if len(event_data.layers) > 1 else (output_name if output_name else domain_layer.name)
            existing_layers = session.layers.list_by_version(song_version_id)
            existing = next((lr for lr in existing_layers if lr.name == layer_name), None)

            if existing is not None:
                layer_record_id = existing.id
                is_main = False
            else:
                layer_record_id = uuid.uuid4().hex
                order = len(existing_layers)
                layer_record = self._make_generated_layer(
                    layer_record_id=layer_record_id,
                    song_version_id=song_version_id,
                    layer_name=layer_name,
                    order=order,
                    pipeline_id=pipeline_id,
                    block_id=block_id,
                    output_name=output_name,
                    data_type="event",
                    execution_id=execution_id,
                    now=now,
                )
                session.layers.create(layer_record)
                is_main = True

            layer_ids.append(layer_record_id)
            layer_event_data = EventData(layers=(domain_layer,))
            take_label = label or f"{domain_layer.name} — {pipeline_id}"

            take = Take.create(
                data=layer_event_data,
                label=take_label,
                origin="pipeline",
                source=TakeSource(
                    block_id=block_id,
                    block_type=block_type,
                    settings_snapshot={"pipeline_id": pipeline_id, "output_name": output_name},
                    run_id=execution_id,
                ),
                is_main=is_main,
            )
            session.takes.create(layer_record_id, take)
            take_ids.append(take.id)
            self._enforce_take_limit(session, layer_record_id)

        return layer_ids, take_ids

    def _handle_persist_as_song_version(
        self,
        audio_data: AudioData,
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
        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = datetime.now(timezone.utc)

        layer_name = output_name if output_name else label or "Audio"
        existing_layers = session.layers.list_by_version(song_version_id)
        existing = next((lr for lr in existing_layers if lr.name == layer_name), None)

        if existing is not None:
            layer_record_id = existing.id
            is_main = False
        else:
            layer_record_id = uuid.uuid4().hex
            order = len(existing_layers)
            layer_record = self._make_generated_layer(
                layer_record_id=layer_record_id,
                song_version_id=song_version_id,
                layer_name=layer_name,
                order=order,
                pipeline_id=pipeline_id,
                block_id=block_id,
                output_name=output_name,
                data_type="audio",
                execution_id=execution_id,
                now=now,
            )
            session.layers.create(layer_record)
            is_main = True

        layer_ids.append(layer_record_id)
        take = Take.create(
            data=audio_data,
            label=label or self._label_from_name(layer_name),
            origin="pipeline",
            source=TakeSource(
                block_id=block_id,
                block_type=block_type,
                settings_snapshot={"pipeline_id": pipeline_id, "output_name": output_name},
                run_id=execution_id,
            ),
            is_main=is_main,
        )
        session.takes.create(layer_record_id, take)
        take_ids.append(take.id)
        self._enforce_take_limit(session, layer_record_id)
        return layer_ids, take_ids

    def _enforce_take_limit(self, session: ProjectStorage, layer_record_id: str) -> None:
        if self._max_takes_per_layer <= 0:
            return
        all_takes = session.takes.list_by_layer(layer_record_id)
        non_main = [t for t in all_takes if not t.is_main and not t.is_archived]
        if len(non_main) > self._max_takes_per_layer - 1:
            to_archive = non_main[:len(non_main) - (self._max_takes_per_layer - 1)]
            for old_take in to_archive:
                from dataclasses import replace as _replace
                archived = _replace(old_take, is_archived=True)
                session.takes.update(archived)
                logger.info(
                    "Archived take '%s' (layer=%s) — exceeded limit of %d",
                    old_take.id, layer_record_id, self._max_takes_per_layer,
                )


AnalysisService = Orchestrator
