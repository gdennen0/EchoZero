"""Pipeline execution orchestrator for persistence-backed song processing.
Exists to run pipelines against stored songs and map outputs into layers and takes.
Connects execution-engine results and pipeline declarations to project persistence.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from echozero.domain.types import AudioData, Event as DomainEvent, EventData, Layer as DomainLayer
from echozero.application.progress import OperationProgressUpdate
from echozero.errors import ValidationError
from echozero.execution import BlockExecutor, ExecutionEngine, GraphPlanner
from echozero.persistence.entities import LayerRecord, PipelineConfigRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import PipelineRegistry
from echozero.progress import RuntimeBus
from echozero.result import Err, Result, err, is_err, ok, unwrap
from echozero.services.provenance import (
    build_analysis_build,
    build_model_artifact,
    initialize_generated_layer_state,
)
from echozero.takes import Take, TakeAnalysisBuild, TakeArtifact, TakeSource

_DEFAULT_TAKE_LABEL_PATTERN = re.compile(r"^take\s+(\d+)$", re.IGNORECASE)
_LAYER_NAME_NORMALIZATION_PATTERN = re.compile(r"\s+")


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
    pipeline_config_id: str | None = None
    analysis_build_id: str | None = None
    execution_id: str | None = None


def _assign_event_instance_id(event: DomainEvent) -> DomainEvent:
    """Create a fresh persisted event instance id while preserving lineage."""

    instance_id = f"evt_{uuid.uuid4().hex}"
    source_event_id = event.source_event_id or event.id
    return replace(
        event,
        id=instance_id,
        source_event_id=source_event_id,
        parent_event_id=event.id,
    )


def _normalize_event_data_for_take(event_data: EventData) -> EventData:
    """Convert pipeline output events into persisted event instances."""

    normalized_layers: list[DomainLayer] = []
    for layer in event_data.layers:
        normalized_layers.append(
            replace(
                layer,
                events=tuple(_assign_event_instance_id(event) for event in layer.events),
            )
        )
    return EventData(layers=tuple(normalized_layers))


def _next_default_take_label(existing_takes: list[Take]) -> str:
    """Compute the next default numbered take label for one layer."""
    next_index = len(existing_takes) + 1
    for take in existing_takes:
        match = _DEFAULT_TAKE_LABEL_PATTERN.fullmatch((take.label or "").strip())
        if match is not None:
            next_index = max(next_index, int(match.group(1)) + 1)
    return f"Take {next_index}"


class Orchestrator:
    """Run persisted pipeline configurations and map outputs back into project storage."""

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
        runtime_bindings: dict[str, Any] | None = None,
        on_progress: Callable[[OperationProgressUpdate], None] | None = None,
    ) -> Result[AnalysisResult]:
        start_time = time.monotonic()

        if on_progress:
            on_progress(
                OperationProgressUpdate(
                    stage="loading_configuration",
                    message="Loading configuration",
                    fraction_complete=0.0,
                )
            )

        with session.locked():
            config = session.pipeline_configs.get(config_id)
            if config is None:
                return err(ValidationError(f"PipelineConfigRecord not found: {config_id}"))

            song_version = session.song_versions.get(config.song_version_id)
            if song_version is None:
                return err(ValidationError(f"SongVersionRecord not found: {config.song_version_id}"))

        if on_progress:
            on_progress(
                OperationProgressUpdate(
                    stage="preparing_pipeline",
                    message="Preparing pipeline",
                    fraction_complete=0.1,
                )
            )

        pipeline = config.to_pipeline()

        from dataclasses import replace as _replace

        from echozero.domain.types import BlockSettings

        effective_audio_file = song_version.audio_file
        if runtime_bindings is not None:
            runtime_audio_file = runtime_bindings.get("audio_file")
            if runtime_audio_file is not None and str(runtime_audio_file).strip():
                effective_audio_file = str(runtime_audio_file)
        resolved_audio_path = self._resolve_audio_path(session, effective_audio_file)
        load_audio_ids = [
            bid for bid, b in pipeline.graph.blocks.items() if b.block_type == "LoadAudio"
        ]
        for block_id in load_audio_ids:
            block = pipeline.graph.blocks[block_id]
            new_settings = {**dict(block.settings), "file_path": resolved_audio_path}
            updated = _replace(block, settings=BlockSettings(new_settings))
            pipeline.graph.replace_block(updated)
        if runtime_bindings:
            for block_id, block in tuple(pipeline.graph.blocks.items()):
                changed = {
                    key: value for key, value in runtime_bindings.items() if key in block.settings
                }
                if not changed:
                    continue
                updated = _replace(
                    block, settings=BlockSettings({**dict(block.settings), **changed})
                )
                pipeline.graph.replace_block(updated)

        if on_progress:
            on_progress(
                OperationProgressUpdate(
                    stage="executing_pipeline",
                    message="Executing pipeline",
                    fraction_complete=0.2,
                )
            )

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
            on_progress(
                OperationProgressUpdate(
                    stage="persisting_results",
                    message="Persisting results",
                    fraction_complete=0.8,
                )
            )

        with session.locked():
            generated_at = datetime.now(timezone.utc)
            analysis_build_id = self._analysis_build_id(config.id, plan.execution_id)
            layer_ids, take_ids = self._persist_outputs(
                pipeline=pipeline,
                raw_outputs=raw_outputs,
                session=session,
                song_version_id=config.song_version_id,
                pipeline_id=config.template_id,
                pipeline_config_id=config.id,
                execution_id=plan.execution_id,
                analysis_build_id=analysis_build_id,
                generated_at=generated_at,
            )

            session.commit()

        if on_progress:
            on_progress(
                OperationProgressUpdate(
                    stage="complete",
                    message="Complete",
                    fraction_complete=1.0,
                )
            )

        duration_ms = (time.monotonic() - start_time) * 1000

        return ok(
            AnalysisResult(
                song_version_id=config.song_version_id,
                pipeline_id=config.template_id,
                layer_ids=layer_ids,
                take_ids=take_ids,
                duration_ms=duration_ms,
                pipeline_config_id=config.id,
                analysis_build_id=analysis_build_id,
                execution_id=plan.execution_id,
            )
        )

    def analyze(
        self,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        bindings: dict[str, Any] | None = None,
        runtime_bindings: dict[str, Any] | None = None,
        on_progress: Callable[[OperationProgressUpdate], None] | None = None,
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
        if runtime_bindings is None:
            return self.execute(session, config.id, on_progress=on_progress)
        return self.execute(
            session, config.id, runtime_bindings=runtime_bindings, on_progress=on_progress
        )

    def _persist_outputs(
        self,
        pipeline: Pipeline,
        raw_outputs: dict[str, Any],
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        pipeline_config_id: str,
        execution_id: str,
        analysis_build_id: str,
        generated_at: datetime,
    ) -> tuple[list[str], list[str]]:
        all_layer_ids: list[str] = []
        all_take_ids: list[str] = []

        custom_mappings = {m.output_name: m for m in self._output_mappings.get(pipeline_id, [])}
        if pipeline_id == "extract_song_sections" and "sections" not in custom_mappings:
            custom_mappings["sections"] = OutputMapping(
                output_name="sections",
                target="layer_take",
                params={"manual_kind": "section"},
            )
        config = session.pipeline_configs.get(pipeline_config_id)
        knob_values = dict(config.knob_values) if config is not None else {}
        selected_extract_song_stems = self._selected_extract_song_drum_stem_outputs(
            pipeline_id=pipeline_id,
            knob_values=knob_values,
        )

        for pipeline_output in pipeline.outputs:
            name = pipeline_output.name
            if (
                selected_extract_song_stems is not None
                and name in self._extract_song_drum_stem_output_names()
                and name not in selected_extract_song_stems
            ):
                continue
            port_ref = pipeline_output.port_ref
            data = self._resolve_output(port_ref, raw_outputs)
            if data is None:
                logger.warning(
                    "Pipeline output '%s' resolved to None (block=%s, port=%s)",
                    name,
                    port_ref.block_id,
                    port_ref.port_name,
                )
                continue

            mapping = custom_mappings.get(name)
            target = self._resolve_target(data, mapping)
            if target is None:
                logger.warning(
                    "Pipeline output '%s' has type %s — no persistence handler, skipping",
                    name,
                    type(data).__name__,
                )
                continue

            label = mapping.label if mapping and mapping.label else ""
            extra_params = mapping.params if mapping else {}
            block = pipeline.graph.blocks.get(port_ref.block_id)
            block_type = block.block_type if block else ""
            source_audio_path = self._resolve_connected_audio_input_path(
                pipeline=pipeline,
                raw_outputs=raw_outputs,
                block_id=port_ref.block_id,
            )

            handler = self._persistence_handlers.get(target)
            if handler is None:
                logger.warning("No handler for target '%s'", target)
                continue

            layer_ids, take_ids = handler(
                data,
                session=session,
                song_version_id=song_version_id,
                pipeline_id=pipeline_id,
                pipeline_config_id=pipeline_config_id,
                execution_id=execution_id,
                analysis_build_id=analysis_build_id,
                generated_at=generated_at,
                block_id=port_ref.block_id,
                block_type=block_type,
                label=label,
                output_name=name,
                source_audio_path=source_audio_path,
                **extra_params,
            )
            all_layer_ids.extend(layer_ids)
            all_take_ids.extend(take_ids)

        return all_layer_ids, all_take_ids

    @staticmethod
    def _extract_song_drum_stem_output_names() -> tuple[str, ...]:
        return ("drums", "bass", "vocals", "other")

    def _selected_extract_song_drum_stem_outputs(
        self,
        *,
        pipeline_id: str,
        knob_values: dict[str, Any],
    ) -> set[str] | None:
        if pipeline_id != "extract_song_drum_events":
            return None
        selected: set[str] = set()
        if bool(knob_values.get("include_drums_stem_layer", False)):
            selected.add("drums")
        if bool(knob_values.get("include_bass_stem_layer", False)):
            selected.add("bass")
        if bool(knob_values.get("include_vocals_stem_layer", False)):
            selected.add("vocals")
        if bool(knob_values.get("include_other_stem_layer", False)):
            selected.add("other")
        return selected

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
    def _resolve_connected_audio_input_path(
        *,
        pipeline: Pipeline,
        raw_outputs: dict[str, Any],
        block_id: str,
    ) -> str | None:
        for connection in pipeline.graph.connections:
            if connection.target_block_id != block_id or connection.target_input_name != "audio_in":
                continue
            upstream_output = raw_outputs.get(connection.source_block_id)
            if isinstance(upstream_output, dict):
                upstream_output = upstream_output.get(connection.source_output_name)
            if isinstance(upstream_output, AudioData):
                candidate = str(upstream_output.file_path).strip()
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _analysis_build_id(pipeline_config_id: str, execution_id: str) -> str:
        return f"{pipeline_config_id}:{execution_id}"

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

    @staticmethod
    def _normalize_layer_name(value: str) -> str:
        return _LAYER_NAME_NORMALIZATION_PATTERN.sub(" ", (value or "").strip().lower())

    @staticmethod
    def _find_matching_layer(
        layers: list[LayerRecord],
        target_name: str,
    ) -> LayerRecord | None:
        if not layers:
            return None
        exact_match = next((lr for lr in layers if lr.name == target_name), None)
        if exact_match is not None:
            return exact_match

        normalized_target = Orchestrator._normalize_layer_name(target_name)
        if not normalized_target:
            return None
        return next(
            (
                lr
                for lr in layers
                if Orchestrator._normalize_layer_name(lr.name) == normalized_target
            ),
            None,
        )

    def _make_generated_layer(
        self,
        *,
        layer_record_id: str,
        song_version_id: str,
        layer_name: str,
        order: int,
        pipeline_id: str,
        pipeline_config_id: str,
        block_id: str,
        block_type: str,
        output_name: str,
        data_type: str,
        execution_id: str,
        analysis_build_id: str,
        now,
        source_audio_path: str | None = None,
        manual_kind: str | None = None,
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
            source_pipeline=None,
            created_at=now,
        )
        generated = initialize_generated_layer_state(
            base,
            pipeline_id=pipeline_id,
            pipeline_config_id=pipeline_config_id,
            output_name=output_name,
            block_id=block_id,
            block_type=block_type,
            data_type=data_type,
            analysis_build_id=analysis_build_id,
            source_song_version_id=song_version_id,
            source_run_id=execution_id,
            generated_at=now,
            source_audio_path=source_audio_path,
        )
        normalized_manual_kind = str(manual_kind or "").strip().lower()
        if normalized_manual_kind:
            updated_state_flags = dict(generated.state_flags or {})
            updated_state_flags["manual_kind"] = normalized_manual_kind
            generated = replace(generated, state_flags=updated_state_flags)
        return generated

    @staticmethod
    def _take_source_artifacts(source_audio_path: str | None) -> tuple[TakeArtifact, ...]:
        if source_audio_path is None or not str(source_audio_path).strip():
            return ()
        return (
            TakeArtifact.from_dict(
                build_model_artifact(
                    role="source_audio",
                    kind="audio_file",
                    locator=str(source_audio_path).strip(),
                    content_type="audio/*",
                )
            ),
        )

    def _handle_persist_as_layer_take(
        self,
        event_data: EventData,
        session: ProjectStorage,
        song_version_id: str,
        pipeline_id: str,
        pipeline_config_id: str,
        execution_id: str,
        analysis_build_id: str,
        generated_at: datetime,
        block_id: str,
        block_type: str,
        label: str = "",
        output_name: str = "",
        source_audio_path: str | None = None,
        manual_kind: str | None = None,
        **_,
    ) -> tuple[list[str], list[str]]:
        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = generated_at

        for domain_layer in event_data.layers:
            layer_name = (
                domain_layer.name
                if len(event_data.layers) > 1
                else (output_name if output_name else domain_layer.name)
            )
            existing_layers = session.layers.list_by_version(song_version_id)
            existing_names = [lr.name for lr in existing_layers]
            existing = self._find_matching_layer(existing_layers, layer_name)
            if (
                existing is not None
                and manual_kind is not None
                and str(existing.state_flags.get("manual_kind", "")).strip().lower()
                != str(manual_kind).strip().lower()
            ):
                existing = None
            if existing is None:
                logger.info(
                    "No matching layer found for pipeline output '%s' (%s) in version '%s'. "
                    "Existing layer names: %s. Creating a new layer.",
                    output_name,
                    layer_name,
                    song_version_id,
                    existing_names,
                )
            elif existing.name != layer_name:
                logger.info(
                    "Matched existing layer by normalized name. "
                    "Requested '%s', found existing '%s' (output='%s', pipeline='%s').",
                    layer_name,
                    existing.name,
                    output_name,
                    pipeline_id,
                )

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
                    pipeline_config_id=pipeline_config_id,
                    block_id=block_id,
                    block_type=block_type,
                    output_name=output_name,
                    data_type="event",
                    execution_id=execution_id,
                    analysis_build_id=analysis_build_id,
                    now=now,
                    source_audio_path=source_audio_path,
                    manual_kind=manual_kind,
                )
                session.layers.create(layer_record)
                is_main = True

            layer_ids.append(layer_record_id)
            layer_event_data = _normalize_event_data_for_take(
                EventData(layers=(domain_layer,))
            )
            take_label = label or _next_default_take_label(
                session.takes.list_by_layer(layer_record_id)
            )
            settings_snapshot: dict[str, Any] = {
                "pipeline_id": pipeline_id,
                "pipeline_config_id": pipeline_config_id,
                "output_name": output_name,
                "data_type": "event",
                "analysis_build_id": analysis_build_id,
                "execution_id": execution_id,
                "generated_at": generated_at.isoformat(),
            }
            if source_audio_path is not None and str(source_audio_path).strip():
                settings_snapshot["source_audio_path"] = str(source_audio_path)
            analysis_build = TakeAnalysisBuild.from_dict(
                build_analysis_build(
                    pipeline_id=pipeline_id,
                    pipeline_config_id=pipeline_config_id,
                    block_id=block_id,
                    block_type=block_type,
                    output_name=output_name,
                    data_type="event",
                    execution_id=execution_id,
                    build_id=analysis_build_id,
                    generated_at=generated_at,
                )
            )

            take = Take.create(
                data=layer_event_data,
                label=take_label,
                origin="pipeline",
                source=TakeSource(
                    block_id=block_id,
                    block_type=block_type,
                    settings_snapshot=settings_snapshot,
                    run_id=execution_id,
                    analysis_build=analysis_build,
                    artifacts=self._take_source_artifacts(source_audio_path),
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
        pipeline_config_id: str,
        execution_id: str,
        analysis_build_id: str,
        generated_at: datetime,
        block_id: str,
        block_type: str,
        label: str = "",
        output_name: str = "",
        source_audio_path: str | None = None,
        **_,
    ) -> tuple[list[str], list[str]]:
        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = generated_at

        layer_name = output_name if output_name else label or "Audio"
        existing_layers = session.layers.list_by_version(song_version_id)
        existing = self._find_matching_layer(existing_layers, layer_name)
        if existing is None:
            logger.info(
                "No matching audio layer found for pipeline output '%s' (%s) in version '%s'. "
                "Existing layer names: %s. Creating a new layer.",
                output_name,
                layer_name,
                song_version_id,
                [lr.name for lr in existing_layers],
            )
        elif existing.name != layer_name:
            logger.info(
                "Matched existing audio layer by normalized name. "
                "Requested '%s', found existing '%s' (pipeline='%s').",
                layer_name,
                existing.name,
                pipeline_id,
            )

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
                pipeline_config_id=pipeline_config_id,
                block_id=block_id,
                block_type=block_type,
                output_name=output_name,
                data_type="audio",
                execution_id=execution_id,
                analysis_build_id=analysis_build_id,
                now=now,
                source_audio_path=source_audio_path,
            )
            session.layers.create(layer_record)
            is_main = True

        layer_ids.append(layer_record_id)
        take_label = label or _next_default_take_label(session.takes.list_by_layer(layer_record_id))
        settings_snapshot: dict[str, Any] = {
            "pipeline_id": pipeline_id,
            "pipeline_config_id": pipeline_config_id,
            "output_name": output_name,
            "data_type": "audio",
            "analysis_build_id": analysis_build_id,
            "execution_id": execution_id,
            "generated_at": generated_at.isoformat(),
        }
        if source_audio_path is not None and str(source_audio_path).strip():
            settings_snapshot["source_audio_path"] = str(source_audio_path)
        analysis_build = TakeAnalysisBuild.from_dict(
            build_analysis_build(
                pipeline_id=pipeline_id,
                pipeline_config_id=pipeline_config_id,
                block_id=block_id,
                block_type=block_type,
                output_name=output_name,
                data_type="audio",
                execution_id=execution_id,
                build_id=analysis_build_id,
                generated_at=generated_at,
            )
        )
        take = Take.create(
            data=audio_data,
            label=take_label,
            origin="pipeline",
            source=TakeSource(
                block_id=block_id,
                block_type=block_type,
                settings_snapshot=settings_snapshot,
                run_id=execution_id,
                analysis_build=analysis_build,
                artifacts=self._take_source_artifacts(source_audio_path),
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
            to_archive = non_main[: len(non_main) - (self._max_takes_per_layer - 1)]
            for old_take in to_archive:
                from dataclasses import replace as _replace

                archived = _replace(old_take, is_archived=True)
                session.takes.update(archived)
                logger.info(
                    "Archived take '%s' (layer=%s) — exceeded limit of %d",
                    old_take.id,
                    layer_record_id,
                    self._max_takes_per_layer,
                )
