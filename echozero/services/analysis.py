"""
AnalysisService: Orchestrates pipeline execution against songs and persists results.
Exists because the engine computes and persistence stores — this service bridges both.
Coordinates: load audio path from SongVersion, get template from registry, apply bindings,
execute pipeline, create Layers + Takes from output.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from echozero.domain.types import EventData
from echozero.errors import ValidationError
from echozero.execution import BlockExecutor, ExecutionEngine, GraphPlanner
from echozero.persistence.entities import LayerRecord
from echozero.persistence.session import ProjectSession
from echozero.pipelines.registry import PipelineRegistry
from echozero.progress import RuntimeBus
from echozero.result import Err, Result, err, is_err, ok, unwrap
from echozero.takes import Take, TakeSource


@dataclass(frozen=True)
class AnalysisResult:
    """Result of running analysis on a song version."""

    song_version_id: str
    pipeline_id: str
    layer_ids: list[str]
    take_ids: list[str]
    duration_ms: float


class AnalysisService:
    """Orchestrates pipeline execution against songs and persists results.

    This is the bridge between the engine (which computes) and persistence
    (which stores). It coordinates: load audio path from SongVersion,
    get template from registry, apply bindings, execute pipeline,
    create Layers + Takes from output.
    """

    def __init__(
        self,
        registry: PipelineRegistry,
        executors: dict[str, BlockExecutor],
    ) -> None:
        self._registry = registry
        self._executors = executors

    def analyze(
        self,
        session: ProjectSession,
        song_version_id: str,
        pipeline_id: str,
        bindings: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> Result[AnalysisResult]:
        """Run a pipeline template against a song version and persist results.

        Steps:
        1. Load SongVersion from session to get audio_file path
        2. Get pipeline template from registry
        3. Merge bindings: user bindings + auto-bindings (audio_file from SongVersion)
        4. Validate bindings against template's promoted params
        5. Build graph from template with bindings applied
        6. Register processors and execute via engine
        7. Collect output EventData from execution results
        8. For each output layer:
           a. Find or create LayerRecord in persistence
           b. Create Take with the EventData
           c. If layer is new, the take is main. If layer exists, take is non-main.
        9. Commit via session
        10. Return AnalysisResult with IDs and timing

        Returns Result[AnalysisResult] — Err on validation failure, engine error, etc.
        """
        start_time = time.monotonic()

        if on_progress:
            on_progress("Loading song version", 0.0)

        # 1. Load SongVersion
        song_version = session.song_versions.get(song_version_id)
        if song_version is None:
            return err(ValidationError(f"SongVersion not found: {song_version_id}"))

        # 2. Get template
        template = self._registry.get(pipeline_id)
        if template is None:
            return err(ValidationError(f"Pipeline template not found: {pipeline_id}"))

        if on_progress:
            on_progress("Preparing pipeline", 0.1)

        # 3. Merge bindings: auto-bindings + user bindings (user overrides auto)
        auto_bindings: dict[str, Any] = {"audio_file": song_version.audio_file}
        merged_bindings = {**auto_bindings, **(bindings or {})}

        # 4. Validate bindings
        errors = template.validate_bindings(merged_bindings)
        if errors:
            return err(ValidationError("; ".join(errors)))

        # 5. Build graph with bindings
        graph = template.build(merged_bindings)

        if on_progress:
            on_progress("Executing pipeline", 0.2)

        # 6. Create engine, register executors, plan, execute
        runtime_bus = RuntimeBus()
        engine = ExecutionEngine(graph, runtime_bus)
        for block_type, executor in self._executors.items():
            engine.register_executor(block_type, executor)

        planner = GraphPlanner()
        plan = planner.plan(graph)
        result = engine.run(plan)

        if is_err(result):
            assert isinstance(result, Err)
            return err(result.error)

        outputs = unwrap(result)

        if on_progress:
            on_progress("Persisting results", 0.8)

        # 7. Find leaf blocks (not a source in any connection) — their outputs are results
        source_block_ids = {c.source_block_id for c in graph.connections}
        leaf_block_ids = [
            bid for bid in plan.ordered_block_ids if bid not in source_block_ids
        ]

        layer_ids: list[str] = []
        take_ids: list[str] = []
        now = datetime.now(timezone.utc)

        # 8. Persist EventData from leaf blocks
        for block_id in leaf_block_ids:
            output = outputs.get(block_id)
            if output is None:
                continue

            block = graph.blocks.get(block_id)
            block_type = block.block_type if block else ""

            if isinstance(output, dict):
                for port_value in output.values():
                    if isinstance(port_value, EventData):
                        self._persist_event_data(
                            session, song_version_id, pipeline_id,
                            plan.execution_id, block_id, block_type,
                            port_value, layer_ids, take_ids, now,
                        )
            elif isinstance(output, EventData):
                self._persist_event_data(
                    session, song_version_id, pipeline_id,
                    plan.execution_id, block_id, block_type,
                    output, layer_ids, take_ids, now,
                )

        # 9. Commit
        session.commit()

        if on_progress:
            on_progress("Complete", 1.0)

        duration_ms = (time.monotonic() - start_time) * 1000

        return ok(AnalysisResult(
            song_version_id=song_version_id,
            pipeline_id=pipeline_id,
            layer_ids=layer_ids,
            take_ids=take_ids,
            duration_ms=duration_ms,
        ))

    def _persist_event_data(
        self,
        session: ProjectSession,
        song_version_id: str,
        pipeline_id: str,
        execution_id: str,
        block_id: str,
        block_type: str,
        event_data: EventData,
        layer_ids: list[str],
        take_ids: list[str],
        now: datetime,
    ) -> None:
        """Create or update LayerRecords and Takes for pipeline output."""
        for domain_layer in event_data.layers:
            # Find existing layer by name for this song version
            existing_layers = session.layers.list_by_version(song_version_id)
            existing = next(
                (lr for lr in existing_layers if lr.name == domain_layer.name),
                None,
            )

            if existing is not None:
                # Layer exists — add new non-main take
                layer_record_id = existing.id
                is_main = False
            else:
                # New layer — create it with main take
                layer_record_id = uuid.uuid4().hex
                order = len(existing_layers)
                layer_record = LayerRecord(
                    id=layer_record_id,
                    song_version_id=song_version_id,
                    name=domain_layer.name,
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

            # Per-layer EventData for the take
            layer_event_data = EventData(layers=(domain_layer,))

            take = Take.create(
                data=layer_event_data,
                label=f"{domain_layer.name} — {pipeline_id}",
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
