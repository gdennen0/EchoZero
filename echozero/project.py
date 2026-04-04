"""
Project: Central application object for EchoZero.

Owns the Graph, Pipeline, Coordinator, Storage, and Orchestrator.
The UI talks to Project. Project talks to everything else.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from echozero.domain.graph import Graph
from echozero.editor.cache import ExecutionCache
from echozero.editor.coordinator import Coordinator, ExecutionHandle
from echozero.editor.pipeline import Pipeline
from echozero.editor.staleness import StaleTracker
from echozero.event_bus import EventBus
from echozero.execution import ExecutionEngine
from echozero.persistence.entities import ProjectSettingsRecord, SongRecord, SongVersionRecord
from echozero.persistence.session import ProjectStorage
from echozero.pipelines.registry import PipelineRegistry, get_registry
from echozero.progress import RuntimeBus
from echozero.result import Result, is_ok
from echozero.services.foundry_orchestrator import FoundryOrchestrator
from echozero.services.orchestrator import AnalysisResult, Orchestrator

logger = logging.getLogger(__name__)


class Project:
    """Central application object — the single thing the UI talks to.

    Bridges engine (Graph, Pipeline, Coordinator) with persistence (ProjectStorage)
    and analysis (Orchestrator). Owns the Graph as single source of truth.
    """

    def __init__(
        self,
        storage: ProjectStorage,
        graph: Graph,
        event_bus: EventBus,
        pipeline: Pipeline,
        coordinator: Coordinator,
        orchestrator: Orchestrator,
        runtime_bus: RuntimeBus,
        foundry: FoundryOrchestrator,
    ) -> None:
        self._storage = storage
        self._graph = graph
        self._event_bus = event_bus
        self._pipeline = pipeline
        self._coordinator = coordinator
        self._orchestrator = orchestrator
        self._runtime_bus = runtime_bus
        self._foundry = foundry

    # -- Factory methods (the ONLY way to create a Project) -----------------

    @classmethod
    def create(
        cls,
        name: str,
        settings: ProjectSettingsRecord | None = None,
        executors: dict[str, Any] | None = None,
        registry: PipelineRegistry | None = None,
        working_dir_root: Path | None = None,
    ) -> "Project":
        """Create a brand new project."""
        event_bus = EventBus()
        storage = ProjectStorage.create_new(
            name=name,
            settings=settings,
            event_bus=event_bus,
            working_dir_root=working_dir_root,
        )
        return cls._build(storage, event_bus, executors, registry)

    @classmethod
    def open(
        cls,
        ez_path: Path,
        executors: dict[str, Any] | None = None,
        registry: PipelineRegistry | None = None,
        working_dir_root: Path | None = None,
    ) -> "Project":
        """Open an existing project from an .ez file."""
        event_bus = EventBus()
        storage = ProjectStorage.open(
            ez_path=ez_path,
            event_bus=event_bus,
            working_dir_root=working_dir_root,
        )
        return cls._build(storage, event_bus, executors, registry)

    @classmethod
    def open_db(
        cls,
        working_dir: Path,
        executors: dict[str, Any] | None = None,
        registry: PipelineRegistry | None = None,
    ) -> "Project":
        """Open from a working directory (recovery or dev use)."""
        event_bus = EventBus()
        storage = ProjectStorage.open_db(working_dir=working_dir, event_bus=event_bus)
        return cls._build(storage, event_bus, executors, registry)

    @classmethod
    def _build(
        cls,
        storage: ProjectStorage,
        event_bus: EventBus,
        executors: dict[str, Any] | None,
        registry: PipelineRegistry | None,
    ) -> "Project":
        """Wire up the full object graph from a ProjectStorage."""
        # Load graph from DB, or create empty
        graph = storage.load_graph() or Graph()

        # Engine components
        runtime_bus = RuntimeBus()
        pipeline = Pipeline(event_bus, graph=graph)
        cache = ExecutionCache()
        stale_tracker = StaleTracker()
        engine = ExecutionEngine(graph, runtime_bus)

        # Register executors
        if executors:
            for block_type, executor in executors.items():
                engine.register_executor(block_type, executor)

        coordinator = Coordinator(
            graph=graph,
            pipeline=pipeline,
            engine=engine,
            cache=cache,
            runtime_bus=runtime_bus,
            stale_tracker=stale_tracker,
        )

        # Wire coordinator to react to graph mutations
        coordinator.subscribe_to_document_bus(event_bus)

        # Orchestrator for analysis workflows
        orchestrator = Orchestrator(
            registry=registry or get_registry(),
            executors=executors or {},
        )

        foundry = FoundryOrchestrator(storage.working_dir, event_bus=event_bus)

        project = cls(
            storage=storage,
            graph=graph,
            event_bus=event_bus,
            pipeline=pipeline,
            coordinator=coordinator,
            orchestrator=orchestrator,
            runtime_bus=runtime_bus,
            foundry=foundry,
        )

        storage.start_autosave(interval_seconds=30.0)
        return project

    # -- Properties ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self._storage.project.name

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    @property
    def storage(self) -> ProjectStorage:
        return self._storage

    @property
    def is_executing(self) -> bool:
        return self._coordinator.is_executing

    @property
    def is_dirty(self) -> bool:
        return self._storage.is_dirty()

    @property
    def stale_tracker(self) -> StaleTracker:
        return self._coordinator.stale_tracker

    @property
    def runtime_bus(self) -> RuntimeBus:
        """Progress reporting bus — UI subscribes for per-block progress updates."""
        return self._runtime_bus

    # -- Graph mutations (via Pipeline) -------------------------------------

    def dispatch(self, command: Any) -> Result[Any]:
        """Dispatch a command to mutate the graph.

        On success, persists the updated graph to the DB.
        """
        result = self._pipeline.dispatch(command)
        # Persist graph after successful mutation
        if is_ok(result):
            self._storage.save_graph(self._graph)
        return result

    # -- Execution (via Coordinator) ----------------------------------------

    def run(self, target: str | None = None) -> Result[str]:
        """Run execution synchronously. Returns execution_id."""
        return self._coordinator.request_run(target)

    def run_async(self, target: str | None = None) -> Result[ExecutionHandle]:
        """Run execution in background thread. Returns ExecutionHandle."""
        return self._coordinator.request_run_async(target)

    def cancel(self) -> None:
        """Cancel in-flight execution."""
        self._coordinator.cancel()

    # -- Analysis (via Orchestrator) ----------------------------------------

    def analyze(
        self,
        song_version_id: str,
        template_id: str,
        knob_overrides: dict[str, Any] | None = None,
        on_progress: Any = None,
    ) -> Result[AnalysisResult]:
        """Run analysis pipeline and persist results."""
        return self._orchestrator.analyze(
            session=self._storage,
            song_version_id=song_version_id,
            pipeline_id=template_id,
            bindings=knob_overrides,
            on_progress=on_progress,
        )

    def execute_config(
        self,
        config_id: str,
        on_progress: Any = None,
    ) -> Result[AnalysisResult]:
        """Execute from an existing PipelineConfigRecord."""
        return self._orchestrator.execute(
            session=self._storage,
            config_id=config_id,
            on_progress=on_progress,
        )

    # -- Foundry (training lane) --------------------------------------------

    def foundry_create_run(
        self,
        dataset_version_id: str,
        run_spec: dict[str, Any],
        *,
        backend: str = "pytorch",
        device: str = "cpu",
    ):
        return self._foundry.create_run(
            dataset_version_id=dataset_version_id,
            run_spec=run_spec,
            backend=backend,
            device=device,
        )

    def foundry_start_run(self, run_id: str):
        return self._foundry.start_run(run_id)

    def foundry_get_run(self, run_id: str):
        return self._foundry.get_run(run_id)

    def foundry_finalize_artifact(self, run_id: str, manifest: dict[str, Any]):
        return self._foundry.finalize_artifact(run_id, manifest)

    def foundry_validate_artifact(
        self,
        artifact_id: str,
        *,
        consumer: str = "PyTorchAudioClassify",
    ):
        return self._foundry.validate_artifact(artifact_id, consumer=consumer)

    def foundry_finalize_artifact_checked(
        self,
        run_id: str,
        manifest: dict[str, Any],
        *,
        consumer: str = "PyTorchAudioClassify",
    ):
        return self._foundry.finalize_artifact_checked(
            run_id=run_id,
            manifest=manifest,
            consumer=consumer,
        )

    # -- Song management (via Storage) --------------------------------------

    def import_song(
        self,
        title: str,
        audio_source: Path,
        artist: str = "",
        label: str = "Original",
        default_templates: list[str] | None = None,
        scan_fn: Any = None,
    ) -> tuple[SongRecord, SongVersionRecord]:
        """Import an audio file as a new song with default pipeline configs.

        Args:
            title: Song title.
            audio_source: Path to the audio file.
            artist: Artist name (optional).
            label: Version label (default "Original").
            default_templates: Template IDs to create configs for. None = all registered.
            scan_fn: Optional injectable for audio scanning (testing).

        Returns:
            (SongRecord, SongVersionRecord) tuple.
        """
        return self._storage.import_song(
            title=title,
            audio_source=audio_source,
            artist=artist,
            label=label,
            default_templates=default_templates,
            scan_fn=scan_fn,
        )

    def add_song_version(
        self,
        song_id: str,
        audio_source: Path,
        label: str | None = None,
        activate: bool = True,
        scan_fn: Any = None,
    ) -> SongVersionRecord:
        """Add a new version of an existing song and copy pipeline configs.

        Args:
            song_id: The existing song to add a version to.
            audio_source: Path to the new audio file.
            label: Human-readable label. Auto-generated if None.
            activate: If True, set the new version as active.
            scan_fn: Optional injectable for audio scanning (testing).

        Returns:
            The newly created SongVersionRecord.
        """
        return self._storage.add_song_version(
            song_id=song_id,
            audio_source=audio_source,
            label=label,
            activate=activate,
            scan_fn=scan_fn,
        )

    # -- Data access (repo shortcuts) ---------------------------------------

    @property
    def songs(self):
        return self._storage.songs

    @property
    def song_versions(self):
        return self._storage.song_versions

    @property
    def layers(self):
        return self._storage.layers

    @property
    def takes(self):
        return self._storage.takes

    @property
    def pipeline_configs(self):
        return self._storage.pipeline_configs

    # -- Lifecycle ----------------------------------------------------------

    def save(self) -> None:
        """Persist current graph state and commit all pending changes."""
        self._storage.save_graph(self._graph)
        self._storage.save()

    def save_as(self, ez_path: Path) -> None:
        """Save to .ez archive. Persists graph first."""
        self._storage.save_graph(self._graph)
        self._storage.save_as(ez_path)

    def close(self) -> None:
        """Close the project. Flushes pending changes, stops autosave, releases DB."""
        self._coordinator.unsubscribe_from_document_bus(self._event_bus)
        # Commit any uncommitted graph changes before closing
        try:
            if not self._storage._closed:
                self._storage.db.commit()
        except Exception:
            pass
        self._storage.close()

    def __enter__(self) -> "Project":
        return self

    def __exit__(self, *args) -> None:
        self.close()
