# EchoZero 2 — Architecture

## Overview

Desktop audio analysis workstation for live lighting design.
Analyzes music → extracts time-stamped events → syncs with grandMA3.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| UI | PyQt6 (in development) |
| Persistence | SQLite + WAL mode |
| Audio analysis | librosa, essentia, demucs, pytorch |
| Audio playback | sounddevice, numpy |
| Serialization | JSON (graphs, pipelines, entities) |

---

## Directory Structure

```
echozero/
  project.py         # Project — central application object
  main.py            # Supporting composition helpers, not the canonical app entrypoint
  takes.py           # Take system (frozen snapshots, TakeLayer, merge)
  execution.py       # ExecutionEngine, GraphPlanner, BlockExecutor protocol
  event_bus.py       # Pub/sub EventBus for domain events
  errors.py          # Shared error types (*Error hierarchy)
  result.py          # Result[T] = Ok[T] | Err — no exceptions across boundaries
  serialization.py   # Graph/Pipeline serialize/deserialize (JSON ↔ domain)
  progress.py        # RuntimeBus for execution progress events

  domain/            # Core types — frozen dataclasses, no deps
    types.py         #   Block, Port, Connection, Event, EventData, AudioData,
                     #   WaveformData, BlockSettings, Layer
    graph.py         #   Graph — DAG with full invariant enforcement
    events.py        #   Domain events (BlockAddedEvent, SettingsChangedEvent, …)
    enums.py         #   BlockCategory, BlockState, Direction, PortType

  editor/            # Pipeline command dispatch and execution coordination
    pipeline.py      #   Pipeline — routes commands, manages event lifecycle
    commands.py      #   Command value objects (AddBlock, RemoveBlock, etc.)
    coordinator.py   #   Coordinator — ties mutation → stale propagation → run
    cache.py         #   ExecutionCache — per-port output cache
    staleness.py     #   StaleTracker — WHY each block is stale (for UI)

  persistence/       # ProjectStorage, entities, repositories, schema, archive
    session.py       #   ProjectStorage — lifecycle manager (open/save/close/autosave)
    entities.py      #   *Record DTOs: ProjectRecord, SongRecord, SongVersionRecord,
                     #   LayerRecord, PipelineConfigRecord
    schema.py        #   SQLite DDL, version tracking, migrations (V1→V2)
    archive.py       #   .ez archive pack/unpack (zip of working dir)
    audio.py         #   Audio import, hash, metadata scanning
    dirty.py         #   DirtyTracker — tracks unsaved changes via EventBus
    base.py          #   BaseRepository helpers
    repositories/
      project.py     #   ProjectRepository
      song.py        #   SongRepository
      song_version.py #  SongVersionRepository
      layer.py       #   LayerRepository
      take.py        #   TakeRepository
      pipeline_config.py # PipelineConfigRepository

  services/          # Application-level workflows (engine + persistence)
    orchestrator.py  #   Orchestrator — pipeline execution → persistence
    setlist.py       #   SetlistProcessor — batch analysis across songs
    waveform.py      #   WaveformService — generate and cache waveform peaks

  pipelines/         # Pipeline builder, template registry, parameter system
    pipeline.py      #   Pipeline — fluent builder (add/output API)
    registry.py      #   PipelineRegistry, @pipeline_template decorator
    block_specs.py   #   BlockSpec descriptors for each processor type
    params.py        #   Knob — typed parameter declarations for templates

  processors/        # 13 block executor implementations
    load_audio.py    #   LoadAudioProcessor — decode audio → AudioData
    detect_onsets.py #   DetectOnsetsProcessor — onset detection → EventData
    separate_audio.py #  SeparateAudioProcessor — stem separation (demucs)
    audio_filter.py  #   AudioFilterProcessor — low/high/bandpass filter
    audio_negate.py  #   AudioNegateProcessor — invert audio signal
    eq_bands.py      #   EQBandsProcessor — multi-band equalizer
    export_audio.py  #   ExportAudioProcessor — write audio to disk
    export_ma2.py    #   ExportMA2Processor — grandMA2 cue export
    export_audio_dataset.py # ExportAudioDatasetProcessor — ML dataset export
    generate_waveform.py    # GenerateWaveformProcessor → WaveformData
    transcribe_notes.py     # TranscribeNotesProcessor — pitch/note detection
    pytorch_audio_classify.py # PyTorchAudioClassifyProcessor — ML classification
    dataset_viewer.py       # DatasetViewerProcessor — debug/inspection

  audio/             # DAW-grade audio playback engine
    engine.py        #   AudioEngine — owns stream, clock, mixer
    transport.py     #   Transport — play/pause/stop/seek state machine
    mixer.py         #   Mixer — layer mixing, per-layer gain/pan
    clock.py         #   Clock — sample-accurate playback position
    layer.py         #   AudioLayer — one decoded audio buffer per layer
    crossfade.py     #   CrossfadeBuffer — equal-power crossfade

  models/            # ML model registry
    registry.py      #   ModelRegistry — enumerate available models
    provider.py      #   ModelProvider protocol

  ui/                # UI constants (tunable design values)
    FEEL.py          #   Timing, sizing, color constants for PyQt6 UI
```

---

## Layered Architecture

```
┌──────────────────────────────────────────────────────┐
│  6. UI Layer           PyQt6 Stage Zero Editor        │
│                        (in development)               │
├──────────────────────────────────────────────────────┤
│  5. Application Layer  App shell runtime + contracts  │
│                        run_echozero.py launcher       │
├──────────────────────────────────────────────────────┤
│  4. Services Layer     Orchestrator                   │
│                        SetlistProcessor               │
│                        WaveformService                │
├──────────────────────────────────────────────────────┤
│  3. Persistence Layer  ProjectStorage                 │
│                        *Record entities               │
│                        *Repository                    │
│                        archive (.ez)                  │
├──────────────────────────────────────────────────────┤
│  2. Engine Layer       Pipeline (commands)            │
│                        ExecutionEngine                │
│                        GraphPlanner                   │
│                        Coordinator                    │
│                        Processors (13 types)          │
├──────────────────────────────────────────────────────┤
│  1. Domain Layer       Graph, Block, Port, Connection │
│                        Event, EventData, AudioData    │
│                        (frozen dataclasses, no deps)  │
└──────────────────────────────────────────────────────┘
```

**Dependency rule:** Lower layers never import from higher layers.
Domain has zero imports. Engine imports domain only.
Persistence imports domain. Services import engine + persistence.
Application imports everything. UI imports application only.

---

## Runtime Boundary

The canonical desktop UI enters through `run_echozero.py`, which builds the
Stage Zero shell from `echozero/ui/qt/app_shell.py`.

`Project` remains an important application object, but the current UI boundary
is more presentation-driven than older docs implied. In practice the desktop
shell talks through app-shell/runtime composition and presentation contracts,
not directly to `Project` as a single universal façade.

## The Project Class

`Project` still owns and wires the core execution/persistence graph:

- `Graph` — single source of truth for the block graph
- `Pipeline` — routes commands, collects/flushes domain events
- `Coordinator` — reacts to graph mutations, manages stale propagation and runs
- `ProjectStorage` — SQLite lifecycle, all repos, autosave
- `Orchestrator` — analysis pipeline execution → persistence

**Factory methods** (the only way to create a Project):

| Method | Description |
|--------|-------------|
| `Project.create(name)` | Brand-new project in temp working dir |
| `Project.open(ez_path)` | Open existing `.ez` archive |
| `Project.open_db(working_dir)` | Open raw working dir (recovery/dev) |

**Core operations:**

| Method | Description |
|--------|-------------|
| `dispatch(command)` | Mutate graph + auto-persist |
| `run(target?)` | Execute pipeline synchronously |
| `run_async(target?)` | Execute in background thread, returns `ExecutionHandle` |
| `cancel()` | Cancel in-flight execution |
| `analyze(song_version_id, template_id)` | Run analysis + persist results |
| `execute_config(config_id)` | Execute from saved `PipelineConfigRecord` |
| `import_song(title, audio_source)` | Import audio, create Song + Version |
| `save()` / `save_as(path)` / `close()` | Lifecycle |

---

## Key Design Principles

| Principle | Description |
|-----------|-------------|
| **FP1: Pipeline-as-Data** | Pipelines are serializable, diffable, inspectable — not just runnable code |
| **FP2: Block Contract** | Typed ports, frozen settings, immutable during execution |
| **FP7: Engine Ignorance** | ExecutionEngine knows nothing about UI, persistence, or domain events |
| **UoW sole write path** | Nothing writes SQLite except through `ProjectStorage` |
| **CommandEnvelope** | Every mutation snapshots before/after state (undo-ready) |
| **Take system** | Frozen snapshots — multiple runs per layer, user curates the main take |
| **Result type** | `Result[T] = Ok[T] \| Err` — no exceptions crossing module boundaries |
| **Crash recovery** | Working dir survives close. On next open, EZ2 detects stale dir and offers recovery |

---

## Entity Model

```
ProjectRecord
  └── SongRecord[]
        ├── active_version_id → SongVersionRecord
        └── SongVersionRecord[]
              ├── LayerRecord[]
              │     └── Take[]
              │           └── EventData (or AudioData)
              └── PipelineConfigRecord[]
                    ├── graph_json (full serialized block graph)
                    ├── knob_values (user-facing settings)
                    └── block_overrides (per-block setting overrides)
```

---

## Naming Conventions

| Suffix | Meaning |
|--------|---------|
| `*Record` | Persistence DTO (SQLite row) |
| `*Repository` | Data access object |
| `*Processor` | Pipeline block executor |
| `*Command` | Graph mutation command |
| `*Event` | Domain event (pub/sub) |
| `*Error` | Exception type |
| *(no suffix)* | Domain or engine type (Block, Graph, Pipeline, …) |

---

## File Formats

| Extension | Description |
|-----------|-------------|
| `.ez` | Project archive (zip of working directory) |
| `project.db` | SQLite database in working directory |
| `project.lock` | PID lockfile (prevents double-open) |
| `audio/` | Imported audio files (within working directory) |
