# GLOSSARY.md — EchoZero Canonical Terminology

**Last verified:** 2026-03-20
**Authority:** Every concept has ONE name. All agents, documentation, UI text, and website copy use these names.

---

## Core Concepts

| Term | Definition | NOT |
|------|-----------|-----|
| **Block** | A processing unit in the pipeline graph. Three categories: Processor, Workspace, Playback. | node, component, module, filter |
| **Event** | A time-stamped marker or region on the timeline. Has time, duration, classifications, metadata. | cue, cue point, hit |
| **Layer** | A named group of events on the timeline. Organized within a song's branch. | track, channel, row |
| **Pipeline** | A saved arrangement of connected blocks that users create and execute. | workflow, chain, preset |
| **Graph** | The internal directed acyclic data structure of blocks and connections. Technical term — not user-facing. | pipeline (user-facing), network, flow |
| **Branch** | An isolated version of event data created by pipeline execution. Git-style: main + branches. | version, snapshot, copy, take |
| **Connection** | A typed link between an output port and an input port on two blocks. | wire, edge, link, pipe |
| **Port** | A typed input or output slot on a block. Types: Audio, Event, OSC, Control. | pin, terminal, socket, endpoint |
| **Project** | A collection of songs with their audio, events, pipelines, and branches. Saved as `.ez` file. | workspace, session, file |
| **Song** | A single audio file with its associated event data, branches, and analysis results. | track (ambiguous), clip, item |
| **Setlist** | An ordered collection of songs for batch processing and show playback. | playlist, project (project contains setlist) |

## Block Categories

| Term | Definition | NOT |
|------|-----------|-----|
| **Processor Block** | Pure transform. Pulls inputs, executes in background, stores outputs. Isolated and idempotent. | worker, transformer, filter |
| **Workspace Block** | Manages its own data. No automatic execution. Manual pull via Take System. Editor and ShowManager. | editor block, interactive block |
| **Playback Block** | Receives data, provides real-time output. No persistence. Ephemeral state only. | player, output block |

## Execution Concepts

| Term | Definition | NOT |
|------|-----------|-----|
| **Execution** | Running a pipeline (or single block) to produce results. Creates a branch. | processing, rendering, run |
| **Staleness** | A block's state indicating its inputs have changed and it needs re-execution. Cascades downstream. | dirty, invalid, outdated |
| **Take** | A manual pull of data into a Workspace block from connected upstream blocks. | sync, import, refresh |

## UI Concepts

| Term | Definition | NOT |
|------|-----------|-----|
| **Timeline** | The horizontal event editor view. Shows layers, events, waveform, playhead. | sequencer, arrangement, editor |
| **Graph View** | The block/pipeline editor view. Shows blocks, ports, connections. | node editor, flow editor, canvas |
| **Inspector** | The properties panel for selected items (block settings, event details). | properties panel, details, sidebar |
| **Playhead** | The vertical line indicating current playback position on the timeline. | cursor, position marker, scrubber |
| **Ruler** | The time display bar at the top of the timeline. Shows time/bars-beats/samples/SMPTE. | header, time bar |
| **Marker** | A user-placed navigation bookmark on the ruler. Song-level, not layer-level. | locator, flag, cue point |
| **Section** | A labeled region of the song (intro, verse, chorus, drop). Soft boundaries — can overlap. | region, zone, part |

## Technical Concepts

| Term | Definition | NOT |
|------|-----------|-----|
| **Core** | The Python engine process. Exposes HTTP + WebSocket API. Knows DAGs, not audio. | backend, server, engine (in user-facing text) |
| **Classification** | A namespaced label assigned to an event by a model. Stored as `classifications: dict[model_category, result]`. | label (too vague), tag, type |
| **Pipeline-as-Data** | FP1: Pipelines are serializable DAGs, not code. Can be saved, shared, composed. | — |
| **Engine Ignorance** | FP7: The execution engine knows about DAGs and blocks, not audio or events or MA3. | — |

## MA3 / Integration Concepts

| Term | Definition | NOT |
|------|-----------|-----|
| **ShowManager** | Workspace block that manages bidirectional sync with grandMA3. Zero network knowledge. | sync manager, MA3 controller |
| **OSC Gateway** | Separate service that handles OSC network communication. ShowManager talks to Gateway, not to MA3 directly. | OSC server, network layer |
| **SyncSnapshot** | Explicit state tracking for MA3 synchronization. Captures what was pushed and what's on the console. | sync state, MA3 state |

---

*When in doubt, use the term from this glossary. When two documents disagree on terminology, this glossary wins.*
