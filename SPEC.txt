# EchoZero — Behavioral Specification

**Purpose:** Comprehensive behavioral specification for ground-up rewrite.  
**Source:** Derived from full codebase analysis of the Python/PyQt6 implementation.  
**Date:** 2026-02-28  

> This document is the single source of truth for expected behavior. Every feature, entity, business rule, edge case, known issue, and design intent captured here is intended to survive the rewrite. Anything not documented here may be lost.

---

## Table of Contents

1. [Domain Model](#1-domain-model)
2. [Data Model](#2-data-model)
3. [Application Services](#3-application-services)
4. [Block Types](#4-block-types)
5. [Editor Behaviors](#5-editor-behaviors)
6. [Node Editor](#6-node-editor)
7. [Show Manager / MA3 Integration](#7-show-manager--ma3-integration)
8. [Project System](#8-project-system)
9. [Persistence Layer](#9-persistence-layer)
10. [UI Architecture](#10-ui-architecture)
11. [Known Issues / TODOs](#11-known-issues--todos)

---

## 1. Domain Model

### 1.1 Block

**Entity:** `Block`

**Fields:**
- `id: str` — UUID, globally unique
- `project_id: str` — parent project
- `name: str` — user-visible display name, unique within a project
- `type: str` — block type identifier (e.g., `"LoadAudio"`, `"Editor"`)
- `ports: List[Port]` — input and output ports, defined by block type
- `metadata: Dict[str, Any]` — type-specific configuration (e.g., file paths, filter settings)
- Status fields managed externally by `BlockStatusLevel` system

**Invariants:**
- `name` must be unique per project (enforced at the DB level with `UNIQUE(project_id, name)`)
- `type` must correspond to a registered block processor
- `ports` are immutable from the domain perspective — they are defined by block type, not user
- `metadata` is the primary configuration store; contents vary per block type

**Lifecycle:**
- Created via `CreateBlockCommand`
- Renamed via `RenameBlockCommand`
- Deleted via `DeleteBlockCommand` (cascades to connections and data items)
- Block position in node editor stored separately in UI state (not in Block entity)

**Status System:**
Each block type defines `BlockStatusLevel` objects with priority ordering:
- Priority 0 = highest severity (Error)
- Priority 1 = warning-level
- Priority 2 = ready/OK
- Each level has a list of `conditions` (callables that check requirements)
- The UI shows the highest-priority status whose conditions are NOT met
- Status colors are hex strings: error `#ff6b6b`, stale/warning `#ffa94d` or `#ffd43b`, ready `#51cf66`

---

### 1.2 Port

**Entity:** `Port`

**Fields:**
- `id: str` — UUID
- `block_id: str` — owning block
- `name: str` — port identifier (e.g., `"audio"`, `"events"`, `"drums"`)
- `direction: PortDirection` — `INPUT` or `OUTPUT`
- `port_type: PortType` — `AUDIO`, `EVENT`, `MANIPULATOR`, or `GENERIC`
- `display_name: str` — optional label shown in node editor

**Port Types:**
- `AUDIO` — carries `AudioDataItem` (audio waveform data)
- `EVENT` — carries `EventDataItem` (time-stamped events)
- `MANIPULATOR` — bidirectional command port (shown in orange in node editor); enables ShowManager ↔ Editor communication
- `GENERIC` — catch-all for other data

**Invariants:**
- Multiple outputs can connect to the same input (removed UNIQUE constraint on connections)
- A `MANIPULATOR` port can be both source and target simultaneously

---

### 1.3 Connection

**Entity:** `Connection`

**Fields:**
- `id: str` — UUID
- `source_block_id: str`
- `source_block_name: str` — denormalized for display
- `source_output_name: str` — output port name on source block
- `target_block_id: str`
- `target_block_name: str` — denormalized for display
- `target_input_name: str` — input port name on target block

**Invariants:**
- Source and target block must be in the same project (implied, not explicitly enforced)
- Source port must be an OUTPUT port; target port must be an INPUT port
- Multiple connections to the same input port are allowed (e.g., Event ports can merge inputs)
- Self-connections (same block as source and target) are not prevented by code but nonsensical

**Lifecycle:**
- Created by drawing a connection in node editor
- Deleted when either block is deleted (cascade) or user disconnects
- Validation: connection type compatibility is checked before creation

---

### 1.4 Project

**Entity:** `Project`

**Fields:**
- `id: str` — UUID
- `name: str` — user-visible, must be unique
- `version: str` — project file format version
- `save_directory: str | None` — absolute path to the `.ez` file's directory
- `created_at: datetime`
- `modified_at: datetime`
- `metadata: Dict[str, Any]` — additional project-level properties

**Invariants:**
- `name` must be unique (enforced by DB `UNIQUE(name)`)
- Projects are the top-level aggregate — blocks, connections, data items all belong to a project
- One setlist per project (enforced by `UNIQUE(project_id)` on setlists table)

**Lifecycle:**
- Created via `create_project(name)` on facade
- Loaded via `load_project(path)` — clears DB, rehydrates from JSON
- Saved via `save_project(project_id)` — serializes to `.ez` (ZIP) file
- Deleted via `delete_project(project_id)`

---

### 1.5 ActionSet and ActionItem

**Entity:** `ActionSet`

**Fields:**
- `id: str`
- `name: str`
- `description: str | None`
- `actions: List[ActionItem]` — serialized as JSON in DB
- `project_id: str | None` — null if global
- `created_at: datetime`
- `modified_at: datetime`
- `metadata: Dict[str, Any]`

**Entity:** `ActionItem`

**Fields:**
- `id: str`
- `action_set_id: str | None`
- `project_id: str`
- `action_type: str` — `"block"` (default) or project-level action type
- `block_id: str | None` — null for project-level actions
- `block_name: str | None` — denormalized
- `action_name: str` — name of the quick action to invoke
- `action_description: str | None`
- `action_args: Dict | None` — optional arguments
- `order_index: int` — zero-based position in the set
- `created_at: datetime`
- `modified_at: datetime`
- `metadata: Dict | None`

**Business Rules:**
- Action items are ordered by `order_index` when executed in a setlist run
- `block_id` is nullable to support project-level (non-block-specific) actions — this was a schema migration; older rows had NOT NULL
- Action sets can be default actions for a setlist or per-song action overrides

---

### 1.6 Setlist and SetlistSong

**Entity:** `Setlist`

**Fields:**
- `id: str`
- `audio_folder_path: str` — root folder containing audio files
- `project_id: str` — one per project (UNIQUE constraint)
- `default_actions: Dict` — JSON-encoded default ActionSet to apply to each song
- `created_at: datetime`
- `modified_at: datetime`
- `metadata: Dict`

**Entity:** `SetlistSong`

**Fields:**
- `id: str`
- `setlist_id: str`
- `audio_path: str` — absolute path to the audio file for this song
- `order_index: int` — position in the setlist (0-based)
- `status: str` — `"pending"`, `"processing"`, `"complete"`, `"error"`
- `processed_at: str | None`
- `action_overrides: Dict | None` — per-song override of default_actions
- `error_message: str | None` — last error if status == "error"
- `metadata: Dict | None`

**Business Rules:**
- Exactly one setlist per project
- Songs are ordered by `order_index`
- Each song can override the setlist's `default_actions` with `action_overrides`
- Song status transitions: pending → processing → complete/error

---

### 1.7 LayerOrder

**Entity:** `LayerOrder`

**Fields:**
- `block_id: str` — the Editor block that owns these layers
- `order_json: str` — JSON encoding of `List[LayerKey]`

**Value Object:** `LayerKey`

**Fields:**
- `group_name: str` — human-readable group (e.g., `"TC 1"`)
- `layer_name: str` — layer name within the group

**Purpose:** Persists the visual ordering of layers in an Editor's timeline widget. Layers are grouped and ordered within groups.

---

### 1.8 SyncLayerEntity (Show Manager Domain)

**Entity:** `SyncLayerEntity`

Represents the mapping between an Editor layer and an MA3 track.

**Fields:** (inferred from show manager code)
- `editor_layer_id: str | None` — the Editor layer name (nullable when disconnected)
- `editor_block_id: str | None` — the Editor block
- `editor_data_item_id: str | None`
- `ma3_track_coord: str` — MA3 coordinate string (e.g., `"tc1_tg1_tr1"` or `"TC1"`)
- `show_manager_block_id: str` — the ShowManager block that owns this entity
- `sync_status: str` — one of `"synced"`, `"out_of_sync"`, `"conflict"`, `"unmapped"`, `"excluded"`, `"disconnected"`
- `sync_type: SyncType` — `"showmanager_layer"` or `"ez_layer"`
- `sync_direction: SyncDirection` — `MA3_TO_EZ`, `EZ_TO_MA3`, `BIDIRECTIONAL`
- `conflict_resolution: ConflictResolution` — `USE_MA3`, `USE_EZ`, `MERGE`, `SKIP`, `PROMPT_USER`

**When an Editor layer is deleted:**
- The sync entity is NOT removed immediately
- It is marked `sync_status = "disconnected"` 
- The user sees it in the ShowManager Layer Sync tab as disconnected
- User can reconnect to another Editor layer or remove the entry entirely

---

## 2. Data Model

### 2.1 DataItem (Base)

**Abstract entity** with fields:
- `id: str` — UUID, assigned on creation
- `block_id: str` — which block produced this item
- `name: str` — display name
- `type: str` — `"Audio"` or `"Event"`
- `created_at: datetime`
- `file_path: str | None` — filesystem path to stored data
- `metadata: Dict[str, Any]` — extensible properties

**Key metadata keys (shared):**
- `output_name: str` — semantic name from the output port (e.g., `"audio:vocals"`, `"audio:drums"`)
- `_source_block_id: str` — which block produced this item (set by engine)
- `_source_audio_id: str` — for event items, the AudioDataItem they came from

---

### 2.2 AudioDataItem

Extends DataItem for audio waveform data.

**Additional fields:**
- `sample_rate: int | None` — e.g., 44100
- `length_ms: float` — duration in milliseconds
- In-memory `_audio_data: np.ndarray | None` — loaded lazily from `file_path`

**Methods:**
- `load_audio(path)` — loads WAV/MP3/FLAC/etc. via librosa into `_audio_data`
- `save_audio(path)` — writes `_audio_data` to WAV file
- `get_audio_data() -> np.ndarray | None` — returns loaded samples
- `set_audio_data(data, sample_rate)` — sets in-memory data

**Waveform service:** After loading, a waveform (amplitude summary) is computed and stored for timeline display. Resolution is configurable (default 50 points). Waveforms are shown in event clips on the timeline only when `show_waveforms_in_timeline` is True (disabled by default for performance). Min event width for waveform display: 30px (LOD threshold).

**Output name convention for multi-stem:** When a Separator block produces multiple stems, each AudioDataItem's `output_name` is `"audio:<stem_name>"` (e.g., `"audio:vocals"`, `"audio:drums"`). AudioFilter blocks preserve this `output_name` from upstream items.

---

### 2.3 EventDataItem

Extends DataItem for time-stamped event data.

**Additional fields:**
- `event_count: int` — total events across all layers
- `_layers: List[EventLayer]` — organized layers of events

**Methods:**
- `get_events() -> List[Event]` — all events across all layers (flattened)
- `get_layers() -> List[EventLayer]` — all layers
- `get_layer_by_name(name) -> EventLayer | None`
- `add_layer(layer)` — adds a new layer; raises if name exists
- `remove_layer(name) -> bool`
- `to_dict() / from_dict()` — serialization

**EventLayer fields:**
- `id: str` — UUID
- `name: str` — layer name (used as event classification bucket)
- `events: List[Event]` — ordered by time
- `metadata: Dict`

Methods: `add_event(event)`, `clear_events()`, `get_events()`, `to_dict()`, `from_dict()`

**Event fields:**
- `id: str` — UUID
- `time: float` — position in seconds
- `duration: float` — length in seconds (0 for instant/marker events)
- `classification: str` — human-readable label (same as layer name for layer-based events)
- `metadata: Dict` — arbitrary event properties

**Key metadata keys on Events:**
- `audio_id: str` — ID of source AudioDataItem (set by DetectOnsets, etc.)
- `_ma3_idx: int` — MA3 event index for sync matching (preserved across snapshots)
- `_synced_from_ma3: bool` — event was received from MA3
- `_ma3_track_coord: str` — MA3 coordinate
- `_show_manager_block_id: str`
- `render_as_marker: bool` — display as marker (point) rather than block (rectangle)
- `_source_audio_id: str`

**Instant vs. Block events:**
- Events with `duration == 0` are "marker" events — displayed as a point/diamond/shape
- Events with `duration > 0` are "block" events — displayed as a rectangle with width proportional to duration

---

### 2.4 Data Flow Through Block Graph

Data flows from upstream blocks to downstream blocks via connections:

1. **Block executes** → produces `DataItem`s → stored in `data_item_repo` with `block_id`
2. **Engine saves** output DataItem IDs into `block_local_state` for the source block's output ports
3. **Downstream block executes** → engine reads the upstream block's `block_local_state` → loads DataItems from `data_item_repo` → passes as `inputs` dict to block processor
4. **Engine handles persistence** — processors return `DataItem`s (unsaved), engine calls `data_item_repo.create()` and updates local state

**Inputs dict key format:** `{input_port_name}` (e.g., `"audio"`, `"events"`)

**Multi-item ports:** If multiple connections arrive at the same input port, inputs may be lists. Processors must handle both single item and `List[DataItem]` inputs.

**Audio resolution from events:** AudioFilter and some other blocks can receive events on their `events` port instead of audio on `audio`. They resolve the source audio by looking up `audio_id` from event metadata in `data_item_repo`. This is the `_lookup_audio_from_events()` pattern.

---

### 2.5 DataState

Used to track whether a block's data is fresh or stale.

**Values:** `FRESH`, `STALE`

**Stale conditions:** A block becomes stale when an upstream block's data changes (e.g., re-execution, configuration change). Status levels check `DataState.STALE` to show the "stale" indicator.

---

### 2.6 Layer Ordering

Layers in the Editor timeline have a persistent visual ordering stored in `layer_orders` table.

**Data structure:** `LayerOrder(block_id, [LayerKey(group_name, layer_name), ...])`

**Layer identity (stable across sessions):**
- For MA3-synced layers: `group_id` uses `tc_N` prefix (e.g., `tc_1`), `group_name` uses `"TC N"` (e.g., `"TC 1"`)
- For non-synced layers: `group_id` is `"{source_block_id}:{item.name}"`, `group_name` is `item.name`
- This identity must survive song switches and re-pulls where EventDataItem UUIDs change

**Layer ordering service:** `layer_order_service` — exposes `add_layer(block_id, group_name, layer_name)`, `remove_layer(...)`, `get_order(block_id)`.

**Reconciliation:** On panel open, `_reconcile_layer_order()` merges persisted order with actual layers. New layers are appended. Missing layers are removed from the order. The result is applied with `_apply_layer_order(order)`.

---

## 3. Application Services

### 3.1 ApplicationFacade

The single application-level API. All UI and external code interacts with the system exclusively through this facade. It delegates to feature services.

**Feature Areas:**

#### Project Management
- `create_project(name) → Result[Project]`
- `load_project(path) → Result[Project]` — clears DB, loads from `.ez` file
- `save_project(project_id) → Result` — saves to `.ez` ZIP
- `save_project_as(project_id, path) → Result`
- `delete_project(project_id) → Result`
- `list_projects() → Result[List[Project]]`
- `describe_project(project_id) → Result[Project]`
- `get_recent_projects() → List[str]` — filesystem paths

#### Block Management
- `create_block(project_id, name, type, metadata) → Result[Block]`
- `describe_block(block_id) → Result[Block]`
- `list_blocks(project_id) → Result[List[Block]]`
- `update_block_metadata(block_id, metadata) → Result`
- `rename_block(block_id, new_name) → Result`
- `delete_block(block_id) → Result`
- `get_block_status(block_id) → Result[StatusInfo]`

#### Connection Management
- `create_connection(source_block_id, source_output, target_block_id, target_input) → Result[Connection]`
- `delete_connection(connection_id) → Result`
- `list_connections(project_id) → Result[List[Connection]]`
- `list_connections_by_block(block_id) → Result[List[Connection]]`

#### Execution
- `run_block(block_id, strategy) → Result` — executes a single block and all dependencies
- `run_project(project_id) → Result` — executes all blocks in topological order
- `get_execution_status(block_id) → Result[ExecutionStatus]`

#### Data Items
- `list_data_items(block_id) → Result[List[DataItem]]`
- `get_data_item(data_item_id) → Result[DataItem]`
- `delete_data_item(data_item_id) → Result`
- `create_data_item(block_id, name, type, metadata) → Result[DataItem]`

#### UI State
- `get_ui_state(state_type, entity_id) → Result[Dict]`
- `set_ui_state(state_type, entity_id, data) → Result`
- `delete_ui_state(state_type, entity_id) → Result`

**UI state types used:**
- `"editor_layers"` — layer list/overrides for Editor blocks (keyed by block_id)
- `"block_position"` — x/y position in node editor (keyed by block_id)
- `"node_editor_viewport"` — scroll/zoom state of node editor

#### Editor Operations (via EditorAPI)
- `get_editor_layers(block_id) → List[LayerDict]`
- `create_editor_layer(block_id, layer_name, properties) → Result`
- `update_editor_layer(block_id, layer_name, properties) → Result`
- `delete_editor_layer(block_id, layer_name) → Result`
- `get_editor_events(block_id, layer_name?, source?) → List[EventDict]`
- `add_editor_events(block_id, events, source) → Result`
- `apply_layer_snapshot(data_item_id, layer_name, events) → Result`

#### Settings / Preferences
- `get_preference(key) → Any`
- `set_preference(key, value) → Result`
- `get_app_settings() → AppSettingsManager`

#### Setlists
- `create_setlist(project_id, audio_folder_path) → Result[Setlist]`
- `get_setlist(project_id) → Result[Setlist]`
- `update_setlist(setlist_id, ...) → Result`
- `add_setlist_song(setlist_id, audio_path, order_index) → Result[SetlistSong]`
- `remove_setlist_song(song_id) → Result`
- `reorder_setlist_songs(setlist_id, song_ids_in_order) → Result`
- `process_setlist(setlist_id) → Result` — runs the block graph for each song

#### Action Sets / Action Items
- `create_action_set(name, project_id, ...) → Result[ActionSet]`
- `list_action_sets(project_id) → Result[List[ActionSet]]`
- `create_action_item(action_set_id, block_id, action_name, ...) → Result[ActionItem]`
- `list_action_items(action_set_id) → Result[List[ActionItem]]`
- `execute_action_items(items) → Result`

#### Show Manager / MA3
- `start_ma3_listener(block_id) → Result`
- `stop_ma3_listener(block_id) → Result`
- `get_ma3_connection_status(block_id) → Result[StatusInfo]`
- `sync_layer_to_ma3(block_id, layer_name) → Result`
- `sync_layer_from_ma3(block_id, ma3_coord) → Result`

---

### 3.2 Command Pattern

All mutations go through the command bus. Commands are undoable/redoable.

**Base class:** `EchoZeroCommand`
- `redo()` — executes the command
- `undo()` — reverses the command
- `text: str` — human-readable description for undo history

**CommandBus:**
- `execute(command)` — runs `command.redo()` and adds to undo history
- `begin_macro(name)` — starts a named macro (groups commands into one undo step)
- `end_macro()` — ends the macro
- Undo limit: 50 steps (configurable via `max_undo_steps` setting)

**Key command groups:**

| Command | What it does |
|---------|--------------|
| `CreateBlockCommand` | Creates a block entity |
| `RenameBlockCommand` | Renames a block |
| `DeleteBlockCommand` | Deletes block + connections + data items |
| `CreateConnectionCommand` | Creates a connection |
| `DeleteConnectionCommand` | Removes a connection |
| `EditorCreateLayerCommand` | Creates a layer in Editor UI state; also syncs to open TimelineWidget immediately |
| `EditorAddEventsCommand` | Adds events to Editor (creates/finds EventDataItem, adds to layers) |
| `EditorUpdateLayerCommand` | Updates layer properties (height, color, visibility, lock) |
| `EditorDeleteLayerCommand` | Deletes layer from UI state, EventDataItem, and marks sync entity disconnected |
| `EditorGetLayersCommand` | Query (non-undoable): builds layer list from EventDataItems + UI overrides |
| `EditorGetEventsCommand` | Query (non-undoable): returns events filtered by layer/source |
| `ApplyLayerSnapshotCommand` | Replaces all events in a layer with a snapshot; preserves `render_as_marker` and user metadata by merging with existing events matched by `_ma3_idx` |
| `CreateEventDataItemCommand` | Creates an EventDataItem entity |
| `AddEventToDataItemCommand` | Adds a single Event to an EventDataItem layer |

**Macro pattern for multi-step operations:** `EditorAddEventsCommand.redo()` opens a macro, executes sub-commands, then closes it. The macro collapses to one undo step.

---

### 3.3 Event Bus

Pub/sub system for decoupled communication between services and UI.

**Key event types:**
- `ProjectCreated(project_id)`
- `ProjectUpdated(project_id)`
- `ProjectDeleted(project_id)`
- `BlockCreated(project_id, data: {id, name, type})`
- `BlockUpdated(project_id, data: {id, name, ...})` — with optional flags: `layers_updated`, `events_updated`, `settings_updated`, `synced_layers_changed`
- `BlockDeleted(project_id, block_id)`
- `ConnectionCreated(project_id, connection_id)`
- `ConnectionDeleted(project_id, connection_id)`
- `MA3MessageReceived(data: {object_type, object_name, change_type, timestamp, ma3_data})`
- `ExecutionStarted(block_id)`
- `ExecutionCompleted(block_id)`
- `ExecutionFailed(block_id, error)`

**Subscriber pattern:** `event_bus.subscribe(EventType, handler_fn)`. Handlers are called synchronously in the publishing thread.

---

### 3.4 Execution Engine

Executes the block graph in dependency order.

**Algorithm:**
1. Build directed acyclic graph from `Connection` entities
2. Topological sort (Kahn's algorithm or DFS-based)
3. Execute each block in sorted order:
   a. Gather inputs: look up `block_local_state` for each input port → load `DataItem`s from repo
   b. Call `BlockProcessor.process(block, inputs, metadata)`
   c. Store returned `DataItem`s in `data_item_repo` (create if new, update if exists)
   d. Update `block_local_state` with output DataItem IDs
   e. Emit `BlockUpdated` event
4. Progress tracking via `ProgressTracker` (emits start, update, complete, error)

**Execution strategies:**
- `ExecutionStrategy.BLOCK` — run only the target block (assume inputs already exist)
- `ExecutionStrategy.FULL` — run target block and all ancestors in dependency order
- `ExecutionStrategy.PROJECT` — run all blocks in the project

**Progress tracking:**
- `ProgressTracker.start(message, total)` — begins tracking
- `ProgressTracker.update(current, total, message)` — incremental
- `ProgressTracker.complete(message)` — done
- UI receives progress via Qt signals (thread-safe): `progress_start_signal`, `progress_update_signal`, `progress_complete_signal`, `progress_error_signal`
- `track_progress(iterable, tracker, label)` — helper that auto-updates tracker per item

**Block execution workers:** The UI runs execution in a background `QThread` (via `RunBlockThread`) to avoid freezing. Progress signals are emitted from the worker and connected to the main thread UI.

---

### 3.5 EditorAPI

Separate application-layer service for Editor-specific operations. Lives in `src/features/blocks/application/editor_api.py`.

Wraps command dispatch and provides the facade's Editor methods. Handles:
- Layer CRUD with immediate TimelineWidget sync (critical for AddEvents following CreateLayer)
- Event queries with layer filtering
- Snapshot application with metadata preservation
- Group identity resolution for stable cross-session layer matching

**Stable group identity rule:**
- If `metadata.group_id` starts with `"tc_"` → use that ID/name directly (MA3 timecode pool)
- Otherwise → `group_id = "{source_block_id}:{item.name}"`, `group_name = item.name`

---

## 4. Block Types

### 4.1 LoadAudio

**Type:** `"LoadAudio"`  
**Category:** Input (blue)  
**Inputs:** none  
**Outputs:** `audio` (Audio)

**Configuration (block.metadata):**
- `file_path: str` — absolute path to audio file
- Supported formats: WAV, MP3, FLAC (anything librosa supports)

**Processing:**
1. Load audio file from `file_path` via librosa
2. Create `AudioDataItem` with sample rate, duration, waveform
3. Compute and store waveform for timeline display
4. Return `{"audio": audio_item}`

**Status levels:**
- Error (0): `file_path` missing OR file does not exist on disk
- Ready (1): file exists

**Quick actions:** Set File (opens file dialog), Reset Data

---

### 4.2 SetlistAudioInput

**Type:** `"SetlistAudioInput"`  
**Category:** Input  
**Inputs:** none  
**Outputs:** `audio` (Audio)

**Configuration:**
- `audio_path: str` — set per-song by the setlist processor

**Processing:** Same as LoadAudio but path comes from `audio_path` metadata key rather than `file_path`.

**Purpose:** Used exclusively in setlist processing. A SetlistAudioInput block in the graph gets its `audio_path` set to each song's file path before execution.

**Status levels:**
- Error (0): `audio_path` missing or file does not exist
- Ready (1): file exists

---

### 4.3 DetectOnsets

**Type:** `"DetectOnsets"`  
**Category:** Analysis (orange)  
**Inputs:** `audio` (Audio)  
**Outputs:** `events` (Event)

**Configuration (block.metadata):**
- `onset_method: str` — `"librosa"`, `"essentia"`, etc.
- `hop_length: int`
- `backtrack: bool`
- Various threshold parameters depending on method

**Processing:**
1. Load audio from input `AudioDataItem`
2. Run onset detection algorithm
3. Create `EventDataItem` with one event per detected onset
4. Each event: `time` = onset time in seconds, `duration = 0` (marker), `classification = "onset"`
5. Event `metadata["audio_id"] = audio_item.id` — allows downstream blocks to resolve source audio
6. Return `{"events": event_data_item}`

**Status levels:**
- Error (0): no audio input connected
- Stale (1): data is STALE
- Ready (2): all clear

---

### 4.4 Separator

**Type:** `"Separator"`  
**Category:** Transform (orange/purple)  
**Inputs:** `audio` (Audio)  
**Outputs:** `drums`, `bass`, `other`, `vocals` (all Audio)

**Configuration:**
- `model: str` — Demucs model name (e.g., `"htdemucs"`, `"mdx_extra"`)
- `device: str` — `"cpu"` or `"cuda"`

**Processing:**
1. Run Demucs source separation on input audio
2. Produce one `AudioDataItem` per stem
3. Each item's `metadata["output_name"] = "audio:<stem>"` (e.g., `"audio:vocals"`)
4. Return `{"drums": ..., "bass": ..., "other": ..., "vocals": ...}`

**Edge case:** If model unavailable or audio is corrupt → `ProcessingError` raised → execution fails with error status.

**Output naming:** When a list of stems is produced on a single port (e.g., `"audio"` returning all stems as a list), each `AudioDataItem` carries its `output_name` in metadata.

---

### 4.5 Editor

**Type:** `"Editor"`  
**Category:** Editing (gray/purple)  
**Inputs:** `events` (Event) — can have multiple connections  
**Outputs:** `events` (Event), `manipulator` (Manipulator — connects to ShowManager)

**Purpose:** The primary interactive editing block. Hosts the Timeline widget. Stores and manages layers of time-stamped events. Receives events from upstream analysis blocks and allows manual editing.

**Processing:** Editor blocks do not process data in the normal sense. They hold and manage EventDataItem state. The "processing" phase copies/merges incoming events into the Editor's own EventDataItems.

**Layers:** Each EventDataItem owned by an Editor block corresponds to one "group" of layers. Each EventLayer within that item is one layer in the timeline.

**Synced layers:** Some layers are marked `is_synced=True` and linked to a ShowManager block and MA3 track coordinate. These layers have special restrictions (see §5 Editor Behaviors).

---

### 4.6 ShowManager

**Type:** `"ShowManager"`  
**Category:** Integration  
**Inputs/Outputs:** `manipulator` port (bidirectional)

**Purpose:** Manages the MA3 synchronization system. Connects to an Editor block via Manipulator port. Maintains the list of synced layer pairs (Editor layer ↔ MA3 track).

**Configuration (block.metadata):**
- `ma3_listen_port: int` — default 9000
- `ma3_listen_address: str` — default `"127.0.0.1"`
- `synced_layers: List[Dict]` — persisted sync entity list
- `target_timecode: str | None` — target TC pool coordinate

---

### 4.7 AudioPlayer

**Type:** `"AudioPlayer"`  
**Category:** Utility/Preview (teal)  
**Inputs:** `audio` (Audio)  
**Outputs:** none

**Purpose:** Embedded audio playback preview block. Displays a compact waveform and transport controls directly in the node editor (wider block, 350px). Connects to Qt Multimedia for playback.

**Configuration:** No user-configurable metadata.

**UI behavior:** Block item embeds waveform widget + play/pause/stop controls. Playback is local to this block and independent of timeline playback.

---

### 4.8 AudioFilter

**Type:** `"AudioFilter"`  
**Category:** Transform (orange)  
**Inputs:** `audio` (Audio) OR `events` (Event — resolves source audio via audio_id)  
**Outputs:** `audio` (Audio)

**Supported filter types:**

| Key | Name | Parameters |
|-----|------|-----------|
| `lowpass` | Low-Pass | cutoff_freq, order |
| `highpass` | High-Pass | cutoff_freq, order |
| `bandpass` | Band-Pass | cutoff_freq, cutoff_freq_high, order |
| `bandstop` | Band-Stop (Notch) | cutoff_freq, cutoff_freq_high, order |
| `lowshelf` | Low-Shelf | cutoff_freq, gain_db, q_factor |
| `highshelf` | High-Shelf | cutoff_freq, gain_db, q_factor |
| `peak` | Peak (Bell) | cutoff_freq, gain_db, q_factor |

**Configuration:**
- `filter_type: str` — one of the keys above (default: `"lowpass"`)
- `cutoff_freq: float` — primary cutoff in Hz (default: 1000.0; clamped 20–nyquist)
- `cutoff_freq_high: float` — upper cutoff for band filters (default: 8000.0)
- `order: int` — Butterworth filter order (default: 4; valid range 1–8)
- `gain_db: float` — gain for shelf/peak filters (default: 0.0)
- `q_factor: float` — Q factor for shelf/peak filters (default: 0.707)

**Filter implementation:**
- Butterworth filters use scipy `butter(..., output="sos")` + `sosfilt()`
- Shelf/peak filters use custom biquad SOS design
- All SOS for numerical stability
- Multi-channel audio: each channel filtered independently

**Input resolution from events:** When input is `events`, resolves source `AudioDataItem`s by reading `audio_id` from each event's metadata (or `_source_audio_id` from item metadata), then looks up those audio items in the data item repo.

**Output:** 
- Single input → single output `AudioDataItem`
- List input → list output
- Each output item preserves `upstream output_name` in metadata (so downstream blocks see the same semantic name)

**Waveform:** Generated for each output item after filtering.

**Output directory:** `<source_file_parent>/<block_name>_filtered/` or `%TEMP%/echozero_filters/`

**Status levels:**
- Error (0): neither `audio` nor `events` input connected
- Stale (1): data STALE
- Ready (2): OK

---

### 4.9 ExportAudio

**Type:** `"ExportAudio"`  
**Category:** Export (teal)  
**Inputs:** `audio` (Audio)  
**Outputs:** none

**Configuration:**
- `output_dir: str` — destination directory (created if missing)
- `audio_format: str` — `"wav"`, `"mp3"`, `"flac"`, `"ogg"`, `"m4a"`, `"aiff"`, `"aif"` (default: `"wav"`)

**Processing:**
1. Create output directory if needed
2. For each input DataItem (single or list):
   - Skip items with no `file_path` or missing source file (warning, not error)
   - Copy source file to `<output_dir>/<item.name>_<item.id>.<format>`
3. Raise `ProcessingError` if zero files were exported

**Status levels:**
- Error (0): `output_dir` not configured or not writable
- Warning (1): audio input not connected
- Ready (2): OK

---

### 4.10 TranscribeNote / NoteExtractorBasicPitch

**Type:** `"TranscribeNote"` or `"NoteExtractorBasicPitch"`  
**Category:** Analysis (green)  
**Inputs:** `audio` (Audio)  
**Outputs:** `events` (Event with pitch/note data)

**Processing:** ML-based note transcription using Basic Pitch model. Events carry pitch, onset, offset, velocity in metadata.

---

### 4.11 TranscribeLib / NoteExtractorLibrosa

**Type:** `"TranscribeLib"` or `"NoteExtractorLibrosa"`  
**Category:** Analysis (lime green)  
**Inputs:** `audio` (Audio)  
**Outputs:** `events` (Event)

**Processing:** Librosa-based note extraction. Faster, lightweight, but less accurate than ML approach.

---

### 4.12 CommandSequencer

**Type:** `"CommandSequencer"`  
**Category:** Utility (yellow)

**Purpose:** Executes a sequence of quick actions in order. Used in automation workflows. Actions are configured as an ordered list of block quick action references.

---

### 4.13 ExportMA2

**Type:** `"ExportMA2"`  
**Category:** Export (teal)

**Purpose:** Exports event data in grandMA2-compatible format. Configuration and format details in metadata.

---

### 4.14 Quick Actions System

Every block type registers quick actions via `@quick_action("BlockType", "Action Name")` decorator.

**QuickAction fields:**
- `name: str` — display name (button text)
- `description: str` — tooltip
- `handler: Callable[[facade, block_id], Any]`
- `category: ActionCategory` — `EXECUTE`, `CONFIGURE`, `FILE`, `EDIT`, `VIEW`, `EXPORT`
- `icon: str | None`
- `primary: bool` — highlighted/prominent in UI
- `dangerous: bool` — destructive styling
- `requires_panel: bool` — opens block panel instead of direct execution
- `keyboard_shortcut: str | None`

**Common actions** (available on all blocks): Delete block, Rename block, Open panel, Run block.

**ActionCategory** determines UI grouping in block context menus and quick action panels.

---

## 5. Editor Behaviors

The Editor block's timeline widget is the most complex UI component.

### 5.1 Timeline Architecture

**Timeline widget components:**
- `TimelineWidget` (main widget) — contains scene, view, layer manager, playback controller, inspector
- `TimelineScene` (QGraphicsScene) — contains all event items
- `TimelineView` (QGraphicsView) — handles viewport, zoom, scroll
- `LayerManager` — manages layer data and ordering
- `PlaybackController` — coordinates audio with 60 FPS playhead updates
- `MovementController` — manages drag/move state machine
- `SnapCalculator` — computes snap positions
- `GridSystem` — manages grid settings and intervals
- `EventInspector` — side panel showing selected event properties

---

### 5.2 Layer Management

**Layer properties (per layer):**
- `name: str` — unique within the Editor block
- `height: int` — in pixels (default 40px; range 20–200px; configurable in settings: `default_layer_height`)
- `color: str | None` — hex color; None = auto-color from palette
- `visible: bool` — default True
- `locked: bool` — default False; when locked, events cannot be moved/edited
- `group_id: str` — stable identity for cross-session matching
- `group_name: str` — human-readable group label
- `group_index: int | None` — position within group
- `is_synced: bool` — linked to MA3 track
- `show_manager_block_id: str | None`
- `ma3_track_coord: str | None`
- `derived_from_ma3: bool` — was created from MA3 sync

**Layer header (left column):**
- Default width: 120px; min 80px, max 300px — user-draggable
- Shows: layer name (editable), eye icon (visibility toggle), lock icon, color swatch, delete button
- Synced layers show MA3 coordinate badge

**Layer column width** persists across sessions via `preferences` table under `timeline.layer_column_width`.

**Duplicate names:** `EditorCreateLayerCommand` automatically suffixes with `_N` (e.g., `"kick_1"`) to ensure unique names. Tracks this for undo.

**Layer creation immediacy:** After `EditorCreateLayerCommand.redo()`, the code immediately tries to restore the layer in any open `EditorPanel` (via `_ensure_layer_restored_in_timeline()`). This is critical because `EditorAddEventsCommand` immediately follows and needs the layer to exist in the `LayerManager`.

**Layer ordering:** Persisted to `layer_orders` table. Displayed order is reconciled with actual layers on panel open. New layers are appended. Stale entries are pruned.

**Layer groups:** Layers with the same `group_id` are visually grouped. A group header row is shown with the `group_name`. Groups can be collapsed/expanded (implied by group_index ordering).

**Synced layer restrictions:** Synced layers (`is_synced=True`) cannot be edited freely — edits propagate to MA3. The sync handler intercepts writes.

---

### 5.3 Event Editing

**Event display modes:**
- **Block events** (`duration > 0`): Rendered as colored rectangles. Width = `duration × pixels_per_second`.
- **Marker events** (`duration == 0` OR `render_as_marker=True` in metadata): Rendered as a configurable shape (diamond default). Width is fixed (default 13px).

**Event item states:** normal, hover, selected, dragging, resize-handle-hover

**Selection:**
- Click to select single event (deselects others)
- Click on empty area to deselect all
- Rubber-band selection: click-drag on empty area selects all events within rectangle
- Multi-select: Ctrl+Click to add/remove from selection (implied by standard Qt behavior)

**Move operation:**
- Click-drag on event body
- Drag threshold: 3 pixels before drag starts (prevents accidental moves)
- State machine: `IDLE → PENDING → DRAGGING → IDLE`
- During drag, all selected events move together maintaining relative offsets
- Escape key cancels drag and restores original positions
- On release: emits `moves_completed(List[EventMoveResult])` → saved to EventDataItem

**Resize operation:**
- Hover near left or right edge of a block event → resize cursor appears
- Drag edge to resize
- Same state machine as move
- Minimum duration enforced (cannot resize to zero or negative)
- On release: emits `resizes_completed(List[EventResizeResult])` → saved

**Layer transfer:** Events can be moved to a different layer by dragging vertically. The movement controller tracks `layer_offset` during drag.

**Create event:** Double-click on empty area in a layer → creates a new event at that time position. Duration determined by drag length (if drag) or default duration.

**Delete event:** Select event(s) → Delete key → removes selected events.

**Slice event:** (Feature: if implemented) Select a block event, place at time position → split into two events at that point.

**Context menu on event:** Right-click → shows classification, time, duration info, plus actions (delete, properties, render_as_marker toggle, etc.).

---

### 5.4 Grid and Snap

**Grid system:**
- `GridSystem` holds `GridSettings`
- Grid intervals are **automatically calculated** from timebase (FPS/seconds) — not user-adjustable for frequency
- Major grid lines: visually distinct, labeled with time
- Minor grid lines: subdivision between major lines
- Grid display color: `TIMELINE_GRID_MAJOR (#50505)` / `TIMELINE_GRID_MINOR (#323237)`

**Grid display:** Configurable via `show_grid_lines: bool` (default True).

**Snap modes** (`snap_interval_mode`):
- `"auto"` — snap to base minor grid interval (computed from zoom level and timebase)
- `"1f"` — snap to exactly 1 frame at configured FPS
- `"2f"` — snap to 2 frames
- `"5f"` — snap to 5 frames
- `"10f"` — snap to 10 frames
- `"1s"` — snap to exactly 1 second

**Snap calculation:**
- In `"auto"` mode: `snapped_time = round(time / minor_interval) × minor_interval` using integer indexing
- In frame mode: `snapped_time = round(time / explicit_interval) × explicit_interval`
- Always `max(0, snapped_time)` — cannot snap to negative time
- Snap can be disabled: `snap_enabled=False` → returns time as-is

**Snap is enabled by default** (`snap_enabled: bool = True`, `snap_to_grid: bool = True`).

---

### 5.5 Zoom and Scroll

**Zoom:**
- `pixels_per_second` (pps): default 100.0; min 10.0; max 1000.0
- Zoom with Ctrl+Scroll wheel or pinch gesture
- Zoom center is the cursor position (not left edge)
- Zoom range enforced on settings set

**Scroll position** persists across sessions via preferences (`last_scroll_x`, `last_scroll_y`) when `restore_scroll_position=True`.

**Playhead follow modes** (`playhead_follow_mode`):
- `"off"` — no automatic scrolling
- `"page"` — scrolls by page when playhead reaches edge
- `"smooth"` — continuously keeps playhead in view
- `"center"` — keeps playhead centered in viewport

---

### 5.6 Playback

**PlaybackController:**
- 60 FPS update timer (16.67ms interval)
- Smooth interpolation between audio backend position updates
- Backend position changes by >1ms → sync immediately (no interpolation for that tick)
- Backend position unchanged → interpolate using elapsed time

**Transport controls:**
- Play / Pause / Stop
- Seek (click on ruler or set position)
- Toggle: Space bar (implied standard behavior)

**Audio backend:** `SimpleAudioPlayer` using Qt Multimedia (`QMediaPlayer`).  
Loading: sets source URL, sets `_play_when_ready` flag, starts playback when `MediaStatus.LoadedMedia` fires.

**Cleanup:** Uses `deleteLater()` for async deletion to prevent blocking. Does NOT call `stop()` during cleanup (can block). Disconnects all signals first.

**Duration:** Set from audio backend. If no backend, duration set manually via `set_duration()`.

---

### 5.7 Event Inspector

Side panel in the timeline. Shows properties of selected events.

**Empty selection:** shows "No events selected" placeholder.

**Single event selected:**
- Count label: "1 event selected"
- Hero section: Time (monospace), Duration (monospace), Classification, Confidence (if present)
- Clip player section: compact waveform (64px height) + play/pause controls for the event's audio clip
- Details section: Event ID, display mode (block/marker), all remaining metadata key-value pairs

**Multi-event selected:** shows count ("N events selected"), aggregate statistics.

**Clip player in inspector:**
- Plays the audio segment corresponding to the selected event (from source AudioDataItem, trimmed to event time/duration)
- Uses librosa for audio slicing (requires `HAS_AUDIO_LIBS=True`)
- Fallback if no audio libs: clip player section hidden

**Inspector visibility** persists via `inspector_visible: bool` (default True). Width persists via `inspector_width: int` (default 220px; range 150–500px).

---

### 5.8 Settings Panel

Accessible from a settings button in the timeline widget toolbar.

**Configurable settings (all persist to preferences):**

| Setting | Default | Range/Options |
|---------|---------|---------------|
| `layer_column_width` | 120px | 80–300px |
| `default_layer_height` | 40px | 20–200px |
| `snap_enabled` | True | bool |
| `snap_to_grid` | True | bool |
| `snap_interval_mode` | `"auto"` | `auto`, `1f`, `2f`, `5f`, `10f`, `1s` |
| `default_pixels_per_second` | 100.0 | 10.0–1000.0 |
| `playhead_follow_mode` | `"page"` | `off`, `page`, `smooth`, `center` |
| `show_grid_lines` | True | bool |
| `show_waveform` | True | bool |
| `waveform_opacity` | 0.5 | 0.0–1.0 |
| `show_waveforms_in_timeline` | False | bool |
| `waveform_resolution` | 50 | 5–1000 points |
| `waveform_min_width` | 30px | 5–200px |
| `show_event_labels` | True | bool |
| `show_event_duration_labels` | False | bool |
| `highlight_current_event` | True | bool |
| `inspector_visible` | True | bool |
| `inspector_width` | 220px | 150–500px |
| `restore_scroll_position` | True | bool |

**Event styling settings (block events):**

| Setting | Default |
|---------|---------|
| `block_event_height` | 32px (16–100) |
| `block_event_border_radius` | 3px (0–20) |
| `block_event_border_width` | 1px (0–5) |
| `block_event_label_font_size` | 10px (6–24) |
| `block_event_border_darken_percent` | 150 (100–200) |
| `block_event_opacity` | 1.0 (0–1) |
| `block_event_z_value` | 0.0 (±1000) |
| `block_event_rotation` | 0.0° (0–360, normalized) |
| `block_event_scale` | 1.0 (0.1–5.0) |
| `block_event_drop_shadow_enabled` | False |
| `block_event_drop_shadow_blur_radius` | 5.0px (0–50) |
| `block_event_drop_shadow_offset_x` | 2.0px (±50) |
| `block_event_drop_shadow_offset_y` | 2.0px (±50) |
| `block_event_drop_shadow_color` | `"#000000"` |
| `block_event_drop_shadow_opacity` | 0.5 (0–1) |

**Marker event styling settings:**

| Setting | Default | Options/Range |
|---------|---------|---------------|
| `marker_event_shape` | `"diamond"` | diamond, circle, square, triangle_up, triangle_down, triangle_left, triangle_right, arrow_up, arrow_down, arrow_left, arrow_right, star, cross, plus |
| `marker_event_width` | 13px (4–50) |  |
| `marker_event_border_width` | 1px (0–5) |  |
| `marker_event_border_darken_percent` | 150 (100–200) |  |
| `marker_event_opacity` | 1.0 (0–1) |  |
| `marker_event_z_value` | 0.0 |  |
| `marker_event_rotation` | 0.0° |  |
| `marker_event_scale` | 1.0 (0.1–5.0) |  |
| Drop shadow settings | same as block events | |

**Ruler label styling:**

| Setting | Default |
|---------|---------|
| `event_time_font_size` | 10px (6–24) |
| `event_time_font_family` | `"monospace"` (monospace, default, small) |
| `event_time_major_color` | `"#F0F0F5"` |
| `event_time_minor_color` | `"#78787D"` |

**Recent layer colors:** Up to 10 most recently assigned colors stored for quick reuse.

---

### 5.9 Keyboard Shortcuts

Default shortcuts (all stored in preferences under `timeline.shortcut_*`):

| Shortcut | Action |
|----------|--------|
| `Left Arrow` | Move selected events left by 1 grid unit |
| `Right Arrow` | Move selected events right by 1 grid unit |
| `Ctrl+Up` | Move selected events up one layer |
| `Ctrl+Down` | Move selected events down one layer |
| `Delete` / `Backspace` | Delete selected events |
| `Escape` | Cancel active drag operation |
| `Space` | Toggle playback (play/pause) |

All shortcuts are user-configurable via the settings panel. Stored as string representations like `"Key_Left"`, `"Ctrl+Key_Left"`.

---

### 5.10 Waveform Display

**In timeline events (clip waveform):**
- Disabled by default (`show_waveforms_in_timeline=False`) for performance
- When enabled: shows amplitude waveform inside event rectangle
- LOD: only rendered when event width ≥ `waveform_min_width` (30px default)
- Resolution: `waveform_resolution` points (default 50)
- Opacity: `waveform_opacity` (default 0.5)

**Waveform service:** `get_waveform_service()` singleton. `compute_and_store(audio_item)` computes and caches waveform data in the item.

---

### 5.11 EditorPanel (Block Panel)

The EditorPanel is the QWidget that hosts the TimelineWidget and wraps it for the dock system. It is opened per-block.

**Initialization sequence:**
1. Panel opens for `block_id`
2. Loads layers via `EditorGetLayersCommand`
3. Reconciles layer order: `_reconcile_layer_order()` → `_apply_layer_order(order)`
4. Restores layer state: `_restore_layer_state()` — creates layer entries in LayerManager
5. Loads events for each layer
6. Renders events on timeline

**Live updates:** Panel subscribes to `BlockUpdated` events for its block_id. On receive:
- If `layers_updated=True`: re-runs `_restore_layer_state()`
- If `events_updated=True`: reloads events
- If `settings_updated=True`: refreshes panel controls

---

## 6. Node Editor

### 6.1 Layout and Visual Representation

**Block visual representation (`BlockItem` in the scene):**
- Fixed width: 150px (default); some blocks have custom widths (AudioPlayer: 350px, AudioNegate: 210px)
- Height: computed from number of ports (min 70px, header 28px, port zone 22px each)
- Header: block type color, block name, status indicator
- Body: left column (input ports), right column (output ports)
- Sharp corners by default (`BLOCK_CORNER_RADIUS = 0`); toggle via `sharp_corners` setting

**Block colors by type:**

| Block Type | Color |
|------------|-------|
| LoadAudio | Blue `(65, 125, 210)` |
| DetectOnsets, TranscribeNote, TranscribeLib | Green `(110, 195, 115)` |
| Separator, AudioFilter | Orange `(215, 135, 65)` |
| ExportAudio, ExportMA2 | Cyan `(95, 195, 195)` |
| Editor, EditorV2 | Purple `(175, 115, 195)` |
| AudioPlayer | Teal `(60, 180, 170)` |
| CommandSequencer | Gray `(135, 135, 140)` |

**Port visual representation:**
- Input ports: left side of block, filled circles (green)
- Output ports: right side, filled circles (red)
- AUDIO ports: blue `PORT_AUDIO = (70, 170, 220)`
- EVENT ports: orange `PORT_EVENT = (230, 150, 70)`
- MANIPULATOR ports: bright orange `PORT_MANIPULATOR = (255, 140, 0)` — bidirectional indicator
- Port label: displayed offset from port circle (10px)

**Grid:**
- Node editor background grid: `GRID_SIZE = 20px`, major lines every 5 cells
- Color: `GRID_LINE = (60, 60, 60)`

**Block positioning:** Stored in UI state as `{"x": float, "y": float}` under `state_type="block_position"`, `entity_id=block_id`. Applied on panel/scene init.

---

### 6.2 Interaction Model

**Block placement:**
- New blocks appear at default position or at cursor/center of viewport
- Auto-layout button arranges blocks in topological order (implied by block flow)

**Block selection:**
- Click to select (shows blue border)
- Rubber-band drag to multi-select
- Click empty area to deselect

**Block movement:**
- Drag selected blocks — moves all selected blocks together
- Position saved to UI state on drop

**Connecting blocks:**
- Click and drag from an output port → connection line follows cursor
- Release on compatible input port → creates `Connection`
- Incompatible port types show visual feedback (e.g., red highlight)
- Existing connection to same input is allowed (no removal of old connection)

**Connection appearance:**
- Bezier curve from output to input port
- Color: `CONNECTION_NORMAL = (120, 120, 125)`
- Hover: `CONNECTION_HOVER = (180, 180, 185)`
- Selected: `CONNECTION_SELECTED = (220, 180, 60)`
- Width: 2px normal, 3px selected, 2.5px hover

**Block context menu (right-click):**
- Open block panel
- Rename block
- Delete block
- Quick actions specific to the block type

**Keyboard shortcuts in node editor:**
- `Delete` — delete selected block(s)
- `Ctrl+Z` / `Ctrl+Y` — undo/redo

---

### 6.3 Node Editor Scene

**NodeScene (QGraphicsScene):**
- Contains all block items and connection items
- Manages selection state
- Handles connection drawing state machine

**NodeEditorWindow:**
- Contains the scene in a QGraphicsView
- Toolbar: Add Block, Auto Layout, Zoom controls
- Block type palette (drag blocks to canvas or use add button)

---

## 7. Show Manager / MA3 Integration

### 7.1 MA3 Communication

**Protocol:** UDP (primary), OSC (future)

**EchoZeroBridge Lua plugin** runs inside grandMA3:
- Hooks into Sequence Pool using `HookObjectChange()`
- Sends pipe-delimited UDP messages: `type=sequence|name=Song1|change=changed|timestamp=1234567890|no=1`

**MA3CommunicationService:**
- Listens on configurable UDP port (default: 9000)
- Address: `127.0.0.1` (localhost default)
- Parses messages into `MA3Message(object_type, object_name, change_type, timestamp, data: Dict)`
- Publishes `MA3MessageReceived` event to event bus
- Supports custom handler registration: `register_handler(object_type, change_type, fn)`

**Settings:**
- `ma3_listen_enabled: bool` — default True
- `ma3_listen_port: int` — default 9000
- `ma3_listen_address: str` — default `"127.0.0.1"`

---

### 7.2 ShowManagerListenerService

Per-ShowManager-block service. Manages the OSC/UDP listener lifecycle.

**State:** running vs stopped. `start()` / `stop()` methods.

**On MA3 message received:** Routes to `MA3EventHandler` which processes the message and updates Editor layers via EditorCommands.

---

### 7.3 SyncSystemManager

Main orchestration point for all sync operations.

**Responsibilities:**
- Maintains list of `SyncLayerEntity` objects for a ShowManager block
- Provides `add_synced_ma3_track(block_id, ma3_coord, editor_layer_id)` — sets up bidirectional link
- Provides `sync_layer(layer_name)` — pushes EZ layer events to MA3
- Provides `sync_from_ma3(ma3_coord)` — pulls MA3 events into EZ layer
- Conflict detection and resolution
- `notify_editor_layer_deleted(layer_name)` — marks sync entity as disconnected

---

### 7.4 Bidirectional Sync

**MA3 → EchoZero:**
1. MA3 sends UDP change notification
2. `MA3CommunicationService` receives → publishes `MA3MessageReceived`
3. `MA3EventHandler` processes → extracts events with timing from MA3 data
4. Creates `EditorAddEventsCommand` with `source="ma3_sync"` OR `ApplyLayerSnapshotCommand`
5. Events land in the corresponding Editor layer

**EchoZero → MA3:**
1. User edits events in timeline
2. `sync_layer(layer_name)` dispatched
3. `SyncLayerCommand` reads events from EventDataItem
4. Sends commands back to MA3 (bidirectional future feature; current focus is MA3→EZ)

**Conflict resolution strategies:**
- `USE_MA3` — MA3 version always wins
- `USE_EZ` — EchoZero version always wins
- `MERGE` — attempt merge (additive)
- `SKIP` — leave as-is, no overwrite
- `PROMPT_USER` — UI dialog to decide per-conflict

**MA3 track coordinate formats:**
- `"tc1_tg1_tr1"` — full coordinate format
- `"TC1"` — short pool reference
- Parsing logic: if starts with `"tc"` followed by digits → extract number → `group_id = "tc_N"`, `group_name = "TC N"`
- If starts with digit: `head = coord.split(".")[0]` → if digit → same TC group extraction

**Synced layer identity in EventDataItem metadata:**
- `_synced_from_ma3: bool`
- `_show_manager_block_id: str`
- `_ma3_track_coord: str`
- `group_name: str`
- `group_id: str` (starts with `"tc_"` for MA3 pools)
- `source: "ma3"` or `"ma3_sync"`

**ApplyLayerSnapshot metadata preservation:**
- When MA3 sends a full snapshot (replacing all events), existing events are matched by `_ma3_idx`
- For matched events: **merge** existing metadata (user fields like `render_as_marker`) with new metadata (MA3 sync fields). Existing metadata wins on conflict (preserves user choices).
- Unmatched events: created fresh

---

### 7.5 ShowManager Settings

`ShowManagerSettingsManager` reads/writes ShowManager block configuration.

Key settings:
- `synced_layers: List[Dict]` — list of sync entity dicts
- `target_timecode: str | None` — TC coordinate for this ShowManager
- OSC listener config (port, address)

Settings are stored in `block.metadata` and persisted via `update_block_metadata`.

---

## 8. Project System

### 8.1 File Format (.ez)

Projects are saved as `.ez` files — **ZIP archives** containing:
- `project.json` — main project manifest
- Blocks, connections, data items, UI state, layer orders
- Setlists and setlist songs
- Action sets and action items

**JSON structure of `project.json` (inferred from ProjectService):**
```json
{
  "project": { "id", "name", "version", "created_at", "modified_at", "metadata" },
  "blocks": [ { "id", "name", "type", "metadata", "ports" }, ... ],
  "connections": [ { "id", "source_block_id", "source_output_name", "target_block_id", "target_input_name", ... }, ... ],
  "data_items": [ { "id", "block_id", "name", "type", "created_at", "file_path", "metadata" }, ... ],
  "ui_state": [ { "id", "type", "entity_id", "data" }, ... ],
  "layer_orders": [ { "block_id", "order_json" }, ... ],
  "block_local_state": [ { "block_id", "inputs_json" }, ... ],
  "setlists": [ { ... } ],
  "setlist_songs": [ { ... } ],
  "action_sets": [ { ... } ],
  "action_items": [ { ... } ]
}
```

**Source of truth:** JSON `.ez` files are the authoritative source. The SQLite database is a runtime session cache only.

---

### 8.2 Save Process

1. Serialize all project entities from repositories
2. Write JSON to temp file
3. ZIP temp file into `.ez` archive
4. Atomic rename to final path
5. Update `recent_projects` list
6. Emit `ProjectUpdated` event

**Save-as:** Same process but to a new path. Updates `project.save_directory`.

**Auto-save:** Not implemented (implied by TODO/FIXME review — no auto-save mention found).

---

### 8.3 Load Process

1. Open `.ez` ZIP file
2. Parse `project.json`
3. `database.clear_runtime_tables()` — wipe session state
4. Rehydrate all entities into repositories (which write to SQLite)
5. Set `current_project_id` on facade
6. Emit `ProjectCreated` event
7. UI panels receive event and refresh

**Audio files:** Not embedded in the `.ez` archive. Referenced by absolute path in `file_path`. If files moved, blocks show error status.

---

### 8.4 Recent Projects

Stored in `RecentProjectsStore` (separate file, not in SQLite). Maintains a list of recently opened `.ez` file paths.

---

### 8.5 Setlist Processing

1. User creates a Setlist and adds songs (by audio file path)
2. User configures `default_actions` (which block quick actions to run per song)
3. `process_setlist(setlist_id)`:
   - For each song (in `order_index` order):
     - Set `SetlistAudioInput` block's `audio_path` to the song's `audio_path`
     - Execute action items from `action_overrides` (or `default_actions` if no override)
     - Update song `status` → "processing" → "complete"/"error"
     - Store `processed_at` timestamp

**Song status flow:** `pending` → `processing` → `complete` or `error`

---

### 8.6 Snapshots (for Setlist Song State)

ProjectService maintains in-memory snapshots keyed by `song_id`. These capture the current project state at a given song and can be restored when switching songs in setlist mode.

**Snapshot cache:** Single-entry fast cache `(song_id, snapshot_dict)` for repeated access to the same snapshot. Dirty snapshots (modified but not yet written to disk) tracked in `_dirty_snapshot_ids`.

---

## 9. Persistence Layer

### 9.1 Database

**File:** File-based SQLite at a configurable path.  
**Mode:** WAL journal mode, foreign keys enabled, `check_same_thread=False` with explicit `RLock`.  
**Role:** Runtime session cache only. All tables cleared on init and project load. Source of truth = `.ez` JSON files.

**Schema version:** 6 (current).

---

### 9.2 Tables

| Table | Purpose |
|-------|---------|
| `projects` | Project entities |
| `blocks` | Block entities |
| `connections` | Connection entities |
| `data_items` | Audio and Event data item records |
| `block_local_state` | Per-block input port → data item ID mapping |
| `ui_state` | Arbitrary UI state blobs (block positions, editor layers) |
| `layer_orders` | Layer ordering per Editor block |
| `preferences` | App-wide key-value settings |
| `session_state` | Transient session key-value state |
| `setlists` | One per project |
| `setlist_songs` | Songs within a setlist |
| `action_sets` | Named sets of actions |
| `action_items` | Individual action entries |

**Indexes:**
- `blocks(project_id)`
- `connections(source_block_id)`, `connections(target_block_id)`
- `data_items(block_id)`
- `block_local_state(block_id)`
- `ui_state(type, entity_id)`, `ui_state(entity_id)`
- `layer_orders(block_id)`
- `setlist_songs(setlist_id)`, `setlist_songs(setlist_id, order_index)`
- `action_sets(project_id)`, `action_sets(name)`
- `action_items(action_set_id)`, `action_items(project_id)`, `action_items(action_set_id, order_index)`

---

### 9.3 Threading

**Thread safety:** All SQLite writes use `TransactionContext` which acquires an `RLock`. Only one writer at a time.  
**Reads:** On the shared connection (no locking; SQLite WAL allows concurrent reads with one writer).

**Transaction pattern:**
```python
with db.transaction() as conn:
    conn.execute(...)
# auto-commits on exit, auto-rollbacks on exception
```

---

### 9.4 Migrations

The database self-migrates on init:

1. **Connections table:** Removed `UNIQUE(target_block_id, target_input_name)` constraint to allow multiple inputs to the same port.
2. **Action items:** Added `action_type` column (default `"block"`). Made `block_id` nullable.
3. **Setlists:** Migrated to one-setlist-per-project schema with `UNIQUE(project_id)`.
4. **Setlist songs:** Added `action_overrides` and `error_message` columns.
5. **Connections:** Added `source_block_name` and `target_block_name` denormalized columns.
6. **Actions table:** Dropped deprecated `actions` table (replaced by `action_sets`/`action_items`).

---

### 9.5 Repository Pattern

Each entity type has a repository:
- `ProjectRepository`, `BlockRepository`, `ConnectionRepository`
- `DataItemRepository` — stores `AudioDataItem` and `EventDataItem` records
- `UIStateRepository`
- `PreferencesRepository` — key-value store for settings
- `SessionStateRepository` — transient session data
- `LayerOrderRepository`
- `SetlistRepository`, `SetlistSongRepository`
- `ActionSetRepository`, `ActionItemRepository`

**Data item repository note:** `DataItemRepository.get(id)` returns the appropriate subtype (`AudioDataItem` or `EventDataItem`) based on the `type` field. `EventDataItem` events are stored as JSON in `metadata["_layers_json"]` or similar embedded structure.

---

### 9.6 Preferences

**Namespace pattern:** Settings keys use `"{namespace}.{key}"` (e.g., `"timeline.layer_column_width"`).

**`BaseSettingsManager`:** Base class for all settings managers. Auto-saves on property change via `_save_setting(key)`. Emits `settings_changed(key)` signal. Emits `settings_loaded()` on load.

**Timeline settings namespace:** `"timeline"`  
**App settings namespace:** implied `"app"` or root-level keys.

**Clearing:** `preferences` and `session_state` are NOT cleared on project load (they persist across projects). Only `ui_state` is project-specific and cleared on load.

---

## 10. UI Architecture

### 10.1 Main Window

**`MainWindow(QMainWindow)`**

**Title:** `"EZ"` (short name)

**Minimum size:** 800×600px

**Dock system:** PyQt6Ads (`CDockManager`) — replaces Qt native docking entirely.
- VSCode-style: drag-to-dock, tab-group, float, auto-hide sidebars
- All panels are equal-citizen `CDockWidget` instances
- Config flags: opaque splitter resize, all tabs have close button, dock areas have close/undock buttons, middle-click closes tab, equal split on insertion, hidden single central widget title bar
- CDockManager's own stylesheet is cleared (overridden by app stylesheet)

**Undo stack:** `QUndoStack` with configurable limit (default 50). Connected to Edit menu Undo/Redo actions.

**App mode:** Production vs Developer mode (via `AppModeManager`).

---

### 10.2 Panel System

Panels are `CDockWidget` instances. The main panels:
- **Node Editor** — block graph editing
- **Execution Panel** — execution log, progress
- **Setlist Panel** — setlist management
- **Block Panels** — per-block panels (Editor, ShowManager, AudioPlayer, etc.)

Block panels are opened on demand (click block → open panel). Multiple block panels can be open simultaneously as tabs.

**WorkspaceManager:** Manages panel lifecycle, layout, and persistence of dock state across sessions.

---

### 10.3 Design System

**Single source of truth:** `ui/qt_gui/design_system.py`

**Color palette (`Colors` class):**
- `BG_DARK = (28, 28, 32)` — main background
- `BG_MEDIUM = (42, 42, 47)` — panels, widgets
- `BG_LIGHT = (56, 56, 62)` — highlights
- `BORDER = (75, 75, 80)` — borders
- `HOVER = (65, 65, 70)` — hover state
- `SELECTED = (85, 85, 90)` — selection
- `TEXT_PRIMARY = (240, 240, 245)`
- `TEXT_SECONDARY = (180, 180, 185)`
- `TEXT_DISABLED = (120, 120, 125)`
- `ACCENT_BLUE = (70, 130, 220)`, `ACCENT_GREEN = (80, 180, 120)`, `ACCENT_RED = (220, 80, 80)`, `ACCENT_YELLOW = (220, 180, 60)`, `ACCENT_ORANGE = (220, 135, 65)`, `ACCENT_PURPLE = (175, 115, 195)`
- Status: `STATUS_SUCCESS = ACCENT_GREEN`, `STATUS_WARNING = ACCENT_YELLOW`, `STATUS_ERROR = ACCENT_RED`, `STATUS_INFO = ACCENT_BLUE`, `STATUS_INACTIVE = TEXT_DISABLED`
- Danger: `DANGER_BG = (58, 32, 32)`, `DANGER_FG = (255, 107, 107)`

**Typography:**
- Default font: `"SF Pro Text, Segoe UI, -apple-system, system-ui"`, 13px
- Heading: `"SF Pro Display, Segoe UI"`, 16px, DemiBold
- Mono: `"SF Mono, Consolas, Monaco, monospace"`, 12px
- Font family and size are configurable via `ui_font_family` and `ui_font_size` settings

**Sizes:**
- Block: 150×100px (min height 70px), header 28px, corner radius 0px (sharp), body padding 6px
- Port: radius 4px, label offset 10px, vertical spacing 20px, zone height 22px
- Grid: 20px cell, major every 5 cells
- Connection line: 2px normal, 3px selected, 2.5px hover

**Spacing constants:** XS=4, SM=8, MD=16, LG=24, XL=32, XXL=48

**Effects:** Shadow blur 8px, offset (0,2), color `(0,0,0,60)`. Animation: fast=150ms, normal=250ms, slow=350ms.

---

### 10.4 Theming

**Theme system:**
- `ThemeRegistry` — stores named themes
- `Colors.apply_theme(name)` — applies theme to all `Colors.*` class attributes
- `on_theme_changed(callback)` — subscribe to theme changes (global signal)
- `ThemeAwareMixin` — mixin for widgets that auto-clear child stylesheets on theme change, then call `_apply_local_styles()`

**Live preview:** `Colors.apply_theme_from_dict(color_dict)` applies from a flat dict without requiring a registered theme. Used by settings dialog for instant preview. Does NOT emit `theme_changed` — caller manages that.

**Sharp corners:** Global toggle `set_sharp_corners(enabled)`. When True, all `border-radius` CSS values become `"0px"`. Use `border_radius(px)` helper in stylesheet generators.

**Stylesheet:** `get_stylesheet()` generates a complete global QSS covering all widget types (menus, toolbars, buttons, inputs, combos, checkboxes, tabs, PyQt6Ads docks, scrollbars, progress bars, views, etc.). Applied to `QApplication`.

**Palette:** `get_application_palette()` builds QPalette from current Colors. Used by QApplication, MainWindow, and splash screen.

**Font application:** `apply_ui_font(app)` sets QApplication default font from settings. Called when applying theme.

---

### 10.5 Settings / Preferences System

**`AppSettingsManager`:** Top-level settings manager for application-wide settings.

**Key app settings:**
- `theme_preset: str` — active theme name (default: `"default dark"`)
- `sharp_corners: bool` — global corner radius toggle
- `ui_font_family: str` — font family override (empty = system default)
- `ui_font_size: int` — font size in pixels (0 = use default 13px)
- `ma3_listen_enabled: bool`
- `ma3_listen_port: int`
- `ma3_listen_address: str`
- `max_undo_steps: int` — default 50

Settings persist via `PreferencesRepository` (SQLite `preferences` table). Keys use namespace prefix.

**Settings dialog:** Allows editing all settings. Theme changes apply live (with `apply_theme_from_dict()` + `MainWindow._apply_theme()`). On save, `theme_changed` is emitted after stylesheet update to prevent visual flash.

---

### 10.6 Splash Screen

Shown on startup with a random "welcome phrase" loaded from `assets/welcome_phrases.txt`. Falls back to `"Welcome to EchoZero."` if file missing.

---

### 10.7 RunBlockThread

`RunBlockThread(QThread)` — background thread for block execution.

Accepts: `facade`, `block_id`, `execution_strategy`

Emits Qt signals (connected to main thread):
- `progress_start_signal(message: str, current: int, total: int)`
- `progress_update_signal(message: str, current: int, total: int)`
- `progress_complete_signal(result_code: int)`
- `progress_error_signal(error_message: str)`
- `subprocess_progress_signal(message: str, percent: int)` — for sub-process progress (e.g., Demucs)

---

## 11. Known Issues / TODOs

Captured from code comments:

1. **Timeline: Multi-select drag** — native snapping mode (`native_snapping=False` by default) is a "POC-verified infrastructure" that exists but isn't the active path. Multi-select continues to use the proven flow.

2. **Block position storage** — block positions in node editor are stored in UI state. If UI state is cleared (project reload), positions may reset.

3. **Audio file references** — audio files are referenced by absolute path. If the user moves files, blocks show error status. No path remapping / search-on-load implemented.

4. **Auto-save** — not implemented. Users must manually save.

5. **Block Processor pattern completeness** — `validate_configuration()` method exists on all processors but is only called selectively, not as part of standard execution flow.

6. **Demucs cleanup** — `SimpleAudioPlayer.cleanup()` comment: "Don't call stop() - it can block indefinitely. Use pause() instead." Known fragility in QMediaPlayer cleanup.

7. **MA3 bidirectional (EZ → MA3)** — marked as future feature in README. Current implementation is primarily MA3 → EZ.

8. **Setlist snapshot dirty flushing** — snapshots modified in memory are tracked in `_dirty_snapshot_ids` but flush timing/mechanism needs explicit management.

9. **Layer override cleanup** — `EditorGetLayersCommand` prunes stale override entries from `editor_layers` UI state. If it encounters duplicate or invalid entries, it logs and cleans up. This is a defensive mechanism for data integrity.

10. **Connections: no type checking** — port type compatibility is described as "checked before creation" but explicit connection type validation code was not found in the examined sources. Rewrite should implement explicit type-gate validation.

11. **`EditorV2`** — referenced in type map and color map as a variant of Editor, but no `editor_v2_block.py` was found. Likely same processor as `Editor`.

12. **`NoteExtractorBasicPitch` / `NoteExtractorLibrosa`** — listed as separate types in visual guide but share functionality with `TranscribeNote` / `TranscribeLib`. Likely aliases or legacy names.

13. **Thread-safe UI updates** — all UI updates from background threads must go through Qt signals. The main window defines typed signals for this. Direct widget manipulation from worker threads is a bug risk.

14. **Database cleared on load** — this means all runtime state (including unsaved UI positions) is lost on project load. The JSON file is the canonical state; any UI-only state not serialized to the `.ez` file is transient.

15. **`EchoZeroBridge.lua` future enhancements** listed in MA3 integration docs: OSC protocol, additional object hooks (cues, executors, timecode), message filtering, automatic hook management, sequence/cue sync, timecode sync.

---

## Appendix A: Port Type Color Reference

| Port Type | Color (RGB) | Hex |
|-----------|------------|-----|
| AUDIO | (70, 170, 220) | `#46AADC` |
| EVENT | (230, 150, 70) | `#E69646` |
| MANIPULATOR | (255, 140, 0) | `#FF8C00` |
| GENERIC | (150, 150, 155) | `#96969B` |
| Input (generic) | (100, 200, 100) | `#64C864` |
| Output (generic) | (200, 100, 100) | `#C86464` |

---

## Appendix B: Block Type Registry Summary

| Type String | Processor Class | Category |
|-------------|----------------|----------|
| `LoadAudio` | `LoadAudioBlockProcessor` | Input |
| `SetlistAudioInput` | `SetlistAudioInputBlockProcessor` | Input |
| `DetectOnsets` | `DetectOnsetsBlockProcessor` | Analysis |
| `TranscribeNote` / `NoteExtractorBasicPitch` | Note transcription processor | Analysis |
| `TranscribeLib` / `NoteExtractorLibrosa` | Librosa note processor | Analysis |
| `Separator` | `SeparatorBlockProcessor` | Transform |
| `AudioFilter` | `AudioFilterBlockProcessor` | Transform |
| `ExportAudio` | `ExportAudioBlockProcessor` | Export |
| `ExportMA2` | `ExportMA2BlockProcessor` | Export |
| `Editor` / `EditorV2` | Editor processor | Editing |
| `AudioPlayer` | `AudioPlayerBlockProcessor` | Utility |
| `CommandSequencer` | `CommandSequencerBlockProcessor` | Utility |
| `ShowManager` | `ShowManagerBlockProcessor` | Integration |

---

## Appendix C: Snap Interval Seconds Mapping

For `snap_interval_mode` to seconds conversion (at standard FPS, e.g., 30fps):
- `"auto"` → computed from current zoom level and grid minor interval
- `"1f"` → `1/fps` seconds (e.g., 0.0333s at 30fps)
- `"2f"` → `2/fps` seconds
- `"5f"` → `5/fps` seconds
- `"10f"` → `10/fps` seconds
- `"1s"` → `1.0` second exactly

FPS is configured in the timebase/grid settings. Default assumed 30fps unless configured otherwise.

---

## Appendix D: MA3 Timecode Coordinate Parsing

Given a `ma3_coord` string, the system extracts a TC group:

```
coord_lower = coord.lower()
if coord_lower.startswith("tc"):
    # Extract digits immediately after "tc"
    num = digits immediately following "tc" (stop at non-digit)
    if num:
        group_id = f"tc_{num}"    # e.g., "tc_1"
        group_name = f"TC {num}"  # e.g., "TC 1"
elif coord_lower[0].isdigit():
    head = coord.split(".")[0]
    if head.isdigit():
        group_id = f"tc_{head}"
        group_name = f"TC {head}"
```

Layer identities using `tc_N` group_id are treated as MA3-synced and persist across session reloads.

---

*End of EchoZero Behavioral Specification*
