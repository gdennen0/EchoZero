# EchoZero Core API Contract
**Version:** 0.1.0-draft  
**Date:** 2026-03-17  
**Status:** Draft — needs review  

---

## Philosophy

This document defines the boundary between Core (engine + blocks) and any client (desktop UI, CLI, cloud, mobile, third-party). 

**Rules:**
1. Core NEVER knows about any specific client. It serves an API.
2. Clients NEVER touch block internals. They send commands, receive results.
3. All state lives in Core. Clients are stateless views.
4. Real-time updates flow via WebSocket. Commands flow via request/response.
5. Every operation that can fail returns a structured error.

---

## Transport

- **Request/Response:** JSON over HTTP (localhost for desktop, network for cloud)
- **Real-Time:** WebSocket at `/ws` — push notifications for progress, state changes, sync events
- **Serialization:** JSON everywhere. Binary payloads (audio) as file references, not inline.

For v1 desktop: Core runs as a local process, API on `localhost:PORT`. UI connects via HTTP + WebSocket. Same machine, zero network latency.

---

## Data Types (Shared Schema)

These types are the shared language. Both Core and clients understand them.

### Project
```json
{
  "id": "uuid",
  "name": "string",
  "created_at": "iso8601",
  "updated_at": "iso8601",
  "min_app_version": "semver",
  "settings": {
    "sample_rate": 44100,
    "bpm": null | number,
    "bpm_confidence": null | number,
    "timecode_fps": null | 24 | 25 | 29.97 | 30
  }
}
```

### Song
```json
{
  "id": "uuid",
  "project_id": "uuid",
  "title": "string",
  "artist": "string | null",
  "order": "integer",
  "active_version_id": "uuid",
  "versions": ["SongVersion"]
}
```

### SongVersion
```json
{
  "id": "uuid",
  "song_id": "uuid",
  "label": "string",
  "audio_file": "string (project-relative path)",
  "duration_seconds": "number",
  "original_sample_rate": "integer",
  "audio_hash": "string (sha256)",
  "created_at": "iso8601"
}
```

### Branch
```json
{
  "id": "uuid",
  "song_version_id": "uuid",
  "name": "string",
  "is_main": "boolean",
  "parent_branch_id": "uuid | null",
  "created_at": "iso8601",
  "source": "analysis | transfer | import | manual",
  "pipeline_id": "string | null",
  "pipeline_params": "object | null"
}
```

### Layer
```json
{
  "id": "uuid",
  "branch_id": "uuid",
  "name": "string",
  "color": "string (hex)",
  "order": "integer",
  "parent_layer_id": "uuid | null",
  "visible": "boolean",
  "locked": "boolean",
  "source_block": "string | null",
  "source_model_category": "string | null"
}
```

### Event
```json
{
  "id": "uuid",
  "layer_id": "uuid",
  "time": "number (seconds, sample-accurate)",
  "duration": "number (seconds, 0 = marker event)",
  "classifications": {
    "model_category": {
      "label": "string",
      "confidence": "number (0-1)",
      "model_id": "string",
      "model_version": "string"
    }
  },
  "origin": "ml | user",
  "artifact_candidate": "boolean",
  "uncertain": "boolean",
  "metadata": "object (extensible)"
}
```

### Section
```json
{
  "id": "uuid",
  "branch_id": "uuid",
  "name": "string",
  "type": "intro | verse | chorus | bridge | drop | outro | transition | excluded | custom",
  "start_time": "number",
  "end_time": "number",
  "confidence": "number | null"
}
```

### Pipeline (Definition)
```json
{
  "id": "string (slug)",
  "name": "string (display)",
  "description": "string",
  "version": "semver",
  "category": "analysis | transform | export | import | utility",
  "input_types": ["Audio", "Events", "Layers", "TimeRange"],
  "output_types": ["Events", "Layers", "Sections", "File"],
  "blocks": [
    {
      "id": "string (instance id within pipeline)",
      "block_type": "string (registered block type)",
      "settings": "object (block-specific)",
      "position": {"x": "number", "y": "number"}
    }
  ],
  "connections": [
    {
      "from_block": "string",
      "from_port": "string",
      "to_block": "string",
      "to_port": "string"
    }
  ],
  "params": [
    {
      "name": "string (user-visible)",
      "key": "string (programmatic)",
      "type": "string | number | boolean | enum",
      "default": "any",
      "constraints": "object | null",
      "maps_to": {
        "block_id": "string",
        "setting_key": "string"
      },
      "description": "string"
    }
  ],
  "sub_pipelines": ["pipeline_id references (for composition)"]
}
```

### BlockSchema (Self-Description)
```json
{
  "type": "string (unique block type identifier)",
  "name": "string (display)",
  "description": "string",
  "category": "input | analysis | classification | transform | export | import | validation | utility",
  "inputs": [
    {"name": "string", "type": "Audio | Events | Sections | Any", "required": "boolean"}
  ],
  "outputs": [
    {"name": "string", "type": "Audio | Events | Sections | File"}
  ],
  "settings": [
    {
      "key": "string",
      "name": "string (display)",
      "type": "string | number | boolean | enum",
      "default": "any",
      "constraints": {"min": null, "max": null, "options": null},
      "description": "string"
    }
  ]
}
```

### Execution
```json
{
  "id": "uuid",
  "pipeline_id": "string",
  "song_version_id": "uuid",
  "branch_id": "uuid (created for this execution)",
  "params": "object (user-supplied parameter values)",
  "status": "queued | running | completed | failed | cancelled",
  "progress": "number (0-1)",
  "current_block": "string | null",
  "started_at": "iso8601 | null",
  "completed_at": "iso8601 | null",
  "error": "string | null"
}
```

### SyncSnapshot (MA3)
```json
{
  "id": "uuid",
  "sync_layer_id": "uuid",
  "direction": "pushed | pulled",
  "events_hash": "string",
  "event_count": "integer",
  "timestamp": "iso8601",
  "events": ["SyncEvent (fingerprint + positional data)"]
}
```

---

## API Endpoints

### Projects

```
POST   /api/projects
  → Create project
  ← Project

GET    /api/projects
  → List projects  
  ← [Project]

GET    /api/projects/:id
  → Get project with full state
  ← Project + Songs + active branch summary

PUT    /api/projects/:id
  → Update project settings
  ← Project

DELETE /api/projects/:id
  → Delete project (moves to trash)
  ← 204

POST   /api/projects/:id/save
  → Save to .ez archive (atomic write, D183)
  ← {path: "string"}

POST   /api/projects/open
  Body: {path: "string"}
  → Open .ez archive
  ← Project (with version check per D184)
```

### Songs & Versions

```
POST   /api/projects/:id/songs
  Body: {title, artist?, audio_file_path}
  → Add song to setlist (resamples if needed per D201)
  ← Song

PUT    /api/songs/:id
  → Update song metadata, reorder
  ← Song

DELETE /api/songs/:id
  → Remove song from setlist
  ← 204

POST   /api/songs/:id/versions
  Body: {label, audio_file_path}
  → Add new version of a song
  ← SongVersion

PUT    /api/songs/:id/active-version
  Body: {version_id}
  → Switch active version (clears ShowManager sync per D242)
  ← Song

POST   /api/songs/:id/align-versions
  Body: {source_version_id, target_version_id}
  → Run DTW alignment (D239), cache WarpMap
  ← {warp_map_id, confidence_summary, preview_data}

POST   /api/songs/:id/transfer-analysis
  Body: {source_version_id, target_version_id, warp_map_id, options}
  → Transfer events to new version branch (D240)
  ← Branch (transfer branch with uncertain flags)
```

### Branches

```
GET    /api/song-versions/:id/branches
  → List branches for a song version
  ← [Branch]

POST   /api/song-versions/:id/branches
  Body: {name, parent_branch_id?}
  → Create branch (manual)
  ← Branch

GET    /api/branches/:id
  → Get branch with layer summary
  ← Branch + [Layer summary]

PUT    /api/branches/:id
  → Rename, update metadata
  ← Branch

DELETE /api/branches/:id
  → Delete branch (not main)
  ← 204

POST   /api/branches/:id/merge
  Body: {target_branch_id, strategy: "accept-source" | "accept-target" | "manual"}
  → Merge branch into target (D169, D203)
  ← {conflicts: [Conflict], merged_count: number} or Branch if no conflicts

POST   /api/branches/:id/switch
  → Switch active branch (pushes to global undo stack per D191)
  ← Branch
```

### Layers & Events

```
GET    /api/branches/:id/layers
  → List layers on branch
  ← [Layer]

POST   /api/branches/:id/layers
  Body: {name, color?, parent_layer_id?}
  → Create layer
  ← Layer

PUT    /api/layers/:id
  → Update layer (name, color, order, visibility, lock)
  ← Layer

DELETE /api/layers/:id
  → Delete layer (undoable per D210)
  ← 204

GET    /api/layers/:id/events
  Query: ?time_start=&time_end=&min_confidence=&classification=
  → Get events with optional filtering (D170)
  ← [Event]

POST   /api/layers/:id/events
  Body: Event (or array for batch)
  → Create event(s) — user-created events become missed-onset training signals (D231)
  ← [Event]

PUT    /api/events/:id
  → Update event (move, resize, reclassify)
  ← Event

DELETE /api/events/:id (or batch: POST /api/events/delete with body)
  → Delete event(s) (D161 confirmation for bulk)
  ← 204

POST   /api/events/:id/correct
  Body: {field: "classification", model_category: "string", correct_label: "string"}
  → Submit user correction (feeds flywheel D130)
  ← Event (updated)
```

### Sections

```
GET    /api/branches/:id/sections
  ← [Section]

POST   /api/branches/:id/sections
  Body: Section
  ← Section

PUT    /api/sections/:id
  ← Section

DELETE /api/sections/:id
  ← 204
```

### Pipelines

```
GET    /api/pipelines
  Query: ?category=&input_type=&context=
  → List available pipelines, optionally filtered by context (D context-driven discovery)
  ← [Pipeline summary]

GET    /api/pipelines/:id
  → Full pipeline definition with block graph + parameters
  ← Pipeline

POST   /api/pipelines
  Body: Pipeline definition
  → Create/register new pipeline (workbench save)
  ← Pipeline

PUT    /api/pipelines/:id
  Body: Pipeline definition
  → Update pipeline
  ← Pipeline

DELETE /api/pipelines/:id
  ← 204

GET    /api/pipelines/:id/params
  → Get just the promoted parameters (for quick UI rendering)
  ← [PipelineParam]
```

### Execution

```
POST   /api/pipelines/:id/execute
  Body: {
    song_version_id: "uuid",
    params: {key: value},
    time_range?: {start, end},
    branch_name?: "string"
  }
  → Start pipeline execution (creates branch, returns immediately)
  ← Execution (status: queued)

GET    /api/executions/:id
  → Check execution status
  ← Execution

DELETE /api/executions/:id
  → Cancel execution
  ← Execution (status: cancelled)

GET    /api/executions
  Query: ?status=running
  → List active executions
  ← [Execution]
```

### Blocks (Registry)

```
GET    /api/blocks
  → List all registered block types
  ← [BlockSchema summary]

GET    /api/blocks/:type
  → Full block schema (I/O, settings, description)
  ← BlockSchema
```

### Models (ML)

```
GET    /api/models
  → List available models (bundled + downloaded + custom)
  ← [ModelInfo]

GET    /api/models/registry
  → Check cloud registry for updates (D123)
  ← {updates_available: [ModelUpdate]}

POST   /api/models/download
  Body: {model_id}
  → Download model from registry
  ← 202 (progress via WebSocket)

POST   /api/models/verify
  Body: {path}
  → Verify custom model file (D226)
  ← {valid: boolean, warnings: [string]}
```

### ShowManager (MA3 Sync)

```
GET    /api/showmanager/status
  → Connection status, sync layer states
  ← {connected, console_ip, sync_layers: [SyncLayerStatus]}

POST   /api/showmanager/connect
  Body: {ip, port}
  ← {status: "connected" | "failed", error?}

POST   /api/showmanager/disconnect
  ← 204

GET    /api/showmanager/sync-layers
  ← [SyncLayer with status, policy, mapped layer]

POST   /api/showmanager/sync-layers
  Body: {layer_id, ma3_track_coordinate, policy}
  → Create sync layer mapping
  ← SyncLayer

PUT    /api/showmanager/sync-layers/:id
  → Update policy, mapping
  ← SyncLayer

POST   /api/showmanager/sync-layers/:id/push
  → Push EZ state to MA3
  ← {pushed_count, snapshot_id}

POST   /api/showmanager/sync-layers/:id/pull
  → Pull MA3 state (creates sync branch if diverged per D238)
  ← {status: "synced" | "diverged", deltas?: [Delta], branch_id?: "uuid"}

GET    /api/showmanager/activity-log
  Query: ?limit=&type=
  ← [ActivityLogEntry]
```

### Undo/Redo

```
POST   /api/undo
  → Undo last action (respects two-stack model D191)
  ← {action_undone: "string description"}

POST   /api/redo
  ← {action_redone: "string description"}

GET    /api/undo/stack
  → Current undo/redo state (for UI display)
  ← {undo: [action descriptions], redo: [action descriptions], current_branch: "uuid"}
```

### Playback

```
POST   /api/playback/play
  Body: {from_time?: number}
  ← 204

POST   /api/playback/pause
  ← 204

POST   /api/playback/stop
  ← 204 (resets to start)

POST   /api/playback/seek
  Body: {time: number}
  ← 204

GET    /api/playback/state
  ← {playing: boolean, position: number, sample_rate: number}

PUT    /api/playback/tracks
  Body: {muted: [track_ids], soloed: [track_ids]}
  ← 204

PUT    /api/playback/event-layers
  Body: {muted: [layer_ids], soloed: [layer_ids]}
  → Mute/solo event playback per D182
  ← 204
```

### System

```
GET    /api/system/status
  ← {version, uptime, platform, gpu_available, models_loaded, disk_space}

GET    /api/system/requirements-check
  ← {meets_minimum: boolean, warnings: [string], specs: {...}}

GET    /api/system/settings
  ← Global app settings

PUT    /api/system/settings
  ← Updated settings
```

---

## WebSocket Protocol (`/ws`)

Real-time push from Core to clients. JSON messages with `type` field.

### Progress
```json
{"type": "execution.progress", "execution_id": "uuid", "progress": 0.45, "current_block": "DetectOnsets", "message": "Detecting onsets (pass 2/3)..."}
{"type": "execution.completed", "execution_id": "uuid", "branch_id": "uuid", "duration_ms": 12340}
{"type": "execution.failed", "execution_id": "uuid", "error": "Out of disk space"}
```

### State Changes
```json
{"type": "branch.created", "branch": Branch}
{"type": "branch.switched", "branch_id": "uuid"}
{"type": "events.changed", "layer_id": "uuid", "change": "created | updated | deleted", "event_ids": ["uuid"]}
{"type": "layer.created", "layer": Layer}
{"type": "section.created", "section": Section}
```

### Playback
```json
{"type": "playback.position", "time": 42.567}
{"type": "playback.state", "playing": true}
```

### ShowManager
```json
{"type": "sync.connected", "console_ip": "192.168.1.50"}
{"type": "sync.disconnected", "reason": "timeout"}
{"type": "sync.divergence", "sync_layer_id": "uuid", "delta_count": 3}
{"type": "sync.drops", "count": 12, "window_seconds": 1.0}
```

### Model Updates
```json
{"type": "models.update_available", "model_id": "string", "current": "3.0.0", "available": "3.1.0"}
{"type": "models.download.progress", "model_id": "string", "progress": 0.7}
```

---

## Error Format

All errors follow:
```json
{
  "error": {
    "code": "string (machine-readable)",
    "message": "string (human-readable)",
    "details": "object | null"
  }
}
```

Common codes:
- `project.not_found`
- `project.version_too_new` (D184)
- `pipeline.invalid_context` (wrong input types)
- `pipeline.cycle_detected` (D229)
- `execution.already_running`
- `execution.disk_space_insufficient` (D207)
- `audio.resample_failed`
- `audio.duration_exceeded` (D178, >10 min)
- `audio.invalid_format` (D160)
- `showmanager.connection_failed`
- `showmanager.divergence_detected`
- `model.verification_failed` (D226)
- `model.download_failed`
- `branch.is_main` (can't delete main)
- `undo.nothing_to_undo`

---

## Design Notes

### Why HTTP + WebSocket (not gRPC, not raw TCP)
- HTTP is universally understood — every language has a client
- WebSocket gives real-time push without polling
- JSON is debuggable (curl, browser dev tools, Postman)
- For v1 desktop: localhost, negligible overhead
- For future cloud: HTTPS + WSS, standard deployment
- If performance becomes an issue: add binary endpoints later for bulk data (waveform peaks, large event sets)

### Why REST-ish (not GraphQL)
- Simpler to implement and debug
- Resources map cleanly to our domain model
- GraphQL's flexibility isn't needed — we control both client and server
- Batch operations handled via dedicated endpoints (not N+1 queries)

### Audio Data
- Audio files are NOT sent through the API as payloads
- API references audio by project-relative path
- Core manages audio files on disk (copy into project dir on import)
- Waveform peaks served via dedicated endpoint when needed:
  `GET /api/song-versions/:id/waveform?resolution=overview|detail&start=0&end=300`

### Pagination
- List endpoints support `?limit=&offset=` 
- Default limit: 100
- Events endpoint supports time-range filtering (more useful than offset pagination)

---

## Versioning

API version in URL: `/api/v1/...`
- Breaking changes = new version
- Additive changes (new endpoints, new fields) = same version
- Deprecated fields marked in schema, removed in next major version

---

*This is a living document. Update as decisions are made.*
