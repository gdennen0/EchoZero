# EchoZero 2 — Entity Model

Describes the persistence-layer entities, their relationships, field-level invariants,
and SQLite table mappings.

---

## Entity Relationship Diagram

```
ProjectRecord
│  id, name, settings, created_at, updated_at
│  (graph_json stored inline — the live block graph)
│
└── SongRecord[]                          projects → songs (1:N)
      │  id, project_id, title, artist,
      │  order, active_version_id
      │
      └── SongVersionRecord[]             songs → song_versions (1:N)
            │  id, song_id, label,
            │  audio_file, duration_seconds,
            │  original_sample_rate, audio_hash,
            │  created_at
            │
            ├── LayerRecord[]             song_versions → layers (1:N)
            │     │  id, song_version_id, name,
            │     │  layer_type, color, order,
            │     │  visible, locked,
            │     │  parent_layer_id (self-ref, optional),
            │     │  source_pipeline, created_at
            │     │
            │     └── Take[]              layers → takes (1:N)
            │           id, layer_id, label,
            │           origin, is_main, is_archived,
            │           source_json, data_json,
            │           created_at, notes
            │
            └── PipelineConfigRecord[]    song_versions → pipeline_configs (1:N)
                  id, song_version_id,
                  template_id, name,
                  graph_json, outputs_json,
                  knob_values, block_overrides,
                  created_at, updated_at
```

---

## ProjectRecord

**Table:** `projects`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID hex |
| `name` | TEXT | NOT NULL | Human-readable project name |
| `sample_rate` | INTEGER | DEFAULT 44100 | Master sample rate |
| `bpm` | REAL | nullable | Project tempo |
| `bpm_confidence` | REAL | nullable | BPM detection confidence (0–1) |
| `timecode_fps` | REAL | nullable | Timecode frame rate |
| `graph_json` | TEXT | nullable | Serialized block graph (live editor state) |
| `created_at` | TEXT | NOT NULL | ISO 8601 UTC |
| `updated_at` | TEXT | NOT NULL | ISO 8601 UTC |

**Invariants:**
- Exactly one `ProjectRecord` per `project.db`
- `graph_json` is updated on every successful `dispatch()` call
- Settings are stored flat (no JSON blob) for queryability

**Python class:** `echozero.persistence.entities.ProjectRecord` + `ProjectSettingsRecord`

---

## SongRecord

**Table:** `songs`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID hex |
| `project_id` | TEXT | FK → projects | Owning project |
| `title` | TEXT | NOT NULL | Song title |
| `artist` | TEXT | DEFAULT '' | Artist name |
| `order` | INTEGER | NOT NULL | Position in setlist (0-indexed) |
| `active_version_id` | TEXT | nullable, FK → song_versions | Which version is active |

**Invariants:**
- `active_version_id` must reference a `SongVersionRecord` with `song_id = this.id`
- `active_version_id` is `NULL` only during the brief window between `SongRecord` creation and first version import
- `order` values are dense integers; gaps are allowed but not preferred

**Indexes:** `idx_songs_project ON songs(project_id)`

**Python class:** `echozero.persistence.entities.SongRecord`

---

## SongVersionRecord

**Table:** `song_versions`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID hex |
| `song_id` | TEXT | FK → songs | Owning song |
| `label` | TEXT | NOT NULL | Human label (e.g., "Original", "Festival Edit") |
| `audio_file` | TEXT | NOT NULL | Relative path within working directory |
| `duration_seconds` | REAL | NOT NULL | Audio duration |
| `original_sample_rate` | INTEGER | NOT NULL | Sample rate at import |
| `audio_hash` | TEXT | NOT NULL | SHA-256 of audio file (content integrity) |
| `created_at` | TEXT | NOT NULL | ISO 8601 UTC |

**Invariants:**
- `audio_file` is always a relative path within the working directory (`audio/<hash>.<ext>`)
- `audio_hash` prevents duplicate imports (checked at import time)
- `duration_seconds` is scanned from the real file at import, never user-supplied

**Indexes:** `idx_versions_song ON song_versions(song_id)`

**Python class:** `echozero.persistence.entities.SongVersionRecord`

---

## LayerRecord

**Table:** `layers`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID hex |
| `song_version_id` | TEXT | FK → song_versions | Owning version |
| `name` | TEXT | NOT NULL | Layer name (matches pipeline output name) |
| `layer_type` | TEXT | CHECK IN ('analysis','structure','manual') | How the layer was created |
| `color` | TEXT | nullable | UI color string |
| `order` | INTEGER | NOT NULL | Z-order in the timeline |
| `visible` | INTEGER | NOT NULL DEFAULT 1 | 1 = visible in UI |
| `locked` | INTEGER | NOT NULL DEFAULT 0 | 1 = locked (no manual edits) |
| `parent_layer_id` | TEXT | nullable, self-FK | For hierarchical layers |
| `source_pipeline` | TEXT | nullable | JSON: `{"pipeline_id": …, "block_id": …}` |
| `created_at` | TEXT | NOT NULL | ISO 8601 UTC |

**Invariants:**
- `layer_type = 'analysis'` → created by a pipeline run
- `layer_type = 'manual'` → created by user interaction
- `layer_type = 'structure'` → structure markers (verse, chorus, etc.)
- `name` is unique per `song_version_id` (enforced by Orchestrator at creation time)
- A `LayerRecord` must have at least one `Take` with `is_main = 1`

**Note:** `LayerRecord` is the **persistence DTO**, distinct from `echozero.domain.types.Layer`
which is the engine-level domain type.

**Indexes:** `idx_layers_version ON layers(song_version_id)`

**Python class:** `echozero.persistence.entities.LayerRecord`

---

## Take

**Table:** `takes`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID (from `uuid.uuid4()`) |
| `layer_id` | TEXT | FK → layers | Owning layer |
| `label` | TEXT | NOT NULL | Human-readable label |
| `origin` | TEXT | CHECK IN ('pipeline','user','merge','sync') | How this take was created |
| `is_main` | INTEGER | NOT NULL DEFAULT 0 | 1 = the active take for this layer |
| `is_archived` | INTEGER | NOT NULL DEFAULT 0 | 1 = archived (hidden from UI) |
| `source_json` | TEXT | nullable | Serialized `TakeSource` provenance |
| `data_json` | TEXT | nullable | Serialized `EventData` or `AudioData` |
| `created_at` | TEXT | NOT NULL | ISO 8601 UTC |
| `notes` | TEXT | DEFAULT '' | User notes |

**Invariants (enforced by `TakeLayer`):**
1. Exactly one `Take` per `LayerRecord` has `is_main = 1`
2. `data` is never mutated after creation. Edit = new Take.
3. Take IDs are globally unique and never reused.
4. Sync only reads/writes the Take where `is_main = 1`.
5. Archived takes cannot be promoted to main (must unarchive first).

**Take retention:** The Orchestrator enforces a limit (default 20 non-main takes per layer).
Oldest non-main takes are archived automatically on re-analysis.

**Indexes:** `idx_takes_layer ON takes(layer_id)`

**Python class:** `echozero.takes.Take`

---

## PipelineConfigRecord

**Table:** `pipeline_configs`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | TEXT | PK | UUID hex |
| `song_version_id` | TEXT | FK → song_versions | Owning version |
| `template_id` | TEXT | NOT NULL | Which template factory created this |
| `name` | TEXT | NOT NULL | Human-readable name |
| `graph_json` | TEXT | NOT NULL | Full serialized block graph |
| `outputs_json` | TEXT | NOT NULL DEFAULT '[]' | Pipeline output declarations |
| `knob_values_json` | TEXT | NOT NULL DEFAULT '{}' | User's current knob values |
| `block_overrides_json` | TEXT | NOT NULL DEFAULT '{}' | Per-block setting overrides |
| `created_at` | TEXT | NOT NULL | ISO 8601 UTC |
| `updated_at` | TEXT | NOT NULL | ISO 8601 UTC |

**Invariants:**
- `graph_json` and `knob_values` are always in sync — updating a knob via `with_knob_value()` also updates the corresponding `BlockSettings` in the graph
- `block_overrides` tracks per-block overrides: `{block_id: [setting_key, …]}`. Global knob updates skip overridden settings.
- Created from a `PipelineTemplate` via `Orchestrator.create_config()` or `ProjectStorage.import_song()`
- After creation, the user owns the config — template changes don't affect existing configs

**Mutation methods** (all return new `PipelineConfigRecord` — immutable dataclass):

| Method | Description |
|--------|-------------|
| `with_knob_value(key, value)` | Update one knob + corresponding block settings |
| `with_knob_values(updates)` | Batch knob update |
| `with_block_setting(block_id, key, value)` | Per-block override (bypasses knob) |
| `with_block_settings(block_id, updates)` | Batch per-block override |
| `clear_block_override(block_id, key)` | Re-link setting back to global knob |
| `to_pipeline()` | Deserialize into runnable `Pipeline` object |

**Indexes:**
- `idx_configs_version ON pipeline_configs(song_version_id)`
- `idx_configs_template ON pipeline_configs(template_id)`

**Python class:** `echozero.persistence.entities.PipelineConfigRecord`

---

## Schema Version History

| Version | Description |
|---------|-------------|
| 1 | Initial schema — `song_pipeline_configs` with flat bindings blob |
| 2 | Replaced `song_pipeline_configs` with `pipeline_configs` (full graph + outputs + knob_values + block_overrides) |

Current: **version 2**

See `echozero/persistence/schema.py` for DDL and migration logic.

---

## Graph Storage

The live editor graph (block nodes + connections) is stored inline in the `projects` table as `graph_json`.
It is **not** part of any `SongRecord` or `PipelineConfigRecord` — it is the application-level graph
the user is editing in real time.

`PipelineConfigRecord.graph_json` is a **separate** serialized graph — the analysis pipeline graph
for a specific song version. These are independent.

---

## Audio File Storage

Imported audio is copied into the working directory under `audio/`:

```
~/.echozero/working/<project-hash>/
  project.db
  project.lock
  audio/
    <sha256[:16]>.<ext>    # imported audio file
    <sha256[:16]>_stem_drums.wav   # stem separation outputs
    …
```

`SongVersionRecord.audio_file` stores the relative path (`audio/<name>.<ext>`).
The working directory root is prepended at runtime to get the full path.
