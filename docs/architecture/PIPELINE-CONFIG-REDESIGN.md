# Pipeline Config Redesign

## Problem

Settings are reconstructed every time from `SongPipelineConfig.bindings` + template code.
The user can't interact with settings as persistent project state. The template rebuilds
the entire pipeline from scratch on every run.

## New Model

The pipeline configuration is a **first-class persistent entity**. When a user adds a
pipeline to a song, the template creates the initial config. After that, the config
lives in the DB and the user owns it.

## Schema

### `pipeline_configs` table (replaces `song_pipeline_configs`)

```sql
CREATE TABLE IF NOT EXISTS pipeline_configs (
    id TEXT PRIMARY KEY,
    song_version_id TEXT NOT NULL REFERENCES song_versions(id) ON DELETE CASCADE,
    template_id TEXT NOT NULL,        -- which template created this
    name TEXT NOT NULL,               -- user-visible name ("Onset Detection")
    graph_json TEXT NOT NULL,         -- full serialized graph (blocks + connections)
    outputs_json TEXT NOT NULL,       -- pipeline output declarations
    knob_values_json TEXT NOT NULL,   -- current knob values {param_name: value}
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### What each column stores

- **template_id**: Which template factory created this. Used for:
  - Schema migration when templates evolve (new params, renamed blocks)
  - "Reset to defaults" button
  - Grouping configs by type in the UI

- **graph_json**: The full graph — blocks, connections, settings. This IS the pipeline.
  Serialized via `serialize_pipeline()` / `deserialize_pipeline()`.

- **outputs_json**: The named output declarations. Which block ports map to layers/takes.

- **knob_values_json**: The user's current knob values. Separate from graph_json because:
  - Knobs are the UI-facing "settings panel" — simple key-value pairs
  - The graph has the fully resolved BlockSettings (knob values baked in)
  - When user changes a knob, we update knob_values_json AND rebuild the affected
    BlockSettings in graph_json
  - This lets us show knob widgets without parsing the full graph

## Lifecycle

### 1. First time: Template creates config

```python
# User adds "Onset Detection" to a song
template = registry.get("onset_detection")
pipeline = template.build_pipeline(defaults)  # factory creates initial pipeline

config = PipelineConfig.from_pipeline(
    pipeline=pipeline,
    template_id="onset_detection",
    song_version_id=sv.id,
    knob_values=template.default_knob_values(),
)
session.pipeline_configs.create(config)
```

### 2. User tweaks settings

```python
# User moves threshold slider to 0.5
config = session.pipeline_configs.get(config_id)
config = config.with_knob_value("threshold", 0.5)  # returns new frozen instance
session.pipeline_configs.update(config)
# → saved immediately, dirty tracking triggers autosave
```

`with_knob_value()` updates BOTH `knob_values_json` and rebuilds the affected
`BlockSettings` in `graph_json`. The graph is always in sync with the knobs.

### 3. User hits analyze

```python
# Orchestrator reads the persisted config
config = session.pipeline_configs.get(config_id)
pipeline = config.to_pipeline()  # deserialize graph_json + outputs_json
result = orchestrator.execute(session, pipeline, song_version_id)
```

No template rebuild. No binding merging. The pipeline is already fully configured.

### 4. Template upgrade migration

When a new version ships with an updated template (new parameter, new block):

```python
def migrate_v2(conn):
    """Add 'min_gap' parameter to all onset_detection configs."""
    configs = conn.execute(
        "SELECT id, knob_values_json, graph_json FROM pipeline_configs "
        "WHERE template_id = 'onset_detection'"
    ).fetchall()
    for config in configs:
        knobs = json.loads(config['knob_values_json'])
        if 'min_gap' not in knobs:
            knobs['min_gap'] = 0.05  # new default
            # Rebuild graph with new default
            ...
            conn.execute(
                "UPDATE pipeline_configs SET knob_values_json = ?, graph_json = ? "
                "WHERE id = ?",
                (json.dumps(knobs), new_graph_json, config['id'])
            )
```

## Custom Pipelines (Stage Zero Editor)

Same table, same entity. The only difference:

- **Built-in**: `template_id = "onset_detection"` — can be reset to defaults, migrated
- **Custom**: `template_id = "custom"` — user built it in the editor, no template to reset to

The graph_json IS the source of truth for custom pipelines. There is no builder function.

## What gets killed

- `song_pipeline_configs` table → replaced by `pipeline_configs`
- `SongPipelineConfig` entity → replaced by `PipelineConfig`
- `PipelineConfigRepository` → replaced by new repo
- `Orchestrator.analyze()` no longer takes `bindings` — takes `config_id`
- Template `build()` / `build_pipeline()` only used for initial creation, not every run

## Entity

```python
@dataclass(frozen=True)
class PipelineConfig:
    id: str
    song_version_id: str
    template_id: str
    name: str
    graph_json: str          # serialized graph
    outputs_json: str        # serialized output declarations
    knob_values: dict[str, Any]  # current knob values
    created_at: datetime
    updated_at: datetime

    def to_pipeline(self) -> Pipeline:
        """Deserialize into a runnable Pipeline."""
        ...

    def with_knob_value(self, key: str, value: Any) -> PipelineConfig:
        """Return new config with updated knob + rebuilt BlockSettings."""
        ...

    @classmethod
    def from_pipeline(cls, pipeline, template_id, song_version_id, knob_values):
        """Create from a freshly-built pipeline (factory method)."""
        ...
```
