# EchoZero 2 — Subagent Audit Reports (2026-03-26)

## Code Quality Audit

### PASS (3/10)
1. ✅ All 6 repos extend BaseRepository[T] with _from_row
2. ✅ Zero conn.commit() in repos
5. ✅ Import hygiene — clean dependency direction, no circular imports

### PARTIAL FAIL (5/10)
3. ⚠️ `schema.py:95` — `get_schema_version` uses `row[0]` instead of `row['value']`
4. ⚠️ `take.py:28` — `data_json` column is nullable but no null guard in `_from_row` (json.loads(None) would TypeError)
6. ⚠️ `registry.py` — 6 public methods missing docstrings (build, register, get, list, ids, get_registry)
9. ⚠️ `is_archived` column orphaned — in schema, written as 0, but Take dataclass has no is_archived field. `graph_json` managed by session not ProjectRepository (intentional but undocumented)
10. ⚠️ Pipeline validate_bindings gaps: no unknown-key detection, no int→float coercion, bindings never actually consumed by builders

### FAIL (2/10)
7. ❌ sqlite3 errors leak through BaseRepository._execute — PersistenceError exists but nothing wraps sqlite3 errors into it. close() swallows exceptions silently without logging.
8. ❌ Threading: `_closed` flag not lock-protected (TOCTOU race). Repo property accessors bypass the lock. close() doesn't acquire _lock.

---

## Test Coverage Audit

### CRITICAL (3)
- C1: `SongVersionRepository` has NO update() method — can never update a version's metadata after creation
- C2: `ProjectSession.open()` (ez_path-based) is completely untested
- C3: No thread-safety tests for autosave + manual save race conditions

### HIGH (5)
- H1: Global pipeline registry singleton leaks state between tests
- H2: `apply_migrations()` infrastructure has zero tests
- H3: `is_archived` field in takes completely untested (and not on domain type)
- H4: Autosave tests are timing-dependent (time.sleep(0.3) with 0.1s interval — flaky on CI)
- H5: No duplicate ID tests for song_versions, layers, or takes (only projects and songs tested)

### MEDIUM (3+)
- M1: Only projects/songs checked for operations-after-close (not versions/layers/takes/pipeline_configs)
- M2: stop_autosave before start_autosave untested
- M3: start_autosave called twice — potential double timer leak untested

---

## Architecture Alignment Audit

### ALIGNED ✅ (12)
- Entity model matches PANEL-R2 exactly (all 5 entities + SongPipelineConfig)
- layer_type enum with CHECK constraint
- Events as JSON blobs in Takes
- Dirty tracking: 5 structural events only, BlockStateChanged excluded
- Pipeline templates: @pipeline_template, PromotedParam, registry, validate_bindings
- SongPipelineConfig replaces ActionSets
- Naming conventions
- WAL + RLock thread safety
- DDL as source of truth
- Kill Branch (never built)
- Manual layers (source_pipeline=null)
- Don't pre-build pipeline storage

### PARTIALLY ALIGNED ⚠️ (6)
- Unit of Work: repos don't commit ✓, but create_new() commits outside UoW boundary
- BaseRepository: only _from_row is abstract (get/delete not abstract as panel specified)
- Schema: graph_json in DDL but NOT in PANEL-R2's spec (added independently)
- Migrations: infrastructure exists but no migrations/ directory or m001_initial.py
- TakeLayer: still present in takes.py, migration incomplete
- DirtyTracker missing `last_saved_at` field (Maya/Sage required it)

### DIVERGENT ❌ (2)
- Module structure: engine files (execution.py, pipeline.py, coordinator.py, cache.py, commands.py) at echozero/ root instead of engine/ subdirectory. shared/ directory doesn't exist.
- API Contract: completely out of date — still references Branch, branch-keyed layers, no Take endpoints. Needs full rewrite.

### NOT YET BUILT (correctly deferred) (6)
- AnalysisService, SetlistProcessor (services layer)
- archive.py, audio.py (Phase 3)
- Full analysis template, stems analysis template
- Section entity

### CONTRADICTIONS FOUND (4)
1. graph_json column exists in code but not in PANEL-R2 DDL
2. Old JSON save_project/load_project still functional alongside new SQLite persistence
3. is_archived in schema but not in Take domain type
4. Bindings validated by pipeline registry but never actually consumed by template builders
