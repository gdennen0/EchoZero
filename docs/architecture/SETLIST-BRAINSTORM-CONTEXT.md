# Setlist Brainstorm Context

## What EchoZero Is
Desktop audio analysis workstation for live production. Takes music, extracts time-stamped events (beats, onsets, classifications), syncs with grandMA3 lighting consoles. Users are lighting designers running live shows.

## Current EZ2 Architecture

### Entity Model
```
Project → Song[] → SongVersion[] → Layer[] → Take[]
                                 → PipelineConfig[] (per-version settings)
```

### What's Built
- **PipelineConfig**: Persistent pipeline settings per song version. Full graph + outputs + knob values stored in DB. Two-level settings: pipeline knobs (global) + per-block overrides.
- **Templates**: onset_detection, stem_separation, full_analysis. Templates are factories — create initial PipelineConfig, then get out of the way.
- **Orchestrator**: `create_config()` (factory), `execute(config_id)` (runs from DB).
- **SetlistProcessor**: Takes list of config_ids, runs them sequentially with error isolation.
- **Song import**: `session.import_song(title, audio_path)` — content-addressed copy, creates Song + SongVersion. Currently does NOT auto-create PipelineConfigs or scan audio metadata.
- **SongVersion**: Has `duration_seconds` and `original_sample_rate` but these are set to 0.0 and 0 on import (not populated).

### What's Missing
- No auto-creation of PipelineConfigs on song import
- No audio scanning (duration, sample rate, waveform) on import
- No processing status on songs
- No batch operations beyond SetlistProcessor
- No "global project settings that flow to all songs"
- No ingest pipeline

## EZ1 Reference (What Worked / What Didn't)

### Worked
- One setlist per project (simple, clear boundary)
- Per-song action overrides (some songs need different settings)
- Sequential processing with error isolation
- Song ordering and reordering

### Didn't Work
- ActionSets were complex (ordered lists of named actions referencing blocks by ID)
- Separate Setlist entity from Project was redundant
- SetlistAudioInput block was a hack (special block type just for setlist mode)
- Pre/post hooks, placeholder resolution — 847 lines of orchestration
- Snapshots for song state switching — complex, fragile

## Griff's Requirements (This Session)

1. Default core pipeline configs should be created automatically
2. Batch automation over the setlist
3. Clear boundaries for what a setlist is
4. "Automatic settings" concept stubbed for later (auto-detect optimal settings per song)
5. Sick ingest/add-song flow with auto-scan (waveform, duration, sample rate)
6. Each song has its own settings, global controls many at once
7. One setlist per project (hard boundary)
8. Song versioning must be rock solid

## Key Design Constraints
- Users never see "pipeline" — they see settings and buttons
- Templates are internal — users configure via knobs/inspector
- PipelineConfig is the settings entity, not ActionSets
- Override protection exists (per-block overrides survive global changes)
- Project = setlist (one setlist per project, hard boundary)
