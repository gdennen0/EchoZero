# Setlist Brainstorm Results

> Authored by: Architecture review session, EchoZero 2
> Context: ~/EchoZero/docs/architecture/SETLIST-BRAINSTORM-CONTEXT.md

---

## 1. Template Factories

### Recommendation
Auto-create **one** PipelineConfig per song on import using the project's default template. Default template is `full_analysis`. No per-template selection during import — just create it and let the user change it after.

Add a `Project.default_pipeline_template` field (string enum: `"onset_detection" | "stem_separation" | "full_analysis"`). This is the only "project default" you need.

The factory call happens inside `session.import_song()` — after song + version row creation, call `orchestrator.create_config(song_version_id, template=project.default_pipeline_template)`. That's it. One config, created immediately, ready to run.

### Why
Lighting designers are running shows. They don't want to think about what pipeline to use — they want to drag files in, hit "Analyze All," and be done. Auto-creating a `full_analysis` config on import means the workflow is: drop files → analyze → done.

Giving them a project-level default covers the 10% case where a whole project is, say, stems-only work. They set it once, everything inherits it. That's the right level of control.

### Entity/API Changes
```python
# Project model addition
class Project(Base):
    default_pipeline_template: str = "full_analysis"  # new field

# Updated import_song
def import_song(self, title: str, audio_path: str) -> Song:
    song = self._create_song(title, audio_path)
    version = self._create_version(song)
    self._run_ingest_pipeline(version, audio_path)  # see topic 6
    config = self.orchestrator.create_config(
        version.id, 
        template=self.project.default_pipeline_template
    )
    return song

# Orchestrator factory (already exists, just wire it up)
def create_config(self, version_id: int, template: str = "full_analysis") -> PipelineConfig:
    ...
```

### What to Skip
- Multiple auto-created configs per song on import (onset + stems + full = 3 configs). Don't do this. One config per song. If they want a different pipeline, they can create it manually.
- "Template wizard" during import. Over-engineering.
- Validating whether the template is appropriate for the audio format on import. That's pipeline execution's job.

---

## 2. Batch Automation Over Setlist

### Recommendation
Three batch operations at the project/setlist level:

1. **`project.analyze_all()`** — queues all songs that have no completed analysis
2. **`project.analyze_stale()`** — queues songs where audio file has changed (mtime or hash mismatch) or PipelineConfig was modified after last run
3. **`project.apply_setting(block_type, knob, value, protect_overrides=True)`** — applies a knob change across all songs' PipelineConfigs

These map directly onto `SetlistProcessor` under the hood. `analyze_all` and `analyze_stale` just filter which `config_ids` go in.

### Why
"Analyze All" is obvious. "Analyze stale only" is critical for any project that lives beyond one session — the user tweaks a threshold, wants to re-run affected songs without nuking the ones that are fine. Apply setting globally is the "global control" story from requirements.

Keep the API flat on `Project` or `Session`. Don't build a `BatchOperationManager` or `SetlistBatchAPI` — that's EZ1 bloat thinking.

### Entity/API Changes
```python
class Session:  # or Project service layer
    def analyze_all(self) -> BatchResult:
        configs = self.db.query(PipelineConfig).filter(
            PipelineConfig.song_version.has(song=Song.project_id == self.project.id),
            PipelineConfig.status != "completed"
        ).all()
        return self.setlist_processor.run([c.id for c in configs])

    def analyze_stale(self) -> BatchResult:
        configs = self.db.query(PipelineConfig).filter(
            PipelineConfig.song_version.has(song=Song.project_id == self.project.id),
            or_(
                PipelineConfig.status == "stale",
                PipelineConfig.status == "pending"
            )
        ).all()
        return self.setlist_processor.run([c.id for c in configs])

    def apply_setting(
        self, 
        block_type: str, 
        knob: str, 
        value: Any,
        protect_overrides: bool = True
    ) -> int:  # returns count of configs updated
        configs = self._get_all_active_configs()
        updated = 0
        for config in configs:
            config.set_knob(block_type, knob, value, 
                           skip_if_overridden=protect_overrides)
            updated += 1
        self.db.commit()
        return updated

# BatchResult (simple)
@dataclass
class BatchResult:
    total: int
    succeeded: int
    failed: int
    errors: dict[int, str]  # config_id → error message
```

**Staleness tracking** — add `PipelineConfig.status` as an enum:
```python
class PipelineStatus(str, Enum):
    PENDING = "pending"       # never run
    RUNNING = "running"       # currently executing
    COMPLETED = "completed"   # done, results valid
    STALE = "stale"           # config or audio changed since last run
    FAILED = "failed"         # last run errored
```

Mark a config `STALE` when:
- Any knob value changes on it
- The audio file mtime changes (check at run-time vs `SongVersion.audio_mtime`)
- The song's `active_version_id` changes

### What to Skip
- Priority queues, cancellation tokens, progress streaming for V1. Run sequentially, report errors. Done.
- "Undo batch apply" — not for V1. If they mess up a batch apply, they set it back.
- Per-song batch operation history.

---

## 3. Setlist Boundaries

### Recommendation
Project = Setlist. Hard boundary, no exceptions. The `Song` table IS the setlist. Ordinal position lives on `Song` as `position: int`. That's the full extent of it.

Song needs two new fields for setlist purposes:
- `position: int` — order in the setlist (1-indexed, gapless)
- `is_enabled: bool` — soft disable without removing (soundcheck track, cut song, etc.)

Processing status lives on `PipelineConfig`, not on `Song`. Song doesn't "have a status" — its configs do.

### Why
EZ1 had a separate `Setlist` entity and it was redundant. A project IS a show. The setlist IS the project's songs in order. Any abstraction on top of that is indirection for no benefit.

`is_enabled` covers a real live production need: a song might be in the project (analysis done, cues built) but cut from tonight's show. You don't want to delete it, but you don't want it in the run-of-show. A soft toggle is the right call.

Processing status on `PipelineConfig` (not `Song`) keeps concerns separate. A song can have multiple configs, each with different status. The song itself isn't "analyzed" — a specific config of a specific version of it is.

### Entity/API Changes
```python
class Song(Base):
    __tablename__ = "songs"
    id: int
    project_id: int
    title: str
    position: int           # NEW: 1-indexed setlist order
    is_enabled: bool = True # NEW: soft-disable for cut songs
    active_version_id: int  # already exists (or should)
    created_at: datetime
    updated_at: datetime

# Reordering
def reorder_songs(self, ordered_ids: list[int]) -> None:
    for position, song_id in enumerate(ordered_ids, start=1):
        self.db.query(Song).filter(Song.id == song_id).update({"position": position})
    self.db.commit()

# Get ordered setlist (respects enabled/disabled)
def get_setlist(self, include_disabled: bool = False) -> list[Song]:
    q = self.db.query(Song).filter(Song.project_id == self.project.id)
    if not include_disabled:
        q = q.filter(Song.is_enabled == True)
    return q.order_by(Song.position).all()
```

### What to Skip
- Setlist entity. Don't add it.
- Song "status" field that mirrors PipelineConfig status. Don't denormalize this.
- Multiple setlists per project for "alternative shows." V1 is one show, one project.

---

## 4. Default Core Pipeline Configs

### Recommendation
Ship **two** defaults, not three:
1. `full_analysis` — the workhorse, what 90% of users use 90% of the time
2. `onset_detection` — lightweight, fast, for quick cue spotting

Skip `stem_separation` as a standalone default. It's a sub-step inside `full_analysis`. If someone wants stems, they run full analysis.

User picks by: project default template (set once in project settings), or right-click a song → "Change Pipeline" → dropdown of available templates. That's it for the selection UI.

### Why
Three defaults implies three parallel workflows the user has to understand. Two is already one too many for most lighting designers who just want "the button that makes it work." Lead with `full_analysis`. Offer `onset_detection` as the "fast/lite" option for quick work.

Stem separation as a standalone config makes sense for power users doing deep work — but that's a "create custom config" flow, not a default. Ship a `stem_separation` template in the code for custom config creation, just don't auto-expose it in the "pick a default" dropdown.

### Entity/API Changes
```python
# Templates registry (internal, not user-facing names)
PIPELINE_TEMPLATES = {
    "full_analysis": FullAnalysisTemplate,
    "onset_detection": OnsetDetectionTemplate,
    "stem_separation": StemSeparationTemplate,  # available but not a "default"
}

# User-facing default options (what shows in project settings dropdown)
DEFAULT_TEMPLATE_OPTIONS = [
    ("full_analysis", "Full Analysis"),
    ("onset_detection", "Onset Detection (Fast)"),
]

# Project field
class Project(Base):
    default_pipeline_template: str = "full_analysis"
```

No changes to how templates work — they're already factories. Just curate which ones are exposed as defaults vs. available-but-advanced.

### What to Skip
- "Smart default" that picks a template based on audio characteristics on import. Save that for the auto-settings stub (topic 5).
- User-created named templates ("My Custom Template"). V1: use the built-ins, tweak per-song.
- Template versioning.

---

## 5. Automatic Settings (Stubbed)

### Recommendation
Define the interface now, implement never (until there's a real ML model to back it). The hook is: after audio scanning in the ingest pipeline, call `AutoSettingsAdvisor.recommend(audio_features) -> SettingRecommendations`. Recommendations are advisory only — they pre-fill knobs but don't block or auto-apply.

```python
@dataclass
class AudioFeatures:
    duration_seconds: float
    sample_rate: int
    estimated_bpm: float | None
    dynamic_range_db: float | None
    frequency_profile: str | None  # "bass_heavy", "mid_range", "bright" — coarse bucket
    has_silence_gaps: bool
    peak_amplitude: float

@dataclass
class SettingRecommendations:
    suggested_template: str | None          # e.g., "onset_detection" for short clips
    knob_suggestions: dict[str, dict[str, Any]]  # block_type → {knob: value}
    confidence: float                       # 0.0-1.0, for UI display
    rationale: str | None                   # human-readable explanation

class AutoSettingsAdvisor:
    def recommend(self, features: AudioFeatures) -> SettingRecommendations:
        # V1: return empty recommendations (no-op)
        return SettingRecommendations(
            suggested_template=None,
            knob_suggestions={},
            confidence=0.0,
            rationale=None
        )
```

The ingest pipeline calls this and stores the result in `SongVersion.auto_settings_recommendations` (JSON blob). UI can display it as a hint panel. User decides whether to apply.

### Why
Stubbing the interface now means:
1. The ingest flow doesn't need to change when the implementation arrives
2. Audio scanning (which you DO need) produces `AudioFeatures` as a natural output
3. The UI can show a "Recommended Settings" panel that's currently always empty — shipping the UX slot before the intelligence

The stub pattern here is: run the scanning (real work), call the advisor (returns nothing), store the nothing (ready for future). Zero behavior change, full future extensibility.

### Entity/API Changes
```python
class SongVersion(Base):
    # Existing
    duration_seconds: float
    original_sample_rate: int
    # New
    audio_mtime: float | None      # file modification time at ingest
    audio_features: dict | None    # JSON: AudioFeatures as dict
    auto_recommendations: dict | None  # JSON: SettingRecommendations as dict
```

The `AutoSettingsAdvisor` lives in `analysis/auto_settings.py`. It's a single class, no base class needed yet — don't over-abstract a stub.

### What to Skip
- Any actual ML/heuristic implementation in V1
- UI for "applying" recommendations (just store them)
- Feedback loop / user-corrected recommendations
- Per-genre models, BPM detection in the ingest pipeline (use a cheap rough estimate only if it's nearly free)

---

## 6. Ingest/Add-Song Flow

### Recommendation
The ingest pipeline is a synchronous, fast, sequential scan that runs immediately on drop. It must complete before the song appears as "ready" in the UI. Target: < 2 seconds per file on typical hardware.

**Steps in order:**

1. **File validation** — is it audio? readable? not already in project? (hash dedup)
2. **Content-addressed copy** — copy to project's audio store, rename by content hash
3. **DB row creation** — `Song` + `SongVersion` with position appended to end of setlist
4. **Audio scan** — open file, read: duration, sample rate, channel count, bit depth, mtime, peak amplitude. **This is real work, not deferred.**
5. **Waveform generation** — downsample to ~1000 points (min/max envelope), store as JSON blob on `SongVersion`. This is what the UI needs to render the waveform immediately.
6. **Metadata extraction** — read ID3/Vorbis tags if present: title override, artist, BPM hint. Update `Song.title` if a better title comes from tags.
7. **AutoSettingsAdvisor** — call with `AudioFeatures` from step 4, store result (noop for now)
8. **PipelineConfig creation** — call `orchestrator.create_config()` using project default template
9. **Emit event** — `song_ingested(song_id)` so UI refreshes

For a folder drop, run steps 1-9 per file sequentially. Emit a `batch_ingest_complete(song_ids)` after all files.

No heavy processing (stems, onset detection) happens during ingest. That's what "Analyze" is for.

### Why
The ingest pipeline needs to be fast and synchronous because users expect instant feedback when they drop files. Waveform + metadata + duration is what you need to show a useful track in the UI. Everything else (actual analysis) is deferred.

Doing it synchronously is fine — audio scanning is I/O bound but fast. For a 50-song setlist drop, worst case is ~30 seconds total, which is acceptable for a one-time operation.

Content-addressed copy solves the "file moved" problem — once ingested, the project owns its audio.

### Entity/API Changes
```python
# New ingest service
class IngestPipeline:
    def ingest_file(self, audio_path: Path, project: Project) -> Song:
        self._validate(audio_path)
        stored_path = self._copy_content_addressed(audio_path, project)
        song, version = self._create_db_rows(stored_path, project)
        features = self._scan_audio(stored_path, version)  # updates version fields
        self._generate_waveform(stored_path, version)       # stores waveform JSON
        self._extract_metadata(stored_path, song, version)  # may update song.title
        recommendations = self.advisor.recommend(features)
        version.auto_recommendations = asdict(recommendations)
        config = self.orchestrator.create_config(version.id, 
                     template=project.default_pipeline_template)
        self.db.commit()
        self.events.emit("song_ingested", song_id=song.id)
        return song

    def ingest_folder(self, folder: Path, project: Project) -> list[Song]:
        audio_files = sorted(self._find_audio_files(folder))
        songs = [self.ingest_file(f, project) for f in audio_files]
        self.events.emit("batch_ingest_complete", song_ids=[s.id for s in songs])
        return songs

# SongVersion additions
class SongVersion(Base):
    duration_seconds: float   # populated at ingest (was 0.0 before)
    original_sample_rate: int # populated at ingest (was 0 before)
    channel_count: int         # NEW
    bit_depth: int | None      # NEW (None for lossy formats)
    audio_mtime: float         # NEW: file mtime at ingest time
    waveform_data: str | None  # NEW: JSON array of {min, max} pairs (~1000 points)
    audio_features: str | None # NEW: JSON AudioFeatures blob
    auto_recommendations: str | None  # NEW: JSON SettingRecommendations blob
```

The audio scanning uses a library like `soundfile` or `librosa.get_duration` — whatever's already a dep. Waveform generation is a fast downsample, not a spectrogram.

### What to Skip
- Async/background ingest with progress bars for V1. Keep it sync.
- Spectrogram generation at ingest. Expensive, not needed for basic UI.
- Duplicate detection beyond content hash (fuzzy audio matching). Way too complex.
- Auto-ordering by filename number on folder drop — do it (it's one line of sorting), but don't build a smart filename parser.

---

## 7. Per-Song Settings with Global Control

### Recommendation
The override system already exists (per-block overrides). The API layer needs two methods:

1. **`project.set_global_knob(block_type, knob, value)`** — applies to all active configs, skips overridden knobs
2. **`song.set_knob(block_type, knob, value)`** — applies to this song's active config only, marks as overridden

"Override" means: this knob was explicitly set per-song. Global changes won't touch it. There's a `song.clear_override(block_type, knob)` to release it back to global control.

The UI story: there's a global settings panel (project-level). Each song has an inspector. When you change a knob in the inspector, it gets the "overridden" badge. When you change it in the global panel, overridden knobs show a "this song is overriding" indicator.

### Why
This is the right abstraction. Lighting designers know some songs need different sensitivity settings — the drop in "Song 7" needs a lower onset threshold than everything else. The override model handles exactly this without requiring per-song template management.

The `protect_overrides=True` default on global changes is critical. Never silently blow away intentional per-song tweaks. Always respect overrides unless the user explicitly says "force all" (an escape hatch, not the default).

### Entity/API Changes
```python
# On Session or Project service
def set_global_knob(
    self, 
    block_type: str, 
    knob: str, 
    value: Any,
    force: bool = False  # if True, override even overridden songs
) -> GlobalKnobResult:
    configs = self._get_all_active_configs()
    results = GlobalKnobResult(total=len(configs), updated=0, skipped=0)
    for config in configs:
        if not force and config.is_knob_overridden(block_type, knob):
            results.skipped += 1
            continue
        config.set_knob(block_type, knob, value)
        config.mark_stale()  # triggers re-analysis needed
        results.updated += 1
    self.db.commit()
    return results

# On Song (or via session.set_song_knob)
def set_song_knob(self, song_id: int, block_type: str, knob: str, value: Any) -> None:
    config = self._get_active_config(song_id)
    config.set_knob(block_type, knob, value)
    config.mark_override(block_type, knob)
    config.mark_stale()
    self.db.commit()

def clear_song_override(self, song_id: int, block_type: str, knob: str) -> None:
    config = self._get_active_config(song_id)
    config.clear_override(block_type, knob)
    # Optionally: re-apply current global value for this knob
    self.db.commit()

@dataclass
class GlobalKnobResult:
    total: int
    updated: int
    skipped: int  # skipped due to override protection
    # UI can show: "Applied to 14 songs, 2 songs have custom overrides"
```

The `PipelineConfig` likely already has override tracking — if not, add `overridden_knobs: dict` (JSON, maps `"block_type.knob"` → True).

### What to Skip
- "Override groups" — V1 doesn't need "Song 7 and Song 12 share overrides." Per-song or global. That's it.
- Override history / audit log
- "Inherit from global then adjust" preview UI. Just ship the override badge.
- Bulk override operations ("override this knob on all songs to different values") — that's per-song iteration in Python, not a new API.

---

## 8. One Setlist Per Project

### Recommendation
No `Setlist` entity. The `Song` table with `project_id` + `position` IS the setlist. Full stop.

The `Project` entity is the setlist container. Methods like `get_setlist()` live on `Session` or a thin `ProjectService`. Nowhere do you create, fetch, or update a `Setlist` row — because it doesn't exist.

`Song.position` is the ordering mechanism. Positions should be dense integers (1, 2, 3...). On reorder, update all positions in one transaction. On delete, compact positions.

### Why
EZ1 had a separate Setlist entity and it added zero value. A lighting designer's mental model is: "I have a project for tonight's show. My show has songs in order." That maps to `Project` + `Song[]` directly.

Adding a `Setlist` entity only makes sense if a project could have multiple setlists (e.g., "main set" vs. "encore"). That's a future feature you're explicitly not building. Don't pre-build the abstraction.

### Entity/API Changes
```python
# What you DON'T need:
# class Setlist(Base): ...  ← skip this entirely

# What you DO need (already covered in topic 3, restated for clarity):
class Song(Base):
    project_id: int   # foreign key to Project
    position: int     # 1-indexed, dense, ordered
    is_enabled: bool  # soft-disable

# Setlist = filtered, ordered query:
def get_setlist(self, include_disabled: bool = False) -> list[Song]:
    q = (self.db.query(Song)
         .filter(Song.project_id == self.project.id)
         .order_by(Song.position))
    if not include_disabled:
        q = q.filter(Song.is_enabled == True)
    return q.all()

# Reorder (atomic)
def reorder_songs(self, ordered_song_ids: list[int]) -> None:
    with self.db.begin():
        for pos, song_id in enumerate(ordered_song_ids, start=1):
            self.db.query(Song).filter(Song.id == song_id).update(
                {"position": pos}, synchronize_session=False
            )

# Remove (and compact positions)
def remove_song(self, song_id: int) -> None:
    song = self.db.query(Song).get(song_id)
    removed_pos = song.position
    self.db.delete(song)
    # compact: shift everything above removed position down
    self.db.query(Song).filter(
        Song.project_id == self.project.id,
        Song.position > removed_pos
    ).update({"position": Song.position - 1}, synchronize_session=False)
    self.db.commit()
```

### What to Skip
- `Setlist` entity. Hard no.
- Multiple setlists per project, setlist templates, setlist copying. Future, not now.
- Position gaps or sparse ordering ("leave room for inserts"). Dense is simpler. Reorder is atomic.

---

## 9. Song Versioning

### Recommendation
Versioning exists for one primary scenario: **the audio file changes**. New recording, different mix, re-edited file. Same song name, different audio content. Analysis results from the old version are preserved but dormant.

`Song.active_version_id` points to the current version. Switching versions means: update `active_version_id`, mark the new version's PipelineConfigs as pending (or stale if previously analyzed), leave old version's data intact.

Version creation flow:
1. User right-clicks song → "Replace Audio" or "Add Version"
2. New `SongVersion` row created with new audio path
3. **Option A** (recommended): Copy PipelineConfig settings from old version → new version's config (user keeps their tweaks, but results are invalidated)
4. `Song.active_version_id` = new version id
5. Old version data is preserved (can be viewed/restored)

### Why
"Same song, different file" is the real-world versioning scenario. A band re-records a track, the LD gets an updated WAV. They don't want to lose their cue work or pipeline settings — they just need the analysis re-run against the new audio.

Preserving old version data is important: what if the new mix introduced a problem? The LD needs to roll back. This is safety net behavior, not complexity for its own sake.

Copying pipeline settings to the new version (Option A) is the right call because the LD's tweaks were made based on the song's musical characteristics, which usually don't change between mixes. Their settings are still valid starting points.

### Entity/API Changes
```python
class SongVersion(Base):
    id: int
    song_id: int
    version_number: int     # auto-increment per song (1, 2, 3...)
    label: str | None       # optional: "v2 mix", "live recording", etc.
    audio_path: str         # content-addressed path
    audio_hash: str         # SHA256 of audio file
    duration_seconds: float
    original_sample_rate: int
    # ... other scan fields from topic 6 ...
    created_at: datetime

class Song(Base):
    active_version_id: int | None  # FK to SongVersion
    # ... other fields ...

# Version management methods
def add_song_version(
    self,
    song_id: int,
    audio_path: Path,
    label: str | None = None,
    copy_settings: bool = True  # copy pipeline settings from current version
) -> SongVersion:
    song = self.db.query(Song).get(song_id)
    old_version = self._get_active_version(song)
    
    # Ingest the new audio
    new_version = self.ingest_pipeline.ingest_version(audio_path, song)
    new_version.label = label
    new_version.version_number = self._next_version_number(song_id)
    
    if copy_settings and old_version:
        # Copy pipeline config settings (not results) to new version
        self._clone_pipeline_config(old_version.id, new_version.id)
    
    # Activate new version
    song.active_version_id = new_version.id
    self.db.commit()
    
    return new_version

def switch_active_version(self, song_id: int, version_id: int) -> None:
    song = self.db.query(Song).get(song_id)
    # Validate version belongs to this song
    version = self.db.query(SongVersion).filter(
        SongVersion.id == version_id,
        SongVersion.song_id == song_id
    ).one()
    song.active_version_id = version_id
    # Mark version's config as stale if results are old
    self._mark_version_stale_if_needed(version)
    self.db.commit()

def _clone_pipeline_config(self, from_version_id: int, to_version_id: int) -> PipelineConfig:
    """Copy settings (knobs, overrides) but not results."""
    source = self.db.query(PipelineConfig).filter(
        PipelineConfig.song_version_id == from_version_id
    ).first()
    if not source:
        return self.orchestrator.create_config(to_version_id)
    
    new_config = source.clone_settings()  # copies knobs + overrides, status=PENDING
    new_config.song_version_id = to_version_id
    self.db.add(new_config)
    return new_config
```

**Staleness on version switch**: when switching to an old version, if the config was last run against a different audio hash than what's now active, mark it `STALE`. This handles the "I'm rolling back to an old mix" case.

```python
def _mark_version_stale_if_needed(self, version: SongVersion) -> None:
    config = self._get_config_for_version(version.id)
    if config and config.last_run_audio_hash != version.audio_hash:
        config.status = PipelineStatus.STALE
```

Add `PipelineConfig.last_run_audio_hash: str | None` to track this.

### What to Skip
- Version branching / version trees. Linear version history only (v1 → v2 → v3).
- Version diffs / "what changed between versions"
- Auto-versioning on file change (file watcher). User-initiated only.
- Version tagging/commenting system for V1 (the `label` field is enough)
- Keeping unlimited version history — for V1, allow pruning old versions manually. Don't auto-delete.

---

## Summary: Entity Changes Required

| Entity | New Fields |
|--------|-----------|
| `Project` | `default_pipeline_template: str` |
| `Song` | `position: int`, `is_enabled: bool`, `active_version_id: int` |
| `SongVersion` | `version_number: int`, `label: str?`, `audio_hash: str`, `audio_mtime: float`, `channel_count: int`, `bit_depth: int?`, `waveform_data: str?`, `audio_features: str?`, `auto_recommendations: str?` |
| `PipelineConfig` | `status: PipelineStatus`, `overridden_knobs: str (JSON)`, `last_run_audio_hash: str?` |

## Summary: New Services/Classes

| Class | Purpose |
|-------|---------|
| `IngestPipeline` | Handles file → Song + Version + Config creation |
| `AutoSettingsAdvisor` | Stub interface for future auto-settings |
| `AudioFeatures` | Dataclass: output of audio scanning |
| `SettingRecommendations` | Dataclass: output of AutoSettingsAdvisor |
| `PipelineStatus` | Enum: pending, running, completed, stale, failed |
| `BatchResult` | Dataclass: result of batch operations |
| `GlobalKnobResult` | Dataclass: result of set_global_knob |

## Summary: New Methods on Session/Project

| Method | Description |
|--------|-------------|
| `analyze_all()` | Queue all pending/failed configs |
| `analyze_stale()` | Queue stale configs only |
| `apply_setting(block, knob, value, protect_overrides)` | Global knob change |
| `set_global_knob(block, knob, value, force)` | Same as above, alias |
| `set_song_knob(song_id, block, knob, value)` | Per-song override |
| `clear_song_override(song_id, block, knob)` | Release per-song override |
| `get_setlist(include_disabled)` | Ordered song list |
| `reorder_songs(ordered_ids)` | Atomic reorder |
| `remove_song(song_id)` | Delete + compact positions |
| `add_song_version(song_id, audio_path, label, copy_settings)` | New audio version |
| `switch_active_version(song_id, version_id)` | Activate old/new version |

---

*No Setlist entity. No ActionSets. No pre/post hooks. No snapshot-based state switching. Keep it flat.*
