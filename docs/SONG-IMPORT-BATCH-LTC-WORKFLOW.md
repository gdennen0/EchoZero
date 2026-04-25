# Song Import Batch + LTC Workflow

Status: implemented behavior map  
Last verified: 2026-04-25 (batch UX pass)

This document captures how song import works today across UI, runtime, settings, and persistence.
It is intentionally implementation-oriented so we can rework safely without re-discovering behavior.

## Scope

Current behavior covered here:

- Drag/drop song files or folders into the setlist/timeline surfaces.
- Batch import dialog choices and ordering guarantees.
- Import-time pipeline action configuration and execution.
- Import-time LTC/program channel split with left/right auto-detection.
- Setlist reorder operations (single move, drag reorder, batch move).
- Versioning behavior for imported versions.

## Operator UX

### Configure import defaults

Use `Preferences -> Application Settings -> Song Import`.

Available controls:

- `Auto Strip LTC Channel`
- `Run <Pipeline Action Label>` toggles for import-safe object actions

The pipeline toggles are descriptor-driven, not hardcoded to two actions.
As of 2026-04-25, the surfaced actions are:

- `timeline.extract_stems`
- `timeline.extract_song_drum_events`
- `timeline.extract_drum_events`
- `timeline.extract_classified_drums`

### Batch folder import

1. Drag a folder (or multiple files) into the song browser or timeline drop targets.
2. The folder is expanded recursively to supported audio files:
   `.wav`, `.mp3`, `.flac`, `.aiff`, `.aif`, `.ogg`.
3. Files are imported in natural sort order (`Song 2` before `Song 10`).
4. If multiple files are present, an import mode dialog appears:
   - import as new songs at end
   - when dropped on a target song: import before/after that song
   - when dropped on a target song: add all as versions of that song
5. If import pipeline actions are configured, a confirmation dialog asks whether to run them for each imported file.
6. For queued import-pipeline runs, batch import now executes in two ordered phases:
   - import all songs/versions first
   - queue per-song pipeline actions in import order
7. Queued pipeline runs reuse the same action-request path as context-menu runs (`request_object_action_run`) and surface progress in the main timeline pipeline status banner.
8. If insertion mode is before/after target, imported songs are repositioned after creation while preserving imported relative order.

### Single-file import

The same import defaults apply to:

- drag/drop single-file import
- `song.add` (Add Song dialog)
- `song.version.add` (Add Version dialog)
- when runtime support is present, configured import pipeline actions for these single-file flows are also deferred through the same queued `request_object_action_run` path used by context-menu runs

## Runtime Contract Behavior

Import actions are passed through runtime method kwargs when available:

- `run_import_pipeline`
- `import_pipeline_action_ids`

If a runtime implementation does not support those kwargs, UI fallback executes configured actions after import via `run_object_action(...)` on the source audio layer.
This keeps older runtimes functional without losing configured import automation.

Action IDs are canonicalized against descriptor metadata, deduped, and filtered to import-safe actions only.

Dialog-driven imports (`song.add`, `song.version.add`) and drop-driven imports now reuse the same canonical runtime invocation helpers for:

- song/version creation call semantics
- import pipeline kwargs passthrough
- legacy fallback import-pipeline execution

For batch imports that queue import-pipeline actions, each queued run is requested through the same context-menu pipeline path and executed sequentially.
When runtime support is present, the same queued path is used for single-file add-song/add-version import actions as well.

## Ordering and Setlist Reorder

Setlist ordering is currently supported by:

- song context actions: move up / move down
- drag reorder within the song list
- batch actions: move selected songs to top / bottom
- explicit full reorder runtime call

Persistence enforces that full reorder operations include exactly the current set of song IDs.

## LTC Import Behavior

When `Auto Strip LTC Channel` is enabled:

1. Stereo source is scanned for an LTC-like channel (`left` or `right`).
2. If no confident LTC channel is found, import continues unchanged.
3. If LTC is detected, both channels are split to mono:
   - program channel artifact
   - LTC channel artifact
4. Program channel artifact becomes the imported song/version audio source.
5. Both artifacts are retained in project scope:
   `audio/split_channels/<hash16>_program_<left|right>.wav`
   `audio/split_channels/<hash16>_ltc_<left|right>.wav`
6. Temporary staged files are cleaned up.

Flipped-channel scenarios are supported (LTC on right, program on left, or vice versa).

## Versioning Integration

Both `import_song(...)` and `add_song_version(...)` go through the same version factory path for audio preprocessing and metadata scanning.

Versioning guarantees that matter here:

- import-time LTC preprocessing applies to both new songs and new versions
- new versions inherit the source version MA3 timecode pool number
- import pipeline actions can run on both song and version import flows

## Important Seams For Rework

These are deliberate callouts for future redesign:

- Import-safe action eligibility is heuristic (`layer` object action with `layer_id`-only required params). This may be too strict or too loose for future actions.
- Import pipeline queue execution is sequential and surfaces state in the main timeline pipeline banner, but it does not yet expose a dedicated import-queue progress/cancel control.
- Multi-import “add all as versions” is target-song-driven (drop on song row). Generic timeline drop defaults to new songs.
- Final normalization in the action router sorts by basename; folders with duplicate filenames in different subdirectories can interleave unexpectedly.
- LTC detection uses fixed heuristics and thresholds; there is no per-project calibration UI yet.
- Retained LTC/program split artifacts are persisted but not yet surfaced in a dedicated operator browser.

## Canonical File Map

UI and drop routing:

- `echozero/ui/qt/song_browser_drop.py`
- `echozero/ui/qt/song_browser_panel.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/widget_actions.py`
- `echozero/ui/qt/timeline/widget_action_contract_mixin.py`

Settings:

- `echozero/application/settings/models.py`
- `echozero/application/settings/page_builder.py`
- `echozero/application/settings/service.py`

Runtime + persistence:

- `echozero/ui/qt/app_shell_project_lifecycle.py`
- `echozero/persistence/audio.py`
- `echozero/persistence/session_versioning_mixin.py`

## Regression Coverage

Primary tests that pin this behavior:

- `tests/ui/test_song_browser_panel.py`
- `tests/ui/timeline_shell_contract_actions_support.py`
- `tests/application/test_app_settings_service.py`
- `tests/test_song_version.py`
