# Application Settings

This package owns machine-local EchoZero preferences.

Use it for settings that belong to the operator environment rather than project truth:
- audio output device and stream behavior
- song import defaults (LTC strip + import pipeline actions)
- MA3 OSC receive/send endpoints
- future local UI or workstation preferences

## Scope Rules

- `ProjectSettingsRecord` remains portable project data.
- `AppPreferences` is local-only and must not be saved into `.ez` files.
- `Session` may reference local preferences or resolved runtime state, but it is not the source of truth for app settings.

## Reusable Pattern

The package is intentionally split into three layers:
- `models.py`: typed persisted/runtime preference models
- `contracts.py`: neutral settings-page contracts for UI rendering
- `page_builder.py`: section/field surfacing for settings-rich pages
- `service.py`: validation, page building, persistence orchestration, and runtime resolution

That lets future settings areas reuse the same pattern without inventing widget-local configuration shapes.

## Runtime Integration

- `run_echozero.py` loads `AppSettingsService` from the local JSON store.
- CLI flags are treated as launch overrides, not the canonical settings source.
- `echozero/ui/qt/launcher_surface.py` opens `PreferencesDialog` from the canonical shell.
- `echozero/ui/qt/timeline/widget_controls.py` exposes a visible shell `Settings` button in the editor chrome.
- `echozero/ui/qt/app_shell_runtime_services.py` resolves `AudioOutputRuntimeConfig` into runtime audio controllers.
- `echozero/infrastructure/osc/*` owns reusable OSC UDP send/receive plumbing.
- `echozero/infrastructure/sync/ma3_osc.py` stays focused on MA3 protocol behavior on top of that generic OSC lane.
- Preferences only edit the local config JSON; they do not live-apply runtime changes from the shell surface.
- MA3 OSC and audio output changes are persisted immediately but apply on the next launcher start.

## Song Import Defaults

The `Song Import` settings section is machine-local and affects both song and version import paths:

- dialog-based add song
- dialog-based add version
- drag/drop add song
- drag/drop add version
- multi-file folder import

Current controls:

- `import.strip_ltc_timecode`: auto-detect one LTC channel in stereo audio and import the program channel.
- `import.pipeline_action.<action_id>` toggles: run configured import-safe pipeline actions after each import.

The action toggle inventory is descriptor-driven via
`import_safe_pipeline_action_descriptors()` and intentionally constrained to import-safe object actions.

For current runtime behavior details and rework seams, see
`docs/SONG-IMPORT-BATCH-LTC-WORKFLOW.md`.

## Surfacing Strategy

Every field can be marked as:
- `primary`
- `advanced`
- `hidden`

This keeps the stored settings model richer than the initial operator surface while preserving a clean way to surface or unsurface fields later without changing persistence shape.

## Expanding Settings Safely

When a new application-level settings area is added:
- add typed persisted/runtime models in `models.py` or a sibling typed module
- render it into the neutral `SettingsPage` contract from `page_builder.py`
- validate and resolve it in `AppSettingsService`
- persist it through `echozero/infrastructure/settings/*`
- decide whether it is config-only, live-applicable, or restart-applied and document that in page warnings

This keeps the settings lane rich by default while still letting the UI surface only the fields that operators truly need first.
