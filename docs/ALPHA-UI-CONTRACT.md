# Alpha UI Contract

_Status: working draft_
_Date: 2026-04-16_

This document freezes the intended Alpha contract for the EchoZero desktop UI.
It is a contract-first build guide, not a paint-first mockup.

If this document conflicts with:
- `STYLE.md`
- `GLOSSARY.md`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- `docs/architecture/DECISIONS.md`

those canonical docs win.

## 1. Purpose

Alpha needs one explicit contract for:
- song selection and version switching
- center timeline behavior
- right-side object info and object actions
- MA3 push/pull flow
- pipeline-triggered UI actions
- boundaries between UI, application orchestration, persistence, and engine

The goal is to stop spreading semantics across:
- `echozero/ui/qt/app_shell.py`
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/presentation/inspector_contract.py`
- `echozero/application/timeline/orchestrator.py`

## 2. Locked Rules

These rules are not negotiable for Alpha:

- Main is truth.
- Takes are subordinate history, comparison, rerun, and candidate lanes only.
- There is no active-take truth model.
- Stable pipeline outputs map to stable layers.
- Reruns append takes by default.
- Downstream staleness changes only when upstream main changes.
- Provenance, freshness, and manual modification are separate concepts.
- MA3 is a transport boundary for push and pull of event data. It is not a take model.
- MA3 writes and sync authority are main-only.
- Engine stays ignorant of timeline, take, song-version, inspector, and MA3 UI semantics.
- FEEL owns tuning constants and UI dimensions.
- UI renders typed application truth and dispatches typed actions. It does not invent parallel truth.

## 3. Alpha Shell

Alpha uses a fixed 3-pane shell:

- Left: `SongsPanel`
- Center: `TimelineWorkspace`
- Right: `ObjectInfoPanel`

Responsibilities:

- `SongsPanel`
  - import songs
  - select song
  - switch viewed version
  - activate version
  - add as version
  - rerun with current settings
  - open per-song pipeline settings
  - copy pipeline settings between songs

- `TimelineWorkspace`
  - render the active song version timeline
  - render layer main rows and take child rows
  - support selection, navigation, editing, and context actions
  - surface pipeline actions from objects
  - surface MA3 push/pull entry points on valid main-layer scopes

- `ObjectInfoPanel`
  - render object identity, facts, warnings, provenance, stale/manual state, and available actions
  - render selection summary for multi-select
  - never become the hidden workflow engine for the whole app

## 4. Canonical Object Path

Alpha should center on this strict path:

`Project -> Song -> SongVersion -> Layer(main truth) -> Take(candidate) -> Event`

Important implications:

- `Song` is the setlist item and parent object.
- `SongVersion` is the audio-specific child object.
- The center timeline is always the currently viewed `SongVersion`.
- A `Layer` owns one main truth lane and zero or more subordinate takes.
- `Event` selection is always understood in the context of its owning main lane or take lane.

## 5. Version Switching

Alpha should use direct version switching.

Recommended default:

- Selecting a song loads that song's currently selected version.
- Choosing a version from the version selector switches the timeline directly to that version.
- There is no separate read-only viewed-version mode in Alpha.
- The selected version is the working version for timeline editing, rerun actions, and pipeline settings inspection.

Why:

- this keeps the shell simple
- this matches the requested operator behavior
- this avoids a second mode system in Alpha

## 6. Song Selection Contract

### Import Song

Import flow:

1. User invokes `Import Song`.
2. App creates `Song`.
3. App imports source audio.
4. App creates first `SongVersion`.
5. App creates default pipeline config records for that version.
6. App selects the new song.
7. App loads the new version into the center timeline.
8. App applies the configured import auto-run policy.

Alpha requirement:

- import must use canonical config creation semantics
- do not keep the current shell shortcut that bypasses default pipeline templates

### Song Row Actions

Each song row should support:

- `Import Song`
- `Select Song`
- `Add As Version`
- `Rerun With Current Settings`
- `Open Pipeline Settings`
- `Copy Settings From...`
- `Apply Settings To...`

### Version Actions

Each visible version should support:

- `Switch To Version`
- `Rename Version`
- `Duplicate Settings From Active`
- `Open Pipeline Settings`

Optional for Alpha, not required on day one:

- inline nested version tree

Recommended Alpha UI:

- keep versions behind a focused chip or dropdown inside the selected song row
- avoid building a full tree browser until the semantics are stable

### Historical Version Guard

Alpha does not use a separate historical-view guard mode.

Version selection switches the working timeline directly.

Guardrails still apply to dangerous actions:

- `pull to main` remains disabled or heavily guarded
- destructive replace-main actions remain explicit and guarded

## 7. Timeline Contract

### Layout

The center workspace should be:

- transport
- ruler
- timeline canvas
- horizontal scroll and timeline-local status surfaces

The timeline canvas should render:

- one main row per layer
- child take rows under expanded layers
- event lanes and waveform lanes as typed block surfaces

### Selection Rules

Alpha selection should be explicit and capability-driven:

- song selection is shell-owned, not timeline-owned
- background click clears timeline object selection but keeps the current song and version loaded
- layer click selects the layer
- layer shift-click selects a contiguous layer range
- layer command-click toggles layer membership
- take click selects the take within its parent layer and clears event selection
- take selection never changes truth
- event click selects one event
- shift-click adds event selection
- command-click toggles event membership

Recommended Alpha constraint:

- multi-event selection should remain scoped to one container path
- do not support cross-layer event multi-select in Alpha

### Highlight Rules

- selected layer highlights the full main row and header
- selected take gets its own clear visual state
- selected event keeps the stronger selected fill and border treatment
- takes must never render like alternate active truth
- mute and solo controls remain on the main row only

### Take Rules

- takes are candidates and history only
- reruns append to the target layer as takes by default
- take actions are editorial actions, not sync truth actions

Recommended take actions:

- `Promote To Main`
- `Merge Into Main`
- `Compare To Main`
- `Rename Take`
- `Delete Take`

## 8. Object Info Contract

The right panel should always have a valid object target.

Priority order:

1. selected event
2. selected take
3. selected layer
4. selection summary
5. current song version

That means when nothing in the timeline is selected, the inspector should fall back to `SongVersion` information, not an empty panel.

### Object Info Responsibilities

For any selected object, the panel should show:

- identity
- object type
- local facts
- provenance
- stale state
- manual modification state
- warnings and guards
- available actions

### Action Taxonomy

Actions should be grouped by type, not by ad hoc widget location:

- inspect
- navigate
- timeline edit
- truth management
- pipeline
- transfer

Examples:

- inspect
  - reveal source layer
  - show source run
  - show stale reason

- navigate
  - seek here
  - zoom to selection
  - reveal downstream dependents

- timeline edit
  - move
  - nudge
  - duplicate
  - quantize
  - relabel
  - mute
  - delete

- truth management
  - promote take to main
  - merge take into main
  - revert main from provenance

- pipeline
  - rerun with current settings
  - rerun with prompt
  - run assigned pipeline action
  - open pipeline settings

- transfer
  - preview push
  - confirm push
  - preview pull
  - apply pull
  - set target mapping

Contract rule:

- object actions must be capability-driven from typed state
- action availability must not be inferred from titles, badges, or string heuristics

## 9. Pipeline Contract

### Core Model

Pipelines should not bind to raw widget events.

Instead:

- UI dispatches typed object actions
- application resolves those actions into pipeline runs or commands
- engine executes the requested graph and returns typed outputs

Recommended action model:

- `ActionDefinition`
  - static app action id
  - scope
  - kind
  - param schema
  - allowed persist modes

- `ActionBinding`
  - data-driven attachment of an action to a pipeline or workflow
  - settings source
  - default target and persist policy

- `ActionResolver`
  - application service that returns available actions for the current object and context

### Pipeline Trigger Model

Alpha needs pipeline triggers at the application layer, not inside the engine.

Recommended trigger keys:

- `song.imported`
- `song.rerun_requested`
- `song.pipeline_settings_updated`
- `layer.pipeline_action_requested`
- `take.pipeline_action_requested`
- `event_layer.pipeline_action_requested`

Each trigger should resolve to:

- subject reference
- song version id
- pipeline config id or template id
- persist mode
- optional target layer id

### Persist Modes

Allowed persist modes should be explicit:

- `new_layer`
- `append_take`
- `replace_main`
- `preview_only`
- `continue_pipeline`

Recommended Alpha default:

- rerun-derived analysis uses `append_take`
- initial first-run analysis uses stable layer creation
- replace-main is explicit and guarded

### Import Auto-Run Policy

Import auto-run must be application-configurable, not shell-hardcoded.

Recommended Alpha policy shape:

- `manual`
- `autorun_default_bundle`
- `autorun_profile(profile_id)`

Default developer bundle for Alpha:

1. source song import
2. stem split
3. drum stem extraction
4. drum event classification
5. stable output layers created or updated

### Per-Song Pipeline Settings

Alpha should treat pipeline settings as:

- song-level defaults
- plus version-level effective copies

Recommended model:

- each `Song` owns a default pipeline profile
- each `SongVersion` receives a copy of those settings when the version is created
- reruns always use the selected version's effective settings
- users may edit a version's settings without mutating older versions
- users may also apply or copy settings across songs as a convenience action

Why:

- per-version effective settings preserve reproducibility for that version
- song-level defaults keep the common case easy
- this supports direct switching and rerunning on any version
- this avoids mutating historical runs when a user tunes settings for a later version

Required behavior:

- each song owns a default pipeline profile
- each song version owns its effective copied settings
- pipeline settings can be edited from the shell
- pipeline settings editing must support both:
  - `Edit This Version Only`
  - `Edit Song Default`
- pipeline settings can be copied from one song to another
- pipeline settings can be copied from one version to another
- settings copy must support partial transfer, not only full-profile overwrite
- copy is a convenience operation over version-owned effective settings plus song defaults

### Settings Copy Semantics

Alpha needs settings transfer at three levels:

- song default -> song default
- version effective settings -> version effective settings
- song default -> version effective settings

Settings transfer must support:

- full copy
- partial copy by pipeline stage
- partial copy by settings section or parameter group
- explicit preview of what will be overwritten

Recommended UI:

- `Copy Settings From...`
- source selector: song or version
- transfer scope:
  - `All Settings`
  - `Selected Pipeline Stages`
  - `Selected Parameters`
- target mode:
  - `Apply To This Version`
  - `Apply To Song Default`

Recommended default:

- default editing mode is `Edit This Version Only`
- `Edit Song Default` is explicit
- partial copy is first-class, not hidden behind advanced tooling

### Settings UI Behavior

Alpha should expose pipeline settings from two entry points:

- song-level shell actions in the left panel
- object-level actions in the timeline and inspector

#### Song-Level Entry Points

The selected song row should provide:

- `Open Pipeline Settings`
- `Rerun With Current Settings`
- `Copy Settings From...`
- `Apply Settings To...`

The version selector context should provide:

- `Edit This Version Only`
- `Edit Song Default`

Recommended interaction:

- clicking `Open Pipeline Settings` opens the settings panel or dialog in the current mode
- default mode is `Edit This Version Only`
- switching to `Edit Song Default` is explicit and visible
- the panel header must always show which scope is being edited

Recommended header examples:

- `Pipeline Settings: Version "Radio Edit"`
- `Pipeline Settings: Song Default`

#### Object-Level Entry Points

Objects in the center timeline may surface settings and rerun actions when they have a stable pipeline origin.

Examples:

- song row
  - `Open Pipeline Settings`
  - `Rerun With Current Settings`

- drum stem layer
  - `Open Drum Settings`
  - `Run Drum Classification`

- snare event layer
  - `Open Classification Settings`
  - `Rerun Snare Classification`

Contract rule:

- object-level settings actions open the relevant stage-focused view of the same underlying settings structure
- they do not create a separate hidden settings system

#### Settings Panel Structure

Recommended Alpha settings layout:

1. scope switch
2. pipeline stage list
3. stage detail editor
4. actions footer

Scope switch:

- `This Version`
- `Song Default`

Stage list:

- one row per pipeline stage
- shows enabled state
- shows whether the stage has overrides relative to song default

Stage detail editor:

- grouped by parameter group
- each group contains its parameters
- output keys are visible for stages that declare outputs

Footer actions:

- `Save`
- `Save And Rerun`
- `Copy Settings From...`
- `Reset This Stage`
- `Reset To Song Default`

Recommended Alpha rule:

- `Save And Rerun` should rerun only the currently relevant scope when the user entered from an object-level action
- entering from the song row may offer whole-song rerun

### Settings Structure

Alpha should structure pipeline settings in three levels:

1. `PipelineProfile`
2. `StageSettings`
3. `ParameterGroup`

Then parameters live inside the parameter groups.

Recommended shape:

- `PipelineProfile`
  - `profile_id`
  - `owner_scope`
    - `song_default`
    - `song_version_effective`
  - `song_id`
  - `song_version_id?`
  - `stages`

- `StageSettings`
  - `stage_key`
  - `label`
  - `enabled`
  - `parameter_groups`
  - `declared_output_keys`

- `ParameterGroup`
  - `group_key`
  - `label`
  - `parameters`

- `Parameter`
  - `parameter_key`
  - `label`
  - `value`
  - `value_type`
  - `default_value`

Alpha should support these edit and copy scopes:

- edit full profile
- edit one stage
- edit one parameter group
- copy full profile
- copy selected stages

Recommended Alpha deferral:

- copying individual parameters may be added later, but it should not complicate the first settings UI pass unless it falls out naturally from the implementation

Why this structure:

- it keeps the UI understandable
- it supports partial settings transfer cleanly
- it supports stage-level rerun and output-scoped rerun
- it avoids treating settings like one unstructured blob

### Partial Copy UI

`Copy Settings From...` should open an explicit transfer flow.

Recommended Alpha flow:

1. choose source
2. choose target scope
3. choose copy scope
4. preview overwrite impact
5. confirm apply

Source options:

- current song default
- another song default
- current version
- another version

Target options:

- current version
- selected version
- current song default

Copy scope options:

- `All Stages`
- `Selected Stages`

Recommended Alpha deferral:

- do not require per-parameter copy in the first pass unless implementation stays simple

Preview should show:

- stages added
- stages overwritten
- stages unchanged

Important rule:

- copy operations mutate settings only
- they do not rerun automatically unless the user explicitly chooses a rerun action afterward

### Rerun Semantics

Examples of required Alpha behavior:

- right-click source song or song row -> `Rerun With Current Settings`
- right-click drum stem -> `Run Classified Drum Extraction With Current Settings`
- output returns into the stable target layer
- rerun appends a new take under that existing layer by default

Alpha also needs output-scoped rerun.

Examples:

- rerun only `snare` classification
- rerun only `kick` classification
- rerun only a selected subset of classifier outputs

That means the pipeline contract must support:

- rerun whole workflow
- rerun one pipeline stage
- rerun one declared output or output subset from a stage

Recommended model:

- classifier stage declares stable named outputs by class
- each class layer keeps a stable mapping to its originating pipeline stage and output key
- object actions may target:
  - the whole stage
  - one stable output key
  - a selected subset of output keys

Example:

- `Snare` event layer action: `Rerun Snare Classification`
- input subject resolves from the correct upstream source, likely the drum stem main take and prior onset results
- output mapping resolves only the `snare` output key
- result appends a new take under the stable `Snare` layer by default

Important boundary:

- UI chooses the object action
- application resolves the subject, stage, output filter, and persist policy
- engine still just executes the requested graph or scoped pipeline run

### Pipeline Stage And Output Identity

To make partial reruns and partial settings copy coherent, pipeline settings need stable structure.

Recommended shape:

- pipeline profile
  - stage key
  - stage settings groups
  - parameter keys
  - declared output keys

Why:

- partial settings transfer needs stable stage and parameter identity
- output-scoped rerun needs stable output identity
- stable layer mapping already depends on stable output names

For the classifier example:

- stage: `drum_classification`
- upstream dependencies:
  - `drum_stem`
  - `onset_detection`
- outputs:
  - `kick`
  - `snare`
  - `tom`
  - `hihat`
  - other declared class outputs

Recommended action examples:

- `song.rerun_current_settings`
- `layer.rerun_stage.drum_classification`
- `layer.rerun_output.drum_classification.snare`
- `layer.rerun_output_subset`

Alpha recommendation:

- support single-output rerun like `snare` first
- multi-output subset rerun can follow if the contract stays clean

### Object Action Surfacing

Object actions should expose rerun and settings affordances from the object that owns the result.

Recommended Alpha examples:

- song row
  - `Rerun With Current Settings`
  - `Open Pipeline Settings`

- drum stem layer
  - `Run Drum Event Extraction`
  - `Open Drum Pipeline Settings`

- classified output layer such as `Snare`
  - `Rerun This Output`
  - `Open Classification Settings`

- take under a classified layer
  - `Compare To Main`
  - `Promote To Main`
  - not direct pipeline settings unless the action clearly targets the parent output layer

Important rule:

- output-layer actions target the stable output mapping, not the currently selected take as alternate truth
- rerun actions should default to appending a new take under the stable target layer

### Classifier Example

For the classifier workflow:

- source song
- stems
- drum stem
- onset detection
- classified onsets
- stable event layer per declared class

Example stable classifier output keys:

- `kick`
- `snare`
- `tom`
- `hihat`

Each resulting class layer should preserve:

- originating stage key
- originating output key
- upstream source references
- run id
- source main revision identity

That lets the UI surface actions like:

- `Rerun Classification For This Layer`
- `Rerun Snare Classification`
- `Open Classification Settings`

And the application can resolve that into:

- the correct upstream subject
- the correct stage
- the correct output filter
- the correct effective settings profile
- the correct stable target layer
- default persist mode `append_take`

## 10. MA3 Push/Pull Contract

MA3 is a transport and remote snapshot surface.
It is not a take system and not an editor truth model.

### MA3 Owns

- connection state
- remote target identity
- remote track discovery
- remote event discovery
- diff inputs for preview
- remote read and write execution

### MA3 Does Not Own

- take semantics
- take promotion
- merge or overwrite-main policy
- layer creation policy
- staleness
- provenance semantics above remote-source metadata
- song-version carry-forward

### Push Flow

1. user selects a main event layer or main events
2. user opens `Push to MA3`
3. user selects a target track mapping
4. app shows diff preview
5. user confirms apply
6. sync transfer service writes to MA3
7. app records transfer metadata and status

Push rule:

- push serializes main only

### Pull Flow

1. user opens `Pull from MA3`
2. user selects one or more MA3 source tracks
3. user selects remote events
4. user maps source tracks to EZ target layers or create-new-layer
5. app sets default import mode to `new_take`
6. app shows diff preview
7. user confirms apply
8. app imports into EZ

Pull rule:

- default import mode is `new_take`
- pull-to-main is explicit and should be deferred or guarded in Alpha

### MA3 UI Rules

- `Push` and `Pull` live on layer or main-row surfaces only
- if a take is selected, the inspector should explain that MA3 transfer uses main only
- transfer status is workflow state, not truth state
- experimental live sync is not part of the Alpha contract

## 11. Engine and Application Boundaries

### Engine Owns

- graph planning
- block execution
- typed input and output handling
- execution ids
- runtime and progress reporting

### Engine Does Not Own

- songs
- song versions
- layers
- takes
- staleness
- promotion or merge
- MA3 semantics
- inspector action language
- UI event bindings

### Application Owns

- truth policy
- pipeline trigger resolution
- output-to-layer mapping
- take append policy
- provenance recording
- stale state updates
- sync transfer workflows
- action resolution for UI objects

### Recommended Service Split

- `TimelineEditService`
  - selection
  - move
  - trim
  - duplicate
  - take promote and merge

- `PipelineRunService`
  - resolve subject
  - load config
  - call engine
  - map outputs
  - persist layers and takes
  - update provenance and stale state

- `SyncTransferService`
  - preview push
  - apply push
  - preview pull
  - apply pull
  - remote snapshot loading

Contract warning:

- `TimelineOrchestrator` is already carrying too much and should not become the permanent god object for Alpha

## 12. Required Presentation Contracts

Alpha needs a shell-level presentation model, not only a timeline presentation.

Minimum recommended types:

- `AppShellPresentation`
  - `project_id`
  - `project_title`
  - `songs`
  - `selected_song_id`
  - `viewed_song_version_id`
  - `timeline`
  - `inspector_target`

- `SongListItemPresentation`
  - `song_id`
  - `title`
  - `order`
  - `is_selected`
  - `active_version_id`
  - `viewed_version_id`
  - `versions`
  - `summary_badges`
  - `available_actions`

- `SongVersionItemPresentation`
  - `song_version_id`
  - `label`
  - `is_active`
  - `is_viewed`
  - `is_editable`
  - `has_results`
  - `pipeline_config_count`
  - `rebuild_mode`

- `SelectionPresentation`
  - `selection_kind`
  - `selection_path`
  - `selection_summary`

- `PipelineActionBinding`
  - `trigger_id`
  - `scope`
  - `subject_kind`
  - `pipeline_config_id`
  - `persist_mode`
  - `enabled`

## 13. Deferred From Alpha

Deliberately defer:

- arbitrary user-extensible UI-event-to-pipeline rule engine
- fully generic automation platform semantics
- rich version remap and arrangement migration tooling
- automatic carry-forward of processed results to new versions
- live sync as a core Alpha truth path
- unconstrained cross-layer event multi-select behavior
- take-level sync actions

## 14. Immediate Build Order

Build in this order:

1. object capability model
2. viewed-version vs active-version contract
3. song import and rerun policy contract
4. shell presentation contract
5. timeline and object info contract
6. MA3 push/pull contract
7. service split for pipeline and transfer orchestration

Do not reverse this order and start with paint or widget composition.

## 15. Decision Checkpoints

These need explicit answers before implementation hardens:

1. Should version switching directly change the working timeline for that version?
2. Should rerun operate on any selected version?
3. Should settings be modeled as song defaults plus version-owned effective copies?
4. Should Alpha import auto-run the default pipeline bundle immediately, while keeping the bundle easy to configure per action?
5. Should `pull to main` be disabled for Alpha?
6. Should versions be shown behind a version chip or dropdown for Alpha?
7. Should copied pipeline configs carry an explicit `pending rerun` state in persistence, or is absence of generated results sufficient for Alpha?

## 16. Current Recommended Answers

Recommended defaults for now:

1. Yes. Version switching should directly change the working timeline.
2. Yes. Rerun should operate on any selected version.
3. Yes. Use song defaults plus version-owned effective copies.
4. Auto-run the developer-configured default bundle on import, with easy add/remove configuration on the song-add action.
5. Yes. Disable or heavily guard `pull to main` for Alpha.
6. Use a version chip or dropdown for Alpha.
7. Add an explicit pending-rerun state if practical; if not, treat it as a near-term follow-up, not a UI-local heuristic.
