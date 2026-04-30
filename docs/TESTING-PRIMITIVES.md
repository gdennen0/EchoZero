# Testing Primitives

Status: reference
Last reviewed: 2026-04-30



This document defines the canonical primitive catalog for testing, demos, and
automation in EchoZero.
It exists so app proof surfaces share one semantic action vocabulary instead of
growing tool-specific command sets.

## Purpose

EchoZero needs one stable operator-intent contract that can be executed through
different proof surfaces:

- `AppShellRuntime`
- the live automation bridge
- `packages/ui_automation/**`
- in-process app harness support
- deterministic simulated GUI coverage
- future agent wrappers and test tools

The primitive catalog is the contract.
Executors and tools are adapters over that contract.

## Canonical Rule

- Primitives are product/operator intents, not widget details.
- The primitive name must still make sense if the UI layout changes.
- New tools must adapt to existing primitive ids before adding new ids.
- Tool-local aliases are allowed only as compatibility shims.
- The live app automation bridge remains the canonical control plane for real
  app automation.

## Non-Goals

- not a replacement for the app automation bridge
- not a permission to bypass app contracts with direct widget mutation
- not a giant action dump copied from every existing surface
- not a demo-only scripting language

## Naming Rules

Use dotted ids grouped by intent area:

- `app.*`
- `song.*`
- `selection.*`
- `timeline.*`
- `transport.*`
- `transfer.*`
- `sync.*`
- `capture.*`

Rules:

- use verb-first ids inside the namespace
- keep ids stable once published
- prefer one primitive with typed params over many narrow aliases
- do not encode tool names in the primitive id
- do not encode UI implementation details in the primitive id

Good:

- `song.add`
- `timeline.extract_stems`
- `timeline.classify_drum_events`
- `transfer.workspace_open`

Bad:

- `add_song_from_path`
- `click_classify_button`
- `widget_open_push_surface`
- `gui_lane_b_select_first_event`

## Primitive Shape

Every executor should be able to consume the same action envelope:

```python
{
    "action_id": "timeline.classify_drum_events",
    "target": {
        "layer_id": "source_audio",
    },
    "params": {
        "model_path": "/tmp/runtime/model.manifest.json",
    },
}
```

Rules:

- `action_id` is mandatory and stable
- `target` is optional when the action is global
- `params` is optional when the action has no extra inputs
- target selectors should prefer stable semantic ids over display labels

## Primitive Catalog V1

### App

#### `app.new`

Create a new project through the canonical app path.

Params:

- none

#### `app.open`

Open an existing project through the canonical app path.

Params:

- `project_path`

#### `app.save`

Save the current project.

Params:

- none

#### `app.save_as`

Save the current project to a new path.

Params:

- `project_path`

### Song

#### `song.add`

Import a song into the current project from a real file path.

Params:

- `title`
- `audio_path`

Notes:

- this replaces `add_song_from_path`
- it is intentionally about song import, not file-dialog mechanics

### Selection

#### `selection.layer`

Select a layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

#### `selection.event`

Select an event.

Target:

- `event_id`
- `layer_id` when needed to disambiguate

#### `selection.first_event`

Select the first event in a target layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

### Timeline

#### `timeline.extract_stems`

Run stem extraction for a target audio layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

#### `timeline.extract_drum_events`

Run drum event extraction for a target drum-derived layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

#### `timeline.classify_drum_events`

Run drum classification for a target drum-derived layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

Params:

- `model_path`

Rules:

- the model path may point to a resolved model file or manifest path
- tests should prefer generated temp assets over repo-tracked prebuilt files

#### `timeline.extract_classified_drums`

Build kick/snare-classified layers from a target drum-derived layer.

Target:

- `layer_id` preferred
- `layer_title` allowed as a compatibility selector in tests

#### `timeline.nudge_selection`

Move selected event data left or right.

Params:

- `direction` in `left|right`
- `steps` optional, integer `>= 1`

#### `timeline.duplicate_selection`

Duplicate the current selection.

Params:

- `steps` optional, integer `>= 1`

### Transport

#### `transport.play`

Start transport playback.

#### `transport.pause`

Pause transport playback.

#### `transport.stop`

Stop transport playback.

### Transfer

#### `transfer.workspace_open`

Open the transfer workspace for a target layer.

Target:

- `layer_id` preferred

Params:

- `direction` in `push|pull`

Notes:

- this replaces `open_push_surface` and `open_pull_surface`

#### `transfer.plan_preview`

Preview the current transfer plan.

#### `transfer.plan_apply`

Apply the current transfer plan.

#### `transfer.plan_cancel`

Cancel the current transfer plan.

### Sync

#### `sync.enable`

Enable live sync.

#### `sync.disable`

Disable live sync.

### Capture

#### `capture.screenshot`

Capture a screenshot artifact for the current app state or a target region.

Params:

- `filename` for artifact-oriented lanes
- `target_id` optional for target-scoped capture

Notes:

- this is a proof artifact primitive, not a product operator intent
- keep capture primitives separate from product actions when possible

## Compatibility Map

Current names should converge toward the catalog above.
Adapters may continue to accept the old names while the repo migrates.

| Current Name | Canonical Primitive |
| --- | --- |
| `add_song_from_path` | `song.add` |
| `extract_stems` | `timeline.extract_stems` |
| `extract_drum_events` | `timeline.extract_drum_events` |
| `classify_drum_events` | `timeline.classify_drum_events` |
| `extract_classified_drums` | `timeline.extract_classified_drums` |
| `select_first_event` | `selection.first_event` |
| `nudge` | `timeline.nudge_selection` |
| `nudge_selected_events` | `timeline.nudge_selection` |
| `duplicate` | `timeline.duplicate_selection` |
| `duplicate_selected_events` | `timeline.duplicate_selection` |
| `open_push_surface` | `transfer.workspace_open` with `direction=push` |
| `open_pull_surface` | `transfer.workspace_open` with `direction=pull` |
| `apply_transfer_plan` | `transfer.plan_apply` |
| `enable_sync` | `sync.enable` |
| `disable_sync` | `sync.disable` |
| `screenshot` | `capture.screenshot` |

## Adapter Rules

Executors may differ in fidelity, but not in vocabulary.

Allowed executors:

- runtime executor
- live bridge executor
- `ui_automation` executor
- in-process harness executor
- simulated GUI executor
- future OpenClaw or MCP wrappers

Rules:

- an executor may reject a primitive if the app state does not support it
- an executor must not silently reinterpret a primitive into a different intent
- executor-specific extras belong in observations or artifacts, not in renamed
  primitives

## Selector Rules

Selectors must converge on semantic ids.

Preferred order:

1. stable object id
2. stable target id
3. stable layer id or event id
4. display label only as a fallback for compatibility

This keeps scenarios resilient when labels, layout, or presentation wording
change.

## Versioning Rules

This catalog is `v1`.

Rules:

- adding a new primitive is allowed when an existing primitive cannot express
  the operator intent cleanly
- renaming a published primitive requires a compatibility alias first
- removing an alias is a cleanup task, not an incidental edit
- scenario formats and providers should declare which primitive catalog version
  they implement once version negotiation exists

## Immediate Migration Target

The near-term standardization target is:

1. keep the live bridge and `packages/ui_automation/**` as the public app
   automation surface
2. teach providers and harnesses to accept canonical primitive ids
3. keep old names as aliases during migration
4. update GUI DSL and demo/test scenarios to emit canonical primitive ids
5. reject new tool-specific action vocabularies unless they extend this catalog

## Out Of Scope For This Doc

- observation/snapshot schema
- executor protocol details
- scenario file schema redesign
- bridge protocol version negotiation

Those should be defined in follow-on docs once the primitive vocabulary is
accepted.
