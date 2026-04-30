# Testing Scenario Schema

Status: reference
Last reviewed: 2026-04-30



This document defines the preferred scenario file shape for EchoZero tests,
demos, and automation flows.
It exists so scenarios compose canonical primitives instead of embedding
tool-specific commands.

## Purpose

A scenario is a declarative flow built from:

- canonical primitive actions
- explicit assertions
- explicit capture steps
- explicit waits

The scenario format should be executor-agnostic.
Different executors may run the same scenario at different fidelity levels.

## Canonical Rule

- scenarios reference canonical primitive ids from
  [docs/TESTING-PRIMITIVES.md](TESTING-PRIMITIVES.md)
- scenarios do not invent executor-local action names
- selector shape must prefer semantic ids over labels
- simulated-only steps must be labeled as such in metadata if they exist

## Preferred Top-Level Shape

```json
{
  "schema": "echozero.testing.scenario.v1",
  "name": "Classify Drum Events",
  "description": "Import a song, extract stems, derive drum events, and classify them.",
  "metadata": {
    "proof_class": "app_path",
    "requires_real_app": false,
    "catalog_version": "v1"
  },
  "steps": []
}
```

Required fields:

- `schema`
- `name`
- `steps`

Optional fields:

- `description`
- `metadata`

## Step Kinds

The preferred step kinds are:

- `invoke`
- `assert`
- `capture`
- `wait`

These map closely to the current legacy E2E scenario model, but the action step
must use canonical primitives.

## Invoke Step

```json
{
  "kind": "invoke",
  "name": "Extract stems",
  "action_id": "timeline.extract_stems",
  "target": {
    "layer_id": "source_audio"
  },
  "params": {}
}
```

Rules:

- `action_id` is required
- `target` is optional for global actions
- `params` is optional
- avoid overloading `params` with selector data when `target` is clearer

## Assert Step

```json
{
  "kind": "assert",
  "name": "Classified layer exists",
  "query": "target.exists",
  "target": {
    "target_id": "timeline.layer:drum_classified_events"
  },
  "expected": true,
  "comparator": "equals"
}
```

Rules:

- assertions must be about semantic state, not screenshots alone
- screenshot-based review is supporting proof, not primary correctness proof

## Capture Step

```json
{
  "kind": "capture",
  "name": "Post-classification screenshot",
  "artifact": "screenshot",
  "params": {
    "filename": "post-classification.png",
    "target_id": "shell.timeline"
  }
}
```

Rules:

- use capture only for proof artifacts or review surfaces
- do not encode product behavior into capture-only steps

## Wait Step

```json
{
  "kind": "wait",
  "name": "Wait for classified layer",
  "until_query": "target.exists",
  "target": {
    "target_id": "timeline.layer:drum_classified_events"
  },
  "expected": true,
  "timeout_ms": 1000,
  "poll_interval_ms": 50
}
```

Rules:

- waits should prefer semantic readiness checks over arbitrary time delays
- raw duration waits are fallback only

## Selector Rules

Preferred selector order:

1. `target_id`
2. domain id such as `layer_id` or `event_id`
3. display label fallback such as `layer_title`

Bad:

```json
{
  "layer_title": "Drums"
}
```

Better:

```json
{
  "layer_id": "drums_layer"
}
```

## Metadata Rules

Recommended metadata keys:

- `proof_class`: `app_path|simulated|fixture`
- `requires_real_app`: boolean
- `catalog_version`: primitive catalog version
- `executor_hints`: optional executor-routing hints

Example:

```json
{
  "metadata": {
    "proof_class": "simulated",
    "requires_real_app": false,
    "catalog_version": "v1",
    "executor_hints": {
      "preferred": "gui_lane_b"
    }
  }
}
```

## Compatibility Guidance

Current scenario surfaces are split:

- `echozero/testing/gui_dsl.py` uses a custom `action` list
- `echozero/testing/e2e/scenario.py` uses `act/assert/capture/wait`
- demo suites use Python functions and presentation mutation helpers

The migration target is:

1. keep `act/assert/capture/wait` as the conceptual step family
2. rename `act` to `invoke` in the canonical schema
3. move all action references to canonical primitive ids
4. keep existing loaders as compatibility readers during transition

## Scenario Example

```json
{
  "schema": "echozero.testing.scenario.v1",
  "name": "Drum Classification Flow",
  "metadata": {
    "proof_class": "app_path",
    "requires_real_app": false,
    "catalog_version": "v1"
  },
  "steps": [
    {
      "kind": "invoke",
      "name": "Import song",
      "action_id": "song.add",
      "params": {
        "title": "Automation Flow Song",
        "audio_path": "__RUN_TEMP__/fixtures/import.wav"
      }
    },
    {
      "kind": "invoke",
      "name": "Extract stems",
      "action_id": "timeline.extract_stems",
      "target": {
        "layer_id": "source_audio"
      }
    },
    {
      "kind": "invoke",
      "name": "Extract drum events",
      "action_id": "timeline.extract_drum_events",
      "target": {
        "layer_title": "Drums"
      }
    },
    {
      "kind": "invoke",
      "name": "Classify drum events",
      "action_id": "timeline.classify_drum_events",
      "target": {
        "layer_title": "Drums"
      },
      "params": {
        "model_path": "__RUN_TEMP__/fixtures/drum-model.pth"
      }
    },
    {
      "kind": "assert",
      "name": "Classified layer exists",
      "query": "target.exists",
      "target": {
        "target_id": "timeline.layer:drum_classified_events"
      },
      "expected": true,
      "comparator": "equals"
    },
    {
      "kind": "capture",
      "name": "Post-classification screenshot",
      "artifact": "screenshot",
      "params": {
        "filename": "lane-b-post-classification.png"
      }
    }
  ]
}
```

## Out Of Scope

- the full query language for asserts
- executor transport selection
- human-only manual verification steps

Those should be added only after the primitive and executor contracts are in
place.
