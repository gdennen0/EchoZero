# Object Pipeline Action Architecture

Status: reference
Last reviewed: 2026-04-30



This document defines the intended object-action architecture for EchoZero.
It exists to restore the original plan: object workflows should be driven by
pipelines assigned to objects, not bespoke runtime helper methods.

## Problem Statement

EchoZero currently has a split model:

- object and inspector surfaces advertise object-scoped actions
- UI and automation harnesses can enumerate those actions
- but execution still falls through a bespoke runtime action layer

Today, the action inventory is object-shaped, but the execution path is still
partly imperative.

That drift shows up most clearly in:

- [echozero/application/presentation/inspector_contract.py](../echozero/application/presentation/inspector_contract.py)
- [echozero/ui/qt/timeline/widget_actions.py](../echozero/ui/qt/timeline/widget_actions.py)
- [echozero/ui/qt/app_shell.py](../echozero/ui/qt/app_shell.py)

## Current State

### Where object actions are defined today

The canonical action inventory is built in the inspector contract layer.

Key location:

- [echozero/application/presentation/inspector_contract.py](../echozero/application/presentation/inspector_contract.py)

Concrete facts:

- `InspectorAction` is the object-facing action descriptor
- `build_timeline_inspector_contract(...)` chooses actions based on the current
  selected or hit object
- object actions are already exposed to UI and automation surfaces through the
  contract

Examples:

- `song.add`
- `timeline.extract_stems`
- `timeline.extract_drum_events`
- `timeline.classify_drum_events`
- `timeline.extract_classified_drums`
- `push_to_ma3`
- `pull_from_ma3`
- `transfer.plan_preview`
- `transfer.plan_apply`

### Where action execution bypasses the intended model

The current execution path is:

1. object contract exposes an `InspectorAction`
2. widget/action router switches on `action_id`
3. router calls a runtime helper or dispatches a special-case intent
4. runtime helper manually chooses a pipeline id plus bindings

Primary bypass points:

- [echozero/ui/qt/timeline/widget_actions.py](../echozero/ui/qt/timeline/widget_actions.py)
- [echozero/ui/qt/app_shell.py](../echozero/ui/qt/app_shell.py)

The clearest examples are:

- [AppShellRuntime.extract_stems](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.extract_drum_events](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.classify_drum_events](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.extract_classified_drums](../echozero/ui/qt/app_shell.py)

These methods are pipeline-backed, but not pipeline-driven at the contract
boundary.

### Why this is architectural drift

The current runtime methods do these jobs at once:

- validate object eligibility
- resolve object-derived bindings
- choose the pipeline template id
- invoke `AnalysisService.analyze(...)`
- refresh/persist presentation state

That means the app shell is acting as:

- object action registry
- object-to-pipeline mapper
- execution coordinator

The original plan was for pipelines to own object-action workflows.
The current shape leaves pipelines subordinate to bespoke runtime methods.

## What Is Already Good

The current codebase already has the parts needed for the correct model.

### 1. A pipeline registry exists

- [echozero/pipelines/registry.py](../echozero/pipelines/registry.py)

### 2. Pipeline templates exist for the relevant workflows

- [echozero/pipelines/templates/drum_classification.py](../echozero/pipelines/templates/drum_classification.py)
- [echozero/pipelines/templates/extract_classified_drums.py](../echozero/pipelines/templates/extract_classified_drums.py)

### 3. The orchestrator already supports template-id plus bindings execution

- [echozero/services/orchestrator.py](../echozero/services/orchestrator.py)

The missing piece is a first-class object-action-to-pipeline layer.

## Target Model

The correct model is:

1. objects expose available actions
2. each action resolves to a pipeline-backed workflow descriptor
3. executors invoke a generic object action runner
4. the runner resolves bindings from object context and executes the assigned
   pipeline

That means EchoZero should move to:

- action descriptors owned by object contracts
- pipeline assignment owned by action/workflow descriptors
- generic object action execution path
- minimal runtime special cases

## Canonical Design

### Object action descriptor

The object-facing contract should advertise actions that include pipeline-backed
workflow metadata.

Preferred shape:

```python
@dataclass(slots=True, frozen=True)
class ObjectActionDescriptor:
    action_id: str
    label: str
    object_types: tuple[str, ...]
    workflow_id: str
    enabled: bool = True
    params_schema: dict[str, object] = field(default_factory=dict)
    static_params: dict[str, object] = field(default_factory=dict)
```

Important rule:

- `workflow_id` is the stable execution identity
- `action_id` is the stable operator/harness identity
- multiple objects may expose the same action id but produce different resolved
  bindings

### Pipeline-backed workflow descriptor

Object actions should resolve to a workflow descriptor, not directly to an app
shell method.

Preferred shape:

```python
@dataclass(slots=True, frozen=True)
class PipelineWorkflowDescriptor:
    workflow_id: str
    pipeline_template_id: str
    binding_resolver: str
    persistence_policy: str
```

Where:

- `pipeline_template_id` maps to the pipeline registry
- `binding_resolver` derives knob bindings from the object and runtime context
- `persistence_policy` defines how outputs refresh object state

### Generic object action executor

The public execution path should become:

```python
run_object_action(
    object_id=...,
    object_type=...,
    action_id=...,
    params=...,
)
```

Responsibilities:

- resolve the current object contract
- resolve the requested action descriptor
- validate params against the action descriptor
- resolve a workflow descriptor
- compute pipeline bindings from object context
- execute the pipeline via `AnalysisService`
- refresh storage and presentation

This should replace bespoke methods like:

- `extract_stems(...)`
- `extract_drum_events(...)`
- `classify_drum_events(...)`
- `extract_classified_drums(...)`

for object-scoped workflows

## Concrete Current-To-Target Mapping

### Current

`timeline.extract_stems`

- declared in inspector contract
- routed in `widget_actions`
- executed by `AppShellRuntime.extract_stems(...)`
- chooses `stem_separation` directly

### Target

`timeline.extract_stems`

- declared in inspector contract with workflow metadata
- routed through generic object action execution
- resolves workflow `layer.audio.extract_stems`
- workflow maps to pipeline template `stem_separation`
- binding resolver derives source audio from the selected layer object

### Current

`timeline.classify_drum_events`

- declared in inspector contract
- routed in `widget_actions`
- executed by `AppShellRuntime.classify_drum_events(...)`
- chooses `drum_classification` directly

### Target

`timeline.classify_drum_events`

- declared in inspector contract with workflow metadata
- generic executor validates `model_path`
- workflow resolver derives `audio_file` from the layer object
- workflow maps to pipeline template `drum_classification`

## Binding Resolver Concept

The critical missing abstraction is the binding resolver.

The resolver should own object-derived parameter filling such as:

- layer source audio path
- current song version id
- installed runtime bundle paths
- transfer or sync context defaults

That logic should not live in `AppShellRuntime` methods.

Preferred shape:

```python
class WorkflowBindingResolver(Protocol):
    def resolve(
        self,
        *,
        timeline: Timeline,
        session: Session,
        object_id: str,
        object_type: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        ...
```

Examples:

- `drum_layer_audio_binding_resolver`
- `binary_drum_bundle_binding_resolver`
- `manual_push_transfer_binding_resolver`

## Harness and Automation Implication

This model gives the harness exactly what it should have:

- every interactable object exposes a public list of actions
- those actions are the same actions the operator sees
- the harness does not need a separate action inventory
- the harness can generically invoke object actions without knowing runtime
  helper names

That aligns with:

- [packages/ui_automation/src/ui_automation/adapters/echozero/provider.py](../packages/ui_automation/src/ui_automation/adapters/echozero/provider.py)

The provider should not need to know about:

- `extract_stems(...)`
- `classify_drum_events(...)`
- `extract_classified_drums(...)`

It should only need:

- object action inventory
- generic object action invocation

## What Should Remain As True Runtime Commands

Not everything is an object pipeline action.

These should remain app/runtime commands:

- `app.new`
- `app.open`
- `app.save`
- `app.save_as`
- `transport.play`
- `transport.pause`
- `transport.stop`

Those are application lifecycle or transport commands, not object workflows.

Object-scoped operations should move out of bespoke runtime methods.

## Refactor Plan

### 1. Introduce workflow descriptors

Create a registry for object-action workflow descriptors.

Scope:

- object action to workflow mapping
- workflow id to pipeline template mapping
- workflow binding resolvers

### 2. Add generic object action execution

Add a single app-shell or application-boundary method:

- `run_object_action(object_id, object_type, action_id, params)`

This becomes the only object workflow execution path.

### 3. Enrich `InspectorAction`

Either:

- add workflow metadata directly to `InspectorAction`

or:

- keep `InspectorAction` lean and resolve workflow metadata from a parallel
  registry keyed by `object_type + action_id`

Preferred approach:

- keep `InspectorAction` operator-facing and lean
- resolve workflow metadata from a separate registry

### 4. Replace widget/runtime special cases

Replace action-specific runtime branches in:

- [echozero/ui/qt/timeline/widget_actions.py](../echozero/ui/qt/timeline/widget_actions.py)

Specifically remove special handling for:

- `timeline.extract_stems`
- `timeline.extract_drum_events`
- `timeline.classify_drum_events`
- `timeline.extract_classified_drums`

and route them all through generic object action execution.

### 5. Delete bespoke object workflow methods

Delete these after the generic path is proven:

- [AppShellRuntime.extract_stems](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.extract_drum_events](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.classify_drum_events](../echozero/ui/qt/app_shell.py)
- [AppShellRuntime.extract_classified_drums](../echozero/ui/qt/app_shell.py)

### 6. Make automation consume object actions only

The automation provider should:

- read object-advertised actions from contracts
- call the generic object action executor
- stop carrying pipeline-action-specific dispatch branches

## Deletion Plan

The runtime action layer should be reduced, not expanded.

### Keep

- project lifecycle methods
- transport methods
- sync enable/disable methods
- raw `dispatch(intent)` for timeline intents

### Delete after migration

- object-scoped pipeline helper methods on `AppShellRuntime`
- object-workflow-specific branches in `widget_actions`
- object-workflow-specific branches in `ui_automation` provider

## Acceptance Criteria

This architecture is in place when:

- every object-facing workflow action is pipeline-backed by descriptor
- the harness can enumerate actions per object without a separate runtime map
- the public automation layer does not hardcode per-pipeline branches
- the app shell does not expose bespoke object workflow helpers
- object actions execute through one generic object-action path

## Bottom Line

Your concern is correct.

EchoZero already exposes object actions as contracts, but it still executes many
object workflows through a bespoke runtime action layer.

That is transitional architecture, not the intended end state.
The correct model is:

- objects advertise actions
- actions resolve to pipeline-backed workflows
- a generic executor runs those workflows from object context

The runtime helper layer for object workflows should be considered deletion
target, not permanent architecture.
