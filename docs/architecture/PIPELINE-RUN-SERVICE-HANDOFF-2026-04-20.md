# Pipeline Run Service Handoff — 2026-04-20

Status: reference
Last reviewed: 2026-04-30


## Status And Authority

This is a coordination note for implementation work.
It is not a new canonical architecture spec.

If anything here conflicts with canonical repo docs, the canonical docs win:

- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
- `docs/UNIFIED-IMPLEMENTATION-PLAN.md`
- `docs/architecture/DECISIONS.md`
- `docs/ALPHA-UI-CONTRACT.md`
- `docs/OBJECT-PIPELINE-ACTION-ARCHITECTURE.md`

## Problem Summary

EchoZero currently has the right high-level ingredients for pipeline-backed
object actions, but the live Stage Zero run path is still transitional.

Today, pipeline-backed object actions are exposed as application contracts, but
execution still falls through a synchronous runtime helper path in the Qt shell.
That makes long-running actions feel hung, and it also keeps execution
ownership in the wrong layer.

Current hot spots:

- `echozero/ui/qt/timeline/widget_actions.py`
- `echozero/ui/qt/app_shell.py`
- `echozero/application/timeline/object_action_settings_service.py`

## Important Conclusion

A simple async patch in the current UI path is not the best permanent fix.

The permanent fix should be:

- a first-class application-owned `PipelineRunService`
- a generic object-action run path
- transient run state owned by the application layer, not by widgets
- thin Qt adapters that request runs and render status

This matches the direction already called for in:

- `docs/OBJECT-PIPELINE-ACTION-ARCHITECTURE.md`
- `docs/ALPHA-UI-CONTRACT.md`
- `docs/architecture/DECISIONS.md`

## What Should Be Ripped Up

The following shape should be treated as transitional and reduced over time:

- synchronous execution directly from `TimelineWidgetActionRouter`
- `AppShellRuntime` owning object-workflow execution semantics
- `ObjectActionSettingsService` owning both settings behavior and live run
  execution
- widget-local run orchestration or ad hoc per-action threading

Specifically, do not keep expanding:

- object-workflow-specific branches in `widget_actions.py`
- object-scoped pipeline helper methods on `AppShellRuntime`
- synchronous `run_object_action(...)` behavior that blocks until refresh

## What Should Be Kept

These parts are aligned with the target model and should be reused:

- object-action descriptors and workflow descriptors
- pipeline registry and template model
- `echozero/services/orchestrator.py`
- `echozero/execution.py`
- `echozero/progress.py`
- existing provenance and take append behavior

Foundry is also useful as a UX reference for background status, but not as the
permanent architecture. Foundry currently demonstrates good responsiveness and
status surfaces, but its worker lifecycle is still UI-managed.

## Permanent Target Model

The desired run flow is:

1. UI emits an object action request.
2. Application resolves the object action and workflow descriptor.
3. Application creates a transient run record and returns a `run_id`
   immediately.
4. Application executes the run off the UI thread.
5. Engine and orchestrator progress are translated into app-owned run state.
6. On completion, application maps outputs, persists results, refreshes
   presentation state, and marks the run complete.
7. UI observes run state and renders status surfaces without owning execution.

The key point is that run lifecycle is application state, not widget state.

## Proposed Service Split

Use the split already implied by `docs/ALPHA-UI-CONTRACT.md`:

- `TimelineEditService`
- `PipelineRunService`
- `SyncTransferService`

For this work, the focus is `PipelineRunService`.

Reasonable first home:

- `echozero/application/timeline/pipeline_run_service.py`

If it grows, split it into a small cluster under:

- `echozero/application/timeline/pipeline_runs/*`

## Responsibilities Of PipelineRunService

`PipelineRunService` should own:

- resolving run subject from `action_id`, `object_id`, and `object_type`
- validating the action and runtime params
- resolving the workflow descriptor
- loading or creating the correct pipeline config
- invoking the orchestrator/engine in background execution
- translating orchestrator and engine progress into transient run state
- enforcing non-global run locks for conflicting subjects
- handling cancellation if supported
- refreshing storage/presentation on completion
- surfacing success/failure to the application and UI

It should not own:

- widget updates
- painting
- timeline truth semantics outside normal application mapping
- persisted layer/take truth beyond the existing output mapping path

## Recommended Run State Model

Create an explicit transient run model owned by the application layer.

Possible shape:

```python
@dataclass(slots=True)
class PipelineRunState:
    run_id: str
    action_id: str
    workflow_id: str
    object_id: str
    object_type: str
    source_layer_id: str | None
    status: str  # queued | resolving | running | persisting | completed | failed | cancelled
    message: str
    percent: float | None
    started_at: float
    finished_at: float | None = None
    can_cancel: bool = False
    error: str | None = None
    output_layer_ids: list[str] = field(default_factory=list)
```

This state should be transient and app-owned.

Acceptable homes:

- session-owned transient state
- a dedicated run store owned by the application runtime

Not acceptable:

- persisting in-flight run state into `LayerStatus`
- using layer titles/badges as truth heuristics
- Qt-local variables as the canonical source of execution state

## Progress Reporting

Use both existing seams:

1. Orchestrator-level coarse progress
   - `Loading configuration`
   - `Preparing pipeline`
   - `Executing pipeline`
   - `Persisting results`
   - `Complete`

2. Engine-level fine-grained progress
   - `RuntimeBus`
   - per-block start/progress/complete reports

The service should aggregate these into one app-facing run state model.

Qt can choose to render:

- a shell status line
- timeline-local status chips
- disabled/running action buttons
- optional block-detail text

But Qt should not be the source of truth for the run lifecycle.

## Threading And Ownership Rules

Keep the decisions already locked in `docs/architecture/DECISIONS.md`:

- execution must be non-blocking
- UI must stay responsive
- locks must not be global
- processor execution must not touch Qt directly

Recommended ownership:

- application service starts background work
- background work emits app-level updates
- Qt observes and renders those updates

Do not push more orchestration into:

- `TimelineWidgetActionRouter`
- `AppShellRuntime`
- dialog classes

## Object Action And Settings Split

`ObjectActionSettingsService` currently does too much.

The clean long-term split is:

- settings service owns settings resolution, scope switching, copy-preview, and
  save behavior
- run service owns execution

That means:

- settings dialog `Run` should request a run through `PipelineRunService`
- direct object action buttons should request a run through the same service
- both entry points should land on one generic object-action run path

## Recommended UI Behavior

The UI should make long runs visibly alive without inventing new truth.

Recommended minimum surfaces:

- shell-level run/status line
- timeline-local status surface near the source layer or action surface
- action button state changes such as `Run` -> `Running...`

Nice-to-have later:

- cancel button
- multi-run queue surface
- expanded per-block detail

The UI contract already leaves room for timeline-local status surfaces.

## Suggested Migration Sequence

1. Extract execution out of the synchronous UI path.
2. Introduce a transient app-owned run state model.
3. Create `PipelineRunService` with a generic `request_run(...) -> run_id` API.
4. Route both direct object-action buttons and settings-dialog runs through that
   API.
5. Feed orchestrator and runtime-bus progress into the run state store.
6. Refresh storage/presentation on completion through the application layer.
7. Delete bespoke object-workflow helpers and branches as they become unused.

## Things To Avoid

Do not:

- add more pipeline-specific routing branches in Qt
- make `AppShellRuntime` the long-term execution coordinator
- persist in-flight status into layer truth
- use a global execution lock
- treat Foundry's UI-managed worker approach as the final architecture for the
  main timeline shell
- hide the problem with a spinner while leaving the synchronous ownership model
  intact

## Acceptance Criteria

This work is in the right place when:

- object actions still come from application contracts
- all object-action execution goes through one generic app-layer run path
- the Qt shell stays responsive during runs
- run status is visible without relying on widget-local truth
- progress is driven by orchestrator/runtime reporting, not fake timers
- results still map into stable layers/takes with correct provenance
- non-main/take truth rules remain untouched
- MA3 sync boundaries remain untouched
- obsolete helper methods and branches are removed, not left as parallel paths

## Tests And Proof Expectations

At minimum, add coverage for:

- `request_run(...)` returns immediately with a `run_id`
- conflicting runs are blocked only at the intended subject scope
- progress transitions move through expected states
- completion refreshes presentation correctly
- failure state is observable and does not silently disappear
- rerun output mapping still appends or maps takes/layers correctly
- provenance and stale behavior remain correct

Visible behavior changes should also include app-path proof, not only unit
tests.

## Short Version

The permanent fix is not "add a spinner."

The permanent fix is:

- pull run execution out of the Qt shell
- make pipeline runs a first-class application service
- keep run state transient and app-owned
- let Qt render status as a thin adapter

That is the clean long-term architecture and the right place to invest now.
