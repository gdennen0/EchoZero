# Testing Executors

_Updated: 2026-04-18_

This document defines the canonical executor interface for EchoZero testing,
demos, and automation.
It exists so the primitive catalog has one execution contract across real app,
test harness, and future tool adapters.

## Purpose

The primitive catalog in
[docs/TESTING-PRIMITIVES.md](/Users/march/Documents/GitHub/EchoZero/docs/TESTING-PRIMITIVES.md:1)
defines what actions mean.
This document defines how an executor consumes those actions and reports
results.

The design goal is extension without ad hoc interfaces.

## Canonical Rule

- Primitive ids are the stable semantic contract.
- Executors differ by fidelity and transport, not by action vocabulary.
- Real app automation remains the canonical proof surface.
- Simulated or internal executors may exist, but they must implement the same
  primitive contract.

## Executor Model

An executor is a component that accepts canonical primitive requests and returns
canonical observations.

Preferred type shape:

```python
class OperatorExecutor(Protocol):
    def snapshot(self) -> OperatorObservation:
        ...

    def invoke(self, request: OperatorActionRequest) -> OperatorObservation:
        ...

    def screenshot(self, request: CaptureRequest | None = None) -> bytes:
        ...

    def close(self) -> None:
        ...
```

This is intentionally close to the existing `AutomationSessionBackend` shape in
`packages/ui_automation/core`, but makes the primitive request envelope
explicit.

## Canonical Request Envelope

```python
@dataclass(slots=True, frozen=True)
class OperatorActionRequest:
    action_id: str
    target: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

Rules:

- `action_id` must use the canonical primitive id
- `target` carries semantic selectors such as `layer_id` or `event_id`
- `params` carries action-specific inputs such as `model_path`
- `metadata` is reserved for executor-neutral trace data, not new semantics

## Canonical Observation Envelope

```python
@dataclass(slots=True, frozen=True)
class OperatorObservation:
    snapshot: AutomationSnapshot
    status: str = "ok"
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
```

Rules:

- `snapshot` is the current semantic app state
- `artifacts` holds executor results such as saved screenshot paths
- `warnings` communicates degraded but non-fatal execution
- failures should be explicit exceptions or structured error returns, never
  silent fallback

## Capability Model

Executors may vary in what they can do, but capabilities must be declared in a
stable way.

Recommended capability surface:

```python
@dataclass(slots=True, frozen=True)
class ExecutorCapabilities:
    can_invoke_primitives: bool
    can_pointer_input: bool
    can_keyboard_input: bool
    can_capture_screenshots: bool
    is_simulated: bool
    is_real_app_path: bool
```

Rules:

- a real app bridge executor should be `is_real_app_path=True`
- GUI Lane B style executors should be `is_simulated=True`
- capability differences do not justify alternate primitive ids

## Preferred Integration With Current Code

The current reusable automation layer already provides a near-match:

- `AutomationSessionBackend.snapshot()`
- `AutomationSessionBackend.invoke(action_id, target_id=None, params=None)`
- `AutomationSessionBackend.screenshot()`
- `AutomationSessionBackend.close()`

The migration target is:

1. keep `AutomationSessionBackend` as the transport-facing surface
2. add a canonical request adapter at the `ui_automation` boundary
3. normalize canonical primitive ids before provider dispatch
4. keep legacy aliases only as compatibility shims

## Executor Categories

### Real App Executor

Use for:

- app-path proof
- agent-driven app control
- launch/runtime/operator flows

Canonical examples:

- live automation bridge
- `packages/ui_automation/adapters/echozero/live_client.py`

### In-Process Harness Executor

Use for:

- fast app-boundary tests
- deterministic setup during development

Canonical examples:

- `HarnessEchoZeroAutomationProvider`
- `echozero/testing/app_flow.py`

Constraint:

- this is internal support, not the public control plane

### Simulated GUI Executor

Use for:

- deterministic GUI regression coverage
- artifact review

Canonical examples:

- `echozero/testing/gui_lane_b.py`

Constraint:

- must be labeled simulated proof

### Legacy Scenario Executor

Use for:

- transitional migration only

Canonical examples:

- `echozero/testing/e2e/**`

Constraint:

- do not treat as the long-term public control surface

## Alias Resolution Rules

Executors may accept aliases, but alias handling must be centralized.

Rules:

- resolve aliases before dispatch
- log or expose the canonical resolved id in debug output
- do not implement alias translation separately in every test tool

Preferred helper shape:

```python
def resolve_primitive_id(action_id: str) -> str:
    ...
```

Examples:

- `add_song_from_path` -> `song.add`
- `open_push_surface` -> `transfer.workspace_open`
- `duplicate` -> `timeline.duplicate_selection`

## Error Rules

- unsupported primitive: raise a clear unsupported-action error
- invalid target: raise a clear target-resolution error
- invalid params: raise a clear validation error
- unavailable runtime state: raise a clear state-precondition error

Do not:

- silently ignore unsupported params
- reinterpret a missing target as a different target
- downgrade real-app proof to simulated proof without saying so

## Immediate Implementation Target

The next concrete implementation steps should be:

1. introduce canonical primitive alias resolution in `packages/ui_automation`
2. adapt the EchoZero provider to expose canonical ids in `AutomationAction`
3. keep old ids accepted during migration
4. update internal harness/scenario executors to consume canonical ids first

## Out Of Scope

- scenario file format details
- observation query/assert DSL
- bridge transport wire format versioning

Those belong in follow-on docs and code changes.
