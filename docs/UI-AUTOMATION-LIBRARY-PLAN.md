# UI Automation Library Plan

_Updated: 2026-04-16_

This document defines the plan for a reusable desktop UI automation library
that EchoZero can adopt without making the library EchoZero-shaped.

It is the companion to [UI Automation Plan](/Users/march/Documents/GitHub/EchoZero/docs/UI-AUTOMATION-PLAN.md:1),
which remains the EchoZero-specific adoption and rollout plan.

## Goal

Build a reusable library that allows OpenClaw and other agent/runtime tools to
control a desktop app in development the way a human can, without relying on
image matching as the primary control model.

The library must support:

- semantic app snapshots
- stable target discovery
- screenshots
- semantic action invocation
- human-style input when needed
- custom-canvas hit testing
- session lifecycle
- transport-agnostic integration

EchoZero is the first adopter, not the shape of the core API.

## North Star

Use a layered architecture:

1. app-agnostic semantic core
2. framework/app adapter
3. bridge transport
4. agent-facing wrapper

The core owns semantics.
Adapters own framework/runtime details.
Transport owns connectivity.
Wrappers own tool-specific exposure such as MCP.

## Non-Goals

- not a generic image-matching bot
- not an arbitrary scripting shell
- not a Qt-only core package
- not an EchoZero-only test harness
- not a public remote-control service

## Core Principles

### App truth wins

The app must publish its own semantic truth:

- stable target ids
- current selection/focus state
- available actions
- custom-canvas hit targets
- object metadata

The library should not infer these from pixels if the app can expose them.

### Semantic first, human input second

Preferred control order:

1. semantic snapshot/query
2. semantic action invoke
3. human-style input
4. OS/native fallback

### Core stays clean

The reusable core must not depend on:

- Qt
- MCP
- EchoZero widget classes
- app-specific data models

### Safe by default

Runtime exposure should start as:

- dev/test only
- localhost only
- explicit opt-in
- session-scoped
- capability-limited

## Recommended Package Boundary

Do not move this into a separate repo immediately.

Incubate it first as a standalone package boundary inside the EchoZero repo so
we can harden the contract before extraction.

Recommended incubation layout:

```text
packages/
  ui_automation/
    core/
    bridge/
    adapters/
      qt/
      echozero/
    mcp/
```

Why:

- the contract is still moving
- EchoZero is the first real proving ground
- extraction before a second adopter is mostly ceremony

Extraction to its own repo should happen only after:

- the core API is stable
- a second app can adopt it without core changes
- packaging and smoke coverage are proven

## Library Layers

### `core/`

Pure Python semantic model and protocol surface.

Owns:

- session lifecycle models
- target ids and selectors
- snapshot schema
- action schema
- screenshot request/response schema
- hit-target schema
- input command schema
- transport-agnostic provider interface

Suggested top-level types:

- `AutomationSession`
- `AutomationSnapshot`
- `AutomationTarget`
- `AutomationAction`
- `AutomationHitTarget`
- `AutomationProvider`
- `AutomationTransport`

### `bridge/`

Local transport between tooling and a running app.

Owns:

- localhost server/client
- request routing
- session attach/detach
- capability gating
- protocol versioning

Recommended first transport:

- localhost socket or localhost HTTP

Keep WebSocket optional.
Keep the bridge protocol independent from MCP.

### `adapters/qt/`

Framework adapter for Qt apps.

Owns:

- widget/window discovery
- `objectName` and accessible-name harvesting
- Qt event injection
- screenshot capture
- focus/window activation hooks

This layer should still be app-agnostic.

### `adapters/echozero/`

EchoZero-specific adapter.

Owns:

- shell region ids
- timeline semantic targets
- timeline hit-target export
- selection-state export
- action inventory mapping
- transport and inspector semantic exposure

This adapter is where EchoZero-specific meaning belongs.

### `mcp/`

Tool-facing wrapper for agent environments.

Owns:

- MCP tool registration
- request/response translation
- session-oriented tool surface

This layer must remain replaceable.
Do not let MCP semantics leak into `core/`.

## Public API Shape

The reusable API should look like this:

```python
session = AutomationSession.connect(...)
snapshot = session.snapshot()
target = session.find("shell.timeline")
session.click(target)
session.drag(start=..., end=...)
session.type_text("Kick on beat 1")
session.press_key("Space")
image = session.screenshot(target="shell.timeline")
session.invoke("timeline.zoom_in")
session.close()
```

Minimum methods:

- `connect(...)`
- `close()`
- `snapshot()`
- `screenshot(target=None)`
- `find(query)`
- `invoke(action_id, params=None)`
- `click(target_or_point)`
- `drag(start, end)`
- `scroll(target_or_point, dx=0, dy=0)`
- `press_key(key, modifiers=None)`
- `type_text(text)`

Minimum snapshot fields:

- window tree
- visible targets
- focused target
- selected targets
- action inventory
- app semantic state
- canvas hit targets where supported

## EchoZero Adoption Model

EchoZero should be the first embedded adopter.

That means:

- EchoZero runtime implements the provider contract
- EchoZero testing harness becomes proof for the provider contract
- EchoZero-specific timeline semantics live only in the EchoZero adapter

Existing seed surfaces:

- [echozero/testing/app_flow.py](/Users/march/Documents/GitHub/EchoZero/echozero/testing/app_flow.py:1)
- [echozero/testing/gui_dsl.py](/Users/march/Documents/GitHub/EchoZero/echozero/testing/gui_dsl.py:1)
- [echozero/testing/gui_lane_b.py](/Users/march/Documents/GitHub/EchoZero/echozero/testing/gui_lane_b.py:1)
- [echozero/ui/qt/app_shell.py](/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/app_shell.py:1)
- [echozero/ui/qt/timeline/widget.py](/Users/march/Documents/GitHub/EchoZero/echozero/ui/qt/timeline/widget.py:1)

The test harness is the seed vocabulary.
The runtime shell and timeline are the first provider implementation.

## Phased Plan

### Phase 0: Freeze the boundary

Goal:

- define the package boundary and the first provider contract

Deliverables:

- package/module layout
- versioned protocol sketch
- minimal `AutomationProvider` interface
- EchoZero adapter boundary defined

Exit criteria:

- core types are importable without Qt or EchoZero imports

### Phase 1: Read-only semantic surface

Goal:

- make the app inspectable before it is controllable

Deliverables:

- stable target ids/names
- `snapshot`
- `screenshot`
- `get_visible_objects`
- `get_selection_state`
- timeline hit-target export

Exit criteria:

- app-flow/UI proof can validate snapshot stability through the app path

### Phase 2: Controlled input surface

Goal:

- add human-style interaction without breaking the semantic model

Deliverables:

- `click`
- `drag`
- `scroll`
- `press_key`
- `type_text`
- `focus_window`
- `invoke_action`

Exit criteria:

- EchoZero can be driven through major shell and timeline flows without image
  matching

### Phase 3: OpenClaw / MCP wrapper

Goal:

- expose the bridge cleanly to agent tooling

Deliverables:

- MCP adapter
- session lifecycle tools
- target query tools
- screenshot tools
- timeline query tools

Exit criteria:

- OpenClaw can drive a real EchoZero dev workflow end to end

### Phase 4: Extract and generalize

Goal:

- prove the core is truly reusable

Deliverables:

- second adopter
- packaged distribution story
- extraction decision memo

Exit criteria:

- second app can use the core without changing core semantics

## First Implementation Slice

Do not start with full control.

Start with:

1. stable shell and timeline automation ids
2. read-only `snapshot`
3. read-only `screenshot`
4. timeline hit-target export

This gives immediate value for:

- agent inspection
- app verification
- UI debugging
- later safe input automation

## Safety Constraints

- localhost only
- explicit dev/test enablement
- read-only runtime attach first
- no arbitrary command execution
- no public remote control surface
- no image matching as the primary locator model
- no duplicate truth store outside the app

## Main Risks

- custom timeline surfaces may be under-modeled
- Qt discovery may drift if ids are not enforced consistently
- MCP wrapper may try to become the core contract
- runtime bridge may become too permissive
- extraction may happen too early and freeze a bad API

## Proof Strategy

Use EchoZero to prove the library.

Primary proof surfaces:

- [tests/testing/test_app_flow_harness.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_app_flow_harness.py:1)
- [tests/testing/test_gui_dsl.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_gui_dsl.py:1)
- [tests/testing/test_gui_lane_b.py](/Users/march/Documents/GitHub/EchoZero/tests/testing/test_gui_lane_b.py:1)
- [tests/ui/test_app_shell_runtime_flow.py](/Users/march/Documents/GitHub/EchoZero/tests/ui/test_app_shell_runtime_flow.py:1)
- [tests/ui/test_timeline_shell.py](/Users/march/Documents/GitHub/EchoZero/tests/ui/test_timeline_shell.py:1)

Do not call the library “proven reusable” until a second app uses it.

## Recommended Decisions

Make these decisions now:

1. incubate as an internal package boundary inside this repo first
2. keep the core semantic and transport-agnostic
3. make EchoZero the first adapter, not the core model
4. ship read-only semantic control before full input control
5. keep MCP as a wrapper, not a dependency of the core

If these hold, we can build a real reusable desktop app-control library instead
of another one-off EchoZero harness.
