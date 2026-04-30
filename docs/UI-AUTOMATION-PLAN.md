# UI Automation Plan

Status: reference
Last reviewed: 2026-04-30



This document defines how EchoZero should expose full app control for OpenClaw
and agents.
It exists to turn the current test harness into a real app-control layer that
can drive the desktop app in development the way a human can.
It connects existing app-flow testing, Qt runtime surfaces, and future
OpenClaw integration into one plan.

The concrete execution board for this redesign lives in
[UI Automation Implementation Board](UI-AUTOMATION-IMPLEMENTATION-BOARD.md).

The short policy for choosing canonical versus internal-only automation surfaces
lives in
[Automation Surface Policy](AUTOMATION-SURFACE-POLICY.md).

For the reusable library boundary that should sit under this app-specific
adoption path, see [UI Automation Library Plan](UI-AUTOMATION-LIBRARY-PLAN.md).

## Goal

Enable OpenClaw to control EchoZero in development with near-human coverage:

- launch the app
- inspect what is on screen
- click, type, press keys, drag, scroll
- trigger app actions and dialogs
- read application state
- capture screenshots
- drive timeline/canvas interactions

This is primarily for:

- UI development
- UI verification
- debugging
- scenario automation
- agent-supervised iteration

## Important Constraint

EchoZero is a Qt desktop app with custom-painted timeline/canvas surfaces.

That means a generic browser-style automation layer is not enough.
To get reliable control, the right approach is:

- semantic app automation where we own the app
- human-style input where needed
- OS-level accessibility only as a fallback

## Current Starting Point

EchoZero already has strong seeds for this:

- `echozero/testing/app_flow.py`
- `echozero/testing/gui_dsl.py`
- `echozero/testing/gui_lane_b.py`
- `tests/testing/test_gui_dsl.py`
- `tests/testing/test_gui_lane_b.py`
- `tests/ui/test_timeline_shell.py`
- `tests/ui/test_app_shell_runtime_flow.py`

These prove that we already have:

- a canonical app runtime harness
- scenario-driven GUI actions
- deterministic sync/test doubles
- some app-level UI exercise paths

So the problem is not “can we start.”
The problem is “how do we promote this into a stable control plane for
OpenClaw.”

## Recommended Architecture

Use a three-layer model.

### 1. Semantic control plane

This is the preferred control layer for development.

The app should expose a structured automation surface with:

- window and panel tree
- stable automation ids
- accessible names/labels
- action inventory
- selected object/state snapshot
- screenshot hooks
- timeline hit-target queries

This should live inside the app, not outside it.

Why:

- we own the app
- semantic control is more reliable than pixels
- custom-painted timeline surfaces need internal hit models anyway

### 2. Human input plane

Add the ability to drive the app with:

- click
- double click
- drag
- hover
- scroll
- key press
- hotkey
- text input

This keeps the harness “human-like” where semantic actions are insufficient or
where we want realistic verification.

### 3. Native/OS fallback plane

Use OS-level automation only when needed for:

- native file dialogs
- window focus/activation
- app switching
- menu bar or system-owned UI

This should be the fallback, not the core design.

## What To Build

### A. Automation bridge inside EchoZero

Add an app-owned automation bridge that runs in dev/test profiles.

It should expose capabilities such as:

- `launch`
- `snapshot`
- `screenshot`
- `list_windows`
- `focus_window`
- `find_object`
- `invoke_action`
- `click`
- `drag`
- `scroll`
- `press_key`
- `type_text`
- `get_timeline_hit_targets`
- `get_visible_objects`
- `get_selection_state`

Transport options:

- local socket
- local HTTP
- local WebSocket
- local MCP server wrapper

The exact transport matters less than the shape of the contract.

### B. Stable automation ids

Every important UI region should have stable ids or names:

- shell regions
- timeline canvas
- ruler
- transport
- inspector
- layer headers
- dialogs
- transfer surfaces

Qt hooks to use:

- `objectName`
- accessible name/description
- explicit app-owned automation ids for custom-painted regions

### C. Timeline hit model

Because the timeline is custom-painted, generic accessibility won’t be enough.

The timeline control layer should expose:

- visible lane list
- visible event/take/layer hit boxes
- screen coordinates for semantic objects
- object lookup at a point
- current zoom/scroll/world mapping

This is the key piece that makes the app controllable “like a human” without
reducing everything to brittle image matching.

### D. Screenshot and visual verification

The bridge should support:

- full-window screenshot
- region screenshot
- timeline-only screenshot
- optionally object-highlight overlays in dev mode

This gives OpenClaw and agents visual confirmation while preserving semantic
control.

### E. Scenario execution mode

Build on the existing GUI DSL so the same action vocabulary can be used for:

- regression scenarios
- agent-driven tasks
- reproducible bug traces
- appflow-style smoke runs

## OpenClaw Integration Path

The cleanest end state is:

- EchoZero exposes an automation bridge
- OpenClaw gets a dedicated EchoZero control tool or MCP connector
- agents can issue high-level app commands plus low-level human input commands

Ideal tool shape:

- `echozero.launch`
- `echozero.snapshot`
- `echozero.find`
- `echozero.invoke`
- `echozero.click`
- `echozero.drag`
- `echozero.type`
- `echozero.press`
- `echozero.screenshot`
- `echozero.timeline.query`

This is better than trying to treat EchoZero like a generic opaque desktop app.

## What Not To Do

- do not rely on image matching as the primary control method
- do not build a browser-only automation mindset around a Qt app
- do not keep all automation locked in test-only internal helpers forever
- do not let the widget layer invent hidden automation semantics separate from app truth

## Phased Plan

### Phase 1: Freeze current harness as the seed

Goal:

- keep `AppFlowHarness` and GUI DSL as the starting control vocabulary

Tasks:

- document current action coverage
- keep existing GUI lanes green
- identify missing actions for real operator flows

### Phase 2: Add stable automation metadata

Goal:

- make important UI regions discoverable and addressable

Tasks:

- add stable object names / automation ids
- add accessible names to key controls
- define timeline semantic object ids

### Phase 3: Build the in-app automation bridge

Goal:

- expose snapshot, action, and screenshot APIs from the running app

Tasks:

- choose transport
- expose app/window snapshot
- expose action invocation and basic input
- expose screenshot capture

### Phase 4: Add timeline semantic control

Goal:

- make the custom canvas reliably controllable

Tasks:

- visible hit-target export
- coordinate mapping
- object-at-point lookup
- semantic click/drag operations for events/layers/takes

### Phase 5: Integrate OpenClaw

Goal:

- allow OpenClaw to drive EchoZero in real development workflows

Tasks:

- wrap the automation bridge in a tool or MCP connector
- add launch/attach/session lifecycle
- support interactive agent-driven UI development loops

## First Implementation Slice

The safest first slice is:

1. add stable automation ids/names to the shell and major UI regions
2. expose a read-only `snapshot` endpoint from the running app
3. expose `screenshot`
4. keep all existing app-flow and GUI proof green

This gives immediate leverage without committing to a full input driver on day one.

## Proof Lanes

Use these as the initial proof surfaces:

- `tests/testing/test_app_flow_harness.py`
- `tests/testing/test_gui_dsl.py`
- `tests/testing/test_gui_lane_b.py`
- `tests/ui/test_app_shell_runtime_flow.py`
- `tests/ui/test_timeline_shell.py`
- `python -m echozero.testing.run --lane appflow`
- `python -m echozero.testing.run --lane gui-lane-b`

When timeline hit control changes, expand proof around selection, drag, transfer,
and follow-scroll behavior.

## Open Questions

1. Should the automation bridge live only in test/dev profiles, or be attachable in a guarded dev mode of the normal app?
2. Do we want native file dialogs in the human-control loop, or should the app provide controlled test/dev substitutes?
3. Should OpenClaw talk directly to the bridge, or should we place an EchoZero-specific MCP server in front of it?

## Bottom Line

Yes, this is possible for EchoZero.

In fact, EchoZero is a better candidate than a random desktop app because we own:

- the runtime
- the widget tree
- the timeline semantics
- the test harness
- the action vocabulary

The right answer is not generic computer use alone.
The right answer is a semantic Qt app-control layer with human-style input where
needed, then connect OpenClaw to that.
