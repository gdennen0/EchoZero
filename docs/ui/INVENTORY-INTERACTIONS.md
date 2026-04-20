# Interaction Inventory

_Updated: 2026-04-19_

This file is the canonical inventory of approved interaction patterns.
Keep entries short.
Only add patterns that are reused, foundational, or risky.

## Entry Format

- `Name`
- `Status`
- `Purpose`
- `Trigger`
- `Outcome`
- `Feedback`
- `Recovery`
- `Canonical Files`
- `Used By`

## Layer Selection

- `Name`: Layer Selection
- `Status`: canonical
- `Purpose`: Select one or more timeline layers for focus and follow-on actions.
- `Trigger`: click, additive modifier, range modifier
- `Outcome`: updates selected layer ids and active selected layer
- `Feedback`: visible selected state in headers and related surfaces
- `Recovery`: click elsewhere, replace selection, clear selection
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`
- `Used By`: timeline workspace, inspector

## Event Selection

- `Name`: Event Selection
- `Status`: canonical
- `Purpose`: Select one or more events for edit, transfer, and inspection.
- `Trigger`: click, additive modifier, range modifier
- `Outcome`: updates selected event ids plus selected layer or take scope
- `Feedback`: event highlight, inspector update, action enablement
- `Recovery`: clear selection, replace selection
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`
- `Used By`: timeline workspace, object info panel

## Event Editor Modes

- `Name`: Event Editor Modes
- `Status`: canonical
- `Purpose`: Let the workspace switch between standard event-editing intents without overloading a single gesture model.
- `Trigger`: local mode bar, keyboard shortcut
- `Outcome`: changes workspace gesture behavior between selection, draw, and erase
- `Feedback`: active mode state, cursor change, interaction preview
- `Recovery`: switch mode again or use the mode shortcut
- `Canonical Files`: `echozero/ui/qt/timeline/widget.py`
- `Used By`: timeline workspace

## Take Selection

- `Name`: Take Selection
- `Status`: canonical
- `Purpose`: Change active take context within a layer without changing truth semantics.
- `Trigger`: take row action or direct take selection gesture
- `Outcome`: updates selected take id for the current layer
- `Feedback`: selected take lane state and related inspector detail
- `Recovery`: select another take or clear take selection
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`
- `Used By`: take rows, inspector

## Active Playback Target

- `Name`: Active Playback Target
- `Status`: canonical
- `Purpose`: Define what layer or take playback acts on.
- `Trigger`: direct selection or dedicated playback-target action
- `Outcome`: updates playback target ids
- `Feedback`: active target indicator distinct from plain selection
- `Recovery`: retarget playback to another object
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/application/timeline/app.py`
- `Used By`: timeline workspace, transport

## Transport

- `Name`: Transport
- `Status`: canonical
- `Purpose`: Control playback and playhead position.
- `Trigger`: play, pause, stop, seek actions
- `Outcome`: updates transport or runtime audio state
- `Feedback`: transport state, playhead motion, button state
- `Recovery`: pause, stop, seek again
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/application/timeline/app.py`
- `Used By`: transport bar, timeline ruler, keyboard shortcuts

## Move Or Drag Events

- `Name`: Move Or Drag Events
- `Status`: canonical
- `Purpose`: Reposition events in time through direct manipulation while preserving canonical selection and edit flow.
- `Trigger`: drag gesture on selected event or explicit move action
- `Outcome`: updates event timing through the application edit path
- `Feedback`: live drag preview, snap indicator, resulting placement
- `Recovery`: cancel drag, reverse move, undo
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/ui/qt/timeline/widget.py`
- `Used By`: event lane

## Create Or Delete Events

- `Name`: Create Or Delete Events
- `Status`: canonical
- `Purpose`: Support standard event authoring directly in event-capable lanes.
- `Trigger`: draw drag, erase click, delete key
- `Outcome`: creates or removes events through the canonical application edit path
- `Feedback`: draw preview, immediate lane update, selection change
- `Recovery`: create again, undo, reselect remaining events
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/ui/qt/timeline/widget.py`
- `Used By`: timeline workspace

## Marquee Event Selection

- `Name`: Marquee Event Selection
- `Status`: canonical
- `Purpose`: Select multiple events in one gesture without changing layer truth semantics.
- `Trigger`: drag on empty event-lane space while in selection mode
- `Outcome`: replaces event selection with the events intersecting the marquee and keeps the relevant layer or take context
- `Feedback`: marquee rectangle, event highlight, inspector update
- `Recovery`: click to replace selection, additive selection, clear selection
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/ui/qt/timeline/widget.py`
- `Used By`: timeline workspace

## Nudge Events

- `Name`: Nudge Events
- `Status`: canonical
- `Purpose`: Apply small controlled movement to selected events.
- `Trigger`: keyboard shortcut or explicit action
- `Outcome`: moves selected events by FEEL-backed increment
- `Feedback`: immediate position update
- `Recovery`: opposite nudge or undo
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`
- `Used By`: timeline workspace

## Duplicate Events

- `Name`: Duplicate Events
- `Status`: canonical
- `Purpose`: Copy selected events forward in time while preserving source content.
- `Trigger`: keyboard shortcut or explicit action
- `Outcome`: creates duplicated events through the application edit path
- `Feedback`: new event instances become visible and selectable
- `Recovery`: undo or delete duplicates
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`
- `Used By`: timeline workspace

## Timeline Snap Or Beat Grid

- `Name`: Timeline Snap Or Beat Grid
- `Status`: canonical
- `Purpose`: Keep event editing aligned to visible timing structure or musical beats when available.
- `Trigger`: snap toggle, grid mode cycle, drag or draw gesture
- `Outcome`: adjusts candidate edit times against visible grid and nearby event edges
- `Feedback`: grid lines, snap indicator, predictable placement
- `Recovery`: toggle snap off, change grid mode, move event again
- `Canonical Files`: `echozero/ui/qt/timeline/time_grid.py`, `echozero/ui/qt/timeline/widget.py`
- `Used By`: timeline workspace

## Transfer Flow

- `Name`: Transfer Flow
- `Status`: candidate
- `Purpose`: Support push, pull, diff preview, and apply flows for MA3 transfer.
- `Trigger`: transfer action, confirm/apply/cancel actions
- `Outcome`: opens bounded transfer workflow and executes canonical apply path
- `Feedback`: plan preview, diff state, result state
- `Recovery`: cancel flow, retry, re-open with adjusted options
- `Canonical Files`: `echozero/application/timeline/intents.py`, `echozero/application/timeline/orchestrator.py`, `echozero/ui/qt/timeline/widget_actions.py`
- `Used By`: inspector, transfer UI, dialogs

## Context Actions

- `Name`: Context Actions
- `Status`: candidate
- `Purpose`: Expose local secondary actions close to the selected object.
- `Trigger`: right-click or explicit local action affordance
- `Outcome`: offers non-primary contextual operations
- `Feedback`: self-describing action list grouped by object context
- `Recovery`: cancel menu, undo action where supported
- `Canonical Files`: `echozero/ui/qt/timeline/widget_actions.py`, `echozero/ui/qt/timeline/object_info_panel.py`
- `Used By`: timeline workspace, object info panel

## Event Clip Preview

- `Name`: Event Clip Preview
- `Status`: canonical
- `Purpose`: Let the inspector audibly confirm the currently selected event clip without switching full transport context.
- `Trigger`: inspector `Play Clip` action for a previewable event selection
- `Outcome`: plays the event time slice from the resolved source audio on the preview path
- `Feedback`: audible preview and existing runtime audio feedback
- `Recovery`: stop preview, change selection, replay
- `Canonical Files`: `echozero/application/presentation/inspector_contract.py`, `echozero/ui/qt/timeline/widget_actions.py`, `echozero/ui/qt/app_shell.py`, `echozero/application/playback/runtime.py`
- `Used By`: object info panel
