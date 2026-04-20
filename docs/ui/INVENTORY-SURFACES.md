# Surface Inventory

_Updated: 2026-04-19_

This file is the canonical inventory of surface roles in EchoZero.
Only define surfaces that are real, reusable, or likely to drift without a
clear boundary.

## Entry Format

- `Surface`
- `Purpose`
- `Primary Responsibilities`
- `Not Allowed`
- `Canonical Files`
- `Notes`

## Workspace

- `Surface`: Workspace
- `Purpose`: Primary object editing and direct manipulation.
- `Primary Responsibilities`: editing, arrangement, immediate object feedback, dense operational context
- `Not Allowed`: global shell policy, long-term setup dialogs, private truth semantics
- `Canonical Files`: `echozero/ui/qt/timeline/widget.py`, `echozero/ui/qt/timeline/blocks/*`
- `Notes`: Primary actions should live with the object being acted on. Event-capable layers own standard editor tooling in the workspace, including mode changes, direct event creation or deletion, marquee selection, drag movement, and timeline snap or grid feedback.

## Inspector

- `Surface`: Inspector
- `Purpose`: Explain and edit the current selection.
- `Primary Responsibilities`: properties, contextual secondary actions, metadata, selection detail
- `Not Allowed`: becoming a second workspace or catch-all workflow engine
- `Canonical Files`: `echozero/application/presentation/inspector_contract.py`, `echozero/ui/qt/timeline/object_info_panel.py`
- `Notes`: Inspector clarifies selection; it may use a bounded scrollable action stack for selection-scoped secondary actions, but global tools still belong in toolbar or other view-scoped surfaces. Event selections may expose a bounded audio clip preview action here when the preview source is unambiguous.

## Toolbar

- `Surface`: Toolbar
- `Purpose`: Hold high-frequency global or view-scoped actions.
- `Primary Responsibilities`: modes, filters, common commands, view toggles
- `Not Allowed`: object-specific clutter that belongs near the object
- `Canonical Files`: `echozero/ui/qt/app_shell.py`
- `Notes`: Stable placement matters for muscle memory. Timeline-local mode controls that change how the workspace interprets gestures may live in a stable local control bar adjacent to transport and ruler surfaces.

## Transport

- `Surface`: Transport
- `Purpose`: Control playback and playhead behavior.
- `Primary Responsibilities`: play, pause, stop, seek, playback feedback
- `Not Allowed`: unrelated object configuration
- `Canonical Files`: `echozero/ui/qt/timeline/blocks/transport_bar_block.py`
- `Notes`: Treated separately because it has unusually high frequency and trust requirements.

## Dialog

- `Surface`: Dialog
- `Purpose`: Handle bounded decisions and temporary flows.
- `Primary Responsibilities`: confirm risky actions, import/export, setup, conflict resolution
- `Not Allowed`: long-term editing responsibility that belongs in the workspace
- `Canonical Files`: `echozero/ui/qt/timeline/widget_actions.py`, `echozero/ui/qt/timeline/manual_pull.py`
- `Notes`: Dialogs must remain bounded and self-contained. Object-level pipeline settings dialogs may show a stage-focused view of the shared settings system, but the editing scope must stay explicit.

## Inline Action Area

- `Surface`: Inline Action Area
- `Purpose`: Expose local actions near the object they affect.
- `Primary Responsibilities`: quick local actions, obvious contextual controls
- `Not Allowed`: hiding essential workflows or global controls
- `Canonical Files`: `echozero/ui/qt/timeline/blocks/layer_header.py`, `echozero/ui/qt/timeline/blocks/take_row.py`
- `Notes`: Powerful secondary actions are valid here when self-describing. Selected layer headers may host local pipeline entrypoints when those actions belong to the layer itself.

## Status Or Monitoring

- `Surface`: Status Or Monitoring
- `Purpose`: Show system condition and background activity.
- `Primary Responsibilities`: progress, sync state, health, warnings, readiness
- `Not Allowed`: replacing primary editing flows
- `Canonical Files`: `echozero/ui/qt/app_shell.py`
- `Notes`: Global state should remain visible without hijacking the workspace.

## Navigation

- `Surface`: Navigation
- `Purpose`: Move between major scopes while preserving orientation.
- `Primary Responsibilities`: project, song, version, and scope movement
- `Not Allowed`: becoming an action panel or duplicate editor
- `Canonical Files`: `echozero/ui/qt/app_shell.py`
- `Notes`: Navigation exposes structure and location, not workflow clutter.
