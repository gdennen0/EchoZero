# UI Interaction Closure Plan - 2026-04-05

## Purpose

This backlog converts the current timeline code into an implementation plan for closing the highest-value UI interaction gaps without inventing new product scope.

Grounding rules used here:
- prioritize actual code and current tests over idealized future UI
- treat `main` as truth and takes as subordinate lanes
- call out only deltas that are already implied by the application layer, existing tests, or locked UX docs

---

## Current Reality Snapshot

What is already real in code:
- the application model supports layers, takes, events, selection, viewport, mute/solo/gain, sync, play/pause/seek, and take actions (`echozero/application/timeline/models.py`, `echozero/application/timeline/orchestrator.py`)
- the assembler enforces "main take is truth" and renders non-main takes as subordinate lanes (`echozero/application/timeline/assembler.py`, `tests/application/test_timeline_assembler_contract.py`)
- the Qt shell can render main rows, take rows, event lanes, waveform lanes, tooltips, follow-scroll, horizontal scroll, and take actions (`echozero/ui/qt/timeline/widget.py`, `tests/ui/test_timeline_shell.py`, `tests/ui/test_follow_scroll.py`)

What is still mostly missing in the widget:
- direct header selection
- clickable mute/solo controls
- transport-bar interaction
- ruler drag / playhead drag
- multi-select and deselect behavior
- event drag / resize / transfer / keyboard edit flows
- zoom commands and context menus

---

## P0 Contract Risks

### P0.1 Expansion state is split across two concepts and one path is effectively dead

Problem:
- `LayerPresentationHints` carries both `collapsed` and `take_selector_expanded`.
- the orchestrator has both `ToggleLayerExpanded` and `ToggleTakeSelector`.
- the assembler and widget only render expansion from `take_selector_expanded` / `is_expanded`.
- current UI dispatch only wires `ToggleTakeSelector`.

User impact:
- expansion behavior can drift between application truth and UI truth
- future work could update `collapsed` and see no visible change
- tests may pass while the wrong intent is being used

Source-of-truth reference:
- `echozero/application/timeline/models.py:24-35`
- `echozero/application/timeline/orchestrator.py:59-67`
- `echozero/application/timeline/assembler.py:119`
- `echozero/ui/qt/timeline/widget.py:287-290`
- `tests/ui/test_timeline_shell.py` (`test_toggle_take_selector_round_trips`)
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md` section "UI implication"

Target files:
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/assembler.py`
- `echozero/ui/qt/timeline/widget.py`
- `tests/application/*timeline*`
- `tests/ui/test_timeline_shell.py`

Acceptance criteria:
- one expansion concept remains authoritative for take-lane visibility
- the widget dispatches only that intent
- no dead expansion field or dead expansion intent remains
- tests prove expand/collapse changes the rendered take-lane state through the full application-to-UI path

### P0.2 Event selection drops take identity and leaves selection state ambiguous

Problem:
- event hit targets only store `(rect, layer_id, event_id)`, even for take lanes
- `SelectEvent` updates `selected_layer_id` and `selected_event_ids`, but not `selected_take_id`
- clicking an event in an alternate take cannot reliably tell the application which take owns that event

User impact:
- inspector or context actions on selected take events will not have stable take context
- take-related UI can show stale selection after event clicks
- future event editing on take lanes is risky because selection truth is incomplete

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:183`
- `echozero/ui/qt/timeline/widget.py:283-285`
- `echozero/ui/qt/timeline/widget.py:428-441`
- `echozero/application/timeline/orchestrator.py:55-58`
- `echozero/application/timeline/models.py:88-92`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `echozero/application/timeline/assembler.py`
- `tests/ui/*.py`
- `tests/application/*timeline*`

Acceptance criteria:
- selecting an event from a take lane preserves enough identity to recover layer, take, and event
- `selected_take_id` is deterministic after event selection
- tests cover main-row event click and take-lane event click separately

### P0.3 Status/provenance UI is currently synthetic, not application-backed

Problem:
- the assembler hardcodes `stale=False` and `manually_modified=False`
- only `source_label` and `sync_label` are surfaced from application state
- architecture docs explicitly separate provenance, freshness, and manual modification, but the UI contract currently fakes two of the three

User impact:
- the UI can appear "clean" while hiding stale or edited state
- users cannot trust the chips or metadata area once real provenance arrives
- closure work on source inspection will sit on a false presentation contract

Source-of-truth reference:
- `echozero/application/timeline/assembler.py:106-111`
- `tests/ui/test_timeline_shell.py` (`test_fixture_exposes_stale_manual_and_sync_signals`)
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md` sections "Freshness / staleness" and "Provenance and manual edits"

Target files:
- `echozero/application/timeline/models.py`
- `echozero/application/timeline/assembler.py`
- `echozero/ui/qt/timeline/blocks/layer_header.py`
- `tests/application/test_timeline_assembler_contract.py`
- `tests/ui/test_timeline_shell.py`

Acceptance criteria:
- stale, manually modified, and provenance/source fields are backed by application data, not hardcoded defaults
- header chips and tooltip metadata reflect real application state
- assembler tests lock the mapping from application state to presentation state

### P0.4 Manual horizontal scroll does not honor the locked follow-mode interruption decision

Problem:
- `set_presentation()` always recomputes follow scroll from presentation state
- manual horizontal scrolling updates `scroll_x` locally, but nothing disables follow mode
- locked UX decision says manual scroll should turn follow off until the user re-enables it

User impact:
- the viewport can "fight" the user during playback
- follow-mode behavior is non-deterministic once live transport updates start driving repaints
- any follow toggle added later will sit on the wrong interruption rule

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:534-545`
- `echozero/ui/qt/timeline/widget.py:568-581`
- `tests/ui/test_follow_scroll.py`
- `docs/UX-DESIGN-DECISIONS.md` D-9
- `docs/UX-MICRO-TESTS.md` `PH-3.3.5`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `tests/ui/test_follow_scroll.py`

Acceptance criteria:
- manual horizontal scroll during active follow disables follow in application-visible state
- subsequent presentation updates do not recenter until follow is explicitly re-enabled
- tests cover manual scroll interruption for at least `PAGE` and `CENTER`

---

## P1 Missing Core Interactions

### P1.1 Layer rows are not directly selectable

Problem:
- `layer_clicked` is declared and connected, but never emitted
- header and empty-row clicks fall through to seek instead of selecting a layer

User impact:
- no reliable way to establish an active layer
- future keyboard commands like `Ctrl+A`, nudge, or move-to-layer lack a stable focus model

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:162`
- `echozero/ui/qt/timeline/widget.py:264-291`
- `echozero/ui/qt/timeline/widget.py:516`
- `echozero/application/timeline/orchestrator.py:46-50`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `tests/ui/test_timeline_shell.py`

Acceptance criteria:
- clicking a layer header selects that layer
- clicking empty content inside a layer row selects the layer without seeking
- clicking empty timeline space outside an event supports deselect or ruler-seek behavior according to the target zone

### P1.2 Mute and solo controls are painted only

Problem:
- header buttons for `M` and `S` are drawn, but no hit targets are stored and no intents are dispatched
- the orchestrator already supports `ToggleMute` and `ToggleSolo`

User impact:
- a visible control suggests functionality that the UI does not provide
- users cannot operate core DAW state from the row they are already looking at

Source-of-truth reference:
- `echozero/ui/qt/timeline/blocks/layer_header.py:34-36`
- `echozero/ui/qt/timeline/blocks/layer_header.py:76-90`
- `echozero/application/timeline/orchestrator.py:86-94`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md` section "UI implication"

Target files:
- `echozero/ui/qt/timeline/blocks/layer_header.py`
- `echozero/ui/qt/timeline/widget.py`
- `tests/ui/*.py`
- `tests/application/*timeline*`

Acceptance criteria:
- mute and solo buttons have explicit hit targets
- clicking them dispatches the corresponding orchestrator intent
- main rows respond; take rows do not expose independent mute/solo controls

### P1.3 The transport bar is visual only

Problem:
- `TransportBarBlock.paint()` returns play/stop rects, but `TransportBar` does not capture or use them
- play/pause behavior exists in the application layer and in demo dispatch tests, but not in the widget itself

User impact:
- transport looks functional while being inert
- users must use external harnesses instead of the timeline shell

Source-of-truth reference:
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py:11-31`
- `echozero/ui/qt/timeline/widget.py:453-468`
- `tests/ui/test_timeline_shell.py` (`test_play_pause_seek_intents_update_presentation`)

Target files:
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py`
- `echozero/ui/qt/timeline/widget.py`
- `tests/ui/test_timeline_shell.py`

Acceptance criteria:
- play/pause and stop controls are clickable from the transport bar
- click behavior dispatches existing transport intents
- widget-level tests prove transport clicks change presentation state

### P1.4 Ruler click/drag and playhead-drag behaviors are not implemented

Problem:
- `TimelineRuler` only paints
- seeking currently happens as a catch-all fallback from `TimelineCanvas.mousePressEvent()`
- there is no ruler drag state and no playhead triangle hit target

User impact:
- seeking is attached to the wrong surface
- locked DAW behavior for ruler click and triangle drag is missing
- empty clicks inside lane content unexpectedly seek

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:264-291`
- `echozero/ui/qt/timeline/widget.py:471-491`
- `echozero/ui/qt/timeline/widget.py:447-450`
- `docs/UX-DESIGN-DECISIONS.md` D-1 and D-7
- `docs/UX-MICRO-TESTS.md` `TR-2.2.1`, `TR-2.2.3`, `PH-3.2.4`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/ruler.py`
- `tests/ui/test_ruler_block.py`
- new widget-level ruler interaction tests

Acceptance criteria:
- ruler click seeks
- ruler drag updates playhead continuously
- playhead head is directly draggable
- lane-body clicks no longer double as generic seek unless that zone is explicitly designated

### P1.5 Selection is still single-item only

Problem:
- selection state can hold multiple event ids, but the widget only ever emits one event id
- no `Shift+Click`, `Ctrl+Click`, rubber band, empty-space deselect, or `Ctrl+A`

User impact:
- current shell cannot perform the baseline selection model already chosen in UX decisions
- downstream editing features would be forced to rebuild selection behavior later

Source-of-truth reference:
- `echozero/application/timeline/models.py:88-92`
- `echozero/application/timeline/orchestrator.py:55-58`
- `echozero/ui/qt/timeline/widget.py:264-291`
- `docs/UX-DESIGN-DECISIONS.md` D-2 and D-15
- `docs/UX-MICRO-TESTS.md` section 6

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `tests/ui/*.py`

Acceptance criteria:
- click selects one event
- `Shift+Click` adds
- `Ctrl+Click` toggles
- dragging empty space creates a rubber-band selection
- `Ctrl+A` selects all visible, unlocked events
- `Escape` clears selection when no drag is active

### P1.6 Event editing flows are absent: drag, resize, layer transfer, nudge, duplicate

Problem:
- the current event lane only paints rectangles and returns click hit boxes
- there is no drag state machine, no resize handles, no keyboard movement, and no duplicate path

User impact:
- the shell remains a viewer plus take-action surface, not an editor
- the largest chunk of locked micro-tests remains unimplemented

Source-of-truth reference:
- `echozero/ui/qt/timeline/blocks/event_lane.py`
- `echozero/ui/qt/timeline/widget.py:264-291`
- `docs/UX-DESIGN-DECISIONS.md` D-3, D-4, D-10, D-11, D-12, D-13, D-14, D-24, D-25
- `docs/UX-MICRO-TESTS.md` sections 7, 9, 11.2, 13

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/event_lane.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `tests/ui/*.py`
- `tests/application/*timeline*`

Acceptance criteria:
- selected events can be dragged horizontally
- event drag clamps at time zero
- event transfer to other event layers is possible and respects locked/hidden rejection rules
- arrow-key nudge and `Ctrl+D` use the locked UX semantics
- undoable data mutations are represented as application intents, not ad hoc widget-only state

### P1.7 Zoom interaction is missing even though zoom state already exists

Problem:
- viewport stores `pixels_per_second`, but the widget only uses it for rendering
- wheel handling only supports `Shift+Wheel` horizontal scroll
- no `Ctrl+Wheel`, keyboard zoom, or zoom-to-fit behavior exists

User impact:
- users cannot reach the intended time-scale workflows from the shell
- follow-scroll and snap behavior cannot be tuned in real use because zoom is static

Source-of-truth reference:
- `echozero/application/timeline/models.py:95-99`
- `echozero/ui/qt/timeline/widget.py:293-300`
- `docs/UX-DESIGN-DECISIONS.md` D-16, D-26, D-27
- `docs/UX-MICRO-TESTS.md` section 1.2 and `KB-11.4.*`
- `echozero/ui/FEEL.py` zoom constants

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/application/timeline/intents.py`
- `echozero/application/timeline/orchestrator.py`
- `tests/ui/test_follow_scroll.py`
- new widget zoom tests

Acceptance criteria:
- `Ctrl+Wheel` zooms with cursor anchoring
- keyboard zoom commands update `pixels_per_second`
- zoom-to-fit is available and testable
- zoom changes remain view-state only, not undo entries

---

## P2 Fidelity / Polish

### P2.1 Context menus and discoverability surfaces are missing

Problem:
- there is no right-click path on events or empty space
- UX docs already narrow v1 context menus to a bounded set of actions

User impact:
- core actions remain less discoverable than the locked design expects
- overlap and selection workflows have no secondary affordance path

Source-of-truth reference:
- `docs/UX-DESIGN-DECISIONS.md` D-18 and D-19
- `docs/UX-MICRO-TESTS.md` `EC-14.4.7`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `tests/ui/*.py`

Acceptance criteria:
- right-click on an event opens an event-scoped menu
- right-click on empty space opens an empty-space menu
- menu contents match current v1 scope, with paste explicitly deferred if unavailable

### P2.2 Header metadata quality is below the intended information density

Problem:
- header tooltip currently shows badges, optional source label, and sync label only
- no surfaced provenance detail beyond source text, and no direct source/parent inspection affordance exists yet

User impact:
- the user sees that status exists, but cannot inspect enough context to act on it
- source/parent inspection remains a documented near-term priority without a UI hook

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:317-327`
- `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md` priority 4

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/layer_header.py`
- `tests/ui/test_layer_header_metadata.py`

Acceptance criteria:
- header metadata exposes the minimum backed provenance/status fields introduced in P0.3
- the user can inspect source/parent context without leaving the timeline shell

### P2.3 Text rendering has visible encoding artifacts in painted UI strings

Problem:
- several UI strings in source are stored with mojibake characters such as `â€¢`, `â–¾`, and `â–¸`

User impact:
- visible polish regression in tooltips, transport metadata, and take-row options
- undermines confidence in otherwise functional UI

Source-of-truth reference:
- `echozero/ui/qt/timeline/widget.py:322`
- `echozero/ui/qt/timeline/blocks/take_row.py`
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py:26`

Target files:
- `echozero/ui/qt/timeline/widget.py`
- `echozero/ui/qt/timeline/blocks/take_row.py`
- `echozero/ui/qt/timeline/blocks/transport_bar_block.py`

Acceptance criteria:
- visible UI labels render with intended glyphs or ASCII-safe equivalents
- tests that assert visible text use stable expected strings

---

## P3 Defers

### P3.1 Copy/paste remains deferred

Problem:
- UX decisions explicitly defer paste despite wanting eventual infrastructure

User impact:
- duplication is the v1 workaround

Source-of-truth reference:
- `docs/UX-DESIGN-DECISIONS.md` D-19

Target files:
- none in this closure sprint set unless shared infrastructure naturally appears

Acceptance criteria:
- no user-facing paste workflow is promised in the P0-P2 sprints
- any clipboard plumbing added is internal-only and does not expand v1 scope

### P3.2 Inspector direct field editing remains deferred

Problem:
- current scope keeps time/duration edits on drag-based interactions, not form fields

User impact:
- precise typed editing waits until after baseline manipulation exists

Source-of-truth reference:
- `docs/UX-DESIGN-DECISIONS.md` D-22

Target files:
- none in this closure sprint set

Acceptance criteria:
- no sprint below depends on editable inspector fields

### P3.3 Advanced overlap selection menu stays v2

Problem:
- the locked decision keeps topmost-on-click in v1 and defers "select from overlapping"

User impact:
- overlap disambiguation remains limited in v1

Source-of-truth reference:
- `docs/UX-DESIGN-DECISIONS.md` D-2

Target files:
- none required for closure beyond preserving topmost selection semantics

Acceptance criteria:
- v1 selection work does not block on overlap menus

---

## Proposed Sprint Breakdown

### Sprint 1 - Contract Cleanup and Selection Plumbing

Why first:
- it removes contradictory state before more interaction code hardens around it
- it gives later event and header work a trustworthy selection model

Scope:
- P0.1 expansion-state unification
- P0.2 event-selection take identity
- P0.3 status/provenance mapping contract
- basic layer selection path from header and row clicks

Exit condition:
- application truth and UI truth agree on expansion, selection, and status fields

### Sprint 2 - Header, Transport, and Ruler Become Real Controls

Why second:
- these are already painted surfaces with existing application intents behind them
- they unlock immediate usability with relatively contained widget work

Scope:
- P1.2 mute/solo click targets
- P1.3 clickable transport
- P1.4 ruler click/drag and playhead drag
- manual-seek zone cleanup so row clicks no longer accidentally seek

Exit condition:
- the shell supports transport, seek, and row-level audio-state control without relying on external harnesses

### Sprint 3 - Baseline Selection and Event Manipulation

Why third:
- once selection plumbing is stable, event editing can be implemented on top of it
- this is the biggest behavior surface and should not start before Sprint 1

Scope:
- P1.5 multi-select and deselect flows
- P1.6 drag, transfer, nudge, duplicate
- locked/hidden drop rejection
- undoable intent boundaries for these mutations

Exit condition:
- the shell crosses from viewer to editor for event layers

### Sprint 4 - Zoom, Follow Interruption, and UX Surface Closure

Why fourth:
- zoom and follow are easiest to validate after direct manipulation exists
- this sprint closes the "feel" gaps without destabilizing earlier contract work

Scope:
- P0.4 follow interruption
- P1.7 zoom controls
- P2.1 context menus
- P2.2 metadata inspection
- P2.3 text cleanup

Exit condition:
- timeline navigation, follow behavior, and discoverability are aligned with the locked UX decisions

---

## Recommended Execution Order Summary

1. Fix contract splits before adding more widget behavior.
2. Wire already-painted controls before building heavy event editing.
3. Add multi-select and event mutations only after selection truth is complete.
4. Finish with zoom/follow/context-menu closure and polish once the editing surface is stable.

---

## Out-of-Scope Notes

- This plan does not assume speculative new architecture beyond the current application/timeline layer.
- Audio-layer cross-layer dragging remains out of scope for v1 editing; the locked decision is that audio stays bound to source.
- Copy/paste and direct inspector field editing remain deferred.
