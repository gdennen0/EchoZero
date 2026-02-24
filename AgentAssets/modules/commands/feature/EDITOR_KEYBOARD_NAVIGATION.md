# Editor: Keyboard Navigation (Layers/Events) and Spacebar Audio Preview

## Summary

Investigation for adding arrow-key navigation (layers and events), a selected-layer accent outline, and spacebar-triggered audio preview in the Editor timeline. Includes council-style evaluation per `AgentAssets/modules/process/council/`.

---

## 1. Problem and Scope

### User request

- **Arrow keys**: Up/Down change the current selected layer (outline in accent); Left/Right move forward/back selecting single events within the selected layer.
- **Standard event editing interactions**: Align with common DAW/editor expectations (keyboard-driven layer and event selection).
- **Spacebar**: Play the selected event(s) audio preview (reuse existing EventInspector clip playback).

### Evidence of need

- Timeline already has keyboard shortcuts for Delete, Escape, Ctrl+A, and configurable move-event (Left/Right, Ctrl+Up/Down). No dedicated **navigation** (change layer / change event in layer).
- EventInspector already has "Play" for the selected event clip; no keyboard shortcut to trigger it from timeline focus.
- Drag target layer is already drawn with accent (ACCENT_BLUE) in `TimelineScene.drawBackground`; no "selected layer" outline when not dragging.

### Scope

| In scope | Out of scope |
|----------|--------------|
| Up/Down = change selected layer, outline in accent | Changing layer order via keyboard |
| Left/Right = previous/next event in selected layer | Multi-select by range (Shift+arrow) in this phase |
| Selected layer = layer of selection, or explicit when empty | Persisting "last selected layer" across sessions |
| Spacebar = play selected event preview when selection exists | Changing timeline Space behavior when no selection (keep current play/pause) |
| EventInspector.play_selected_preview() callable from timeline | New audio backend; reuse ClipAudioPlayer |

---

## 2. Current State (Findings)

### Timeline view keyboard handling (`view.py`)

- **Space**: Always emits `space_pressed` -> `_playback_controller.toggle_playback` (play/pause timeline).
- **Delete/Backspace**: Forwarded to scene (delete selected events).
- **Ctrl+A / Escape**: Select all / Deselect all.
- **With selection**: Left/Right and Ctrl+Up/Down are configured as move-event shortcuts (nudge, move layer).
- **Without selection**: Home/End/Left/Right move playhead.

No Up/Down for layer selection; no plain Left/Right for "previous/next event in layer" (those keys are used for nudge when selection exists).

### Timeline scene (`scene.py`)

- `get_selected_event_ids()`, `get_selected_items()`, `select_event()`, `select_events()`, `deselect_all()`.
- No API to get "events in layer sorted by time" or "selected layer" or "layer above/below".
- `drawBackground()` draws track rects; drag target layer gets `ACCENT_BLUE` outline; no "current/selected layer" outline.

### LayerManager (`layer_manager.py`)

- `get_all_layers()` (ordered), `get_layer_index(layer_id)`, `get_first_layer_id()`, `get_layer(id)`.
- Order is `_order` (list of layer IDs). Layer "above" = lower index, "below" = higher index.

### EventInspector (`inspector.py`)

- `_on_play_clip_clicked(event_dict)` plays one event's clip via `_clip_player.play_clip(audio_id, audio_name, clip_start, clip_end)`.
- No public `play_selected_preview()`; selection is in `_selected_events` (list of event dicts).

### Design system

- `Colors.ACCENT_BLUE` (and theme overrides) used for focus/selection accents; scene already uses it for drag target. Use same for selected-layer outline.

---

## 3. Proposed Design

### 3.1 Selected layer and outline

- **Selected layer**: In this MVP, derive from selection: layer of the first selected event. If no selection, use a stored "current layer" (e.g. last layer that had selection, or first layer).
- **Stored current layer**: When selection is empty, keep a `_current_layer_id` (or equivalent) so Up/Down still have a target; initialize from first layer or last selected layer.
- **Drawing**: In `TimelineScene.drawBackground()`, after drawing track backgrounds, if there is a selected/current layer (and it is visible), draw its track rect with an accent outline (reuse same pattern as drag target: `QPen(Colors.ACCENT_BLUE, 2)` or theme accent). Prefer design_system `Colors` so themes apply.

### 3.2 Keyboard behavior (view + scene)

- **Up (no Ctrl)**: "Layer above"
  - If we have a current/selected layer, move to layer above (lower index). If no event selection, optionally select first event in that layer; else just change current layer and redraw outline.
  - If no current layer, set current layer to first layer.
- **Down (no Ctrl)**: "Layer below" (same idea, higher index).
- **Left (no Ctrl, no move shortcut consumed)**: "Previous event in layer"
  - Resolve current layer (from selection or `_current_layer_id`). Get events in that layer sorted by start_time. Select previous event (by time) relative to first selected event; if none selected, select last event in layer.
- **Right (no Ctrl)**: "Next event in layer" (select next by time; if none selected, select first event in layer).

Conflict with existing shortcuts:

- Left/Right are currently "nudge event" when there is selection (from settings: `shortcut_move_event_left` / `shortcut_move_event_right`). Options: (A) make arrow-nav take precedence when only one event is selected and we want "navigate"; (B) use Alt+Left/Right for event nav and keep Left/Right for nudge; (C) keep current move shortcuts and use e.g. J/K or other keys for event nav. **Recommendation**: (A) is simplest and matches "navigate vs edit": when user has one event selected, Left/Right = previous/next event; nudge could stay on Ctrl+Left/Right or remain configurable. If we keep nudge on plain Left/Right, then (B) Alt+Left/Right for event nav is a clear compromise.

For this doc we assume: **Up/Down = layer nav (no conflict). Left/Right = event-in-layer nav**; existing "move event left/right" can be remapped to Ctrl+Left/Right or kept as-is and we document that "navigate" takes precedence when we add it (single-event selection: Left/Right = navigate; multi: Left/Right = nudge, or we keep both and prefer navigate only when no modifier). Implementation can start with: **no modifier = navigate (event/layer); Ctrl+Left/Right = nudge (existing)** so there is no conflict.

### 3.3 Spacebar: play selected preview

- **When timeline view has focus**: If at least one event is selected, Space = "play selected event(s) audio preview" (first selected only). If no event selected, Space = current behavior (timeline play/pause).
- **Implementation**: View checks selection; if any, emit e.g. `play_preview_requested` (or call into widget); widget calls `event_inspector.play_selected_preview()`. EventInspector exposes `play_selected_preview()`: if `_selected_events` has one or more, call same logic as `_on_play_clip_clicked(_selected_events[0])` (toggle stop if already playing). No new audio backend.

### 3.4 Scene API additions

- `get_events_in_layer_sorted(layer_id) -> List[BaseEventItem]` (by start_time).
- `get_selected_layer_id() -> Optional[str]`: layer of first selected event; if no selection, return None (widget can then use stored current layer).
- `select_layer_at_index(direction: int)`: move "current layer" up (-1) or down (+1), optionally select first event in that layer; return new layer_id.
- `select_previous_event_in_layer()` / `select_next_event_in_layer()`: use current layer (from selection or widget-provided layer_id), get sorted events, select prev/next; scroll into view if needed.

Widget (TimelineWidget) can hold `_current_layer_id` when selection is empty and pass it into scene for Up/Down and Left/Right, so scene stays stateless for "current layer when no selection."

### 3.5 EventInspector

- Add public method `play_selected_preview() -> bool`: if `_selected_events` non-empty and first event has clip timing/audio, run same path as "Play" button (toggle stop if playing); return True if preview started or toggled, False otherwise. No new UI.

---

## 4. Council Evaluation

### Architect

**Problem understanding**: Add keyboard-driven layer/event navigation and selected-layer visual, and reuse existing clip playback for Space.

**Concerns**:  
- Keeping "current layer" in one place: either scene or widget. Prefer widget holding `_current_layer_id` and scene providing pure "get events in layer," "select event," "layer at index" so scene does not duplicate layer ordering logic.  
- Clear boundary: view handles key events and delegates to widget; widget coordinates scene + inspector.

**Alternatives**:  
- Store current layer in scene vs widget: widget is better (widget already owns selection_changed and inspector updates).  
- Left/Right: navigate vs nudge – use modifier (Ctrl = nudge) to avoid two meanings for same key.

**Vote**: **Approve with Conditions**  
- Current layer when no selection lives on widget; scene exposes only selection and layer-index/layer-id helpers.  
- Left/Right: no modifier = event nav; Ctrl+Left/Right = nudge (or keep existing configurable shortcuts and document precedence).

---

### Systems

**Problem understanding**: Same as above; no new services, only UI/keyboard and existing playback.

**Concerns**:  
- Spacebar: ensure we don’t start timeline playback and preview at once; branch clearly: selection -> preview, no selection -> play/pause.  
- Clip player is already used by inspector; single thread, no new resources.

**Vote**: **Approve**  
- No new failure modes if we only call existing play_clip path and guard on selection.  
- Resource usage unchanged.

---

### UX

**Problem understanding**: Users can move between layers (Up/Down) and between events in a layer (Left/Right), see which layer is "current" (accent outline), and play preview with Space.

**Concerns**:  
- Discoverability: document in UI or help that Up/Down = layer, Left/Right = event.  
- Consistency: accent outline matches existing drag-target and theme accent.  
- Space: "Space = preview when something selected" is intuitive and matches "play this clip" in inspector.

**Vote**: **Approve**  
- Standard DAW-like navigation; outline reinforces current layer; Space for preview reduces context switching.

---

### Pragmatic

**Problem understanding**: Implement nav, outline, and Space preview with minimal new code and no new deps.

**Scope**:  
- Scene: `get_events_in_layer_sorted`, optional `get_selected_layer_id`; widget: `_current_layer_id`, `select_layer_up/down`, `select_prev/next_event_in_layer` (or scene methods called from widget with current layer).  
- View: Up/Down/Left/Right (with modifier rule) and Space branch.  
- Scene: draw selected-layer outline in `drawBackground`.  
- EventInspector: `play_selected_preview()`.

**Testing**: Unit tests for "events in layer sorted," selection changes; manual test for key bindings and Space.

**Vote**: **Approve with Conditions**  
- Implement in small steps: (1) outline + current layer tracking, (2) Up/Down layer nav, (3) Left/Right event nav, (4) Space preview.  
- Resolve Left/Right vs nudge once (e.g. Ctrl = nudge) and stick to it.

---

## 5. Recommendation

**RECOMMENDATION: Proceed with Modifications**

- **Proceed**: Arrow-key layer/event navigation, selected-layer accent outline, and Spacebar-triggered audio preview are in scope and align with existing architecture and EchoZero values.
- **Modifications**:
  1. **Current layer**: Keep `_current_layer_id` (or equivalent) on TimelineWidget; scene provides only selection and sorted-events-in-layer + layer-at-index helpers.
  2. **Left/Right**: Use unmodified Left/Right for "previous/next event in layer"; keep move-event shortcuts on Ctrl+Left/Right (or as configured) so nudge and navigate do not conflict.
  3. **Space**: When there is at least one selected event, Space triggers EventInspector `play_selected_preview()`; when there is no selection, keep current Space = timeline play/pause.
  4. **Outline**: Draw selected/current layer track with theme accent (e.g. `Colors.ACCENT_BLUE`) in `TimelineScene.drawBackground()`, consistent with drag target.

### Conditions

- [ ] Widget owns "current layer when no selection"; scene stays stateless for that.
- [ ] EventInspector exposes `play_selected_preview()` and timeline connects Space (when selection exists) to it.
- [ ] Shortcut policy for Left/Right (navigate vs nudge) decided and documented (e.g. Ctrl = nudge).
- [ ] No new dependencies; reuse ClipAudioPlayer and existing inspector play path.

### Next steps

1. Add `get_events_in_layer_sorted(layer_id)` and layer-index helpers on scene (or use LayerManager + scene event items).
2. Add selected-layer outline in `TimelineScene.drawBackground()` using design_system accent color.
3. Add `_current_layer_id` and Up/Down handling in TimelineWidget; view forwards Up/Down to widget.
4. Add Left/Right event-in-layer navigation (view -> widget -> scene); ensure shortcut conflict with nudge is resolved (Ctrl+Left/Right = nudge).
5. Add `EventInspector.play_selected_preview()` and connect Space (when selection) to it from timeline widget/view.
6. Manual test and short doc/comment on keyboard behavior.

---

**Document status**: Investigation and council evaluation complete. Ready for implementation planning (tickets or small PRs).
