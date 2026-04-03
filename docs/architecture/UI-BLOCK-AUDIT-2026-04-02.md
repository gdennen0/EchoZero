# UI Block Audit — 2026-04-02

## Purpose

Audit the Stage Zero timeline shell against EZ2 UI principles:
- Structure reveals truth
- Truth first, chrome second
- Small blocks, narrow models, explicit intents, no sideways state access
- Named structural regions instead of ad hoc drawing

This audit focuses on regionization, block boundaries, and extraction order.

---

## Current state

### Good / aligned

#### 1. Top-level truth model is reflected better than before
- Main rows are visually privileged over take lanes
- Take lanes are subordinate rows
- Ruler exists as its own concept
- Transport exists as its own concept
- Scroll clipping protects the header area from content overlap

#### 2. Block architecture has begun
Currently extracted:
- `RulerBlock`
- `LayerHeaderBlock`
- `WaveformLaneBlock`

This is the correct direction. `widget.py` is beginning to behave as a composer instead of a gravity well.

#### 3. Presentation layer exists
`TimelinePresentation`, `LayerPresentation`, `TakeLanePresentation`, `LayerStatusPresentation`, and related structs are already carrying useful UI-facing state.

---

## Main architectural finding

The shell is no longer primarily blocked by missing features.
It is blocked by **under-regionized blocks**.

In plain terms:
- we started extracting blocks
- but those blocks still own internal layout implicitly
- and several future micro-widgets have nowhere explicit to live yet

So the next leap is not just “extract more blocks.”
It is:

## **Define named structural regions inside blocks and rows**

That is what will allow future controls/status chips/context affordances to be added without layout spaghetti.

---

## Region audit

### A. Timeline screen regions

#### Already somewhat explicit
- transport/top bar
- ruler
- left header column
- right content viewport

#### Missing next-level structure
Need explicit layout structs for:
- transport title cluster
- transport control cluster
- time readout cluster
- meta/status cluster
- ruler header label region
- ruler tick/content region

Recommendation:
Create `TransportLayout` next if top bar continues to evolve.

---

### B. Main row regions

This is the most important area.

#### Current named regions (good start)
Inside `HeaderSlots`:
- `title_rect`
- `subtitle_rect`
- `status_rect`
- `controls_rect`
- `toggle_rect`
- badge origin/y

#### Missing named regions
Need to evolve into a fuller row composition model:
- `header_rect`
- `content_rect`
- `title_row_rect`
- `subtitle_row_rect`
- `status_row_rect`
- `controls_row_rect`
- `badges_row_rect`
- `toggle_slot_rect`

Current state still partly mixes row layout assumptions directly in widget composer.

Recommendation:
Replace loose `HeaderSlots` values with a dedicated `MainRowLayout` struct that contains both:
- header subregions
- content viewport region

This will make future widgets like source badges, sync indicators, or context anchors easier to add.

---

### C. Take row regions

#### Current state
Take rows are understructured.
They currently have:
- row background
- take name text
- content viewport

#### Missing regions
Need explicit regions for:
- take label slot
- provenance/source summary slot
- future context-menu affordance slot
- future selection/action affordance slot
- content viewport rect

Recommendation:
Introduce `TakeRowLayout` before adding more take interactions.
Otherwise take rows will become the next mini-spaghetti zone.

---

### D. Content lane regions

#### Waveform lane
Good:
- extracted into `WaveformLaneBlock`

Missing:
- explicit viewport rect in its presentation/layout
- future overlay regions (selection, cursor handles, clip bounds, transient highlights)
- real waveform data hooks

Recommendation:
Add a `content_rect` / viewport rect into waveform lane layout before deepening interactions.

#### Event lane
Current state:
- still embedded in `widget.py`

This is the clearest next extraction target.
Need:
- `EventLaneBlock`
- event hit targets owned by the block
- explicit event viewport region
- future drag/trim handles as named overlay regions

Recommendation:
Extract next.

---

### E. Interaction zones

Current hit testing lives in `TimelineCanvas` with lists of rect tuples:
- `_take_rects`
- `_toggle_rects`
- `_event_rects`

This is acceptable as an intermediate step but not a final structure.

Need to evolve toward block-owned hit regions or block-returned hit targets.

Recommendation:
As each block is extracted, it should eventually own:
- `paint(...)`
- `hit_targets(...)`

This will reduce the widget’s role to composition/dispatch.

---

## What to keep

Keep and build on:
- `TimelinePresentation` family as UI-facing input structs
- content clipping boundary between header and timeline content
- block extraction direction (`RulerBlock`, `LayerHeaderBlock`, `WaveformLaneBlock`)
- realistic fixture direction (song + stems + classifiers + takes + stale/edited/sync examples)

---

## What to fix next

### 1. Regionize row composition more formally
Create layout structs:
- `MainRowLayout`
- `TakeRowLayout`

These should explicitly describe all subregions, not just a few freehand rects.

### 2. Extract `EventLaneBlock`
This is the next obvious block still trapped in `widget.py`.

### 3. Add block-owned hit-target shape over time
Do not leave all hit-testing permanently centralized as raw rect lists.

### 4. Formalize transport regions if top bar expands
Currently acceptable, but it will need structuring once transport grows beyond play/stop/time.

---

## What to extract next

### Immediate next extraction order
1. `EventLaneBlock`
2. `TakeLaneBlock` (or `TakeRowBlock`) composition
3. `MainRowLayout` / `TakeRowLayout` structs
4. optionally `TransportBarBlock` once top bar grows more controls

This is the cleanest next path.

---

## What to defer

Defer until the above is cleaner:
- full XML/DSL-style declarative UI layer
- advanced transport controls
- right-click context menus everywhere
- deeper provenance inspector widgets
- richer waveform overlays
- full event editing gestures

These should not land on top of under-regionized row architecture.

---

## Candidate future micro-widgets

These now have plausible homes if row layouts are formalized:
- M button widget
- S button widget
- stale chip widget
- edited chip widget
- sync badge widget
- provenance/source chip widget
- take context anchor widget
- row menu anchor widget
- zoom cluster widget
- current time cluster widget

This is why regionization matters: future little widgets need named homes.

---

## Recommended next build order

### Phase 1
- Introduce `MainRowLayout`
- Introduce `TakeRowLayout`
- Move existing header/content rect logic into those structs

### Phase 2
- Extract `EventLaneBlock`
- keep `widget.py` as composer only

### Phase 3
- Extract `TakeRowBlock` composition
- prepare for future context affordances without inline clutter

### Phase 4
- If needed, extract `TransportBarBlock`
- especially once zoom controls / follow controls / mode toggles appear

---

## Bottom line

The next problem is not “missing widgets.”
It is **missing named structural regions inside rows and blocks**.

That is the lever that will prevent future spaghetti and let us add many more little widgets safely.
