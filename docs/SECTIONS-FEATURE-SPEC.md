# Sections Feature Spec

Status: proposed implementation target
Last verified: 2026-04-26

This spec defines the canonical `Sections` concept for EchoZero.

Use this document when the question is:
- what `Sections` are
- how `Sections` differ from markers and generic regions
- how section ownership works across time
- what transport/import/export data must exist
- how the timeline should render and edit sections

Use [STATUS.md](STATUS.md) for repo truth.
Use [architecture/DECISIONS.md](architecture/DECISIONS.md) for prior locked
architecture decisions around song structure, annotations, and locators.

## Why This Exists

EchoZero already has nearby concepts:
- song-structure decisions and section backdrops
- freeform annotation markers
- ruler locators
- generic timeline regions

What it did not yet have was one concrete product contract for the user's
desired cue-owned song-part model:

- one cue marks the left edge of a song part
- that cue owns the timeline area until the next cue
- cue numbering does not imply order
- the pre-first-cue gap can be real and unowned

This spec locks that model down.

## Product Goal

Create one first-class `Sections` workflow where the operator can:
- define song-part boundaries by placing named cues
- see section ownership immediately across the timeline
- preserve imported cue identity exactly
- push and pull section cues through the MA transport lane
- reason about song parts as structure, not as generic point markers

## Locked Rules

These are the current source-of-truth rules.

1. User-facing name is `Sections`.
2. Data model is `section cues`, rendered as derived `section regions`.
3. The gap before the first cue is valid and must render neutral/unowned.
4. Cue refs are preserved exactly as entered/imported; no renumbering and no
   sequential requirement.
5. Region ownership is purely time-based: each section owns from its cue time
   until the next cue time; the last section owns to song end.

Additional locked implications:
- cues may start a few seconds into the song
- cue refs may go backward or be non-sequential
- time order controls ownership, not cue numbering
- transport remains cue-based; region spans are derived locally in EchoZero
- `Sections` are a new concept; existing markers remain their own concept

## Canonical Vocabulary

Use these terms in code, docs, and UI copy:

- Section:
  the whole user-facing song-part concept
- Section cue:
  the left-edge boundary object with identity and metadata
- Section region:
  the derived owned span from one section cue until the next cue
- Unsectioned gap:
  the neutral area before the first cue
- Cue ref:
  the preserved imported/entered cue identifier, e.g. `Q7`, `12`, `3.5`

## What "Song Part Ownership Model" Means

The ownership model is:

- every section cue starts ownership at its exact time
- ownership continues rightward until the next section cue
- the next cue owns its own boundary from its own start time onward
- after the last cue, that section owns to song end
- before the first cue, nothing owns the song yet; that area is neutral

Example:

- `00:12  Q7 Verse`
- `00:41  Q3 Chorus`
- `01:10  Q9 Breakdown`

Derived ownership:

- `00:00 - 00:12` unsectioned gap
- `00:12 - 00:41` owned by `Q7 Verse`
- `00:41 - 01:10` owned by `Q3 Chorus`
- `01:10 - song end` owned by `Q9 Breakdown`

The crucial rule:
- cue ref does not determine order
- time determines order

## Relationship To Existing Concepts

### Sections vs Annotation Markers

Annotation markers remain:
- freeform notes
- bookmarks
- rehearsal comments
- lighting ideas

Sections are not just renamed markers.
Sections are structure-defining cues with derived ownership spans.

### Sections vs Ruler Locators

Ruler locators remain:
- song-level navigation bookmarks
- quick jumps
- DAW-style locator behavior

Sections are not just locator labels.
Sections define owned song parts.

### Sections vs Generic Regions

Current generic regions span across all layers and already power some timeline
overlay behavior.

Locked interpretation:
- `Sections` are their own product concept
- generic regions are not the user-facing source of truth for this feature
- existing region rendering/editor primitives may be reused as implementation
  substrate where helpful
- long-term, generic regions may stay as lower-level plumbing or may be
  deprecated later, but `Sections` must not be explained to the user as
  "just regions"

## Canonical Data Model

The canonical persisted/user-edited object is the section cue.

Recommended shape:

```text
SectionCue
- id
- start_seconds: float
- cue_number: int | float | None
- cue_ref: str
- name: str
- color: str | None
- notes: str | None
- payload_ref: str | None
```

Derived shape:

```text
SectionRegion
- cue_id
- start_seconds: float
- end_seconds: float
- cue_ref: str
- name: str
- color: str | None
- notes: str | None
- payload_ref: str | None
```

Important:
- `end_seconds` is derived from the next cue, not independently authored as the
  primary truth
- a neutral pre-first-cue gap is implicit, not stored as a fake section
- `name` is always string data
- `cue_number` is numeric cue data when present; `cue_ref` is the preserved
  display/reference identity EchoZero shows or reconstructs around that number

## Cue Ref Contract

`cue_ref` must be treated as an opaque preserved string.

Valid examples:
- `Q7`
- `7`
- `3.5`
- `B12`

Rules:
- never renumber automatically
- never sort by cue ref
- never enforce monotonic cue numbering
- display the preserved cue ref exactly, except for optional harmless UI
  formatting such as spacing normalization

## MA Transport Contract

The MA side technically maps both section cues and regular events back to the
same underlying event object family.

Locked interpretation:
- section cue metadata should use a shared transport metadata envelope that
  regular events can also carry when available
- this does not make all events sections
- it means EchoZero should standardize the common metadata fields so import and
  export stay coherent for both

Recommended shared metadata fields:

```text
- cue_ref
- label / name
- color
- notes
- payload_ref
```

Implications:
- section cues require these fields where applicable
- regular events may preserve or surface the same fields if MA provides them
- transport normalization should not fork into incompatible "section-only" and
  "event-only" cue metadata shapes

## Push / Pull Rules

### Push

Push exports section cues as cue-start objects.

Push must send:
- cue time
- cue ref
- cue name
- optional shared metadata fields supported by the transport lane

Push must not require a separate explicit region-end object unless MA has a
real canonical equivalent we decide to support later.

### Pull

Pull imports cue-start objects and reconstructs section regions locally.

Pull must:
- preserve cue refs exactly
- sort ownership by imported cue time
- derive each section end from the next imported cue
- leave the pre-first-cue area neutral

Operator workflow requirement:
- manual pull must offer both `+ Create New Layer...` and
  `+ Create Section Layer...`

Current implementation boundary:
- the transport and pull models still type `cue_number` as `int` in several
  places, so float cue numbers are not yet preserved end-to-end without loss

## Timeline Rendering Contract

Sections should not read like tiny marker pills only.

The required visual model is:
- a strong left-edge cue boundary line
- a readable cue label such as `Q7 Chorus`
- a subtle region wash from this cue until the next cue
- neutral pre-first-cue area

Recommended rendering:
- dedicated `Sections` lane for cue identity and editing
- full-timeline backdrop ownership overlay across all layers
- optional centered or repeated region label when zoom level permits

The user should be able to answer, at a glance:
- where does this section start
- what cue owns this time
- what song part am I in right now

## Editing Rules

### Create

Adding a section cue:
- inserts a new section start at the chosen time
- splits the previously owned section at that point if one exists
- does not backfill a fake section before it

### Delete

Deleting a section cue:
- removes that section start
- merges its owned span into the previous cue's ownership if one exists
- if it was the first cue, the song regains a neutral pre-first-cue gap

### Move

Moving a section cue:
- moves the left boundary of that section
- correspondingly changes the end of the previous section and the start of this
  one

### Rename / Metadata Edit

Editing cue ref, name, color, or notes:
- updates the owning section identity
- does not change order unless the time changes

## Selection / Interaction Expectations

The selected section should expose:
- cue ref
- section name
- start time
- derived end time
- derived duration
- optional notes/color/payload reference

Clicking inside a section backdrop may later select the owning section, but the
first implementation should prioritize clear left-edge cue targeting and visual
readability.

## Migration / Compatibility

Locked rule:
- `Sections` are new

Therefore:
- do not silently reinterpret all existing marker layers as sections
- existing markers stay markers
- any later migration should be explicit, e.g. `Convert markers to sections`

## Implementation Direction

Preferred path:
- build `Sections` as their own canonical feature
- reuse existing region/backdrop primitives where helpful
- keep markers and locators conceptually separate
- standardize MA cue metadata for both section cues and regular events

## Not In This Spec's Done Bar

Not required for the first integrated sections milestone:
- deleting generic regions from the codebase
- live claims about MA hardware behavior without proof
- forcing all existing markers to migrate
- inferred auto-sections from ML as the primary workflow
- hard requirement that the first section start at `0.0`
