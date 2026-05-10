# Fixture And Preset Mental Model

This note captures the practical MA3 mental model EchoZero should use when
reasoning about fixtures, presets, phasers, and recipes.

It is intentionally operator-facing. It does not try to restate the full MA3
manual.

## Core Model

Think about MA3 in this order:

1. Fixture type defines what a device is capable of.
2. Patch instantiates that type into the show.
3. Attributes are the controllable semantics.
4. Feature groups organize attributes and map to preset pools.
5. Presets, phasers, and recipes author behavior against those attributes.

That means preset reasoning should be attribute-first, not only object-first.

## Preset Storage Modes

Use this practical shorthand:

- `Selective`: stores values for individual fixtures.
- `Global`: stores values per fixture type.
- `Universal`: stores one shared value that can apply across fixtures.

This is the operator-facing meaning we should optimize for in tooling and docs.

Implementation detail:

- native MA3 object fields may expose this through `PresetMode`,
  `PresetModeInternal`, `StoredData`, and mixed states such as
  `Selective/Global`
- do not assume there is one flat universal storage path underneath every
  preset

## Show Assumption

In many real shows, presets are already maintained well enough to at least
support the fixture types in the current rig at a global level.

That should be the default expectation when browsing or editing show presets:

- first assume the show author intended fixture-type-safe reuse
- then inspect whether the specific preset also contains selective or universal
  data

## Fixture Hierarchy

Use MA3 language when reasoning about fixture structure.

Do not think of a fixture as one undifferentiated control surface.

A patched fixture may expose:

- the main fixture at the top
- one or more intermediate parent subfixtures underneath it
- child subfixtures below those parents

Some fixtures do not expose subfixtures at all. They are still valid fixtures
and should not be forced into a hierarchy they do not have.

Loose example:

- a 20-pixel fixture may have a main fixture
- beneath that, a parent subfixture for the pixel bank
- beneath that, 20 child subfixtures where each child subfixture is one RGB
  pixel
- the same fixture may also expose another parent subfixture for shared
  controls such as shutter across all pixels

This matters because two attributes with similar names may belong to different
levels of the same fixture hierarchy.

## What To Inspect

When EchoZero inspects a preset or phaser, it should try to surface:

- preset pool / feature-group context
- storage mode: `Selective`, `Global`, `Universal`, or mixed
- parent preset properties
- child `Recipe` lines when present
- affected attributes
- affected fixture type or fixture/subfixture scope when recoverable

For phasers and recipes, the authored behavior often lives on child `Recipe`
objects even when the parent preset carries the user-facing name and mode.

For browsing, the default view should be:

- by preset pool / preset type
- optionally filtered by song range so the operator does not need to scroll
  through the full show pool

Treat the song range as a narrowing filter, not as the primary organization.

## Editing Rule

When editing preset or phaser behavior:

- first identify the attribute family
- then identify the storage mode
- then identify the fixture hierarchy level the value belongs to
- then decide whether the write belongs on the parent `Preset` or a child
  `Recipe`

Do not reduce the task to "set one property on one preset object" unless the
inspection data proves that is really what the show object is doing.

## Phaser Editing

For operator-facing editing, treat phasers as step-based presets.

Practical rule:

- phaser editing is step editing

Even when native MA3 structure exposes supporting `Recipe` objects, the
operator-facing edit model should stay centered on steps.

Useful shorthand:

- a phaser is a preset with multiple steps
- the edit surface should present those steps directly

Default operator preference for this repo:

- prefer `Universal` phasers unless the task explicitly calls for
  `Selective`, `Global`, or mixed storage behavior

## Why This Matters For EchoZero

This mental model is enough to keep the agent from making the most common MA3
mistakes:

- confusing parent preset metadata with authored recipe behavior
- treating all preset values as fixture-agnostic
- ignoring subfixture/module structure on multi-part fixtures
- assuming `Global`, `Selective`, and `Universal` are just labels instead of
  storage behaviors
