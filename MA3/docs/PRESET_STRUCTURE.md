# Preset Structure Documentation

This document captures live grandMA3 preset behavior discovered through the
native MA terminal.

For the operator-facing fixture/preset mental model that should sit beside
these structural findings, see `MA3/docs/FIXTURE_PRESET_MENTAL_MODEL.md`.

## Core Finding

Do not treat preset behavior as a flat property list on the `Preset` object.

A significant part of preset semantics lives in child `Recipe` objects.

Observed live patterns:

- a `Preset` in the `Preset Recipes` pool can have one child of class `Recipe`
- a `Preset` in `Phaser Presets` can also have one child of class `Recipe`
- some phaser-style presets in the `Phasers` pool also expose one child
  `Recipe`

So for agent reasoning:

- `Preset` is the container / user-facing object
- `Recipe` is often where recipe-style or phaser-style behavioral values live

## Pool-Level Findings

Live `PresetPools:Children()` exposed these relevant pools:

- `Phasers`
- `Phaser Presets`
- `Preset Recipes`

Important MA quirk:

- these pools were discovered reliably through `DataPool()[4]:Children()`
- direct numeric indexing like `DataPool()[4][14]` was not reliable for this
  pool family in the live target

So for preset-pool traversal, prefer ordered `:Children()` access over
assuming direct numeric handle indexing.

## Preset Object Findings

Across the sampled show state:

- `PresetMode` varies across `Universal`, `Selective`, and `Global`
- `StoredData` varies across `Universal`, `Selective`, and
  `Selective/Global`
- many presets have non-empty `References`
- `RecipeTemplate` remained `No` in the observed objects, even in the
  `Preset Recipes` pool

Interpretation:

- recipe behavior cannot be inferred from `RecipeTemplate = Yes` alone
- references are common and appear to encode meaningful preset relationships
- pool membership and child structure matter at least as much as one flag

## Recipe Child Findings

Observed `Recipe` child properties include:

- selection fields:
  `SELECTION`, `SELECTIONMODE`
- source/link fields:
  `PRESET`, `MATRICKS`, `FILTER`, `GENERATOR`, `VALUES`
- behavior flags:
  `ENABLED`, `SELECTIONFROMVALUE`
- phaser-capable fields:
  `PHASERTRANSFORM`, `SPEEDFROMX`, `PHASEFROMX`, `FADEFROMX`, `DELAYFROMX`
  and the corresponding Y/Z variants
- matricks-style fields:
  `X`, `Y`, `Z`, `XWINGS`, `YWINGS`, `ZWINGS`, `XWIDTH`, `YWIDTH`, `ZWIDTH`

This is the critical abstraction:

- the parent `Preset` tells you broad mode and ownership state
- the child `Recipe` tells you how the recipe/phaser behavior is actually
  configured

## Live Examples

### Preset Recipes Pool

Observed object:

- `Preset Recipes -> Base Recipe -> Recipe 1`

Findings:

- parent `PresetMode = Selective`
- parent `OwnDataPresent = No`
- child class = `Recipe`
- child contains recipe-targeting fields like `PRESET`, `MATRICKS`, `FILTER`,
  `GENERATOR`, `VALUES`

### Phaser Preset With Active Child Behavior

Observed object:

- `Phasers -> intro_phaser_Hurricane#2 -> Recipe 1`

Findings:

- parent `PresetMode = Selective`
- parent `OwnDataPresent = No`
- child `Recipe` had populated behavior fields including:
  `XWINGS = 2`
  `SPEEDFROMX = 47.50 BPM`
  `PHASEFROMX = 180°`
  `PHASETOX = 0.00`

This is strong proof that phaser-like behavior can live materially on the
`Recipe` child, even when the parent preset itself looks sparse.

### Phaser Presets Pool

Observed object:

- `Phaser Presets -> RGB_Sin -> Base Recipe`

Findings:

- parent `PresetMode = Universal`
- parent `OwnDataPresent = Yes`
- child `Recipe` exists even when many child phaser fields are still `None`

Interpretation:

- the `Recipe` child is structural, not only present when heavily populated

## Agent Rules

For agent reasoning and future harness work:

1. To understand a preset, inspect both the parent `Preset` and any child
   `Recipe` objects.
2. Do not classify a preset as “not a recipe” just because
   `RecipeTemplate = No`.
3. Use pool context:
   `Phasers`, `Phaser Presets`, and `Preset Recipes` are semantically
   meaningful.
4. Treat non-empty `References` as evidence of cross-preset dependency.
5. Treat `Recipe` child presence as a first-class structural signal.

## Open Questions

- whether a live object with `RecipeTemplate = Yes` exists elsewhere in the
  show or in another show state
- how `PRESET`, `MATRICKS`, `FILTER`, `GENERATOR`, and `VALUES` are populated
  in more fully authored recipe examples
- whether multi-recipe presets occur in the wild or if one `Recipe` child is
  the common pattern
- how phaser-specific child fields correlate with `PresetMode`,
  `StoredData`, and `References`
