# Cue Recipe Structure Documentation

This document captures live grandMA3 cue-part recipe behavior discovered from
real sequences in the `1000+` range.

## Core Finding

Cue-part recipes are real structural children in live show data.

Observed live pattern:

- `Sequence`
  - `Cue`
    - `Part`
      - `Recipe`

This mirrors the preset-side pattern:

- `Preset`
  - `Recipe`

So the `Recipe` class is not preset-only. It is also a first-class cue-part
child type.

## Sequence Scan Result

Live scan artifact:

- `artifacts/ma3-terminal-crawl/sequence-1000-scan/scan.json`
- `artifacts/ma3-terminal-crawl/sequence-1000-scan/scan.md`

Summary from the first 30 sequences with sequence numbers `>= 1000`:

- sequences scanned: `30`
- cue parts with child objects: `33`
- sequences with non-empty sampled cue parts: `26`
- all observed cue-part child objects in this scan were class `Recipe`

This is strong evidence that recipe-bearing cue parts are common in actual show
content, not a rare edge case.

## Representative Live Example

Observed live object path:

- `Sequences -> 1200 'BadIdea' -> 1 'Mark' -> 0 'Mark' -> Recipe 1`

Object path from MA:

`Root/ShowData/DataPools/1 'Default'/Sequences/1200 'BadIdea'/1 'Mark'/0 'Mark'/Recipe 1`

Important discovered fields on the `Recipe` child:

- `SELECTION = 4 'Scenic Spot'`
- `SELECTIONMODE = Normal`
- `PRESET = 2 'Position'.28 'SCENIC FOCUS#2'`
- `VALUES = FeatureGroup 2 'Position'.Preset 28 'SCENIC FOCUS#2'`
- `ENABLED = Yes`
- `PRESETMODE = Selective`
- `PRESETMODEINTERNAL = Selective`

Interpretation:

- the cue part is not just storing opaque values
- the recipe child can explicitly reference a selection target
- it can explicitly reference a preset
- it can explicitly expose the resolved value source

That is exactly the kind of structure we need for agent reasoning about whether
a cue is using direct values, preset references, or recipe-driven output.

## Important Contrast

In the sampled `AUTOMATOR` sequence, cue parts did not show child recipes.

In the `1000+` show-content sequences, cue parts frequently did.

So:

- utility or control sequences are not representative
- real show sequences are a better source of MA behavior truth

## Recipe Count Per Cue Part

Observed cue parts ranged from:

- `1` recipe child
- up to `20` recipe children in the sampled `Ballad -> Mark` cue part

That means a cue part can aggregate many recipe lines/children, not just one.

## Agent Rules

For agent reasoning:

1. To understand a cue, inspect its `Part` objects, not only the `Cue`.
2. To understand a cue part, inspect both:
   - the part’s own timing/value fields
   - any child `Recipe` objects
3. Do not assume cues are only direct-value containers.
4. Do not assume recipe behavior exists only in preset pools.

## Open Questions

- when cue parts use direct stored values without recipe children, what exact
  fields most clearly prove that mode
- whether some cue parts mix direct values and recipe children
- how recipe ordering / recipe-child count maps to MA’s UI “recipe lines”
- how cue-part recipes interact with phaser-capable fields and cooked output
