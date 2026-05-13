# MA3 Preset / Recipe / Phaser Model Notes

This document is a living evidence log for understanding how grandMA3 exposes presets, recipe lines, cue parts, phasers, and tracked cue state to Lua.

The goal is not to guess from property names. The goal is to build a tested model from live show data and use that model to make EchoZero cue-only / copy-status behavior trustworthy.

## Current Evidence

Source show context: live MA3 old/rich show, `Sequence 1200` (`BadIdea`).

### Recipe line example: Sequence 1200 Cue 2

The cue has 3 recipe lines. The lines reference normal preset objects:

| Recipe | Selection | Referenced preset | Observed properties |
|---:|---|---|---|
| 1 | `Group 2` / `Spot Floor` | `Preset 1.7` / `60` | `Type=Preset`, `Mode=0`, `Relative=true`, `PresetMode=1`, `PresetModeInternal=1` |
| 2 | `Group 4` / `Scenic Spot` | `Preset 1.4` / `Full` | `Type=Preset`, `Mode=0`, `Relative=true`, `PresetMode=1`, `PresetModeInternal=1` |
| 3 | `Group 43` / `Pulsar Col Pix` | `Preset 1.4` / `Full` | `Type=Preset`, `Mode=0`, `Relative=true`, `PresetMode=1`, `PresetModeInternal=1` |

Additional observed recipe properties on those lines:

- `RelativeFade = true`
- `RelativeDelay = true`
- `RelativePhase = true`
- `RelativeSpeed = false`
- `SelectionMode = Normal`
- `ValuesMode = nil`
- `StoredData = empty`
- `PresetData = empty`
- `OwnDataPresent = false`
- `OwnNonCookedDataPresent = false`

## Important Correction

Do **not** currently interpret recipe `Relative=true` as “the referenced preset value is relative.”

That is not proven and is likely misleading for dimmer presets like `Preset 1.7 '60'` and `Preset 1.4 'Full'`, which Griffin expects to be absolute values.

Current safer terminology:

- `recipe_relative_flag`: the recipe object property `Relative`.
- `referenced_preset`: the actual preset object referenced by `Preset` / `Values`.
- `value_semantics`: unknown until we inspect the referenced preset object and/or its stored/cooked data.
- `recipe_mode_raw`: raw `Mode` property, e.g. `0`.
- `preset_mode_raw`: raw `PresetMode` / `PresetModeInternal`, e.g. `1`.

## Open Questions

1. Why does a recipe line referencing dimmer preset `60` show `StoredData` empty?
2. What exactly does `PresetMode = 1` mean?
3. What exactly does `PresetModeInternal = 1` mean?
4. What exactly does recipe `Mode = 0` mean?
5. Why does `Relative = true` appear when the referenced dimmer preset appears absolute?
6. Are `RelativeFade`, `RelativeDelay`, `RelativePhase`, and `RelativeSpeed` timing/phaser modifiers rather than value semantics?
7. Where does the actual preset value live for referenced presets: on the recipe line, on the preset object, in cooked data, in fixture/attribute children, or only through MA's internal evaluation?
8. How should recipe lines be keyed when both absolute and relative behavior may apply to the same part/group/preset type?
9. How do phaser presets and recipe references interact with cue status and tracking?
10. How do cue parts alter status semantics? A part must be treated as a first-class dimension, not just a display suffix.

## Current Report/Analyzer Guidance

Until the model is proven:

- Reports should show raw fields separately instead of collapsing them into `mode:relative`.
- Analyzer keys should not claim value semantics from `Relative=true` alone.
- Prefer keys like:
  - `part:<n>`
  - `group:<name>`
  - `preset_type:<n>`
  - `recipe_relative_flag:<true/false>`
  - `recipe_mode_raw:<value>`
  - `preset_mode_raw:<value>`
- If a semantic lane is needed, call it `value_semantics:unknown` until proven.

## Next Investigation Steps

1. Dump referenced preset objects directly for `Preset 1.7`, `Preset 1.4`, and a known relative/phaser preset.
2. Compare recipe line properties against referenced preset properties.
3. Find where actual dimmer value `60` is stored/exposed.
4. Inspect children of preset objects and any cooked/stored data handles.
5. Compare an absolute dimmer preset, a relative preset, a phaser preset, and a recipe line that references each.
6. Update EchoZero report labels to separate raw properties from inferred semantics.
7. Only then update copy/status analyzer state keys.

## Evidence Files

- Raw broken recipe line dump: `/Users/march/Documents/GitHub/ma3-harness/artifacts/broken-recipe-lines-dump.txt`
- Human-readable sequence report: `/Users/march/Documents/GitHub/ma3-harness/artifacts/sequence-1200-cue-recipe-report.md`
- EZ special function test: `/Users/march/Documents/GitHub/ma3-harness/artifacts/ez-special-cue-functions-test.txt`

## 2026-05-13 Deep Extraction Findings

Artifacts generated from live `Sequence 1200`:

- `/Users/march/Documents/GitHub/ma3-harness/artifacts/preset-recipe-phaser-deep-report.md`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/preset-recipe-phaser-deep-raw.txt`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/preset-recipe-phaser-deep-summary.json`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/exported-preset-value-summary.md`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/exported-preset-value-summary.json`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/exported-presets/*.xml`

### Recipe rows vs preset data

All 175 recipe lines scanned from `Sequence 1200` shared this recipe-row pattern:

```text
Type=Preset | Mode=0 | Relative=true | PresetMode=1 | ValuesMode=nil | OwnData=<empty>
```

This strongly suggests a recipe row is a reference wrapper when `Type=Preset`. The recipe line points at a preset object through `Preset`/`Values`, but the recipe row does not own the value payload locally.

That explains why a recipe row referencing `Preset 1.7 '60'` has `StoredData` empty: the row is not the preset; it is a reference to the preset.

### Exported preset XML exposes actual values

Lua `:Children()` returned zero children for preset objects, but MA export reveals actual data in XML `<PresetData>`.

38 unique referenced presets from Sequence 1200 were exported and parsed:

- Total phasers: `3016`
- Total steps: `4000`
- Step value fields observed:
  - `Absolute`: `3135`
  - `Relative`: `709`
  - `RelativePhys`: `36`
  - `AbsolutePhys`: `214`
  - timing/shape fields: `Accel`, `Decel`, `Trans`, `Width`, `Integrated`, etc.

### Simple presets are still represented as phasers

Even static presets are stored as `<Phaser>` entries with one or more `<Step>` entries.

Examples:

```xml
<Preset Name="60" PresetModeInternal="Universal" Mode="0">
  <PresetData Size="94">
    <Phaser IDType="2" ID="1" Attribute="Dimmer">
      <Step Function="Dimmer" Absolute="60"/>
    </Phaser>
    ...
  </PresetData>
</Preset>
```

`Preset 1.7 '60'`:

- XML `PresetModeInternal="Universal"`
- XML `Mode="0"`
- 94 phasers
- 94 steps
- every step has `Absolute="60"`

`Preset 1.4 'Full'`:

- 94 phasers
- 94 steps
- every step has `Absolute="100"`

This proves these are absolute-value presets even though Lua reports `Relative=true` on the preset object and recipe reference.

### Relative value examples

`Preset 2.15 'Relative Stomp'`:

```xml
<Step Function="Tilt" RelativePhys="0.000000"/>
```

- 8 phasers
- 8 steps
- every step uses `RelativePhys`

`Preset 1.19 'Rel Release'`:

```xml
<Step Function="Dimmer" Relative="Specials:Release"/>
```

- 4 phasers
- 4 steps
- every step uses `Relative="Specials:Release"`

`Preset 1.2 'Full Relative Stomp'`:

```xml
<Step Function="Dimmer" Absolute="100" Relative="0"/>
```

- has both absolute and relative fields on each step.

### Phaser examples

`Preset 21.1200 'IntroRnd'` is a phaser-style preset:

```xml
<Phaser Attribute="Dimmer" Speed="16777216" Phase="0" Measure="16777216">
  <Step Absolute="100" Trans="100" Width="5.96046e-06" .../>
  <Step Absolute="0" Trans="100" Width="100" .../>
</Phaser>
```

- 94 phasers
- 188 steps
- two steps per phaser
- stores absolute dimmer values plus timing/shape fields.

`Preset 21.1204 'Chorus Wipe#2'` stores relative phaser steps:

- 94 phasers
- 188 steps
- step fields include `Relative`, `Trans`, `Width`, `Accel`, `Decel`.

### Current model update

Use these distinctions:

- `recipe_relative_flag`: Lua property on a recipe row. Not value semantics.
- `recipe_preset_mode_raw`: Lua recipe row `PresetMode`, observed as `1` on all scanned recipe references.
- `preset_mode_internal`: actual preset object mode, observed in XML as `Universal` for most, `Global` for some.
- `step_value_semantics`: derived from exported `<Step>` fields:
  - `Absolute` / `AbsolutePhys` => absolute data
  - `Relative` / `RelativePhys` => relative data
  - both fields present => mixed absolute + relative data
- `phaser_shape`: derived from multiple `<Step>` entries and fields like `Speed`, `Phase`, `Measure`, `Trans`, `Width`, `Accel`, `Decel`.

### Implication for EchoZero analyzer keys

Do not key by `mode:relative` from recipe `Relative=true`.

Better key dimensions:

```text
part:<part>
group:<selection>
preset_type:<type>
referenced_preset:<type.no>
recipe_preset_mode:<raw or enum>
recipe_relative_flag:<true/false>
value_semantics:<absolute|relative|mixed|unknown>
phaser_shape:<static|multi_step|unknown>
```

For `Sequence 1200 Cue 2`:

- recipe row says `Relative=true`, `PresetMode=1`, `StoredData=empty`
- referenced preset `1.7 '60'` exports as absolute dimmer 60
- referenced preset `1.4 'Full'` exports as absolute dimmer 100

So the user’s concern was correct: the values are absolute even though the recipe row had `Relative=true`.

## EchoZero implementation update

`MA3/plugins/presets.lua` now applies the preset/recipe model in the cue recipe analyzer:

- recipe rows keep raw recipe metadata (`recipe_relative_flag`, `recipe_mode_raw`, `preset_mode`, `preset_mode_internal`) separate from value semantics.
- referenced presets are exported to temporary XML (`ez_sem_preset_<type>_<no>.xml`) and parsed to derive:
  - `value_semantics`: `absolute`, `relative`, `mixed`, or `unknown`
  - `phaser_shape`: `static`, `multi_step`, or `unknown`
  - `absolute_step_count`, `relative_step_count`, `phaser_count`, `step_count`
- state keys use value lanes, e.g. `part:0|group:Spot Floor|feature:PresetType 1|lane:absolute`, instead of `mode:relative` from the recipe row flag.
- mixed presets expand into both absolute and relative lanes for contributor tracking.
- absolute lanes replace prior contributors; relative lanes stack.
- cue parts remain part of the key.

Critical correction: EZ Store Cue Only and Copy Cue Status are **recipe-only features**, not wrappers around native MA `Store`/`Copy` cue syntax. Do not use native `Store Sequence ... /CueOnly`, `Copy Sequence ... /CopyCueSource "Status"`, or equivalent MA cue commands as the implementation.

Correct direction:

- `Recipe Copy Cue Status`: resolve recipe contributors/status using the analyzer, then author/copy recipe lines into the destination cue using recipe-line operations only.
- `Recipe Store Cue Only`: derive recipe data from the programmer/current recipe context, author recipe lines into the target cue, then restore recipe tracking in the following cue using recipe-line operations only.
- Until recipe-line authoring/restoration is implemented, mutating functions must return unsupported instead of executing native MA cue commands.

Validation artifacts:

- `/Users/march/Documents/GitHub/ma3-harness/artifacts/ez-semantic-state-test.txt`
- `/Users/march/Documents/GitHub/ma3-harness/artifacts/ez-semantic-relative-test.txt`


## Recipe line authoring proof

Standalone harness plugin:

```text
/Users/march/Documents/GitHub/ma3-harness/plugins/RecipeCueStatus/recipe_cue_status.lua
```

Proven recipe-only write primitive:

- destination cue/part must already exist,
- clear destination recipe children via recipe handle deletion,
- append recipe children with `part:Append('Recipe')`,
- copy source recipe data with `destRecipe:Copy(sourceRecipe)`,
- never call native MA cue `Copy Sequence ...` or `Store Sequence ... /CueOnly` for the feature implementation.

Current exported API:

```lua
RecipeCueStatus.ApplyCopyCueStatusToSequence(sourceSequenceNo, sourceCueNo, destSequenceNo, destCueNo)
RecipeCueStatus.ApplyCopyCueStatus(sequenceNo, sourceCueNo, destCueNo)
RecipeCueStatus.ApplyStoreCueOnlyFromSourceCue(sourceSequenceNo, sourceCueNo, targetSequenceNo, targetCueNo)
RecipeCueStatus.WriteRowsToCue(rows, destSequenceNo, destCueNo, options)
```

Store Cue Only restore needs tombstones for recipe keys introduced by the target/programmer state but absent from the previous following-cue state. Copying only the old reference rows is insufficient because new absolute keys would keep tracking through. Current proof uses disabled copied recipe lines (`Enabled=No`, named `EZ Tracking Release`) as EchoZero recipe-tracking tombstones. `presets.lua` now includes `recipe_enabled` and treats disabled recipe rows as cancellation markers in the recipe tracking model.

Validation artifact:

```text
/Users/march/Documents/GitHub/ma3-harness/artifacts/recipe-line-write-real-sequence-test.txt
```

Observed against real song source Sequence 1200, scratch write Sequence 9897:

- Copy Cue Status `1200 cue 2 -> 9897 cue 1`: wrote 17 recipe lines, post-analysis 17 contributors, source/status match `true`.
- Store Cue Only proof `source 1200 cue 3 -> 9897 cue 1`, restore into `9897 cue 2`: target wrote 20 recipe lines and matched source `true`; restore wrote 20 lines including tombstones, post-analysis restored to 17 contributors, restore reference match `true`.
