# EchoZero Visual Lab

Status: support-only external visual harness

Visual Lab is a small Qt preview and capture harness for aggressive visual iteration outside the canonical EchoZero runtime path. Timeline previews now build compact current app timeline/session models and assemble them through `TimelineAssembler`, then render production Qt timeline components where practical.

The lab is now organized as a component catalog, not one preview scene. Every visual object should eventually be independently previewable by stable ID, with composites listing the smaller parts they contain. The intended workflow should feel like stepping through a folder of UI objects: primitives first, then rows/headers/canvas parts/chrome/panels/forms/waveforms, then composed screens.

This is a foundation, not a broad production refactor. New entries can use lab-only wrappers while production EchoZero UI migrates toward catalogable components over time, but synthetic state must structurally match current production models.

## Tweak Tokens

Use the `Style Tokens` panel inside `python -m dev.visual_lab.preview`, or edit
`dev/visual_lab/tokens.toml` directly.

The first-pass vocabulary covers:

- global_colors: shared app background, surface, text, accent, success/warning/error colors
- palette: object-level window, panel, row, selected/muted row, waveform, accent, sync/stale/status colors
- fonts: family, sizes, weights
- metrics: radii, border width, glow strength, row heights, transport height, spacing

The in-lab editor is generated from token dataclasses and catalog entry metadata.
When an item is selected, the right-side editor shows only the knobs declared for
that object. The editor also exposes a part tree for nested style targets such
as `timeline.layer_row.header.title`, `timeline.layer_row.header.badge`, and
`transport.button.play.icon`; selecting a part filters the visible controls to
that sub-element. Edits apply live to the active preview; `Apply` is retained as
an explicit retry path for manual text edits, and `Save` writes the active
values back to the selected token TOML file.

Font-family fields render as editable dropdowns populated from Qt's available
font families, with the current token value preserved even when that family is
not installed. Use `Import Font` to copy `.ttf`, `.otf`, or `.ttc` files into
`dev/visual_lab/assets/fonts/` and register them with
`QFontDatabase.addApplicationFont` for the current lab session. The font
dropdowns refresh after import when Qt exposes family names from the file.

Color fields have a swatch plus manual `#RRGGBB` text field. Double-click the
swatch or the text value to open a `QColorDialog`. Automated tests monkeypatch
that picker path, so the test suite never requires an interactive color dialog.

## Run Preview

```bash
python -m dev.visual_lab.preview
```

Open a specific catalog object:

```bash
python -m dev.visual_lab.preview --item timeline.header.stale-cues
```

Use a custom token file:

```bash
python -m dev.visual_lab.preview --tokens /path/to/tokens.toml
```

## Capture Screenshots

Artifacts are written under ignored `artifacts/visual-lab/`.

```bash
python -m dev.visual_lab.preview --capture
```

Capture a specific object through the same catalog selection path:

```bash
python -m dev.visual_lab.preview --capture --item primitive.status-chips --no-peekaboo
```

Peekaboo is preferred when `/opt/homebrew/bin/peekaboo` is installed and the window is visible. The runner automatically falls back to Qt widget grab if Peekaboo is unavailable or cannot capture the window.

Force the fallback:

```bash
python -m dev.visual_lab.preview --capture --no-peekaboo
```

## Catalog Philosophy

Catalog entries live behind `build_visual_lab_catalog()` and carry:

- stable `entry_id`
- folder/category
- human-readable name and description
- `kind`: primitive, element, chrome, or composition
- `source_kind`: `production-backed`, `current-model synthetic`, or `lab-only experimental`
- `source_path`: import path or factory path that owns the preview truth
- render factory
- optional `part_ids` for composites and decomposed row/header relationships
- optional `editable_token_paths` for the generated live style editor
- optional `style_targets` for nested component -> part/subpart -> property editing

Prefer adding a small catalog entry before adding a new monolithic scene. If a visual object is meaningful in the app, the long-term target is that it can be previewed alone and also as part of larger compositions.

Coverage should move toward the whole current UI surface:

- top chrome/header equivalents, currently the production timeline editor toolbar
- transport, ruler, scroll/status chrome, and other timeline shell controls
- timeline canvas parts: rows, headers, ruler, waveform states, event lanes, take rows
- side panels such as the setlist browser and object info palette
- representative dialogs and reusable settings/forms
- basic reusable controls, cards, buttons, chips, and lab-only wrappers when production extraction is not yet clean

When a production widget cannot be separated cleanly, add a small lab-only/current-model wrapper and label it with `source_kind`. Do not pretend a wrapper is production-backed unless it directly instantiates or paints the current production component.

## Source Mapping

Current catalog entries use these source contracts:

- Timeline rows: `production-backed`, rendered by `echozero.ui.qt.timeline.widget_canvas.TimelineCanvas` using presentation data assembled by `echozero.application.timeline.assembler.TimelineAssembler`.
- Timeline headers: `production-backed`, rendered by `echozero.ui.qt.timeline.blocks.layer_header.LayerHeaderBlock`.
- Timeline ruler: `production-backed`, rendered by `echozero.ui.qt.timeline.widget_controls.TimelineRuler`.
- Chrome toolbar and transport: `production-backed`, rendered by `TimelineEditorModeBar` and `TransportBar`.
- Panels: `production-backed`, rendered by `SongBrowserPanel` and `ObjectInfoPanel`.
- Forms: `production-backed`, rendered by the reusable `SettingsPageForm` with a compact current-shape settings page.
- Waveforms: `current-model synthetic` when backed by `dev.visual_lab.waveforms`, which registers fun sine/pulse peak data in the same `CachedWaveform` shape consumed by production waveform preview widgets.
- Status chips and control primitives: `lab-only experimental`, retained only for token and primitive exploration.

The compact sample state lives in `dev/visual_lab/current_state.py`. It intentionally creates a tiny current-model timeline/session pair instead of importing deleted demo apps, realistic timeline JSON fixtures, or screenshot-only mutation shims. Synthetic waveform data lives in `dev/visual_lab/waveforms.py`; it is useful preview data, not a claim about user audio.

## Extend State

Add or adjust compact current-model data in `dev/visual_lab/current_state.py`, then add or update catalog entries in `dev/visual_lab/catalog_entries.py`. Update `dev/visual_lab/widgets.py` only when a preview needs a new lightweight wrapper or a production paint block needs an isolation harness. Add synthetic waveform/cache providers in `dev/visual_lab/waveforms.py` when a preview needs stable waveform data without local audio files.

For every new entry:

- choose the narrowest useful component or object state
- set `source_kind` to `production-backed`, `current-model synthetic`, or `lab-only experimental`
- include a concrete `source_path`
- add `part_ids` when the entry is a composition or decomposes a larger surface
- add `editable_token_paths` for the tokens the object owns or visibly consumes
- add `style_targets` for every meaningful sub-element that should be addressable
- update focused tests for category coverage, metadata completeness, and stale demo-fixture import guards

Editable token mappings can be representative when a production widget is not
fully tokenized yet. Prefer a small explicit tuple such as `HEADER_TOKENS`,
`TIMELINE_ROW_TOKENS`, `STATUS_TOKENS`, or `CONTROL_TOKENS` in
`dev/visual_lab/catalog_entries.py`, then broaden it as the preview starts using
more tokens. Global/theme entries should expose `GLOBAL_COLOR_TOKENS`; object
entries should avoid unrelated knobs so selection keeps the sidebar focused.

Nested style targets live in `dev/visual_lab/style_targets.py`. Each target is a
component plus part path plus property list, for example:

```python
style_target(
    "timeline.layer_row",
    "header.title",
    "Header title",
    (("color", "palette.text"), ("font_family", "fonts.family")),
)
```

The first pass maps these part properties onto the current token vocabulary.
That is intentional: the schema is already deep enough for exhaustive
customization, while the token set can grow gradually into per-sub-element paths
such as `timeline.layer_row.header.title.color` or
`transport.button.play.icon.color`.

Keep production EchoZero paths free of lab-only styling and screenshot logic.
