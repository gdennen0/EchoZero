# Timeline Style Integration Pass 1

Date: 2026-04-07

## Migrated in this pass

- Expanded `echozero/ui/qt/timeline/style.py` from Object Palette-only tokens into typed surface groups for:
  - timeline canvas shell
  - playhead
  - transport bar
  - layer header
  - take row
  - event lane
  - ruler
  - existing object palette and scroll area shell styling
- Replaced hardcoded presentational literals in:
  - `echozero/ui/qt/timeline/blocks/transport_bar_block.py`
  - `echozero/ui/qt/timeline/blocks/layer_header.py`
  - `echozero/ui/qt/timeline/blocks/take_row.py`
  - `echozero/ui/qt/timeline/blocks/event_lane.py`
  - `echozero/ui/qt/timeline/blocks/ruler.py`
  - `echozero/ui/qt/timeline/widget.py`
- Kept layout/geometry in FEEL and existing layout logic. This pass moved colors, text colors, control fills/borders, corner radii, and painter font styling into the style module only.
- Added focused token-plumbing tests to verify the timeline blocks and shell default to shared style tokens instead of embedding local literals.

## Remaining after this pass

- `echozero/ui/qt/timeline/blocks/waveform_lane.py` still owns its lane paint styling directly.
- Tooltip/body copy strings remain local to widget logic; this pass did not treat content strings as style.
- Painter offsets tied to text centering such as `adjusted(0, -1, 0, -1)` remain local because they are rendering alignment behavior, not tokenized theme values.
- No visual regression or image snapshot coverage was added in this pass.

## Focused verification

Command:

```powershell
py -3.12 -m pytest tests\ui\test_timeline_style.py tests\ui\test_timeline_shell.py
```

Result:

```text
40 passed in 1.58s
```
