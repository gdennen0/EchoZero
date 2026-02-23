# EchoZero Theme System Overhaul: Socratic Analysis and Migration Plan

**Status:** Analysis complete; implementation not started  
**Date:** February 2025

---

## Phase 1: Audit Results

### 1.0 Audit Summary Table

Every file with theming issues, categorized by severity:

| File | Severity | Issue | Action |
|------|----------|-------|--------|
| `block_item.py` | High | 2 hardcoded QColor (overlay, text contrast) | Add tokens; replace |
| `audio_player_block_item.py` | High | 4 hardcoded QColor (overlays, alpha) | Add tokens; replace |
| `eq_bands_block_item.py` | High | 4 hardcoded QColor + 1 rgba in setStyleSheet | Add FILTER_*, OVERLAY_*; replace |
| `audio_filter_block_item.py` | High | 4 hardcoded QColor (shelf/peak) | Add FILTER_*; replace |
| `node_scene.py` | Medium | 1 hardcoded QColor (grid) | Add GRID_LINE; replace |
| `command_history_dialog.py` | Medium | 2 QColor("white") | Use Colors.TEXT_PRIMARY |
| `timeline/core/style.py` | Medium | 31 QColor - domain-specific | Keep structure; ensure apply_theme syncs shared tokens |
| `timeline/settings/storage.py` | Low | Default hex in config | Document; keep as settings |
| `timeline/events/items.py` | Low | Uses settings for shadow color | OK - config-driven |
| `block_panel_base.py` | OK | Uses Colors.name(); has _on_theme_changed | No change |
| `main_window.py` | OK | Uses get_stylesheet, Colors, StyleFactory | No change |
| `settings_dialog.py` | OK | ThemeAwareMixin; Colors.name() | No change |
| `setlist_view.py` | OK | ThemeAwareMixin; StyleFactory | No change |
| Other ThemeAwareMixin users (12 files) | OK | Participate in theme refresh | No change |
| Block panels (30+ files) | OK | Extend BlockPanelBase; get theme updates | No change |
| Panels without ThemeAwareMixin | Low | May have stale styles if they set local stylesheet at init | Audit; add theme subscription if needed |

### 1.1 How Colors Actually Works

**Pattern:** Mutable class-level `QColor` attributes. `Colors.apply_theme()` reassigns them:

```python
cls.BG_DARK = theme.bg_dark
cls.BORDER = theme.border
# ... etc.
```

**Implications:**

| Scenario | Behavior |
|----------|----------|
| Widget reads `Colors.BG_DARK` at **paint time** | Correct. Gets current theme value. |
| Widget reads `Colors.BG_DARK` at **construction** and caches it | **Stale after theme switch.** Cached reference holds the old `QColor`. |
| QSS using `Colors.BG_DARK.name()` | **Stale.** Global stylesheet is regenerated on theme change via `get_stylesheet()`, so the *new* QSS has correct values. But any widget that set `setStyleSheet(f"color: {Colors.BG_DARK.name()}")` at init and never re-applies will keep the old hex string. |

**Critical finding:** `BlockItem._color` is set once from `Colors.get_block_color(block_type)` at construction. After theme switch, `_color` still holds the previous theme's block color. The header bar is painted with this cached value. The current workaround: `main_window._apply_theme()` calls `node_editor_window.refresh()`, which does a full `scene.refresh()` and recreates all block items. That works but is heavy (reloads from DB, recreates all items).

---

### 1.2 ThemeAwareMixin Behavior

**Implementation (design_system.py:84-93):**

```python
def _on_theme_changed_mixin(self):
    for child in self.findChildren(QWidget):
        if child.styleSheet():
            child.setStyleSheet("")
    if hasattr(self, 'setStyleSheet') and self.styleSheet():
        self.setStyleSheet("")
    self._apply_local_styles()
```

**Analysis:**

- **Destructive:** Clears *all* child stylesheets unconditionally. Any widget using stylesheet for layout (e.g. `min-width`, `padding`) or conditional formatting loses it.
- **Scope:** Only 15 files use `ThemeAwareMixin`. Block panels use `BlockPanelBase` which has its own `_on_theme_changed` with the same clear-then-reapply pattern.
- **Files using ThemeAwareMixin:** `settings_dialog`, `setlist_view`, `setlist_action_config_panel`, `action_set_editor`, `add_editor_layer_dialog`, `setlist_error_summary_panel`, `add_ma3_track_dialog`, `add_action_dialog`, `setlist_processing_dialog`, `command_history_dialog`, `action_picker_dialog`, `event_filter_dialog`, `properties_panel`, `actions_panel`.
- **Other theme subscribers:** `BlockPanelBase` (all block panels), `TimelineWidget`, `ProgressBar` (via `on_theme_changed`).

---

### 1.3 get_stylesheet() Coverage

**Widget types styled:** QMainWindow, QWidget, QMenuBar, QMenu, QToolBar, QToolButton, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QRadioButton, QTextEdit, QPlainTextEdit, QTabWidget/QTabBar, PyQt6Ads dock widgets (ads--CDockWidget, ads--CDockWidgetTab, ads--CDockAreaTitleBar, etc.), QGroupBox, QScrollBar, QProgressBar, QTreeView/QListView/QTableView, QHeaderView, QScrollArea, QToolTip, QGraphicsView, QLabel, QStatusBar, QDialog.

**Potentially missing:** QSlider, QDateEdit, QTimeEdit, QDateTimeEdit (if used), some ADS edge cases.

**Granularity:** Applied to `QApplication.setStyleSheet()`. Child widgets inherit; local `setStyleSheet()` overrides with higher specificity.

---

### 1.4 Hardcoded Colors Audit

| File | Instances | Category | Recommendation |
|------|-----------|----------|---------------|
| `block_item.py` | 2 | Missing tokens | Add `Colors.OVERLAY_SUBTLE` (QColor(255,255,255,18)), `Colors.TEXT_ON_DARK` / `Colors.TEXT_ON_LIGHT` |
| `audio_player_block_item.py` | 4 | Missing tokens | Add `Colors.OVERLAY_FEINT` (255,255,255,15), `Colors.OVERLAY_DIM` (255,255,255,20); waveform alpha from accent is domain logic |
| `eq_bands_block_item.py` | 4 | Semantic (shelf vs peak) | Add `Colors.FILTER_SHELF` (80,200,100), `Colors.FILTER_PEAK` (220,140,60) or use ACCENT_GREEN/ACCENT_ORANGE |
| `audio_filter_block_item.py` | 4 | Same as eq_bands | Same tokens |
| `node_scene.py` | 1 | Grid line | Add `Colors.GRID_LINE` or use `Colors.BORDER` |
| `timeline/core/style.py` | 31 | Domain-specific | Keep PLAYHEAD, SELECTION, SNAP_LINE, GRID_MAJOR/MINOR, LAYER_COLORS as TimelineStyle; sync shared tokens from Colors (already done via `apply_theme()`) |
| `eq_bands_block_item.py` (setStyleSheet) | 1 | Alternate row | `rgba(255,255,255,6)` -> `Colors.OVERLAY_VERY_SUBTLE` |
| `command_history_dialog.py` | 2 | Bug | `QColor("white")` -> `Colors.TEXT_PRIMARY` |
| `timeline/settings/storage.py` | default hex | Config defaults | Keep as settings; ensure they map to theme tokens where appropriate |
| `design_system.py` (Effects.SHADOW_COLOR) | 1 | Token | Keep in design_system; already centralized |

---

### 1.5 setStyleSheet() Pattern Analysis

**Totals:** ~70 files with `setStyleSheet`; ~350+ calls.

** categorization (sample-based):**

| Pattern | Count (approx) | Theme-aware? |
|---------|----------------|--------------|
| `StyleFactory.method()` | ~110 | Yes (reads Colors at call time) |
| `Colors.X.name()` in f-string | ~250+ | Yes if re-applied on theme change |
| Hardcoded hex/rgba | ~5 | No |
| Qt property / inherits from global | Many | Yes |

**Files with hardcoded hex/rgba in stylesheet:**
- `eq_bands_block_item.py`: `rgba(255,255,255,6)`
- `audio_player_block_item.py`: `rgba(...)` built from Colors (theme-aware)
- `command_history_dialog.py`: `QColor("white")` in setForeground (not stylesheet, but hardcoded)

**Conclusion:** The vast majority of stylesheet usage is theme-aware. The main problem is widgets that set stylesheet once at init and never re-apply when theme changes. `ThemeAwareMixin` and `BlockPanelBase._on_theme_changed` address this by clearing and re-applying. Widgets that *don't* use these (e.g. some dialogs, panels that bypass the base) may have stale styles.

---

## Phase 2: Architecture Evaluation

### 2.1 Mutable Class Attribute Pattern

**Research (Qt docs, Stack Overflow):** QPalette alone is unreliable for cross-platform theming; QSS is preferred. The "placeholder tokens" pattern (format stylesheet with current colors at theme-switch time) is standard.

**Verdict:** The mutable-class-attribute pattern is sound for PyQt6:

1. **Central definition:** Colors is the single source; Theme defines presets.
2. **Live switching:** `apply_theme()` updates class attributes; global QSS is regenerated; `theme_changed` triggers local re-application.
3. **Custom-painted widgets:** Must read `Colors.X` at **paint time**, not cache at construction.
4. **QSS widgets:** Global QSS is regenerated; local overrides must be re-applied via `theme_changed` (StyleFactory/`_apply_local_styles`).

**Required fix:** Custom-painted widgets that cache colors (e.g. `BlockItem._color`) must either subscribe to `theme_changed` and update, or be recreated. Prefer subscription for performance.

---

### 2.1b Benchmark: Comparison to Existing Projects

Research into Qt theming implementations and design systems yields these comparisons and borrowable ideas:

| Project | Approach | EchoZero Comparison | Ideas to Borrow |
|---------|----------|---------------------|-----------------|
| **Qt-Material** (PyQt6) | XML theme files, `apply_stylesheet()`, 18 built-in themes, custom accent + font | We use Theme dataclass + ThemeRegistry; they use XML. We have 20+ presets; similar scale. | **Export to .qss:** Allow exporting current theme to a standalone .qss file for sharing or C++ builds. **Density scale:** Optional accessibility knob (e.g. 0.8x, 1.0x, 1.2x) for spacing/size. **Extra dict:** Per-widget overrides (e.g. `extra={"QMenu": {"height": 50}}`) for edge cases. **Env vars:** Expose read-only theme vars (e.g. `QTMATERIAL_PRIMARYCOLOR`) for debugging or tooling. |
| **QDarkStyleSheet v3** | Theme framework, refined palette (Spyder collaboration), SCSS for programmatic access | We use Python classes; they use SCSS/build step. | **Minimal base palette:** v2 had "only 9 colors" – derive others programmatically. We could reduce Theme dataclass to ~9 base tokens and compute OVERLAY_*, FILTER_* from them. **SCSS-like layer:** A build step or generator that produces QSS from a minimal token set could simplify maintenance. |
| **Qt Color Editor Factory** | `QItemEditorFactory` + `QStandardItemEditorCreator` for color cells in `QTableView` | We plan table + color delegates. | **Use `QItemEditorFactory`:** Register a custom color editor so *any* table cell with `Qt.ItemDataRole.BackgroundRole` or `UserRole` color automatically gets a color picker. Reduces custom delegate code. Official Qt pattern. |
| **Design Tokens (Web)** | CSS variables, JSON as canonical format, Style Dictionary for multi-platform export | We use Python dataclasses. | **JSON canonical:** Consider a `tokens.json` as the single source; Python/ThemeRegistry loads from it. Enables design-tool workflows. **Named semantic tokens:** "color.background.dark" vs raw hex – we already have BG_DARK; ensure naming is consistent and documented. |
| **Style Dictionary** | JSON in → CSS/Swift/XML/etc. out. Build-time transformation. | We are runtime only. | **Documentation output:** A build step that generates a style reference (HTML/Markdown) from our tokens could help designers and developers. Optional, not required. |
| **Spyder IDE** | Preferences > Appearance; dark/light; syntax themes; ini for custom themes | We have Settings > Theming. | **Preview before apply:** Spyder lets you preview syntax themes before applying. We could add a "Preview" split: left = current, right = edited, Apply to commit. **Ini/JSON export:** Allow exporting theme to a portable file for sharing. |

**Summary of borrowable ideas:**
1. **QItemEditorFactory** for color cells – use Qt's built-in pattern instead of hand-rolled delegate.
2. **Export to .qss** – one-click export of current theme for sharing or debugging.
3. **Density scale** – optional accessibility setting (future).
4. **Preview-before-apply** – side-by-side or A/B preview for theme edits.
5. **Minimal base palette** – consider deriving overlay/filter colors from base tokens to reduce Theme surface area.
6. **JSON canonical format** – optional; design-tool round-trip if valuable later.

---

### 2.2 ThemeAwareMixin

**Problems:**
- Nuclear clear removes non-theme stylesheet usage.
- Opt-in means easy to forget; new widgets may not participate.

**Recommendation:**
- **Narrow the clear:** Only clear stylesheets that are known to contain theme tokens. This is non-trivial (would require tracking which widgets have theme-derived styles). Simpler: document that theme-aware widgets must not use stylesheet for layout-only purposes, or use a separate mechanism (e.g. `setMinimumWidth`).
- **Alternative:** Make theme reactivity automatic via global QSS cascade. Reserve ThemeAwareMixin for widgets that *must* set local StyleFactory/Colors overrides (primary buttons, status labels). Widgets that rely purely on global QSS need no mixin.
- **Practical approach:** Keep current behavior but add a `_preserve_stylesheet_for(widget)` or similar for layout-only styles. Low priority; most theme-aware widgets use StyleFactory which is re-applied.

---

### 2.3 StyleFactory

**Usage:** ~15 files use it heavily; ~60 import design_system. Many use inline `f"color: {Colors.X.name()}"` for one-off overrides.

**Verdict:** StyleFactory is useful and should remain. Expand where it reduces duplication (e.g. status label variants). Developers bypass it for one-off overrides; that's acceptable if they use `Colors.X.name()` and the widget participates in theme refresh.

---

### 2.4 Custom-Painted Widgets (QPainter)

**Current:** Node editor blocks, timeline items, waveform draw with `QPainter`. They read `Colors.X` at paint time.

**BlockItem exception:** `self._color` is cached from `Colors.get_block_color()` at construction. Header bar uses this; it stays stale until `refresh()` recreates the block.

**Recommendation:** BlockItem (and similar) should subscribe to `theme_changed` and do:
```python
self._color = Colors.get_block_color(self.block.type)
self.update()
```

---

## Phase 3: Target State

### 3.1 Non-Negotiable Requirements

1. Every visible UI element responds to theme changes.
2. No hardcoded colors outside `design_system.py` (or a documented exception list).
3. Theme switching is instant (no restart).
4. Custom-painted widgets participate (read Colors at paint, or update cache on theme change).
5. New widgets naturally "fall into the pit of success" (convention, docs, base classes).
6. **Initialize-on-create (no flip):** Windows and dialogs are created with correct theme values from the first paint. No visible "flash" or "flip" where a window briefly shows default/wrong colors before switching to the correct theme. Theme must be applied before any window is shown.
7. **Global theme editor:** A single table-based menu from which every theme option (Colors, Typography, Spacing, Sizes, flags) can be edited. Highly editable—click-to-edit, no nested dialogs for simple values. All UI refactored to obey the design system; no hardcoded values anywhere.

---

### 3.2 Initialize-on-Create: No Visible Flip

**Goal:** Every window appears with the correct theme from its first frame. No brief display of default colors followed by a correction.

**Current gaps:**
- **Splash screen:** Created and shown before `Colors.apply_theme()` runs. Uses Colors class defaults (hardcoded in class body), not user's saved theme.
- **Login dialog:** Shown during `_authenticate()`, before any theme application. Same issue.
- **Main window:** Theme is applied in `MainWindow.__init__` before `show()`, so it is correct when first visible.
- **Dialogs opened later:** Created after MainWindow; Colors already has correct theme. No flip if they read Colors at construction.

**Required changes:**
1. **Earliest possible theme application:** Call `Colors.apply_theme()` immediately after `QApplication` creation in `main_qt.py`, before splash or login. At that moment `app_settings` does not exist (DB loads during `initialize_services`, which runs after login). `apply_theme()` will fall back to "default dark". Splash and login will use a consistent baseline—no flip between them.
2. **User theme before MainWindow:** When `initialize_services` completes and `app_settings` is available, apply the user's theme preset *before* creating MainWindow. `qt_application.initialize()` already does `_setup_dark_theme()` before MainWindow; extend this to call full `Colors.apply_theme(theme_name)` with `app_settings.theme_preset`. MainWindow will then appear with the correct theme from its first frame.
3. **Acceptable transition:** Splash and login will always use "default dark" (user theme is in DB, which loads after login). The transition from splash (default dark) to MainWindow (user's theme) may be visible if the user has a non-default preset. This is acceptable; the alternative would require loading theme from persistence before login, which adds complexity. The critical guarantee: no *within-window* flip (a window never appears with wrong colors then corrects itself).
4. **Document the contract:** Any code that creates a window must assume `Colors` is already synced. Theme-changed signals are for *reactivity* when the user switches themes mid-session, not for initial setup.

**Migration checklist item:** Add "Initialize-on-create sequence" to Phase 4, with specific edits to `main_qt.py` and `qt_application.py`.

---

### 3.3 Minimal Architecture

| Component | Role |
|-----------|------|
| **Colors** | Single source of truth; mutable class attributes. |
| **get_stylesheet()** | Global QSS on QApplication; handles ~90% of standard widgets. |
| **ThemeAwareMixin** | For dialogs/panels that hold variant overrides (primary button, etc.). Clear child stylesheets, then `_apply_local_styles()`. |
| **StyleFactory** | Recommended for variant styles (primary, danger, table, etc.). |
| **on_theme_changed** | Block items, timeline, progress bar, etc. update cached colors and repaint. |

---

### 3.3 Missing Color Tokens

| Token | Default | Use Case |
|-------|---------|----------|
| `OVERLAY_SUBTLE` | QColor(255,255,255,18) | Block border when not selected |
| `OVERLAY_FEINT` | QColor(255,255,255,15) | Center line, dividers |
| `OVERLAY_DIM` | QColor(255,255,255,20) | Zoom bar background |
| `OVERLAY_VERY_SUBTLE` | QColor(255,255,255,6) | Alternate row highlight |
| `TEXT_ON_LIGHT` | QColor(20,20,25) | Header text on light block color |
| `TEXT_ON_DARK` | QColor(255,255,255) | Header text on dark block color |
| `FILTER_SHELF` | QColor(80,200,100) | Shelf/boost in EQ/filter viz |
| `FILTER_PEAK` | QColor(220,140,60) | Peak in EQ/filter viz |
| `GRID_LINE` | QColor(60,60,60) | Node scene grid |

**TimelineStyle tokens (stay in TimelineStyle, optionally synced):**
- PLAYHEAD_COLOR, SELECTION_COLOR, SNAP_LINE_COLOR, GRID_LINE_MAJOR, GRID_LINE_MINOR, LAYER_COLORS.

---

### 3.4 Global Theme Editor: Table-Based, Highly Editable

**Goal:** A single, simple menu/panel from which every theme-related option in Qt can be edited. All UI across the application must read from the design system—no exceptions.

**UI Design:**
- **Layout:** A table. Each row = one setting. Columns = [Setting Name | Value (editable) | Preview/Swatch (for colors)].
- **Simple and scannable:** No nested groups, no tabs for core options. One flat list. Optional: group rows by category (Colors, Typography, Spacing, Sizes) via section headers or a "Category" column, but the primary interaction is row-by-row editing.
- **Highly editable:** Every cell is directly editable. Click-to-edit, no extra dialogs for simple values. Color cells: swatch + click opens picker (or inline hex editor). Numeric cells: QSpinBox or QDoubleSpinBox. String cells: QLineEdit. Boolean: QCheckBox.

**Scope — All theme options globally:**
- **Colors:** Every token in `Colors` (BG_DARK, BG_MEDIUM, ..., OVERLAY_SUBTLE, FILTER_SHELF, etc.). ~35+ rows.
- **Typography:** Font family, font size. (heading_font, mono_font if overridable via settings.)
- **Spacing:** XS, SM, MD, LG, XL, XXL (if made user-editable).
- **Sizes:** Block dimensions, port sizes, corner radius, etc. (if made user-editable.)
- **Flags:** Sharp corners (bool).
- **Effects:** Shadow color, animation durations (if exposed).

**Persistence:** Edits apply live via `Colors.apply_theme_from_dict()` (or equivalent). Preset selector at top: load built-in or custom preset, then override any row. Save as new preset or update current custom preset.

**Refactor mandate:**
- **All UI obeys this system.** No widget may hardcode a color, font, spacing, or size. Every visual value flows from `design_system` (Colors, Typography, Spacing, Sizes, etc.) or from the theme editor's persisted state. Refactor scope is unbounded—if a file uses hardcoded values, it gets refactored regardless of size.
- **Single source of truth:** The theme editor reads from and writes to the same design system that all UI consumes. No duplicate definitions, no "settings-only" values that bypass Colors.

**Implementation notes:**
- Replace the current Theming tab (grouped swatches, separate font controls) with the table-based editor. May be a dedicated "Theme Editor" window/dialog, or the main Theming tab redesigned.
- Use `QTableWidget` or `QTableView` with custom delegates: color delegate (swatch + picker), spinbox delegate, etc. Qt's `QStyledItemDelegate` supports per-cell editor widgets.
- The table's data model can be a simple list of `(key, value, type)` tuples generated from `ThemeRegistry.COLOR_FIELDS` plus Typography/Sizes/Spacing keys. One source, one UI.

---

## Phase 4: Migration Plan

### 4.1 What Stays As-Is

- Colors class as single source of truth.
- ThemeRegistry for presets and persistence.
- ThemeSignals / on_theme_changed.
- get_application_palette().
- StyleFactory concept and API.
- get_stylesheet() structure.

---

### 4.2 Refactors

| Item | Change |
|------|--------|
| **Startup theme sequence** | Apply theme before first window (splash/login); apply user preset before MainWindow. Ensure no flip on create. |
| ThemeAwareMixin | Document limitations; consider targeted clear (future) |
| get_stylesheet() | Add QSlider, etc. if used; verify ADS coverage |
| Colors | Add missing tokens (OVERLAY_*, TEXT_ON_*, FILTER_*, GRID_LINE) |
| Theme dataclass | Add optional theme overrides for new tokens |
| BlockItem, AudioPlayerBlockItem, etc. | Subscribe to theme_changed; update cached colors; call update() |
| node_editor refresh on theme | Consider lighter repaint (invalidate + update) instead of full scene.refresh() |
| **Theme editor UI** | Replace current Theming tab with table-based editor. Rows = settings, columns = name/value/preview. All design system values editable. |

---

### 4.3 Deletions / Replacements

| Location | Replacement |
|----------|-------------|
| block_item.py: QColor(255,255,255,18) | Colors.OVERLAY_SUBTLE |
| block_item.py: QColor(255,255,255) / QColor(20,20,25) | Colors.TEXT_ON_DARK / Colors.TEXT_ON_LIGHT |
| audio_player_block_item.py: QColor(255,255,255,15) | Colors.OVERLAY_FEINT |
| audio_player_block_item.py: QColor(255,255,255,20) | Colors.OVERLAY_DIM |
| eq_bands_block_item.py: QColor(80,200,100), QColor(220,140,60) | Colors.FILTER_SHELF, Colors.FILTER_PEAK |
| eq_bands_block_item.py: rgba(255,255,255,6) | Colors.OVERLAY_VERY_SUBTLE.name() with alpha |
| audio_filter_block_item.py: same | Same tokens |
| node_scene.py: QColor(60,60,60) | Colors.GRID_LINE |
| command_history_dialog.py: QColor("white") | Colors.TEXT_PRIMARY |

---

### 4.4 File-by-File Migration Order

**Tier 0 – Initialize-on-Create (No Flip)**
1. `main_qt.py` – Call `Colors.apply_theme()` immediately after `QApplication` creation, before splash. Ensures splash and login use at least default-dark consistently.
2. `qt_application.py` – In `initialize()`, after storing `app_settings` on the app and before creating MainWindow, call `Colors.apply_theme()` with the user's theme preset from `app_settings.theme_preset`. This ensures MainWindow is created with the correct theme from the first frame.
3. Verify splash and login receive correct values; document the startup theme sequence in code comments.

**Tier 1 – Design System & Tokens**
1. `design_system.py` – add OVERLAY_SUBTLE, OVERLAY_FEINT, OVERLAY_DIM, OVERLAY_VERY_SUBTLE, TEXT_ON_LIGHT, TEXT_ON_DARK, FILTER_SHELF, FILTER_PEAK, GRID_LINE.
2. `theme_registry.py` – add optional fields for new tokens; ensure `apply_theme` / `theme_from_dict` handle them.

**Tier 1b – Global Theme Editor (table-based)**
3. Replace `_create_theming_tab()` in `settings_dialog.py` (or create dedicated ThemeEditorDialog) with table-based UI.
4. Table: rows = each design system token (Colors, Typography, Spacing, Sizes, sharp_corners). Columns = [Name | Value | Preview].
5. Use `QItemEditorFactory` + custom color editor for color cells (Qt's standard pattern); numeric/string/bool delegates for other types.
6. Preset selector at top; load/save presets; live preview via `Colors.apply_theme_from_dict()`.
7. Ensure Typography, Spacing, Sizes are persisted in app_settings if they become editable; wire design system to read from settings.
8. (From benchmark) Add **Export to .qss** button; optional **Preview-before-apply** split for larger edits.

**Tier 2 – Node Editor (high visibility)**
8. `block_item.py` – use new tokens; subscribe to theme_changed; update `_color` and `update()`.
9. `audio_player_block_item.py` – use new tokens.
10. `eq_bands_block_item.py` – use FILTER_SHELF, FILTER_PEAK, OVERLAY_VERY_SUBTLE.
11. `audio_filter_block_item.py` – same.
12. `node_scene.py` – use Colors.GRID_LINE.
13. `node_editor_window.py` – on theme change, consider scene.invalidate() + view.update() instead of full refresh (optional performance improvement).

**Tier 3 – Dialogs & Panels**
14. `command_history_dialog.py` – QColor("white") -> Colors.TEXT_PRIMARY.
15. Review block panels for any remaining hardcoded colors (grep-driven).

**Tier 4 – Timeline (domain-specific)**
16. `timeline/core/style.py` – ensure apply_theme() syncs any new shared tokens; keep timeline-specific tokens as-is.
17. Timeline settings defaults – document mapping to theme tokens where relevant.

**Tier 5 – Full UI obedience (refactor mandate)**
18. Audit all ~70 UI files for any remaining hardcoded colors, fonts, spacing, or sizes. Refactor every violation to use design system. Scope is unbounded; no exceptions.

---

## Validation Criteria

1. **Theme switch test:** Change theme 3 times (e.g. Default Dark -> NASA -> Arctic White). Every visible element (blocks, docks, buttons, tables, timeline, dialogs) updates immediately.
2. **No hardcoded colors:** `grep -r "QColor([0-9]" ui/` excludes design_system.py, theme_registry.py, and documented exceptions.
3. **No hardcoded hex in stylesheets:** `grep -r "#[0-9a-fA-F]\{6\}" ui/` and `rgba(` in setStyleSheet only in design_system or with Colors-derived values.
4. **Custom-painted refresh:** Node editor blocks repaint correctly on theme change without requiring project reload.
5. **Regression:** All 20+ theme presets load and apply without error.
6. **Initialize-on-create (no flip):** No window ever shows wrong colors then "flips" to correct ones within its lifetime. Splash and login use consistent default-dark (user theme loads after login). MainWindow and all dialogs opened thereafter appear with correct theme from first frame.
7. **Theme editor:** Table shows every design system setting. Each row is editable inline. Color edits preview live. Preset load/save works. No theme-related value exists in the app that cannot be edited from this table.

---

## Summary

The architecture (Colors, ThemeRegistry, StyleFactory, ThemeAwareMixin) is fundamentally sound. The overhaul is a completion and cleanup effort:

1. Add missing color tokens.
2. Replace ~15 hardcoded color usages with tokens.
3. Fix BlockItem and similar to update cached colors on theme change.
4. Fix command_history_dialog hardcoded "white".
5. Implement initialize-on-create so no window flips on first paint.
6. Replace the theme UI with a table-based, highly editable menu (rows = settings, columns = name/value/preview).
7. Refactor all UI to obey the design system; no hardcoded values anywhere.
8. Optionally narrow ThemeAwareMixin’s clear behavior and lighten node editor theme refresh.

No new dependencies; no ground-up rewrite. The refactor mandate (item 7) may be large but is explicit and non-negotiable.
