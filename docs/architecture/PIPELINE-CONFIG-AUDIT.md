# PipelineConfig Audit — What-Ifs

Date: 2026-03-28

## Findings

### 🟡 WI-A: Phantom knob values (low risk)

**Scenario:** `with_knob_value("nonexistent_knob", 999)` succeeds silently.
The value gets stored in `knob_values` but never touches any block settings.

**Impact:** Orphaned data in knob_values. No functional harm — the knob just
does nothing. But it's messy.

**Fix:** `with_knob_value()` should validate that the key matches at least one
block setting OR is a known knob in the template. For now: acceptable. Template
validation at `create_config` time already catches unknown bindings.

**Status:** Deferred — not a real user scenario since knobs come from the UI,
not typed by hand.

---

### 🔴 WI-B: Shared setting names across blocks (design issue)

**Scenario:** `full_analysis` has 4 DetectOnsets blocks (drums, bass, vocals, other).
All share `threshold` as a setting name. `with_knob_value("threshold", 0.9)` blasts
ALL of them to 0.9.

**Impact:** User can't set different thresholds per stem. "Make drums more sensitive
but keep vocals less sensitive" is impossible.

**Root cause:** Knob-to-block mapping is by setting name, not by block ID. The knob
system was designed for single-block pipelines.

**Fix options:**
1. **Namespaced knobs:** `drums_threshold`, `bass_threshold` — template declares
   which block each knob maps to. Knob gets a `block_id` field.
2. **Per-block settings UI:** Each block in the graph has its own settings panel.
   Knobs are just the "quick settings" for the most common tweaks.
3. **Both:** Knobs for global settings, per-block panel for fine-tuning.

**Recommendation:** Option 3. Add `maps_to_block: str | None` field on Knob.
When set, `with_knob_value` only updates that specific block. When None (default),
updates all blocks with that setting (current behavior = "global" knob).

**Status:** Needs fix before full_analysis ships.

---

### 🟡 WI-C: Last-write-wins on concurrent edits (expected)

**Scenario:** Two reads of the same config, each modifies a different knob, both
write back. Second write overwrites first change.

**Impact:** In a desktop app with a single UI thread, this basically can't happen.
The UI always reads → modifies → writes sequentially. Only a concern if we ever
have collaborative editing or background services modifying configs.

**Fix:** Not needed for V1. If needed later: optimistic locking via `updated_at`
column (reject write if `updated_at` doesn't match what you read).

**Status:** Acceptable for V1.

---

### ✅ WI-D: Serialization performance (no issue)

**Results:**
- `graph_json` size: 2.9 KB (full_analysis with 8 blocks)
- `with_knob_value`: 0.09ms per call (deserialize + modify + reserialize)
- `to_pipeline`: 0.05ms per call
- DB update + read round-trip: 0.02ms

**Impact:** None. Even a 50-block custom pipeline would be well under 1ms for
any operation. Graph serialization is not a bottleneck.

---

### 🟡 WI-E: Template version migration (needs strategy)

**Scenario:** User saves a config with v1 template (threshold, method). We ship
v2 template that adds `min_gap` knob. User's saved config is missing it.

**Current behavior:** Execution still works — the block has default settings
baked in from creation time. But the UI won't show the new knob (it's not in
`knob_values`), and the user can't tweak it.

**Fix:** Template upgrade migrations. When the app detects that a config's
`template_id` has a newer version than what the config was created with:
1. Compare template knobs vs config knob_values
2. Add missing knobs with defaults
3. Optionally rebuild graph from template (preserving user's knob values)
4. Update config in DB

**Implementation:** Add `template_version` to PipelineConfig and PipelineTemplate.
Migration runs on project open when version mismatch detected.

**Status:** Deferred to pre-release. Not needed during development since we
control all test data.

---

## Summary

| Issue | Severity | Action |
|-------|----------|--------|
| WI-A: Phantom knobs | 🟡 Low | Deferred — no real user path |
| WI-B: Shared setting names | 🔴 Design | Fix before full_analysis ships |
| WI-C: Concurrent edits | 🟡 Low | Acceptable for desktop V1 |
| WI-D: Perf | ✅ None | No action needed |
| WI-E: Template migration | 🟡 Medium | Add template_version pre-release |
