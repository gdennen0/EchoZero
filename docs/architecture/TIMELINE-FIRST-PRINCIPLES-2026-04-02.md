# Timeline First Principles — 2026-04-02

Status: reference
Last reviewed: 2026-04-30


## Purpose

Lock the application model for layers, takes, pipelines, provenance, staleness, MA3 sync, and song-version evolution before more UI is built.

This note supersedes any UI assumptions that treat takes as selectable alternates. Main is truth.

---

## 1. Truth model

### Main is truth
Main layer content is the only truth for:
- playback/output
- editor truth
- MA3 sync
- export
- freshness/staleness comparison

### Takes are subordinate
Takes are:
- rerun results
- frozen candidates
- history
- comparison material
- sources to promote/merge/subtract into main

Takes are NOT independently active/live truth.

### UI implication
- parent row = main
- expanded child rows = takes
- mute/solo only on main row
- no active-take UX model

---

## 2. Pipeline/application boundary

### Engine responsibilities
Engine stays ignorant of editor/timeline semantics.
It only:
- executes graphs
- returns typed outputs
- knows nothing about layers, takes, stale state, song-version carry-forward, or sync

### Application/orchestrator responsibilities
Application maps outputs into editor state.
It must:
- resolve pipeline outputs to stable result layers
- append takes by default on rerun
- record provenance
- track freshness relative to upstream main
- leave downstream untouched unless user explicitly reruns/promotes

### Output contract
Pipelines must declare outputs explicitly.
This is critical for separator-style pipelines where different models may emit different output sets.

Application must support:
- stable mapping when a declared output already has an existing layer
- create-on-first-run when it does not
- append-take-on-rerun by default

---

## 3. Rerun semantics

### Default
Reruns create a new NON-main take.
They do not auto-promote.

Future UI may offer:
- rerun and create take (default)
- rerun and prompt
- rerun and replace main

But default behavior is always:
- create take
- user explicitly promotes or merges later

### Stable layers
Named pipeline outputs map to stable result layers.
Examples:
- separator output `drums` maps to `Drums` audio layer
- classifier output `kick` maps to `Kick` event layer

Rerunning should not create sibling layer clutter when a stable mapping exists.

---

## 4. Freshness / staleness

### Stale only on upstream main change
A downstream derived layer becomes stale only when its upstream source MAIN changes.

A new upstream non-main take does NOT make downstream stale.

### Why
A take has no semantic effect until promoted/pushed into main.
Therefore stale is tied to main-truth transitions only.

### Stale is not the same as provenance
A layer can:
- still be derived from a source
- have been manually edited
- not currently be stale

These are separate concepts.

---

## 5. Provenance and manual edits

Every generated layer/take should preserve provenance.
At minimum this should support:
- source layer id
- source song version id
- source main revision identity
- pipeline/template id
- output name
- execution/run id

Manual edits must not erase provenance.
They should mark the result as manually modified.

### Distinct concepts
Need separate concepts for:
- provenance: where did this come from?
- freshness: does this still match current upstream main?
- manual modification: did a user alter this after generation?

These must not be collapsed into one flag.

---

## 6. MA3 sync boundary

MA3 sync reads/writes MAIN only.

Non-main takes do not sync directly.
Promoting/merging into main changes sync truth.

This is a simplifying boundary and should remain strict.

---

## 7. Song versions

### New song version starts as blank slate
A new SongVersion should begin as a blank editor state.
Do not silently carry previous processed layers into the new version as if they are current truth.

### What gets carried forward now
Only pipeline configs/settings should be copied forward by default.
This allows the user to rerun pipelines to reproduce similar state intentionally.

### Future carry-forward/remap direction
Need to leave room for tools that help migrate previous processed info into a new version:
- rerun unchanged generated layers
- manual section moves
- beat/tempo realignment tools
- automatic remap/alignment algorithms

These are NOT required now, but the architecture must not block them.

### Important rule
Until remap exists, previous-version results are not auto-truth for the new version.
The user should intentionally rerun or manually rebuild state.

---

## 8. Real-world version changes

Examples that must be respected:
- intro shortened
- pre-chorus extended by 8 bars
- tempo changed by +2 BPM
- arrangement sections added/removed

This means future version carry-forward must be treated as:
- rerun
- manual intervention
- auto remap with confidence/review

Not as naive copy-forward.

---

## 9. Current codebase audit delta

### Good / aligned
- persisted take system with exactly one main take per layer
- reruns already append non-main takes for EventData outputs
- song versions already exist as first-class persistence entity
- pipeline configs copy forward on new song versions

### Missing / needs correction
- AudioData outputs are not yet persisted as timeline layers/takes
- no first-class provenance/freshness/manual-modified model on persistence/application side
- new UI shell still carries old active-take assumptions
- song version flow copies configs only, but has no explicit remap/rerun strategy note in app services
- no source/parent inspection surface yet

### Immediate next implementation priorities
1. Persist audio outputs as stable audio layers + takes, not disk-cache-only
2. Introduce provenance/freshness/manual-modified concepts in persistence/application models
3. Refactor UI away from active-take semantics toward main + take lanes
4. Add source/parent inspection affordances
5. Keep song-version creation as blank slate + copied configs, with remap hooks deferred but designed for

---

## 10. Anti-bandaid rule

Do not patch the UI around a false mental model.
If a UI assumption conflicts with these rules, change the model first, then the UI.
