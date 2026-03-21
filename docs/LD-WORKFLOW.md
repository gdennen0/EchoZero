# LD Workflow Spec
**Source:** Griff brain dump, 2026-03-20
**Last verified:** 2026-03-20

---

## Snap Behavior
- **Toggle snap on/off** (keyboard shortcut)
- **Grid-only snap.** No snapping to other events, layers, onsets, or markers.
- Grid based on timeline's current time units (Reaper-style)
- No other snap targets needed for v1

## Typical Workflow
1. **Add song** to setlist (or add as version of existing song)
2. **Right-click song → "Extract All"** — runs the full default pipeline automatically:
   - Separate stems (Demucs)
   - Detect onsets on each stem
   - Classify events
   - Results appear as layers in the Editor
3. **Review pass** — listen through, make corrections:
   - Delete false positive events (most frequent action)
   - Move events that are slightly off-time
   - Reclassify events (kick → snare, etc.)
   - Add events manually where detection missed
   - Split/merge events occasionally
4. **Push to MA3** — commit-style push, not live sync

## Extraction Options
- **One-click "Extract All"** — full pipeline, default settings. Primary workflow.
- **Granular menu** — choose what to extract (stems only, onsets only, classify only, specific stems). Available but secondary.
- Clean, simple UX. Progressive disclosure — simple default, power user options accessible but not in the way.

## Review Pass — High-Frequency Actions
All must be **instant** — no spinners, no confirmation dialogs:
- Delete selected events (Delete/Backspace)
- Move events (drag, snap to grid)
- Reclassify events
- Add events manually (double-click or shortcut)
- Split/merge (occasional)

## Keyboard Shortcuts
- **Configurable** but not user-facing key mapper for v1
- Use **intuitive defaults** (Delete = delete, Space = play/pause, etc.)
- Internal **shortcut registry** that makes it easy for developers to add new shortcuts
- Not a user-facing preferences panel yet — just clean developer infrastructure

## MA3 Integration (v1)
- **NO live bidirectional sync** for v1
- **Push/pull/commit model** — like git for your console
  - Push: send events from EchoZero → MA3
  - Pull: read current state from MA3 → EchoZero
  - Commit: explicit user action, not automatic
- Live sync = future experimental feature, not v1 scope
- This significantly simplifies ShowManager architecture

## Implications for Architecture
- ShowManager becomes simpler: no real-time sync loop, no conflict resolution during live operation
- OSC Gateway still needed for push/pull, but no persistent connection monitoring
- "Extract All" needs a **default pipeline template** that auto-constructs the right block graph
- Pipeline-as-data (FP1) enables this: the default pipeline is a serialized DAG, not hardcoded logic
- Shortcut registry = lightweight mapping of key combos → command names, stored in a config-like structure

---

*This is Griff's actual workflow. Build for this. Everything else is secondary.*
