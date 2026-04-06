# UI Source Of Truth - 2026-04-05

Purpose: define the practical source-of-truth for Stage Zero timeline planning, reconcile current doc drift, and keep follow-on work pointed at one contract.

## Precedence Order

Use this order when documents conflict:

1. `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md`
   Reason: this is the governing model for truth, takes, provenance, staleness, sync boundary, and song-version behavior.
2. `docs/architecture/DECISIONS.md`
   Reason: later architecture decisions refine product-wide rules and source-of-truth ownership, but do not override First Principles unless explicitly doing so.
3. `echozero/ui/FEEL.py`
   Reason: authoritative for current UI tuning constants only. It does not override architecture or interaction semantics.
4. `docs/UX-DESIGN-DECISIONS.md`
   Reason: authoritative for chosen interaction behavior when not in conflict with First Principles or current FEEL-backed runtime constants.
5. `docs/UX-MICRO-TESTS.md`
   Reason: verification target, not primary authority. When it disagrees with decided behavior, update tests to match decisions.
6. Audit docs
   Files: `docs/UI-CONTRACT-AUDIT-2026-04-03.md`, `docs/DISTILLATION-CONFORMANCE-AUDIT-2026-04-04.md`, `docs/architecture/UI-BLOCK-AUDIT-2026-04-02.md`
   Reason: these are evidence and drift detectors, not canonical behavior specs.
7. `docs/architecture/TIMELINE-TASKS.md`
   Reason: historical build-plan document. Useful for extraction order and FEEL discipline, but stale as a Stage Zero contract.

## Canonical Stage Zero Position

- Main is truth. Takes are subordinate comparison/history lanes, never alternate live truth.
- Stable pipeline outputs map to stable layers; reruns append takes by default.
- Staleness is driven only by upstream main changes.
- MA3 sync boundary is main-only.
- SongVersion starts blank; configs carry forward, processed results do not.
- FEEL owns dimensions, spacing, timing constants, and visual tuning.
- UX decisions should favor the locked decisions doc over older TBDs in micro-tests.
- Current UI planning should assume row/block regionization is still needed before deeper affordance growth.

## Drift Matrix

| Doc / section | Status | Why | Action |
|---|---|---|---|
| `TIMELINE-FIRST-PRINCIPLES` sections 1-8 | current | Matches latest architecture intent and recent audits | Treat as top authority |
| `DECISIONS.md` D266-D267 | current | Confirms glossary/source-of-truth discipline | Reference when docs disagree |
| `FEEL.py` layer/ruler constants | current | Now reflects active 320/72/44 timeline shell values; older audits say otherwise | Use FEEL for current sizing |
| `UX-DESIGN-DECISIONS` D6 hover delay | stale | Says immediate hover; `FEEL.py` still sets `HOVER_DELAY_MS = 150` | Update decision doc or FEEL after Griff confirms desired behavior |
| `UX-DESIGN-DECISIONS` D14 vs D24 nudge-when-snap-off | stale | Internal conflict: fixed small amount vs 1 frame; later D24 is more specific | Canonicalize to D24 and patch doc |
| `UX-MICRO-TESTS` ruler/playhead/follow TBDs | partial | Contains pre-decision TBD language now resolved in design decisions | Update tests to decisions before using as acceptance gate |
| `UI-CONTRACT-AUDIT` C6-C7 FEEL drift findings | stale | Audit says FEEL diverges from runtime, but current `FEEL.py` matches active shell dimensions | Keep as historical evidence; do not treat as current blocker |
| `DISTILLATION-CONFORMANCE-AUDIT` A4 active-take leak | current | Still a real architecture risk per audit evidence | Keep as implementation blocker for Stage Zero |
| `DISTILLATION-CONFORMANCE-AUDIT` A7 FEEL integration pass | partial | "No FEEL integration" is no longer fully true, but FEEL-as-single-source still needs broader wiring proof | Re-audit after timeline modules import FEEL consistently |
| `UI-BLOCK-AUDIT` extraction/regionization guidance | current | Still aligned with next-step UI structure work | Use as structural implementation guide, not behavior contract |
| `TIMELINE-TASKS` overall plan | stale | Written for earlier from-scratch build path and older FEEL defaults | Use only for extraction heuristics; not for contract decisions |

## Immediate Planning Guidance

For Stage Zero timeline work, treat the following as the working contract:

- Semantics: `TIMELINE-FIRST-PRINCIPLES`
- Product-level source-of-truth discipline: `DECISIONS.md`
- Runtime tunables and dimensions: `FEEL.py`
- Interaction choices: `UX-DESIGN-DECISIONS`, except where marked stale above
- Structural cleanup order: `UI-BLOCK-AUDIT`
- Known implementation risks: `DISTILLATION-CONFORMANCE-AUDIT`, then `UI-CONTRACT-AUDIT`

## Griff Decisions Required

Only these still need Griff input to remove real ambiguity:

1. Hover delay: keep immediate hover from `UX-DESIGN-DECISIONS` D6, or keep current `FEEL.py` 150ms delay?
2. Nudge when snap is off: lock to 1 frame (`UX-DESIGN-DECISIONS` D24) and discard the older fixed-small-amount fallback from D14?

If no response is needed for planning, use the current implementation-facing defaults:
- hover delay = `FEEL.py`
- snap-off nudge = latest specific design decision (`1 frame`)
