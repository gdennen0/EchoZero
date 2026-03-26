# EchoZero Rewrite — Session Resume Briefing

**Last session:** 2026-03-01  
**Read this first** to pick up where we left off.

---

## What We Did

Used S.A.S.S. (panel of 4 AI agents debating architecture) to distill SPEC.md section by section. Covered:
- Domain Model entities (Block, Port, Connection, etc.)
- Block entity deep dive (execution, data flow, settings, dynamic items)
- Editor as Workspace block (layers, groups, classification, pull behavior)
- Execution flow (pull → execute → store pipeline)

## Key Architecture Decisions (29 total in DECISIONS.md)

**The big ones:**
- **Block types:** Processor (pure transform) vs Workspace (manages own data, like Editor)
- **Copy-on-pull:** TRUE full copy — metadata AND files. Each block is an isolated island.
- **Events:** Pure time data. Audio bundled as a snapshot copy through connections. No hidden `audio_id` cross-references.
- **Settings:** Individual keyed DB rows per block (not `metadata: Dict`). Same interface for hardcoded + user-added.
- **Persistence:** SQLite as real DB. `.ez` file is export/import format.
- **Ports:** Typed declarations, validated at connection time. GENERIC eliminated. MANIPULATOR extracted as own abstraction (channel, not port).
- **Execution:** Non-blocking background worker. Per-block lock. Processor can't touch Qt/event bus. FULL strategy stops at Workspace boundaries.
- **Editor pull:** Manual only. Replace (nuke upstream-derived) or Keep Existing (add alongside). Dismiss to clear notification. Merge deferred until stable event identity designed.
- **Groups:** First-class entity. LayerOrder table eliminated.
- **Classification:** Per-event `classification_state` enum (UNCLASSIFIED/TRUE/FALSE/USER_TRUE/USER_FALSE). Single layer, opacity-driven display.

## Where We Stopped

Completed Block entity distillation. Ready to continue with:

### Immediate Next Topics
1. **Command Bus / Event Bus / Facade architecture** — foundational application layer. How commands flow, undo/redo, how facade coordinates everything. CommandSequencer and AudioPlayer have conflicts with background worker model.
2. **ShowManager sync layers in Editor** — how MA3 bidirectional sync works in the new Workspace model
3. **Port entity** — typed declarations, validation rules, compatibility matrix
4. **Remaining SPEC.md entities** — Connection, Project, ActionSet/ActionItem, Setlist/SetlistSong, LayerOrder (mostly replaced)

### Open Questions (in DECISIONS.md Pending section)
- MANIPULATOR runtime behavior / protocol
- UPSTREAM_ERROR block status
- Layer dirty tracking granularity
- Engine component decomposition
- Subprocess isolation for ML blocks
- Full metadata typing coverage

## Key Files
- `SPEC.md` — Full behavioral spec from legacy codebase
- `docs/architecture/DECISIONS.md` — 29 decisions + pending items ← **primary reference**
- `docs/architecture/ARCHITECTURE.md` — Vertical feature module architecture
- `docs/architecture/OBSERVABILITY.md` — Universal observability design
- `sass/seminars/` — All S.A.S.S. panel analyses and syntheses

## Important Rules
- We are documenting **full desired functionality**, NOT version planning (no V1/V2)
- We're painting the complete picture, then deciding implementation phasing later
- S.A.S.S. panel = spawn 4 agents (Maya/Architect, Dr. Voss/Perfectionist, Sage/Data Modeler, Rex/Devil's Advocate) to debate, then synthesize
