# EchoZero 2

Audio analysis workstation for live lighting design.

## Quick Links

- [Status](STATUS.md) — Canonical current-state map for the repo
- [Getting Started](GETTING-STARTED.md) — Set up your dev environment
- [Agent Context](AGENT-CONTEXT.md) — Compact orientation for coding agents
- [OpenClaw / Codex Prompting](OPENCLAW-CODEX-PROMPTING.md) — Canonical prompt patterns and task-shaping rules for EchoZero agent work
- [LLM Cleanup Board](LLM-CLEANUP-BOARD.md) — Cleanup campaign for LLM readability and canonical-path organization
- [Execution Plan](EXECUTION-PLAN.md) — Ordered remediation plan for the current cleanup issues
- [Backlog Clearance Plan](BACKLOG-CLEARANCE-PLAN.md) — Ordered cleanup, streamlining, doc, and release backlog plan after the remediation pass
- [Worker Roles](WORKER-ROLES.md) — `lead-dev` and disposable worker contract
- [Testing](TESTING.md) — Verification lanes and proof map
- [Testing Primitives](TESTING-PRIMITIVES.md) — Canonical action vocabulary for tests, demos, and automation
- [Testing Executors](TESTING-EXECUTORS.md) — Standard executor interface for automation and proof surfaces
- [Testing Scenario Schema](TESTING-SCENARIO-SCHEMA.md) — Preferred scenario file shape built on canonical primitives
- [Testing Migration Map](TESTING-MIGRATION-MAP.md) — Explicit migration path from current test/demo surfaces
- [Object Pipeline Action Architecture](OBJECT-PIPELINE-ACTION-ARCHITECTURE.md) — Refactor spec for object-owned pipeline workflows and generic action execution
- [Song Import Batch + LTC Workflow](SONG-IMPORT-BATCH-LTC-WORKFLOW.md) — Implemented behavior map for folder import, import actions, and LTC stripping
- [Release Checklist](RELEASE-CHECKLIST.md) — Release signoff checklist
- [Unified Plan](UNIFIED-IMPLEMENTATION-PLAN.md) — Canonical implementation direction
- [Architecture](ARCHITECTURE.md) — System design and structure  
- [UI Standard](UI-STANDARD.md) — North Star for desktop UI design and implementation
- [UI Automation Plan](UI-AUTOMATION-PLAN.md) — Make the desktop app controllable by OpenClaw and agents
- [UI Automation Library Plan](UI-AUTOMATION-LIBRARY-PLAN.md) — Separate reusable desktop app-control library plan
- [UI Cleanup Map](UI-CLEANUP-MAP.md) — Preserve/extract/delete/prove-next for the UI stack
- [UI Engine Redevelopment Plan](UI-ENGINE-REDEVELOPMENT-PLAN.md) — Reusable UI engine extraction and rebuild path
- [Project Class](PROJECT-CLASS.md) — The UI developer's API guide
- [Entity Model](architecture/ENTITY-MODEL.md) — Data relationships
- [LD Workflow](LD-WORKFLOW.md) — How lighting designers use EchoZero
- [Cleanup Map](EZ2-CODEBASE-CLEANUP-MAP.md) — Current keep/remove posture

## Subsystem Maps

- [Timeline Application](../echozero/application/timeline/README.md) — Canonical timeline contract and proof surfaces
- [Presentation Layer](../echozero/application/presentation/README.md) — UI-facing presentation and inspector contract map
- [Timeline Qt Surface](../echozero/ui/qt/timeline/README.md) — Stage Zero UI runtime versus support-only surfaces
- [Foundry](../echozero/foundry/README.md) — Training, artifacts, and validation lane

## What is EchoZero?

EchoZero analyzes music and extracts time-stamped events (beats, onsets, classifications) that sync with grandMA3 lighting consoles. It's built for lighting designers who need precise, reliable show programming tools.

## Current Status

- **Canonical app:** `run_echozero.py` launches the active EZ2 desktop shell
- **Core runtime:** `echozero/` contains the engine, application layer, UI, and Foundry code
- **Current truth map:** `docs/STATUS.md` is the fastest current-state orientation doc
- **Docs set:** trimmed to active architecture, delivery, and MA3/Foundry context
- **Repo posture:** legacy EZ1 code and historical doc surfaces have been removed from this branch

## Tech Stack

- Python 3.11+
- PyQt6 (UI)
- SQLite + WAL (persistence)
- librosa, essentia, demucs, pytorch (ML/audio)
