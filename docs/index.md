# EchoZero 2

Audio analysis workstation for live lighting design.

## Quick Links

- [Getting Started](GETTING-STARTED.md) — Set up your dev environment
- [Agent Context](AGENT-CONTEXT.md) — Compact orientation for coding agents
- [Worker Roles](WORKER-ROLES.md) — `lead-dev` and disposable worker contract
- [Testing](TESTING.md) — Verification lanes and proof map
- [Release Checklist](RELEASE-CHECKLIST.md) — Release signoff checklist
- [Unified Plan](UNIFIED-IMPLEMENTATION-PLAN.md) — Canonical implementation direction
- [Architecture](ARCHITECTURE.md) — System design and structure  
- [Project Class](PROJECT-CLASS.md) — The UI developer's API guide
- [Entity Model](architecture/ENTITY-MODEL.md) — Data relationships
- [LD Workflow](LD-WORKFLOW.md) — How lighting designers use EchoZero
- [Cleanup Map](EZ2-CODEBASE-CLEANUP-MAP.md) — Current keep/remove posture

## What is EchoZero?

EchoZero analyzes music and extracts time-stamped events (beats, onsets, classifications) that sync with grandMA3 lighting consoles. It's built for lighting designers who need precise, reliable show programming tools.

## Current Status

- **Canonical app:** `run_echozero.py` launches the active EZ2 desktop shell
- **Core runtime:** `echozero/` contains the engine, application layer, UI, and Foundry code
- **Docs set:** trimmed to active architecture, delivery, and MA3/Foundry context
- **Repo posture:** legacy EZ1 code and historical doc surfaces have been removed from this branch

## Tech Stack

- Python 3.11+
- PyQt6 (UI)
- SQLite + WAL (persistence)
- librosa, essentia, demucs, pytorch (ML/audio)
