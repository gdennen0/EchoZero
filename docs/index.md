# EchoZero 2

Audio analysis workstation for live lighting design.

## Quick Links

- [Getting Started](GETTING-STARTED.md) — Set up your dev environment
- [Architecture](ARCHITECTURE.md) — System design and structure  
- [Project Class](PROJECT-CLASS.md) — The UI developer's API guide
- [Entity Model](architecture/ENTITY-MODEL.md) — Data relationships
- [LD Workflow](LD-WORKFLOW.md) — How lighting designers use EchoZero

## What is EchoZero?

EchoZero analyzes music and extracts time-stamped events (beats, onsets, classifications) that sync with grandMA3 lighting consoles. It's built for lighting designers who need precise, reliable show programming tools.

## Current Status

- **Engine:** Complete (1523 tests passing)
- **Persistence:** Complete (SQLite + WAL + .ez archives)  
- **Processors:** 13 block types (audio loading, onset detection, stem separation, filtering, export, classification)
- **Application Layer:** Complete (Project class bridges everything)
- **UI:** PyQt6 Stage Zero Editor (in development)

## Tech Stack

- Python 3.11+
- PyQt6 (UI)
- SQLite + WAL (persistence)
- librosa, essentia, demucs, pytorch (ML/audio)
