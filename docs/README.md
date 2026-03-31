# EchoZero 2 — Documentation

Desktop audio analysis workstation for live lighting design.
Analyzes music → extracts time-stamped events → syncs with grandMA3.

---

## Quick Start

| Need | Doc |
|------|-----|
| **Architecture overview** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Getting started / setup** | [GETTING-STARTED.md](GETTING-STARTED.md) |
| **Project class API** | [PROJECT-CLASS.md](PROJECT-CLASS.md) |
| **Entity model (DB schema)** | [architecture/ENTITY-MODEL.md](architecture/ENTITY-MODEL.md) |
| **Behavioral spec** | [../SPEC.md](../SPEC.md) |

---

## Architecture & Design

| Doc | Description |
|-----|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Master architecture — stack, layers, directory structure, naming |
| [architecture/ENTITY-MODEL.md](architecture/ENTITY-MODEL.md) | Persistence entities, SQLite schema, invariants |
| [architecture/DECISIONS.md](architecture/DECISIONS.md) | Architectural decisions log (ADRs) |
| [architecture/PIPELINE-CONFIG-AUDIT.md](architecture/PIPELINE-CONFIG-AUDIT.md) | PipelineConfig system audit |
| [architecture/PIPELINE-CONFIG-REDESIGN.md](architecture/PIPELINE-CONFIG-REDESIGN.md) | PipelineConfig redesign notes |
| [architecture/OBSERVABILITY.md](architecture/OBSERVABILITY.md) | Logging, metrics, and observability design |

---

## Workflow & Process

| Doc | Description |
|-----|-------------|
| [LD-WORKFLOW.md](LD-WORKFLOW.md) | Lighting designer workflow and EchoZero integration |
| [packaging/PACKAGING.md](packaging/PACKAGING.md) | Build, packaging, and distribution |

---

## Audits

Recent codebase audits live in `docs/audits/`:

| Doc | Description |
|-----|-------------|
| [audits/architecture-audit-2026-03-30.md](audits/architecture-audit-2026-03-30.md) | Architecture audit — March 30, 2026 |
| [audits/codebase-audit-2026-03-30.md](audits/codebase-audit-2026-03-30.md) | Full codebase audit |
| [audits/codebase-audit-2026-03-30-part2.md](audits/codebase-audit-2026-03-30-part2.md) | Codebase audit part 2 |
| [audits/audio-engine-audit.md](audits/audio-engine-audit.md) | Audio engine audit |
| [audits/model-registry-audit.md](audits/model-registry-audit.md) | Model registry audit |

---

## What's in `echozero/`

```
echozero/
  project.py       # Project — the single entry point for the UI
  main.py          # Production bootstrap
  domain/          # Graph, Block, Port, Connection, Event (frozen dataclasses)
  editor/          # Pipeline commands, Coordinator, ExecutionCache, StaleTracker
  persistence/     # ProjectStorage, *Record entities, repositories, schema, archive
  services/        # Orchestrator, SetlistProcessor, WaveformService
  pipelines/       # Pipeline builder, template registry, knobs
  processors/      # 13 block executor implementations
  audio/           # AudioEngine, transport, mixer, clock
  models/          # ML model registry
  takes.py         # Take system (frozen snapshots, TakeLayer)
  ui/              # FEEL.py (UI constants)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full directory breakdown and layered architecture diagram.

---

## What's NOT here (by design)

- `src/` — old EZ1 code (legacy, not part of EZ2)
- Tauri / web UI — killed. UI will be PyQt6.
- Old `OverrideStore` — replaced by the Take system
- `ProjectSession` — renamed to `ProjectStorage`

---

*Last updated: 2026-03-30*
