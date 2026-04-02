# EchoZero Architecture Note

_This is the current best alignment, not immutable law._

Use this note to keep implementation moving in the same direction without treating architecture like scripture. If reality exposes a better structure, challenge the current one directly with a strong reason.

## Core Split

EchoZero is organized into four responsibility zones:

- **Engine** — reusable, app-agnostic processing/runtime capability
- **Application** — EchoZero product meaning and orchestration
- **UI** — rendering and interaction only
- **Infrastructure** — persistence, adapters, backends, platform details

## Canonical Nouns

Use these consistently:

- Project
- Session
- Song
- SongVersion
- Timeline
- Layer
- Take
- Event
- Transport
- Mixer
- Playback
- Sync
- Presentation
- Assembler
- Orchestrator
- Repository

Avoid vague core nouns like `Track`, `Manager`, `Controller`, and `Data` unless they are truly the right word.

## Structural Rules

- **Layer** is the primary timeline row
- **Take** is a child of a layer
- **Event** belongs to a take
- Takes are **not peer rows by default**
- Transport, Mixer, Playback, and Sync are separate application concepts/services
- UI consumes presentation and emits intents
- Engine must not learn EchoZero product nouns
- Repositories persist only; they do not interpret business meaning

## Anti-Spaghetti Guardrails

Reject these patterns:

- giant god objects (`TimelineController`, `AppManager`, mega `ProjectSession`)
- widgets reaching directly into repositories
- UI assembling business meaning ad hoc
- engine code importing EchoZero product semantics
- core product logic hidden in loose metadata forever

## Self-Refinement Rule

Future coding agents are explicitly allowed — and expected — to challenge the current architecture when there is a strong reason.

A strong challenge should answer:
1. What boundary problem does this solve?
2. What coupling does this reduce?
3. What future complexity does this prevent?
4. Does it preserve or improve the core principles and canonical nouns?

## Current Implementation Priority

Build in this order:
1. shared foundations
2. core application models
3. service contracts
4. presentation models
5. timeline assembler
6. intents + orchestrator
7. repositories + infrastructure
8. read-only UI
9. UI intent wiring
10. editing/playback/sync later
