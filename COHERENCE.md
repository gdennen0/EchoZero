# COHERENCE.md — EchoZero Codebase Patterns & State

**Last verified:** 2026-03-20
**Updated by:** Chonch (after every merge)

---

## Established Patterns

*Updated as modules are merged. New agents must follow these patterns.*

### Enums
- Use `enum.Enum` with `auto()`. One-line docstring per enum class.
- Pattern: `class PortType(Enum):`

### Value Objects
- Frozen dataclasses (`@dataclass(frozen=True)`).
- No IDs on value objects that are identified by their content (e.g., Connection is identified by endpoint tuple).
- Value objects with identity (Event, Layer, Block) use `id: str` (UUID string).

### Aggregate Root
- Graph is the aggregate root — mutable, owns blocks and connections.
- All invariant validation lives on Graph, not on individual entities.
- Methods: `add_X()` validates then stores. `remove_X()` cascades as needed.

### Error Handling
- Domain violations raise `ValidationError` (from `echozero.errors`).
- Error messages include the offending value: `f"Duplicate block ID: {block.id}"`

### Collections
- Immutable sequences use `tuple[X, ...]` (not `list`).
- Mutable collections on aggregate roots use `dict` (keyed by ID) or `list`.

---

## Module Registry

*Every module in the codebase, with purpose and owner.*

| Module | Purpose | Lines | Tests | Added | Agent |
|--------|---------|-------|-------|-------|-------|
| `echozero/errors.py` | Typed error hierarchy (DomainError, ValidationError, etc.) | 28 | 2 (smoke) | 2026-03-20 | Chonch |
| `echozero/result.py` | Result[T] type for pipeline operations | 68 | 4 (smoke) | 2026-03-20 | Chonch |
| `echozero/domain.py` | Core domain entities: Block, Event, Connection, Graph | 248 | 39 | 2026-03-20 | Claude Code |
| `echozero/ui/FEEL.py` | Human-tunable UI constants | 120 | 0 | 2026-03-20 | Chonch |

---

## Decision Queue

*Tier 3 items waiting for Griff. Max 5. Each has a 48-hour default.*

| ID | Question | Asked | Default (48h) | Status |
|----|----------|-------|---------------|--------|
| *(empty)* | | | | |

---

## Known Technical Debt

*Tracked with decision references. Each item has a plan to resolve.*

| Item | Decision Ref | Priority | Plan |
|------|-------------|----------|------|
| *(none yet)* | | | |

---

## Source of Truth Map

| Concept | Authoritative Source | References |
|---------|---------------------|------------|
| Agent onboarding / compact context | `AGENTS.md`, `docs/AGENT-CONTEXT.md` | README, docs index, work-unit context |
| Timeline truth model | `docs/architecture/TIMELINE-FIRST-PRINCIPLES-2026-04-02.md` | `docs/UNIFIED-IMPLEMENTATION-PLAN.md`, timeline app/UI code |
| Architecture decisions | `docs/architecture/DECISIONS.md` | unified plan, architecture docs |
| App delivery and release gates | `docs/APP-DELIVERY-PLAN.md` | `echozero/testing/run.py`, `.github/workflows/test.yml` |
| UI behavior constants | `echozero/ui/FEEL.py` | Qt implementation, UI tests |
| User-facing terminology | `GLOSSARY.md` | UI strings, docs, code review |
| Code style | `STYLE.md` | Every agent session |
| Repo cleanup / keep boundaries | `docs/EZ2-CODEBASE-CLEANUP-MAP.md` | README, hygiene decisions |
| Foundry workflow | `docs/FOUNDRY-TRAINING.md` | `echozero/foundry/*` |
| MA3 reference / pitfalls | `MA3/README.md`, `MA3/MA3_INTEGRATION_PITFALLS.md` | `MA3/docs/*`, sync work |

---

## Sync Points

| Milestone | Target Week | Status |
|-----------|------------|--------|
| Feature Lock | Week 20 | Not started |
| Copy Lock | Week 30 | Not started |
| Design Lock | Week 32 | Not started |

---

*This file is the living record of the codebase. If it's not in here, it doesn't exist yet.*
