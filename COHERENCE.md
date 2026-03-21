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
| Block types and schemas | `docs/architecture/DISTILLATION.md` Part 2 | domain.py, STYLE.md, GLOSSARY.md |
| API endpoints | `docs/architecture/API-CONTRACT.md` | Core server, UI client |
| Architecture decisions | `docs/architecture/DECISIONS.md` | DISTILLATION.md (summary), panel transcripts |
| UI behavior constants | `echozero/ui/FEEL.py` | Qt implementation, behavioral spec |
| User-facing terminology | `GLOSSARY.md` | UI strings, website copy, docs |
| Build patterns | `COHERENCE.md` (this file) | Work unit specs, agent context |
| Code style | `STYLE.md` | Every agent session |
| First Principles (FP1-7) | `docs/architecture/DISTILLATION.md` Part 2 | STYLE.md, DECISIONS.md |

---

## Sync Points

| Milestone | Target Week | Status |
|-----------|------------|--------|
| Feature Lock | Week 20 | Not started |
| Copy Lock | Week 30 | Not started |
| Design Lock | Week 32 | Not started |

---

*This file is the living record of the codebase. If it's not in here, it doesn't exist yet.*
