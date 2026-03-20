# COHERENCE.md — EchoZero Codebase Patterns & State

**Last verified:** 2026-03-20
**Updated by:** Chonch (after every merge)

---

## Established Patterns

*Updated as modules are merged. New agents must follow these patterns.*

*(No patterns established yet — Sprint 0)*

---

## Module Registry

*Every module in the codebase, with purpose and owner.*

| Module | Purpose | Lines | Tests | Added | Agent |
|--------|---------|-------|-------|-------|-------|
| *(none yet)* | | | | | |

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
