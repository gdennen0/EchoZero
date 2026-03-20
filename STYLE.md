# STYLE.md — EchoZero Development Standards

**Last verified:** 2026-03-20
**Authority:** This document governs all code in the EchoZero repository. Every agent session receives this as mandatory context.

---

## Naming

### Functions
- Verb-first: `detect_onsets()`, `load_audio()`, `save_project()`
- Boolean functions: `is_stale()`, `has_connections()`, `can_execute()`
- Never: `get_X()` for expensive operations. `get` implies cheap/cached. Use `compute_X()` or `build_X()` for work.
- Event handlers: `on_block_added()`, `on_execution_complete()`
- Factory methods: `create_block()`, `create_connection()`

### Variables
- Descriptive over terse: `onset_times` not `ot`, `block_count` not `bc`
- Allowed abbreviations: `id`, `db`, `ui`, `ws` (websocket), `ipc`, `osc`, `ma3`, `bpm`, `fps`, `lod`
- Collections are plural: `events`, `blocks`, `connections`
- Single items are singular: `event`, `block`, `connection`
- Booleans read as assertions: `is_modified`, `has_audio`, `should_cascade`

### Classes
- Noun-based: `Block`, `Connection`, `EventBus`
- No `Manager` suffix unless it actually manages lifecycle (ShowManager is valid — it manages MA3 sync lifecycle)
- No `Helper`, `Utils`, `Misc` — these are code smells. Put the function where it belongs.

### Files/Modules
- One concept per file. `event_bus.py` contains the event bus. Not the event bus AND the progress reporter.
- If a file exceeds 300 lines, it's probably doing too much. Split by sub-concept.
- Test files mirror source: `echozero/event_bus.py` → `tests/test_event_bus.py`

---

## Structure

### Module Size
- **Target:** 100-300 lines per module
- **Hard limit:** 500 lines. If you hit this, split.
- **Exception:** Domain model (`domain.py`) may be larger because it's a single aggregate.

### Function Size
- **Target:** 10-30 lines per function
- **Hard limit:** 50 lines. If you hit this, extract helpers.
- **One level of nesting preferred.** Two max. Three = refactor.

### Imports
- Standard library first, blank line, third-party, blank line, local imports.
- No wildcard imports. Ever.
- Prefer explicit imports: `from echozero.domain import Block, Connection` over `import echozero.domain`

---

## Documentation

### Every File Gets a Header
Three lines. What, why, how-it-connects.

```python
"""
EventBus: Decoupled notification system for domain events.
Exists because blocks must stay isolated (FP1) — no block knows about other blocks.
Consumers subscribe by event type; events publish breadth-first after UoW commits.
"""
```

### Every Public Class Gets a Docstring
One line. A non-coder should understand the purpose.

```python
class StalenessService:
    """Marks downstream blocks as needing re-execution when upstream results change."""
```

### Every Public Function Gets a Docstring
What it does, not how.

```python
def detect_onsets(audio: AudioData, sensitivity: float) -> list[Event]:
    """Find percussive moments in audio. Higher sensitivity = more detections, more false positives."""
```

### Comments
- **Why, never what.** If code needs a "what" comment, the code isn't clear enough — rename things.
- **Exception:** Complex algorithms (DSP, ML) get step-by-step comments because the math isn't self-documenting.
- **Never:** Commented-out code. Delete it. Git remembers.

---

## First Principles Compliance

Every module must be traceable to EchoZero's First Principles:

- **FP1 (Pipeline-as-Data):** If you're writing code that only works when called in a specific order by a specific caller, you're violating pipeline-as-data. Pipelines are serializable DAGs.
- **FP2 (Block Contract):** Every block declares its I/O and settings schema. If your block doesn't have a schema, it's not a block.
- **FP3 (Context Discovery):** Pipelines are filtered by context. If your pipeline only works in one context, question whether it's a pipeline or a hardcoded workflow.
- **FP4 (Parameter Promotion):** User-facing settings come from promoted block parameters. If you're adding a setting that doesn't trace to a block parameter, justify why.
- **FP5 (Composition):** Pipelines contain other pipelines. If your pipeline is a monolith, break it up.
- **FP6 (Branch-Per-Execution):** Every pipeline run creates a branch. If your code mutates main, it's wrong.
- **FP7 (Engine Ignorance):** The engine knows DAGs, not audio/events/MA3. If your engine code imports audio-specific types, it's leaking.

---

## Anti-Patterns (Explicitly Banned)

1. **No God Objects.** No class that "knows everything." If a class has 10+ dependencies, it's doing too much.
2. **No stringly-typed interfaces.** Use enums and typed dataclasses, not dicts and strings.
3. **No silent failures.** Every error path either raises an exception or returns a Result type. Never `except: pass`.
4. **No global mutable state.** No module-level variables that change at runtime. Pass state explicitly.
5. **No circular imports.** If module A imports B and B imports A, your abstraction boundaries are wrong.
6. **No "temporary" code without a TODO.** If it's temporary, mark it: `# TODO(D###): Replace when X is implemented`
7. **No reimplementing existing modules.** Before writing a utility function, check if it already exists in the codebase.

---

## Consistency Rules

- **One pattern per concept.** If events use the EventBus pattern, all events use EventBus. No one-off callback registrations.
- **Error handling:** Domain errors are `DomainError` subclasses. Infrastructure errors are `InfrastructureError` subclasses. Never raise bare `Exception`.
- **Result type:** Operations that can fail return `Result[T]`, not exceptions, at the pipeline boundary. Internal methods may raise.
- **Serialization:** JSON everywhere. No pickle, no custom binary formats, no msgpack. JSON.
- **Configuration:** Settings dataclasses with defaults. No raw dicts for config.

---

## Testing Rules

- **Every test must assert on output values, not just execution success.** A test that only checks "did it run without crashing" is not a test — it's a smoke check. Tests must verify that the output is correct, not just that the function returned.
- **Bad:** `result = detect_onsets(audio)` (no assertion — just ran it)
- **Bad:** `assert result is not None` (proves nothing about correctness)
- **Good:** `assert len(result.events) == 10` (verifies count)
- **Good:** `assert abs(result.events[0].time - 0.1) < 0.01` (verifies timing within tolerance)
- **Golden file tests compare against known-correct output.** When a golden file exists, use `assert_matches_golden()` helper. Tolerances for floating-point comparisons (audio timing) should be explicit, never exact equality.
- **User-only tests** (audio playback timing, hardware interaction) are marked `@pytest.mark.manual` and skipped in CI. They require Griff to run locally.
- **Test functions should be under 20 lines.** Extract setup into fixtures. If a test needs a paragraph of setup, the setup belongs in conftest.py.

---

## Agent Exit Checklist

Before submitting any work unit:

- [ ] All tests pass
- [ ] mypy --strict passes
- [ ] No file exceeds 300 lines (500 hard limit)
- [ ] Every public class/function has a docstring
- [ ] Every file has a 3-line header (what/why/how)
- [ ] No wildcard imports
- [ ] No commented-out code
- [ ] No global mutable state
- [ ] Error handling uses DomainError/InfrastructureError, not bare Exception
- [ ] Naming follows conventions above
- [ ] No imports from forbidden modules listed in work unit spec
- [ ] GLOSSARY.md terminology used consistently
