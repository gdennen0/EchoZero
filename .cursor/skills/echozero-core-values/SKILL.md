---
name: echozero-core-values
description: EchoZero core values and decision framework. Use when evaluating proposals, making architectural decisions, refactoring, adding features, or when the user asks about EchoZero principles, "best part is no part", simplicity, or design philosophy.
---

# EchoZero Core Values

## The Two Pillars

### 1. "The Best Part is No Part"
- Question every new class, method, abstraction, feature, or dependency
- Can we delete code instead of adding it?
- Does this solve a real problem or anticipated complexity?
- Three simple functions beat one complex abstraction

### 2. "Simplicity and Refinement are Key"
- Simple is what remains after removing everything unnecessary
- Code should read like prose; no cleverness for its own sake
- Explicit over implicit; boring is good
- One obvious way to do things

## Derived Principles

| Principle | Action |
|-----------|--------|
| Default to Deletion | First ask "What can I remove?" |
| Question New Abstractions | Pattern emerged 3+ places? Rule of Three before abstracting |
| Make Common Things Easy | Optimize for 80% use case |
| Explicit Over Implicit | No magic; easy to debug and test |
| Composition Over Inheritance | Build from simple independent parts |
| Optimize for Reading | Code is read 10x more than written |
| Errors Impossible or Obvious | Prevent via types or make immediately visible |
| Data Over Code | Prefer data-driven solutions |

## Decision Framework

Apply these filters in order:

1. **Is This Necessary?** What problem? Do users have it? Can we not do this?
2. **Can We Remove Instead?** Net complexity change?
3. **What's the Simplest Solution?** Dumbest thing that could work?
4. **What's the Cost?** LOC, dependencies, concepts, testing, maintenance?
5. **Is It Reversible?** Can we undo this later?

## Anti-Patterns to Reject

- Premature generalization ("we might need X someday")
- Resume-driven development (cool tech for its own sake)
- Abstraction addiction (interfaces with one implementation, factories for one type)
- Feature creep ("it would be cool if...")
- "More flexible", "cleaner", "best practices" without concrete problem

## Mantras

1. "Can I delete this instead?"
2. "What's the simplest thing that could work?"
3. "Will I be happy debugging this at 3am?"
4. "Am I being clever or clear?"
5. "Does this make the system simpler or more complex?"

## Reference

Full CORE_VALUES.md: `AgentAssets/CORE_VALUES.md`
