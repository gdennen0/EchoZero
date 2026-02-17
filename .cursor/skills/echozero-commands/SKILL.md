---
name: echozero-commands
description: EchoZero workflow commands - feature, refactor, cleanup, research. Use when planning features, evaluating refactors, implementing cleanup, or investigating ideas before implementation. Use @feature, @refactor, @cleanup, or @research tags.
---

# EchoZero Workflow Commands

**Flow:** Research -> Feature -> Implementation -> Refactor -> Cleanup

Avoid "fallbacks" - define exact workflow instead of handling edge case piles.

## @research

**When:** Idea is vague, multiple approaches, unclear if needed, assumptions need validation.

**Use for:** Exploring before committing, comparing approaches, validating feasibility.

**Flow:** Research -> Findings -> Feature (when ready to implement)

## @feature

**When:** Problem clearly defined with evidence, solution evident, ready to plan.

**Phases:**
1. Planning: Define problem, MVP, alternatives, scope, council if major
2. Design: Architecture, components, interfaces
3. Implementation: Domain -> Application -> Infrastructure -> UI -> Tests
4. Integration: Register, document, E2E test
5. Validation: Review against core values

**Common types:**
- New block: Processor, BlockRegistry, panel, cleanup(), tests
- Settings: Schema, manager, UI integration (see echozero-settings-abstraction)
- Command: EchoZeroCommand, CommandBus, ApplicationFacade
- UI: BlockPanelBase, design system, cleanup

## @refactor

See **echozero-refactor** skill for evaluation framework.

## @cleanup

See **echozero-cleanup** skill for resource cleanup patterns.

## Related Skills

- echozero-block-implementation - Block creation
- echozero-core-values - Decision framework
- echozero-council - Major proposal review
