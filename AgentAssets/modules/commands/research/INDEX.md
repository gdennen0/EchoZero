# Research Module

## Purpose

Provides a structured approach for investigating and exploring new feature ideas before committing to implementation. Helps flush out concepts, validate assumptions, and gather information needed to make informed decisions.

## When to Use

### Use Research Command When:
- Idea is vague or exploratory ("What if we...")
- Multiple approaches possible (need to compare)
- Unclear if feature is actually needed
- Assumptions need validation before proceeding
- Technical feasibility is uncertain
- Investigating existing solutions or patterns
- Exploring problem space before defining solution

### Use Feature Command When:
- Problem is clearly defined with evidence
- Solution approach is evident
- Ready to plan implementation details
- Research has already been completed
- Moving from investigation to execution

### Flow:
**Research → Findings/Validation → Feature → Implementation**

Use research to explore, validate, and inform. Use feature to plan and implement.

## Quick Start

1. Review **PRESET.md** for the research investigation template
2. Follow **GUIDE.md** for step-by-step research workflow
3. Use research findings to inform feature development decisions
4. Transition to `modules/commands/feature/` when ready to implement

## Contents

- **PRESET.md** - Research templates (Basic for quick research, Comprehensive for complex investigation)
- **GUIDE.md** - Step-by-step research workflow and methodologies
- **EXAMPLES.md** - Real-world examples of research in action

## Related Modules

- [`modules/commands/feature/`](../feature/) - Use research findings to inform feature development
- [`modules/process/council/`](../../process/council/) - Present research findings for major feature decisions
- [`modules/patterns/block_implementation/`](../../patterns/block_implementation/) - Research block-related patterns
- [`modules/commands/refactor/`](../refactor/) - Research refactoring opportunities

## Documentation Links

- [Architecture](../../../docs/ARCHITECTURE.md) - Understanding system structure for research
- [Show Manager Sync](../../../docs/show_manager_sync_system.md) - MA3 sync system documentation

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Questioning necessity first** - Research whether the problem is real before solving it
- **Exploring alternatives** - Investigate existing solutions before building new ones
- **Validating assumptions** - Test ideas before committing resources
- **Informing decisions** - Use research to make informed choices about what to build

Research helps avoid building unnecessary features by thoroughly investigating ideas before implementation.
