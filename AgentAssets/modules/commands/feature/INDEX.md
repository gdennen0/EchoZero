# Feature Module

## Purpose

Provides a standardized workflow for developing new features in EchoZero, ensuring consistency, quality, and alignment with core values.

## When to Use

- When starting development on a new feature
- When planning feature implementation
- When evaluating feature scope and complexity
- When determining if a feature needs council review

## Quick Start

1. Review **PRESET.md** for the feature development template
2. Follow **GUIDE.md** for step-by-step workflow
3. Check **EXAMPLES.md** for real feature implementations
4. Use council process if feature is significant

## Contents

- **PRESET.md** - Feature development template and checklist
- **GUIDE.md** - Step-by-step feature development workflow
- **EXAMPLES.md** - Examples of completed features
- **PROGRESS_TRACKING_SYSTEM.md** - Centralized progress tracking system proposal (multiple approaches)
- **async_block_execution.md** - Async block execution feature guide
- **SET_TARGET_TIMECODE_COMMAND.md** - Command to set ShowManager target timecode number

## Related Modules

- [`modules/process/council/`](../../process/council/) - For major feature decisions requiring evaluation
- [`modules/patterns/block_implementation/`](../../patterns/block_implementation/) - When feature involves new blocks
- [`modules/patterns/settings_abstraction/`](../../patterns/settings_abstraction/) - When feature needs settings
- [`modules/commands/refactor/`](../refactor/) - When feature requires refactoring

## Documentation Links

- [Architecture](../../../docs/ARCHITECTURE.md) - Understanding system structure
- [Progress Tracking](../../patterns/progress_tracking/) - Progress tracking pattern
- [Block Implementation](../../patterns/block_implementation/) - Block creation patterns

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Questioning necessity** - Does this feature solve a real problem?
- **Minimal scope** - Start with MVP, iterate based on usage
- **Incremental delivery** - Small, testable steps
- **Deletion consideration** - Can we remove something instead?

Feature development should add value while maintaining simplicity, not accumulating complexity.

