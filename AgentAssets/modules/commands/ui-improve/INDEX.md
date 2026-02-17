# UI Improve Module

## Purpose

Provides a standardized workflow for improving existing UI panels, dialogs, and embedded node controls in EchoZero. Ensures visual consistency, readability, and responsive layouts that work in narrow side-panel contexts.

## When to Use

- When improving readability of an existing panel or dialog
- When making controls more compact for narrow layouts
- When adding embedded controls (knobs, sliders) to node editor items
- When applying the design system consistently across UI surfaces
- When evaluating whether a panel needs a custom node editor item

## Quick Start

1. Review **PRESET.md** for the UI improvement evaluation template
2. Follow **GUIDE.md** for step-by-step workflow
3. Check **PATTERNS.md** for proven layout patterns and code examples

## Contents

- **PRESET.md** - UI improvement evaluation template and checklist
- **GUIDE.md** - Step-by-step UI improvement workflow
- **PATTERNS.md** - Proven layout patterns, compact form techniques, and node editor embedding patterns

## Related Modules

- [`modules/patterns/ui_components/`](../../patterns/ui_components/) - Base UI component patterns (BlockPanelBase, quick actions)
- [`modules/patterns/settings_abstraction/`](../../patterns/settings_abstraction/) - Settings that drive UI panels
- [`modules/commands/refactor/`](../refactor/) - When UI improvement requires structural refactoring
- [`modules/process/council/`](../../process/council/) - For major UI redesigns requiring evaluation

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Removing visual noise** - Strip unnecessary labels, borders, and chrome
- **Information hierarchy** - Surface what matters, de-emphasize debug info
- **Responsive by default** - Panels must work narrow; avoid fixed-width assumptions
- **Reuse over invention** - Use existing design system tokens, not ad-hoc styles
- **Question every element** - If the user never looks at it, hide or remove it
