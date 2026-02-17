# Refactor Module

## Purpose

Provides a framework for evaluating whether refactoring is justified and how to execute refactoring safely.

## When to Use

- When considering refactoring existing code
- When evaluating a refactoring proposal
- When determining if code changes are justified
- When assessing net complexity impact of changes

## Quick Start

1. Read **PRESET.md** to understand the evaluation framework
2. Answer the required questions (concrete problem, simpler fix, deletion opportunity)
3. Check red flags and green flags
4. Use the output format to document your evaluation

## Contents

- **PRESET.md** - Refactoring evaluation framework with questions, red flags, green flags, and output format

## Related Modules

- [`modules/commands/cleanup/`](../cleanup/) - Resource cleanup patterns (often needed after refactoring)
- [`modules/process/council/`](../../process/council/) - For major refactoring decisions requiring council review
- [`modules/patterns/block_implementation/`](../../patterns/block_implementation/) - When refactoring involves block structure

## Encyclopedia Links

- [Architecture](../../../docs/ARCHITECTURE.md) - Understanding current architecture before refactoring

## Core Values Alignment

This module embodies "the best part is no part" by:
- **Requiring evidence of actual pain** - No refactoring without concrete problems
- **Preferring deletion** - Always ask "Can we DELETE instead of reorganize?"
- **Questioning necessity** - Reject "more flexible", "cleaner", "best practices" without concrete problems
- **Measuring net complexity** - Refactoring must reduce or maintain complexity, not increase it

The module ensures refactoring serves simplicity, not aesthetics or imagined future needs.

