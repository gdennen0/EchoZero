# Execution Feature Module

Handles graph execution, progress tracking, and topological sorting.

## Overview

The execution engine processes the block graph in topological order,
tracking progress and handling errors.

## Architecture

```
execution/
├── application/
│   ├── execution_engine.py    # Main execution coordinator
│   ├── progress_tracker.py    # Progress reporting
│   └── topological_sort.py    # Dependency ordering
├── domain/
│   └── (execution domain objects)
└── infrastructure/
    └── (execution infrastructure)
```

## Key Components

- **ExecutionEngine** - Coordinates block execution
- **ProgressTracker** - Reports execution progress
- **TopologicalSort** - Orders blocks by dependencies

## Execution Flow

1. Build dependency graph from connections
2. Topologically sort blocks
3. Execute blocks in order
4. Track progress and status
5. Handle errors and rollback

## Related

- [Blocks](../blocks/README.md) - Block entities
- [Connections](../connections/README.md) - Dependency graph
- [Block Processors](../../application/blocks/README.md) - Processing logic
