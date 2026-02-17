# Blocks Feature Module

The blocks feature manages block entities, their state, and expected outputs.

## Overview

Blocks are the core processing units in EchoZero. Each block:
- Has a type that determines its behavior
- Manages input/output ports for data flow
- Tracks processing status
- Stores configuration and results

## Architecture

```
blocks/
├── application/          # Services and API
│   ├── block_service.py       # Block CRUD operations
│   ├── block_state_helper.py  # State management
│   ├── block_status_service.py # Status tracking
│   ├── editor_api.py          # Editor integration
│   └── expected_outputs_service.py
├── domain/               # Entities and interfaces
│   ├── block.py              # Block entity
│   ├── block_repository.py   # Repository interface
│   ├── block_status.py       # Status enum
│   └── port.py               # Port definitions
└── infrastructure/       # Implementations
    └── block_repository_impl.py
```

## Key Components

- **BlockService** - CRUD operations for blocks
- **BlockStatusService** - Manages block processing status
- **BlockStateHelper** - Handles block state transitions
- **ExpectedOutputsService** - Manages expected output tracking

## Usage

```python
from src.application.api.application_facade import get_facade

facade = get_facade()

# Create a block
block = facade.create_block("LoadAudio")

# Get block status
status = facade.get_block_status(block.id)
```

## Related

- [Block Processors](../../application/blocks/README.md) - Processing implementations
- [Connections](../connections/README.md) - Block connections
- [Execution](../execution/README.md) - Block execution
