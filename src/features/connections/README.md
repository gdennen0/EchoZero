# Connections Feature Module

Manages connections between blocks for data flow.

## Overview

Connections link output ports of one block to input ports of another,
enabling data to flow through the processing graph.

## Architecture

```
connections/
├── application/
│   ├── connection_commands.py  # Command pattern operations
│   └── connection_service.py   # Connection management
├── domain/
│   ├── connection.py           # Connection entity
│   ├── connection_repository.py # Repository interface
│   └── connection_summary.py   # Summary data class
└── infrastructure/
    └── connection_repository_impl.py
```

## Key Components

- **ConnectionService** - Create, delete, query connections
- **Connection** - Entity representing a link between ports

## Usage

```python
from src.application.api.application_facade import get_facade

facade = get_facade()

# Connect two blocks
facade.connect_blocks(source_block_id, target_block_id, 
                      source_port, target_port)

# Get connections for a block
connections = facade.get_block_connections(block_id)
```

## Related

- [Blocks](../blocks/README.md) - Block entities
- [Execution](../execution/README.md) - How connections affect execution order
