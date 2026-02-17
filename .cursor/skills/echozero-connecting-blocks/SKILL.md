---
name: echozero-connecting-blocks
description: Connect blocks and manage connections in EchoZero. Use when connecting blocks, creating connections between blocks, validating port compatibility, or when the user asks about block connections, data flow, or port types.
---

# Connecting Blocks

## Connection Model

A connection links one block's output port to another block's input port:

```
Source block + output port -> Target block + input port
```

Connection entity: `src/features/connections/domain/connection.py`

```python
@dataclass
class Connection:
    id: str
    source_block_id: str
    source_output_name: str
    target_block_id: str
    target_input_name: str
```

## Creating Connections

Via ApplicationFacade:

```python
facade.connect_blocks(
    source_block_id=source_id,
    source_output="audio",
    target_block_id=target_id,
    target_input="audio"
)
```

Via ConnectionService: `src/features/connections/application/connection_service.py`

## Port Compatibility

- Port types must be compatible (source output type compatible with target input type)
- Use `PortType.is_compatible_with()` for validation
- Multiple connections to same input port allowed for Event ports (multi-merge)
- Bidirectional ports: 1:1 only

## Port Structure on Block

Blocks store ports in `block.ports` with composite keys:

- Format: `"{direction}:{port_name}"` (e.g., `"input:audio"`, `"output:events"`)
- Directions: `PortDirection.INPUT`, `OUTPUT`, `BIDIRECTIONAL`
- Port types: `AUDIO_TYPE`, `EVENT_TYPE`, etc. from `PortType`

## Key Files

- Connection entity: `src/features/connections/domain/connection.py`
- ConnectionService: `src/features/connections/application/connection_service.py`
- Connection commands: `src/features/connections/application/connection_commands.py`
- CreateConnectionCommand / DeleteConnectionCommand for undo
- ConnectionsAPI: `src/application/api/features/connections_api.py`

## Disconnecting

```python
facade.disconnect_blocks(connection_id)
```

## Events

- `ConnectionCreated` - Published when connection created
- `ConnectionRemoved` - Published when connection deleted
- `BlockChanged` - Emitted for both source and target (affects status)

## Data Flow

Execution engine uses connections to:
1. Build dependency graph (topological sort)
2. Gather inputs: map (target_block_id, target_input) -> (source_block_id, source_output)
3. Pull DataItems from source block outputs
4. Pass to target block processor as `inputs` dict
