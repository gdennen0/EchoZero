# Show Manager Feature Module

Manages show-based workflows with bidirectional sync to GrandMA3 lighting consoles.

## Overview

Show Manager provides show-oriented workflows, organizing content into layers and maintaining bidirectional synchronization with GrandMA3 lighting consoles via OSC.

## Architecture

```
show_manager/
├── application/
│   ├── sync_system_manager.py      # Main sync orchestration
│   ├── sync_layer_manager.py       # Layer sync management
│   ├── sync_safety.py              # Conflict resolution & safety
│   ├── sync_subscription_service.py # Change subscriptions
│   ├── show_manager_listener_service.py  # OSC listener
│   ├── show_manager_state_service.py     # Connection state
│   ├── ma3_event_handler.py        # MA3 event processing
│   ├── ma3_track_resolver.py       # Track name resolution
│   └── commands/                   # Sync commands
│       ├── sync_layer_command.py
│       ├── batch_sync_command.py
│       └── ...
└── domain/
    ├── sync_layer_entity.py       # Unified sync entity
    ├── sync_state.py               # Sync state tracking
    └── layer_sync_types.py         # Type definitions
```

## Key Components

- **SyncSystemManager** - Single orchestration point for all sync operations
- **SyncLayerEntity** - Unified entity representing synced layers (MA3 or Editor)
- **ShowManagerListenerService** - Manages OSC listener for MA3 communication
- **ShowManagerStateService** - Tracks connection state and status polling
- **SyncSafety** - Conflict resolution, validation, and backup management

## Features

- **Bidirectional Sync** - Sync Editor layers ↔ MA3 tracks
- **Real-time Updates** - OSC-based event synchronization
- **Conflict Resolution** - Automatic conflict detection and resolution
- **Safety Checks** - Validation and backup before destructive operations
- **Layer Mapping** - Flexible mapping between EchoZero layers and MA3 tracks

## Usage

```python
from src.features.show_manager.application import SyncSystemManager

# Add a synced layer
sync_manager.add_synced_ma3_track(
    block_id="show_manager_1",
    ma3_coord="tc1_tg1_tr1",
    editor_layer_id="layer_1"
)

# Sync changes
sync_manager.sync_layer("layer_1")
```

## Related

- [MA3 Integration](../ma3/README.md) - GrandMA3 integration details
- [Blocks](../blocks/README.md) - Block system
- [Connections](../connections/README.md) - Block connections
