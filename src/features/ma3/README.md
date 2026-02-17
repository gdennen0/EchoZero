# MA3 Feature Module

GrandMA3 lighting console integration.

## Overview

This module provides integration with GrandMA3 lighting consoles,
enabling synchronized playback and control.

## Architecture

```
ma3/
├── application/
│   ├── ma3_service.py         # Main MA3 service
│   ├── ma3_commands.py        # Command operations
│   └── ma3_connection.py      # Connection management
├── domain/
│   ├── ma3_connection.py      # Connection entity
│   └── ma3_events.py          # MA3 events
└── infrastructure/
    └── ma3_client.py          # OSC/Network client
```

## Key Components

- **MA3Service** - High-level MA3 operations
- **MA3Connection** - Connection state management
- **MA3Client** - Network communication

## Configuration

See [docs/MA3_INTEGRATION.md](../../../docs/MA3_INTEGRATION.md) for setup.

## Related

- [Show Manager](../show_manager/README.md) - Show-based workflows
- [MA3 Plugins](../../../ma3_plugins/) - GrandMA3 Lua plugins
