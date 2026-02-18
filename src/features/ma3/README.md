# MA3 Feature Module

GrandMA3 lighting console integration.

## Overview

This module provides integration with GrandMA3 lighting consoles,
enabling synchronized playback and control.

## Architecture

```
ma3/
├── application/
│   ├── ma3_communication_service.py   # OSC messaging and connection
│   ├── ma3_sync_service.py            # Sync state with MA3
│   ├── ma3_routing_service.py         # Message routing
│   ├── ma3_layer_mapping_service.py   # Layer-to-track mapping
│   ├── osc_message_dispatcher.py      # OSC message dispatch
│   └── ...
├── domain/
│   ├── osc_message.py                 # OSC message model
│   ├── ma3_sync_state.py              # Sync state
│   └── ma3_event.py                   # MA3 event model
└── infrastructure/
    └── osc_parser.py                  # OSC parsing
```

## Key Components

- **MA3CommunicationService** - OSC messaging and connection to MA3
- **MA3SyncService** - Bidirectional sync state with MA3
- **MA3LayerMappingService** - Maps EchoZero layers to MA3 tracks

## Configuration

See [MA3 Integration Pitfalls](../../../MA3/MA3_INTEGRATION_PITFALLS.md) and [MA3 docs](../../../MA3/docs/) for setup and development.

## Related

- [Show Manager](../show_manager/README.md) - Show-based workflows
- [MA3 Plugins](../../../MA3/) - GrandMA3 Lua plugins and integration docs
