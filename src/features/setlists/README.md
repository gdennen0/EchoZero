# Setlists Feature Module

Manages setlists for batch audio processing workflows.

## Overview

Setlists enable batch processing of multiple audio files through
the same block graph, tracking state for each item.

## Architecture

```
setlists/
├── application/
│   ├── setlist_service.py     # Setlist operations
│   └── setlist_commands.py    # Command pattern operations
├── domain/
│   ├── setlist.py             # Setlist entity
│   ├── setlist_item.py        # Individual items
│   └── setlist_repository.py  # Repository interface
└── infrastructure/
    └── setlist_repository_impl.py
```

## Key Components

- **SetlistService** - CRUD and processing operations
- **Setlist** - Container for items to process
- **SetlistItem** - Individual audio file entry

## Related

- [Show Manager](../show_manager/README.md) - Show-based workflows
- [Execution](../execution/README.md) - Processing engine
