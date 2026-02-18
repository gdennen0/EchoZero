# Setlists Feature Module

Manages setlists for batch audio processing workflows.

## Overview

Setlists enable batch processing of multiple audio files through
the same block graph, tracking state for each item.

## Architecture

```
setlists/
├── application/
│   ├── setlist_service.py           # Setlist CRUD and operations
│   ├── setlist_processing_service.py # Batch execution across setlist
│   └── setlist_snapshot_service.py   # Per-song state snapshots
├── domain/
│   ├── setlist.py             # Setlist entity
│   ├── setlist_song.py        # Individual song/item
│   ├── setlist_repository.py  # Repository interface
│   └── setlist_song_repository.py
└── infrastructure/
    ├── setlist_repository_impl.py
    └── setlist_song_repository_impl.py
```

## Key Components

- **SetlistService** - CRUD and setlist operations
- **SetlistProcessingService** - Batch execution across setlist items
- **Setlist** - Container for songs to process
- **SetlistSong** - Individual audio file entry

## Related

- [Show Manager](../show_manager/README.md) - Show-based workflows
- [Execution](../execution/README.md) - Processing engine
