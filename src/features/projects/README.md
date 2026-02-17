# Projects Feature Module

Manages project persistence, loading, and state.

## Overview

Projects are the top-level container for blocks, connections, and data.
This module handles saving, loading, and project lifecycle.

## Architecture

```
projects/
├── application/
│   ├── project_service.py     # Project operations
│   └── recent_projects.py     # Recent projects tracking
├── domain/
│   ├── project.py             # Project entity
│   ├── project_repository.py  # Repository interface
│   └── project_events.py      # Project-related events
└── infrastructure/
    ├── project_repository_impl.py
    └── project_file_handler.py  # .ez file handling
```

## Key Components

- **ProjectService** - Create, save, load projects
- **Project** - Entity containing blocks and connections
- **ProjectFileHandler** - Serialization to .ez format

## Usage

```python
from src.application.api.application_facade import get_facade

facade = get_facade()

# Save project
facade.save_project("/path/to/project.ez")

# Load project
facade.load_project("/path/to/project.ez")

# New project
facade.new_project()
```

## Related

- [Blocks](../blocks/README.md) - Blocks within projects
- [Connections](../connections/README.md) - Connections within projects
