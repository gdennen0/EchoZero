# EchoZero Architecture

## Overview

EchoZero uses a **vertical feature module** architecture with clear separation of concerns. Code is organized by feature rather than by technical layer, making it easier to understand, modify, and test each feature independently.

## Directory Structure

```
src/
    features/           # Feature modules (vertical slices)
        blocks/         # Block entities and services
        connections/    # Block connections
        execution/      # Execution engine
        ma3/           # grandMA3 integration
        projects/       # Project management
        setlists/       # Setlist batch processing
        show_manager/   # Show-based workflows & MA3 sync
    
    application/       # Application services
        blocks/         # Block processors (execution logic)
        commands/       # Command pattern (undo/redo)
        settings/       # Settings system
        api/            # ApplicationFacade
        bootstrap.py    # Service initialization
    
    shared/            # Cross-cutting concerns
        application/
            events/    # EventBus, domain events
            registry/  # ModuleRegistry
            settings/  # SettingsRegistry
            status/    # StatusPublisher
            validation/ # Validation framework
        domain/
            entities/  # Shared entities (DataItem, EventDataItem, etc.)
        infrastructure/
            persistence/ # BaseRepository
        utils/         # Logging, paths
    
    infrastructure/    # Persistence layer
        persistence/
            sqlite/    # SQLite implementations

ui/
    qt_gui/           # PyQt6 interface
        block_panels/  # Block configuration UI
        node_editor/   # Visual graph editor
        widgets/       # Reusable widgets (timeline, etc.)
        views/         # Main views (setlist, etc.)
        dialogs/       # Dialog windows
```

## Feature Module Structure

Each feature module follows a consistent three-layer pattern:

```
feature_name/
    __init__.py          # Public API, re-exports
    domain/              # Entities, value objects, repository interfaces
        __init__.py
    application/         # Services, commands, processors
        __init__.py
    infrastructure/      # Repository implementations, external integrations
        __init__.py
```

## Import Conventions

### Preferred (New Code)

```python
# Feature imports
from src.features.blocks import Block, BlockService, BlockProcessor
from src.features.connections import Connection, ConnectionService
from src.features.projects import Project, ProjectService

# Shared imports
from src.shared.application.events import EventBus, BlockAdded
from src.shared.utils import Log, get_user_data_dir
```

### Legacy (Existing Code)

```python
# These still work for backwards compatibility
from src.domain.entities.block import Block
from src.application.services.block_service import BlockService
from src.Utils.message import Log
```

## Core Patterns

### 1. BaseRepository

Generic CRUD operations for all repositories:

```python
from src.shared.infrastructure.persistence import BaseRepository

class MyRepository(BaseRepository[MyEntity]):
    def _entity_to_row(self, entity): ...
    def _row_to_entity(self, row): ...
```

### 2. StatusPublisher

Unified status updates across the application:

```python
from src.shared.application.status import StatusPublisher, StatusLevel

class MyService(StatusPublisher):
    def process(self):
        self.update_status("Processing...", StatusLevel.INFO)
```

### 3. ModuleRegistry

Component registration with auto-discovery:

```python
from src.shared.application.registry import register_component

@register_component("blocks", "LoadAudio", tags=["audio", "input"])
class LoadAudioProcessor:
    pass
```

### 4. SettingsRegistry

Settings auto-discovery with validation:

```python
from src.shared.application.settings import register_block_settings

@register_block_settings("LoadAudio")
@dataclass
class LoadAudioSettings(BaseSettings):
    file_path: str = validated_field(default="", required=True)
```

### 5. Validation Framework

Declarative input validation:

```python
from src.shared.application.validation import validate, RequiredValidator, RangeValidator

result = validate(value, [RequiredValidator(), RangeValidator(0, 100)])
if not result.is_valid:
    print(result.errors)
```

## Feature APIs

For cleaner access to application functionality:

```python
from src.application.api.features import ProjectsAPI, BlocksAPI

# Projects
projects = ProjectsAPI(facade)
projects.create_project("My Project")

# Blocks
blocks = BlocksAPI(facade)
blocks.add_block("LoadAudio", settings={"file_path": "audio.wav"})
```

## Design Principles

1. **"The best part is no part"** - Question every addition, prefer removal
2. **Explicit over implicit** - Clear, obvious code
3. **Data over code** - Prefer declarative approaches
4. **Feature-first** - Organize by feature, not by technical layer

## Application Bootstrap

The application initializes through `src/application/bootstrap.py`:

1. **Database initialization** - SQLite database setup
2. **Event bus** - Domain event system
3. **Repositories** - Data persistence layer
4. **Services** - Business logic services
5. **Execution engine** - Block execution system
6. **Block processors** - Auto-registered processors
7. **Application facade** - Unified API

Entry points:
- `main_qt.py` - Qt GUI application (with splash screen and progress tracking)
- `main.py` - CLI application

## Current State

- **17+ Block Types** - Fully implemented processors
- **7 Feature Modules** - Organized by domain
- **Vertical Architecture** - Feature-first organization
- **Command Pattern** - Undo/redo support
- **Settings System** - Type-safe settings with validation
- **Event System** - Domain events for decoupled communication

## Migration Notes

The codebase uses vertical feature modules. Import paths:

| Feature | Import Path |
|---------|-------------|
| Blocks | `src.features.blocks` |
| Connections | `src.features.connections` |
| Execution | `src.features.execution` |
| Projects | `src.features.projects` |
| Setlists | `src.features.setlists` |
| Show Manager | `src.features.show_manager` |
| MA3 | `src.features.ma3` |

Shared utilities:
- `src.shared.application.events` - EventBus
- `src.shared.infrastructure.persistence` - BaseRepository
- `src.shared.utils` - Logging, paths
