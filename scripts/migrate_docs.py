#!/usr/bin/env python3
"""
Documentation Migration Script for EchoZero EZ2.1

This script implements the documentation reorganization plan:
- Archives outdated planning/audit docs
- Creates co-located READMEs in feature modules
- Simplifies AgentAssets
- Organizes project-wide docs

Run with --dry-run to preview changes without executing them.
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
ARCHIVE_DIR = PROJECT_ROOT / ".archive"
SRC_DIR = PROJECT_ROOT / "src"
UI_DIR = PROJECT_ROOT / "ui"
AGENT_ASSETS_DIR = PROJECT_ROOT / "AgentAssets"

# File categorization
ARCHIVE_PLANNING: List[str] = [
    "BLOCK_FILTER_CLEANUP_PLAN.md",
    "BLOCK_FILTER_VALIDATION_DESIGN.md",
    "EDITOR_LAYER_SYNC_COORDINATED_PLAN.md",
    "LAYER_SYNC_EXECUTION_FIX.md",
    "LAYER_SYNC_FUNDAMENTAL_ISSUES_PLAN.md",
    "LAYER_SYNC_REFACTOR_PLAN.md",
    "LAYER_SYNC_SIMPLIFICATION.md",
    "LAYER_SYNC_STANDARDIZATION.md",
    "LAYER_SYNC_UPDATE_CHECKLIST.md",
    "LAYER_CREATION_UNIFIED_APPROACH.md",
    "LOADING_EFFICIENCY_IMPROVEMENTS.md",
    "LOADING_OPTIMIZATION_PLAN.md",
    "MIGRATION_EXECUTION_PLAN.md",
    "ORGANIZATION_PATTERNS_COMPARISON.md",
    "PANEL_ONLY_WORKFLOW_PREPARATION.md",
    "PROGRESS_TRACKING_IMPLEMENTATION.md",
    "QT6_EXAMPLES_IMPROVEMENTS.md",
    "REFACTORING_OPPORTUNITIES.md",
    "DIRECTORY_STRUCTURE_VERTICAL_MODULES.md",
    "FILE_MAPPING_VERTICAL_MODULES.md",
    "SPLASH_SCREEN_IMPLEMENTATION.md",
    "BULLETPROOF_STATE_SWITCHING.md",
    "BULLETPROOF_STATE_SWITCHING_IMPLEMENTED.md",
    "BULLETPROOF_STATE_SWITCHING_STANDARDS.md",
    "STALE_DATA_HANDLING.md",
]

ARCHIVE_AUDITS: List[str] = [
    "APPLICATION_LOADING_AUDIT.md",
    "BLOCK_FILTER_SYSTEM_AUDIT.md",
    "BLOCK_FILTER_UI_ERROR_HANDLING.md",
    "BLOCK_SAVE_LOAD_VERIFICATION.md",
    "BLOCK_STATUS_CASE_STUDIES.md",
    "BLOCK_STATUS_LEVELS_IMPLEMENTATION.md",
    "LOGGING_IMPROVEMENTS.md",
    "REORGANIZATION_AUDIT.md",
    "SETLIST_BULLETPROOF_IMPLEMENTATION_SUMMARY.md",
    "SETLIST_IMPROVEMENTS.md",
    "SETLIST_MODULE_BREAKDOWN.md",
    "SETLIST_STATE_MANAGEMENT_ANALYSIS.md",
    "SHOWMANAGER_STATUS_SYNC_FIX.md",
    "STANDARDIZATION_AUDIT.md",
    "TENSORFLOW_CLASSIFY_INFO_MESSAGES.md",
]

# Feature module READMEs to create
FEATURE_READMES: Dict[str, str] = {
    "src/features/blocks/README.md": """# Blocks Feature Module

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
""",

    "src/features/connections/README.md": """# Connections Feature Module

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
""",

    "src/features/execution/README.md": """# Execution Feature Module

Handles graph execution, progress tracking, and topological sorting.

## Overview

The execution engine processes the block graph in topological order,
tracking progress and handling errors.

## Architecture

```
execution/
├── application/
│   ├── execution_engine.py    # Main execution coordinator
│   ├── progress_tracker.py    # Progress reporting
│   └── topological_sort.py    # Dependency ordering
├── domain/
│   └── (execution domain objects)
└── infrastructure/
    └── (execution infrastructure)
```

## Key Components

- **ExecutionEngine** - Coordinates block execution
- **ProgressTracker** - Reports execution progress
- **TopologicalSort** - Orders blocks by dependencies

## Execution Flow

1. Build dependency graph from connections
2. Topologically sort blocks
3. Execute blocks in order
4. Track progress and status
5. Handle errors and rollback

## Related

- [Blocks](../blocks/README.md) - Block entities
- [Connections](../connections/README.md) - Dependency graph
- [Block Processors](../../application/blocks/README.md) - Processing logic
""",

    "src/features/projects/README.md": """# Projects Feature Module

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
""",

    "src/features/setlists/README.md": """# Setlists Feature Module

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
""",

    "src/features/show_manager/README.md": """# Show Manager Feature Module

Manages show-based workflows with cues, layers, and MA3 integration.

## Overview

Show Manager provides a show-oriented view of audio processing,
organizing content into cues and layers for live performance.

## Architecture

```
show_manager/
├── application/
│   ├── show_manager_service.py  # Main service
│   ├── cue_service.py           # Cue management
│   ├── layer_service.py         # Layer management
│   └── ma3_sync_service.py      # MA3 integration
└── domain/
    ├── show.py                  # Show entity
    ├── cue.py                   # Cue entity
    └── layer.py                 # Layer entity
```

## Key Components

- **ShowManagerService** - Orchestrates show operations
- **CueService** - Manages cues within shows
- **LayerService** - Manages layers and sync
- **MA3SyncService** - GrandMA3 integration

## Related

- [MA3](../ma3/README.md) - GrandMA3 integration details
- [Setlists](../setlists/README.md) - Batch processing
""",

    "src/features/ma3/README.md": """# MA3 Feature Module

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
""",

    "src/application/commands/README.md": """# Command System

Implements the Command pattern for undoable operations.

## Overview

All state-changing operations use commands, enabling:
- Undo/redo functionality
- Operation logging
- Transactional behavior

## Architecture

```
commands/
├── base_command.py       # Base command class
├── command_bus.py        # Command dispatcher
├── block_commands.py     # Block operations
├── editor_commands.py    # Editor operations
├── timeline_commands.py  # Timeline operations
├── layer_sync/           # Layer sync commands
└── ma3/                  # MA3 commands
```

## Creating Commands

```python
from src.application.commands.base_command import BaseCommand

class MyCommand(BaseCommand):
    def __init__(self, param):
        super().__init__()
        self.param = param
        
    def execute(self) -> bool:
        # Perform the operation
        self.old_value = get_current_value()
        set_new_value(self.param)
        return True
        
    def undo(self) -> bool:
        # Reverse the operation
        set_new_value(self.old_value)
        return True
```

## Usage

```python
from src.application.commands.command_bus import get_command_bus

bus = get_command_bus()
bus.execute(MyCommand(new_value))
bus.undo()  # Reverts the command
bus.redo()  # Re-applies the command
```

## Related

- [Encyclopedia: Command System](../../../docs/encyclopedia/01-architecture/command-system.md)
""",

    "src/application/settings/README.md": """# Settings System

Centralized settings management with persistence.

## Overview

The settings system provides:
- Type-safe settings access
- Persistence to database
- Category-based organization
- UI integration support

## Architecture

```
settings/
├── base_settings.py      # Base settings class
├── settings_service.py   # Settings operations
├── settings_provider.py  # Settings access
└── [feature]_settings.py # Feature-specific settings
```

## Creating Settings

```python
from src.application.settings.base_settings import BaseSettings

class MyFeatureSettings(BaseSettings):
    CATEGORY = "my_feature"
    
    @property
    def my_option(self) -> str:
        return self.get("my_option", "default_value")
    
    @my_option.setter
    def my_option(self, value: str):
        self.set("my_option", value)
```

## Usage

```python
from src.application.settings.my_feature_settings import MyFeatureSettings

settings = MyFeatureSettings()
print(settings.my_option)
settings.my_option = "new_value"
```

## Related

- [Encyclopedia: Settings](../../../docs/encyclopedia/)
""",

    "ui/qt_gui/README.md": """# Qt GUI Layer

PyQt6-based user interface implementation.

## Overview

The Qt GUI provides the graphical interface for EchoZero:
- Node editor for block graph visualization
- Block panels for configuration
- Timeline for event editing
- Dialogs and widgets

## Architecture

```
qt_gui/
├── main_window.py        # Main application window
├── qt_application.py     # Qt app initialization
├── design_system.py      # Visual design tokens
├── theme_registry.py     # Theme management
├── node_editor/          # Block graph visualization
├── block_panels/         # Block configuration UI
├── widgets/              # Reusable widgets
│   └── timeline/         # Timeline components
├── views/                # View controllers
├── dialogs/              # Modal dialogs
└── core/                 # Core UI components
```

## Key Components

- **MainWindow** - Application shell
- **NodeEditor** - Visual block graph
- **BlockPanels** - Per-block configuration
- **Timeline** - Event editing

## Design System

See [DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) for visual design guidelines.

## Related

- [Node Editor](./node_editor/README.md)
- [Block Panels](./block_panels/README.md)
- [Timeline](./widgets/timeline/README.md)
""",

    "ui/qt_gui/widgets/timeline/README.md": """# Timeline Widget

Qt-based timeline for event editing and visualization.

## Overview

The timeline provides:
- Event visualization and editing
- Layer management
- Time-based navigation
- Playback integration

## Architecture

```
timeline/
├── timeline_widget.py     # Main widget
├── timeline_scene.py      # QGraphicsScene
├── timeline_view.py       # QGraphicsView
├── event_item.py          # Event graphics items
├── layer_manager.py       # Layer handling
├── grid_system.py         # Time grid
├── movement_controller.py # Drag/drop
└── playback_controller.py # Playback cursor
```

## Key Components

- **TimelineWidget** - Container widget
- **TimelineScene** - Manages events and layers
- **TimelineView** - Viewport with zoom/pan
- **EventItem** - Individual event visualization

## Usage

```python
from ui.qt_gui.widgets.timeline import TimelineWidget

timeline = TimelineWidget()
timeline.set_events(events)
timeline.set_layers(layers)
```

## Related

- [Encyclopedia: Timeline](../../../../docs/encyclopedia/04-ui/timeline/README.md)
""",
}


def ensure_dirs():
    """Create necessary directories."""
    dirs = [
        ARCHIVE_DIR / "planning",
        ARCHIVE_DIR / "audits" / "ui",
        ARCHIVE_DIR / "superseded",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def archive_files(dry_run: bool = True) -> List[Tuple[Path, Path]]:
    """Archive outdated planning and audit docs."""
    moves = []
    
    # Archive planning docs
    for filename in ARCHIVE_PLANNING:
        src = DOCS_DIR / filename
        dst = ARCHIVE_DIR / "planning" / filename
        if src.exists():
            moves.append((src, dst))
    
    # Archive audit docs
    for filename in ARCHIVE_AUDITS:
        src = DOCS_DIR / filename
        dst = ARCHIVE_DIR / "audits" / filename
        if src.exists():
            moves.append((src, dst))
    
    # Archive entire ui_audit_reports folder
    ui_audit_src = DOCS_DIR / "ui_audit_reports"
    if ui_audit_src.exists():
        moves.append((ui_audit_src, ARCHIVE_DIR / "audits" / "ui"))
    
    if not dry_run:
        for src, dst in moves:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                
    return moves


def create_feature_readmes(dry_run: bool = True) -> List[Path]:
    """Create co-located README files in feature modules."""
    created = []
    
    for path, content in FEATURE_READMES.items():
        full_path = PROJECT_ROOT / path
        
        # Don't overwrite existing files
        if full_path.exists():
            print(f"  Skipping (exists): {path}")
            continue
            
        created.append(full_path)
        
        if not dry_run:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            
    return created


def simplify_agent_assets(dry_run: bool = True) -> Dict[str, List[Path]]:
    """Simplify AgentAssets to essential context."""
    result = {"archive": [], "keep": [], "delete": []}
    
    # Files to archive (move to .archive/superseded)
    to_archive = [
        "MASTER_REFACTORING_PLAN.md",
        "DEPLOYMENT_README.md",
        "DEPLOYMENT_SUMMARY.md",
        "FINAL_DEPLOYMENT_SUMMARY.md",
        "IMPLEMENTATION_GUIDE.md",
        "MODULE_INDEX.md",
        "MODULE_STANDARD.md",
        "STRUCTURE.md",
        "AUTOMATIC_SYNC_GUIDE.md",
        "SELF_REFINING_GUIDE.md",
    ]
    
    for filename in to_archive:
        src = AGENT_ASSETS_DIR / filename
        if src.exists():
            result["archive"].append(src)
            if not dry_run:
                dst = ARCHIVE_DIR / "superseded" / "AgentAssets" / filename
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    
    # Archive modules folder
    modules_dir = AGENT_ASSETS_DIR / "modules"
    if modules_dir.exists():
        result["archive"].append(modules_dir)
        if not dry_run:
            dst = ARCHIVE_DIR / "superseded" / "AgentAssets" / "modules"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(modules_dir), str(dst))
    
    # Keep core folder, scripts, data
    for keep in ["core", "scripts", "data", "README.md"]:
        path = AGENT_ASSETS_DIR / keep
        if path.exists():
            result["keep"].append(path)
    
    return result


def print_summary(
    archived: List[Tuple[Path, Path]],
    created: List[Path],
    agent_changes: Dict[str, List[Path]],
    dry_run: bool
):
    """Print migration summary."""
    prefix = "[DRY RUN] " if dry_run else ""
    
    print(f"\n{prefix}Documentation Migration Summary")
    print("=" * 50)
    
    print(f"\n{prefix}FILES ARCHIVED: {len(archived)}")
    for src, dst in archived[:10]:
        print(f"  {src.name} -> {dst.relative_to(PROJECT_ROOT)}")
    if len(archived) > 10:
        print(f"  ... and {len(archived) - 10} more")
    
    print(f"\n{prefix}READMES CREATED: {len(created)}")
    for path in created:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    
    print(f"\n{prefix}AGENT ASSETS CHANGES:")
    print(f"  Archived: {len(agent_changes['archive'])} items")
    print(f"  Kept: {len(agent_changes['keep'])} items")
    
    if dry_run:
        print("\n" + "=" * 50)
        print("This was a dry run. Run with --execute to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Migrate EchoZero documentation")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the migration (default is dry-run)"
    )
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    print("EchoZero Documentation Migration")
    print("=" * 50)
    
    if dry_run:
        print("[DRY RUN MODE - No changes will be made]")
    else:
        print("[EXECUTE MODE - Changes will be applied]")
        ensure_dirs()
    
    print("\n1. Archiving outdated docs...")
    archived = archive_files(dry_run)
    
    print("\n2. Creating co-located READMEs...")
    created = create_feature_readmes(dry_run)
    
    print("\n3. Simplifying AgentAssets...")
    agent_changes = simplify_agent_assets(dry_run)
    
    print_summary(archived, created, agent_changes, dry_run)


if __name__ == "__main__":
    main()
