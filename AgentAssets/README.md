# EchoZero AI Agent Context

Quick-reference context for AI agents working on EchoZero.

## Cursor Skills (Primary)

EchoZero context is available as **Cursor Skills** in `.cursor/skills/`. Skills are applied automatically when relevant. Prefer skills for:

- Block implementation, settings, progress tracking
- MA3 integration, UI components, council framework
- Feature/refactor/cleanup/research workflows

See `.cursor/skills/README.md` for the skill index.

AgentAssets remains the source for detailed guides; skills provide concise, discoverable summaries.

---

## Core Values

**"The best part is no part"**
- Question every addition
- Prefer removal over addition
- Every feature must justify its existence

**"Simplicity and refinement are key"**
- Simple is what remains after removing everything unnecessary
- Refine existing features before adding new ones
- Polish beats scope

---

## Work Tags

When using `@research`, `@refactor`, `@feature`, or `@cleanup`, avoid creating "fallbacks." Define the exact workflow for how the system should work, load, and operate instead of handling a pile of edge cases. Fallback-heavy code leads to chaotic behavior and unclear expectations.

### Command Flow

- `@research` - Investigate and explore ideas before committing
- `@feature` - Plan and implement validated features
- `@refactor` - Evaluate and execute code improvements
- `@cleanup` - Resource management and cleanup patterns

**Typical workflow:** Research -> Feature -> Implementation -> Refactor -> Cleanup

---

## Project Structure

```
EchoZero/
├── src/
│   ├── features/               # Vertical feature modules
│   │   ├── blocks/             # Block entities, services, editor API
│   │   ├── connections/        # Connection management
│   │   ├── execution/          # Graph execution engine + progress tracking
│   │   ├── projects/           # Project persistence + snapshots
│   │   ├── setlists/           # Setlist processing + song switching
│   │   ├── show_manager/       # MA3 sync system (layer sync, divergence)
│   │   └── ma3/                # GrandMA3 OSC communication
│   ├── application/
│   │   ├── api/                # ApplicationFacade + feature-specific APIs
│   │   │   └── features/       # BlocksAPI, ConnectionsAPI, ProjectsAPI, SetlistsAPI
│   │   ├── blocks/             # Block processors (execution logic) + quick actions
│   │   ├── commands/           # Command pattern (undo/redo via QUndoCommand)
│   │   ├── processing/         # BlockProcessor base class + type validation
│   │   ├── settings/           # Settings system (app, block, show manager)
│   │   ├── events/             # Domain events (project, block, execution, setlist, MA3)
│   │   ├── services/           # Application services + progress re-exports
│   │   └── bootstrap/          # Application initialization
│   ├── shared/
│   │   ├── application/
│   │   │   ├── events/         # EventBus + event dispatcher
│   │   │   ├── services/       # Progress tracking (context, store, models)
│   │   │   ├── registry/       # Module registry
│   │   │   ├── settings/       # Settings registry
│   │   │   ├── status/         # Status publishing
│   │   │   └── validation/     # Validation framework
│   │   ├── domain/
│   │   │   ├── entities/       # AudioDataItem, EventDataItem, EventLayer
│   │   │   └── repositories/   # Repository interfaces
│   │   └── infrastructure/
│   │       └── persistence/    # Repository implementations
│   ├── infrastructure/         # SQLite persistence layer
│   └── utils/                  # Utility modules (paths, message, etc.)
├── ui/
│   └── qt_gui/                 # PyQt6 interface
│       ├── block_panels/       # Block configuration UI panels
│       │   └── components/     # Reusable panel components (filters, outputs)
│       ├── core/               # Base components, progress bar, actions panel
│       ├── node_editor/        # Visual graph editor
│       ├── dialogs/            # Dialog windows (filters, setlist, tracks)
│       ├── views/              # Setlist views, action set editor
│       ├── widgets/            # Reusable widgets (timeline, settings, logs)
│       │   └── timeline/       # Timeline widget (events, waveform, playback)
├── tests/
│   ├── application/            # Application layer tests
│   ├── integration/            # Integration tests
│   └── unit/                   # Unit tests
├── docs/                       # Project documentation
├── MA3/                        # GrandMA3 Lua plugins and integration docs
└── AgentAssets/                # You are here
```

---

## Common Patterns

### Creating a Block Processor

BlockProcessors follow a 5-step execution flow managed by the execution engine:
1. `step_clear_local_data()` - Clear owned data items
2. Pull upstream data (engine-managed)
3. `step_pre_process()` - Pre-processing hook
4. `process()` - Main processing (required)
5. `step_post_process()` - Post-processing hook

```python
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.application.blocks import register_processor_class

class MyBlockProcessor(BlockProcessor):
    def get_block_type(self) -> str:
        return "MyBlockType"

    def can_process(self, block) -> bool:
        return block.type == self.get_block_type()

    def process(self, block, inputs, metadata=None):
        # Validate inputs, process data, return outputs dict
        audio = inputs.get("audio")
        if audio is None:
            raise ProcessingError("Missing required input: audio")
        # ... processing logic ...
        return {"output": result}

    def cleanup(self, block) -> None:
        """Clean up resources (timers, media players, etc.)"""
        pass

register_processor_class(MyBlockProcessor)
```

### Creating a Command

Commands use Qt's QUndoCommand for undo/redo support:

```python
from src.application.commands.base_command import EchoZeroCommand

class MyCommand(EchoZeroCommand):
    def __init__(self, facade, param):
        super().__init__(facade, f"My Operation: {param}")
        self._param = param
        self._original_state = None

    def redo(self):
        if self._original_state is None:
            self._original_state = self._get_current_state()
        self._facade.do_something(self._param)

    def undo(self):
        if self._original_state is not None:
            self._facade.restore_state(self._original_state)

# Execute via command bus:
# facade.command_bus.execute(MyCommand(facade, param))
```

### Using ApplicationFacade

```python
from src.application.api.application_facade import get_facade

facade = get_facade()
facade.create_block("LoadAudio")
facade.save_project("/path/to/project.ez")
```

### Progress Tracking (Context Managers)

Two progress systems exist:
- **Simple**: `ProgressTracker` for block-level subprocess progress events
- **Advanced**: `ProgressContext` for hierarchical operations (setlist processing)

```python
from src.application.services import get_progress_context

progress = get_progress_context()

with progress.operation("my_operation", "My Operation Name") as op:
    op.set_total(len(items))
    for item in items:
        with op.level("item", item.id, item.name) as ctx:
            ctx.update(message="Processing...")
            # ... do work ...
            # Automatically completed when exiting
            # Automatically failed if exception raised
```

### Quick Actions

Quick actions provide block-level operations accessible from the UI:

```python
from src.application.blocks.quick_actions import quick_action, ActionCategory

@quick_action("MyBlock", "My Action", category=ActionCategory.EXECUTE)
def my_action(facade, block):
    # Action logic
    pass
```

---

## Key Interfaces

| Interface | Location | Purpose |
|-----------|----------|---------|
| ApplicationFacade | `src/application/api/application_facade.py` | Unified API for all operations |
| BlockProcessor | `src/application/processing/block_processor.py` | Block execution logic (5-step flow) |
| EchoZeroCommand | `src/application/commands/base_command.py` | Undoable operations (QUndoCommand) |
| BaseSettingsManager | `src/application/settings/block_settings.py` | Type-safe settings with persistence |
| BlockSettingsManager | `src/application/settings/block_settings.py` | Block-level settings (metadata storage) |
| EventBus | `src/shared/application/events/` | Domain event dispatch |
| ProgressContext | `src/shared/application/services/progress_context.py` | Hierarchical progress tracking |
| ProgressEventStore | `src/shared/application/services/progress_store.py` | Centralized progress state |
| EditorAPI | `src/features/blocks/application/editor_api.py` | Unified editor layer/event operations |
| SyncSystemManager | `src/features/show_manager/application/sync_system_manager.py` | MA3 sync orchestration |
| SetlistService | `src/features/setlists/application/setlist_service.py` | Setlist processing coordinator |

---

## Key Domain Events

| Event | Module | When |
|-------|--------|------|
| `ProjectLoaded` / `ProjectCreated` | `application/events/` | Project lifecycle |
| `BlockAdded` / `BlockUpdated` / `BlockRemoved` | `application/events/` | Block changes |
| `ExecutionStarted` / `ExecutionCompleted` | `application/events/` | Execution lifecycle |
| `SubprocessProgress` | `application/events/` | Block-level progress |
| `SetlistProcessingStarted` / `SetlistProcessingCompleted` | `application/events/` | Setlist lifecycle |
| `ConnectionCreated` / `ConnectionRemoved` | `application/events/` | Connection changes |
| `StatusChanged` | `application/events/` | Block status updates |

---

## Feature Modules

| Feature | Location | Purpose |
|---------|----------|---------|
| blocks | `src/features/blocks/` | Block entities, services, editor API, expected outputs |
| connections | `src/features/connections/` | Connection management and commands |
| execution | `src/features/execution/` | Execution engine, progress tracker, topological sort |
| projects | `src/features/projects/` | Project service, snapshot service |
| setlists | `src/features/setlists/` | Setlist service, processing service, snapshot switching |
| show_manager | `src/features/show_manager/` | Sync system, layer entities, MA3 event handling |
| ma3 | `src/features/ma3/` | MA3 communication, OSC messaging, routing |

---

## Known Pitfalls (READ FIRST)

These documents capture hard-won knowledge about subtle bugs:

| Document | Topic |
|----------|-------|
| `MA3/MA3_INTEGRATION_PITFALLS.md` | MA3 state persistence, OSC timing, track indexing |

**Most important pitfall**: MA3 Lua plugin state persists across EchoZero restarts. Always send current data with responses, even if state says "already done".

---

## Documentation

- **Co-located READMEs** - Feature modules and block processors have README.md files
- **Architecture docs** - `docs/architecture/ARCHITECTURE.md`, `MA3/docs/show_manager_sync_system.md`
- **Core Values** - `AgentAssets/CORE_VALUES.md`
- **MA3 Integration** - `MA3/` for GrandMA3 Lua plugins and integration docs
- **Progress Tracking** - `docs/progress_tracking.md`

### Finding Documentation

1. **Cursor Skills** - `.cursor/skills/` for block implementation, settings, progress, MA3, etc.
2. **Feature-specific** - Check the feature module's README.md
3. **Architecture** - `docs/architecture/ARCHITECTURE.md`
4. **Show Manager sync** - `MA3/docs/show_manager_sync_system.md`
5. **Progress tracking** - `AgentAssets/modules/patterns/progress_tracking/` or echozero-progress-tracking skill
6. **Block implementation** - `AgentAssets/modules/patterns/block_implementation/` or echozero-block-implementation skill
7. **Settings patterns** - `AgentAssets/modules/patterns/settings_abstraction/` or echozero-settings-abstraction skill
8. **MA3 Integration** - `MA3/MA3_INTEGRATION_PITFALLS.md` or echozero-ma3-integration skill

---

## Scripts

Automation tools in `AgentAssets/scripts/`:

| Script | Purpose |
|--------|---------|
| `quality_checks.py` | Pre-commit code quality |
| `context_cli.py` | Get AI context |
| `decision_tracker.py` | Track decisions |
| `auto_sync.py` | Auto-sync AgentAssets with codebase |
| `ai_feedback_system.py` | AI agent feedback tracking |
| `code_generator.py` | Code generation utilities |

---

## Quick Commands

```bash
# Get context for a question
python AgentAssets/scripts/context_cli.py context "How do I create a block?"

# Run quality checks
python AgentAssets/scripts/quality_checks.py src/my_file.py

# Track a decision
python AgentAssets/scripts/decision_tracker.py record feature "New Feature" "Description"
```

---

*Last Updated: February 2026*
