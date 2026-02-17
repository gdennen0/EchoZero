---
name: echozero-undo-redo
description: Undo/redo command system in EchoZero using QUndoCommand. Use when creating undoable operations, implementing EchoZeroCommand, command bus, or when the user asks about undo, redo, or command pattern.
---

# Undo/Redo Command System

## Standard

All undoable operations use `EchoZeroCommand` (subclass of Qt's `QUndoCommand`).

**Rule:** Execute via `facade.command_bus.execute(cmd)` - never push directly to stack.

## Command Template

```python
from src.application.commands.base_command import EchoZeroCommand

class MyCommand(EchoZeroCommand):
    def __init__(self, facade: "ApplicationFacade", param1, param2):
        super().__init__(facade, f"My Operation: {param1}")
        self._param1 = param1
        self._param2 = param2
        self._original_state = None

    def redo(self):
        if self._original_state is None:
            self._original_state = self._get_current_state()
        self._facade.do_something(self._param1, self._param2)

    def undo(self):
        if self._original_state is not None:
            self._facade.restore_state(self._original_state)

# Execute
facade.command_bus.execute(MyCommand(facade, p1, p2))
```

## Key Concepts

- **redo()** = execute. Qt calls redo() when command is first pushed.
- **undo()** = reverse. Must restore exact state before redo().
- Store original state in first redo() before making changes.
- Commands are atomic, reversible, descriptive.

## Command Categories

| Category | Examples |
|----------|----------|
| Block | AddBlockCommand, DeleteBlockCommand, RenameBlockCommand, MoveBlockCommand, UpdateBlockMetadataCommand |
| Connection | CreateConnectionCommand, DeleteConnectionCommand |
| Editor | EditorAddEventsCommand |
| Data Item | Data item commands |
| Timeline | Timeline commands |

## Macros (Multiple Commands = One Undo Step)

```python
facade.command_bus.begin_macro("Delete Selection")
facade.command_bus.execute(DeleteBlockCommand(facade, block1_id))
facade.command_bus.execute(DeleteBlockCommand(facade, block2_id))
facade.command_bus.end_macro()
```

## Command Merging

Override `id()` and `mergeWith()` for mergeable commands (e.g., drag operations):

```python
def id(self) -> int:
    return 1000  # Block position changes
def mergeWith(self, other: QUndoCommand) -> bool:
    if isinstance(other, MoveBlockCommand) and other._block_id == self._block_id:
        self._new_x, self._new_y = other._new_x, other._new_y
        return True
    return False
```

## Block Panel Metadata Updates

Use commands for undoable metadata changes:

- `UpdateBlockMetadataCommand` - single key
- `BatchUpdateMetadataCommand` - multiple keys
- `ConfigureBlockCommand` - block configuration

BlockPanelBase provides `set_block_metadata_key()` which wraps these.

## Key Files

- Base: `src/application/commands/base_command.py`
- Command bus: `src/application/commands/command_bus.py`
- Block commands: `src/application/commands/block_commands.py`
- Connection commands: `src/features/connections/application/connection_commands.py`
