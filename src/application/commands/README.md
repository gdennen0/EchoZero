# Command System

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

Commands use Qt's QUndoCommand. Inherit from `EchoZeroCommand`:

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
```

## Usage

```python
from src.application.api.application_facade import get_facade

facade = get_facade()
# Execute via command bus (facade provides undo/redo)
facade.command_bus.execute(MyCommand(facade, new_value))
facade.command_bus.undo()  # Reverts the command
facade.command_bus.redo()  # Re-applies the command
```

## Related

- [Undo/Redo skill](../../../.cursor/skills/echozero-undo-redo/SKILL.md)
