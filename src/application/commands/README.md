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
