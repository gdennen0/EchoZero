"""
Base Command Class - EchoZero Command Standard

Foundation for all undoable commands in EchoZero.
Uses Qt's QUndoCommand for industry-standard undo/redo support.

COMMAND STANDARD
================

All commands in EchoZero MUST follow this pattern:

1. INHERIT from EchoZeroCommand
2. STORE state needed for undo in __init__ or first redo()
3. IMPLEMENT redo() to execute the operation
4. IMPLEMENT undo() to reverse the operation
5. USE facade.command_bus.execute() to run commands (never push directly)

TEMPLATE
--------
```python
class MyCommand(EchoZeroCommand):
    '''
    Brief description of what the command does.
    
    Redo: What happens when executed/redone
    Undo: What happens when undone
    '''
    
    def __init__(self, facade: "ApplicationFacade", param1, param2):
        super().__init__(facade, f"My Operation: {param1}")
        
        # Store parameters
        self._param1 = param1
        self._param2 = param2
        
        # State to restore on undo (captured in first redo)
        self._original_state = None
    
    def redo(self):
        '''Execute the operation.'''
        # Capture original state (first time only)
        if self._original_state is None:
            self._original_state = self._get_current_state()
        
        # Apply the change
        self._facade.do_something(self._param1, self._param2)
    
    def undo(self):
        '''Reverse the operation.'''
        if self._original_state is not None:
            self._facade.restore_state(self._original_state)
```

GUIDELINES
----------
1. Commands should be ATOMIC - one logical operation
2. Commands should be REVERSIBLE - undo must restore exact state
3. Commands should be DESCRIPTIVE - text shows in UI
4. Commands should FAIL GRACEFULLY - no crashes on invalid state
5. Commands should LOG errors - use Log.error() for failures

COMMAND CATEGORIES
------------------
- Block Commands: AddBlockCommand, DeleteBlockCommand, RenameBlockCommand, etc.
- Connection Commands: CreateConnectionCommand, DeleteConnectionCommand
- Position Commands: MoveBlockCommand (for drag operations)
- Metadata Commands: UpdateBlockMetadataCommand

See HANDBOOK.md Section 12 for complete command reference.
"""

from typing import TYPE_CHECKING, Optional
from PyQt6.QtGui import QUndoCommand

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class EchoZeroCommand(QUndoCommand):
    """
    Base class for all EchoZero undoable commands.
    
    All commands MUST inherit from this class and implement:
        - redo(): Execute (or re-execute) the operation
        - undo(): Reverse the operation
    
    Optional overrides:
        - id(): Return unique ID for command merging (default: -1, no merge)
        - mergeWith(): Merge with another command (for drag operations)
    
    Key Concepts:
        - Qt calls redo() when command is first pushed to stack
        - There is no separate execute() method - redo() IS the execute
        - Store original state in redo() before making changes
        - undo() must restore the exact original state
    
    Usage:
        from src.application.commands import MyCommand
        
        cmd = MyCommand(facade, param1, param2)
        facade.command_bus.execute(cmd)  # This pushes and executes via redo()
    """
    
    # Command type identifier (override in subclasses for grouping)
    COMMAND_TYPE: str = "base"
    
    def __init__(
        self, 
        facade: "ApplicationFacade", 
        description: str, 
        parent: Optional[QUndoCommand] = None
    ):
        """
        Initialize command.
        
        Args:
            facade: ApplicationFacade for executing operations
            description: Human-readable description (shown in Edit menu and Command History)
            parent: Optional parent command (for macro grouping)
        """
        super().__init__(description, parent)
        self._facade = facade
        self._executed = False  # Track if first execution completed
    
    def redo(self):
        """
        Execute or re-execute the operation.
        
        Called by QUndoStack when:
            1. Command is first pushed to stack (initial execution)
            2. User triggers redo after undo
        
        Implementation Requirements:
            - First call: Capture original state, then apply change
            - Subsequent calls: Apply change (state already captured)
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement redo()")
    
    def undo(self):
        """
        Reverse the operation.
        
        Called by QUndoStack when user triggers undo.
        
        Implementation Requirements:
            - Restore the exact state that existed before redo()
            - Handle edge cases (e.g., entity was deleted externally)
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement undo()")
    
    def id(self) -> int:
        """
        Return command ID for merging.
        
        Commands with the same non-negative ID can be merged.
        Return -1 to disable merging (default).
        
        Override for commands that should merge (e.g., repeated position changes).
        
        Common IDs (convention):
            - 1000: Block position changes
            - 2000: Block metadata changes (same key)
            - -1: No merging (default)
        """
        return -1
    
    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Attempt to merge with another command.
        
        Called by QUndoStack when a new command has same id() as top command.
        Merging combines multiple commands into one undo step.
        
        Example: Multiple position changes while dragging become one "Move Block" command.
        
        Args:
            other: The new command to potentially merge
            
        Returns:
            True if merged (other is discarded), False otherwise
        
        Override for commands that should merge (e.g., drag operations).
        """
        return False
    
    @property
    def facade(self) -> "ApplicationFacade":
        """Access to the application facade."""
        return self._facade
    
    def _log_error(self, message: str):
        """Log an error during command execution."""
        from src.utils.message import Log
        Log.error(f"[{self.__class__.__name__}] {message}")
    
    def _log_warning(self, message: str):
        """Log a warning during command execution."""
        from src.utils.message import Log
        Log.warning(f"[{self.__class__.__name__}] {message}")
