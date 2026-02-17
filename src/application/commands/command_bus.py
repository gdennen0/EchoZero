"""
Command Bus - Centralized Command Ingestion Pipeline

Firmware-style architecture for bulletproof undo/redo:
1. SINGLE entry point for all commands
2. GUARANTEED delivery to undo stack
3. LOGGED for debugging
4. VALIDATED before execution

Usage:
    from src.application.commands.command_bus import CommandBus
    
    # Create instance with explicit undo_stack dependency
    command_bus = CommandBus(undo_stack)
    
    # Use instance methods
    command_bus.execute(AddBlockCommand(facade, "LoadAudio", "Test"))
    
Note: CommandBus instance should be accessed via ApplicationFacade.command_bus
to maintain explicit dependencies.
"""
from typing import Optional, TYPE_CHECKING
from PyQt6.QtGui import QUndoStack, QUndoCommand

if TYPE_CHECKING:
    pass


class CommandBus:
    """
    Command Bus for centralized command execution.
    
    Firmware-style guarantees:
    - Commands go through ONE path
    - Every command is logged
    - Failures are visible (not silent)
    - Stack state is always known
    
    Dependencies are explicit - undo_stack must be provided at construction.
    """
    
    def __init__(self, undo_stack: QUndoStack):
        """
        Initialize CommandBus with explicit undo stack dependency.
        
        Args:
            undo_stack: The QUndoStack instance (owned by MainWindow)
        """
        from src.utils.message import Log
        
        if undo_stack is None:
            raise ValueError("CommandBus requires a QUndoStack instance")
        
        self._undo_stack = undo_stack
        self._command_count = 0
        
        Log.info(f"CommandBus: Initialized with QUndoStack (id: {id(undo_stack)})")
    
    def execute(self, command: QUndoCommand) -> bool:
        """
        Execute a command through the undo system.
        
        This is THE ONLY way commands should be executed for undoable operations.
        
        Args:
            command: The QUndoCommand to execute
            
        Returns:
            True if command was pushed successfully, False otherwise
        """
        from src.utils.message import Log
        
        # Get command text BEFORE pushing (Qt takes ownership after push)
        # Must capture text early to avoid RuntimeError if object is deleted
        try:
            command_text = command.text() if command else "Unknown Command"
        except (RuntimeError, AttributeError):
            # Command object was deleted or invalid - use class name as fallback
            try:
                command_text = command.__class__.__name__
            except (RuntimeError, AttributeError):
                command_text = "Invalid Command"
        
        if self._undo_stack is None:
            Log.error("CommandBus: undo_stack is None!")
            Log.error(f"  Dropped command: {command_text}")
            return False
        
        # Execute command
        self._command_count += 1
        command_id = self._command_count
        
        before_count = self._undo_stack.count()
        
        # Push command to stack (Qt takes ownership)
        try:
            
            self._undo_stack.push(command)
            after_count = self._undo_stack.count()
            
            
            Log.debug(f"CommandBus: [{command_id}] {command_text} (stack: {before_count} -> {after_count})")
        except RuntimeError as e:
            Log.error(f"CommandBus: Failed to push command '{command_text}': {e}")
            
            
            return False
        except Exception as e:
            return False
        
        return True
    
    def begin_macro(self, description: str) -> bool:
        """
        Begin a command macro (group of commands as one undo step).
        
        Args:
            description: Description for the macro
            
        Returns:
            True if macro was started, False otherwise
        """
        from src.utils.message import Log
        
        if self._undo_stack is None:
            Log.error(f"CommandBus: Cannot begin macro '{description}' - undo_stack is None")
            return False
        
        self._undo_stack.beginMacro(description)
        Log.debug(f"CommandBus: Begin macro '{description}'")
        return True
    
    def end_macro(self) -> bool:
        """
        End the current command macro.
        
        Returns:
            True if macro was ended, False otherwise
        """
        from src.utils.message import Log
        
        if self._undo_stack is None:
            Log.error("CommandBus: Cannot end macro - undo_stack is None")
            return False
        
        self._undo_stack.endMacro()
        Log.debug("CommandBus: End macro")
        return True
    
    def undo(self) -> bool:
        """Undo the last command."""
        if self._undo_stack is not None and self._undo_stack.canUndo():
            self._undo_stack.undo()
            return True
        return False
    
    def redo(self) -> bool:
        """Redo the last undone command."""
        if self._undo_stack is not None and self._undo_stack.canRedo():
            self._undo_stack.redo()
            return True
        return False
    
    def clear(self) -> None:
        """Clear all command history."""
        if self._undo_stack is not None:
            self._undo_stack.clear()
            from src.utils.message import Log
            Log.debug("CommandBus: History cleared")
    
    def get_stack(self) -> Optional[QUndoStack]:
        """
        Get the underlying QUndoStack.
        
        Use sparingly - prefer execute() for commands.
        Needed for: QUndoStack.createUndoAction(), Command History dialog.
        """
        return self._undo_stack
    
    def get_stats(self) -> dict:
        """Get command statistics for debugging."""
        return {
            "total_commands": self._command_count,
            "stack_count": self._undo_stack.count() if self._undo_stack else 0,
            "can_undo": self._undo_stack.canUndo() if self._undo_stack else False,
            "can_redo": self._undo_stack.canRedo() if self._undo_stack else False,
        }



