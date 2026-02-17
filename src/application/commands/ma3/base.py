"""
Base MA3 Command Class

Foundation for all bidirectional commands between EchoZero and grandMA3.
Extends the standard EchoZero command pattern with:
- validate(): Pre-execution validation
- to_osc(): Convert command to OSC message format
- Target support: Execute on Editor (local) or MA3 (via OSC)

COMMAND STANDARD
================

All MA3 commands MUST:
1. INHERIT from BaseMA3Command
2. IMPLEMENT validate() to check preconditions
3. IMPLEMENT _do_execute() for the actual operation
4. IMPLEMENT _do_undo() to reverse the operation
5. IMPLEMENT to_osc() to convert to OSC format

Usage:
    # Create and validate command
    cmd = AddEventCommand(context, time=1.5, classification="kick")
    result = cmd.validate()
    if not result.valid:
        Log.error(f"Validation failed: {result.errors}")
        return
    
    # Execute on target
    result = cmd.execute("editor")  # or "ma3"
    if not result.success:
        Log.error(f"Execution failed: {result.error}")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade


class CommandTarget(Enum):
    """Target for command execution."""
    EDITOR = "editor"  # Execute on Editor block (local)
    MA3 = "ma3"        # Execute on grandMA3 (via OSC)


@dataclass
class CommandContext:
    """
    Context for MA3 command execution.
    
    Contains references to services and entities needed for command execution.
    """
    facade: "ApplicationFacade"
    
    # Editor block context (for local execution)
    editor_block_id: Optional[str] = None
    
    # MA3 context (for OSC execution)
    timecode_no: int = 1
    track_group_idx: int = 0
    
    # OSC bridge service reference (optional, set when MA3 target is used)
    osc_bridge: Optional[Any] = None


@dataclass
class ValidationResult:
    """Result of command validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def success(cls) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(valid=True)
    
    @classmethod
    def failure(cls, error: str) -> "ValidationResult":
        """Create a failed validation result with single error."""
        return cls(valid=False, errors=[error])
    
    @classmethod
    def failures(cls, errors: List[str]) -> "ValidationResult":
        """Create a failed validation result with multiple errors."""
        return cls(valid=False, errors=errors)
    
    def add_error(self, error: str) -> "ValidationResult":
        """Add an error (makes result invalid)."""
        self.errors.append(error)
        self.valid = False
        return self
    
    def add_warning(self, warning: str) -> "ValidationResult":
        """Add a warning (does not affect validity)."""
        self.warnings.append(warning)
        return self


@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    @classmethod
    def ok(cls, data: Optional[Dict[str, Any]] = None) -> "CommandResult":
        """Create a successful result."""
        return cls(success=True, data=data)
    
    @classmethod
    def fail(cls, error: str) -> "CommandResult":
        """Create a failed result."""
        return cls(success=False, error=error)


class BaseMA3Command(ABC):
    """
    Base class for all MA3 bidirectional commands.
    
    Unlike standard EchoZero commands (which use QUndoCommand),
    MA3 commands are standalone and can execute on multiple targets.
    
    Key Methods:
        validate(): Check if command can execute
        execute(target): Execute on editor or ma3
        undo(): Reverse the operation
        to_osc(): Convert to OSC message format
    
    Subclass Requirements:
        - Implement validate()
        - Implement _do_execute() for actual operation
        - Implement _do_undo() to reverse operation
        - Implement to_osc() for OSC message format
    """
    
    # Command type identifier (override in subclasses)
    COMMAND_TYPE: str = "base"
    
    # OSC address prefix (override in subclasses)
    OSC_ADDRESS: str = "/echozero/unknown"
    
    def __init__(self, context: CommandContext):
        """
        Initialize command.
        
        Args:
            context: Command context with references to services and entities
        """
        self._context = context
        self._executed = False
        self._undo_data: Optional[Dict[str, Any]] = None
    
    @property
    def context(self) -> CommandContext:
        """Access to command context."""
        return self._context
    
    @property
    def facade(self) -> "ApplicationFacade":
        """Shortcut to application facade."""
        return self._context.facade
    
    @abstractmethod
    def validate(self) -> ValidationResult:
        """
        Validate command parameters before execution.
        
        Called before execute() to check preconditions.
        Should NOT modify any state.
        
        Returns:
            ValidationResult with valid=True if command can execute,
            otherwise valid=False with error messages.
        """
        pass
    
    def execute(self, target: str = "editor") -> CommandResult:
        """
        Execute command on specified target.
        
        Args:
            target: "editor" for local execution, "ma3" for OSC execution
            
        Returns:
            CommandResult with success status and any returned data
        """
        # Parse target
        try:
            target_enum = CommandTarget(target.lower())
        except ValueError:
            return CommandResult.fail(f"Invalid target: {target}. Use 'editor' or 'ma3'.")
        
        # Validate first
        validation = self.validate()
        if not validation.valid:
            return CommandResult.fail(f"Validation failed: {', '.join(validation.errors)}")
        
        # Execute based on target
        try:
            if target_enum == CommandTarget.EDITOR:
                result = self._execute_on_editor()
            else:
                result = self._execute_on_ma3()
            
            if result.success:
                self._executed = True
            
            return result
            
        except Exception as e:
            from src.utils.message import Log
            Log.error(f"[{self.__class__.__name__}] Execution error: {e}", exc_info=True)
            return CommandResult.fail(str(e))
    
    def _execute_on_editor(self) -> CommandResult:
        """Execute command locally on Editor block."""
        return self._do_execute()
    
    def _execute_on_ma3(self) -> CommandResult:
        """Execute command on grandMA3 via OSC."""
        if not self._context.osc_bridge:
            return CommandResult.fail("OSC bridge not configured")
        
        address, args = self.to_osc()
        try:
            self._context.osc_bridge.send_message(address, *args)
            return CommandResult.ok()
        except Exception as e:
            return CommandResult.fail(f"OSC send error: {e}")
    
    @abstractmethod
    def _do_execute(self) -> CommandResult:
        """
        Perform the actual command operation.
        
        Called by execute() after validation.
        Should store any data needed for undo in self._undo_data.
        
        Returns:
            CommandResult with success status
        """
        pass
    
    def undo(self) -> CommandResult:
        """
        Reverse the command operation.
        
        Returns:
            CommandResult with success status
        """
        if not self._executed:
            return CommandResult.fail("Command was not executed")
        
        try:
            result = self._do_undo()
            if result.success:
                self._executed = False
            return result
        except Exception as e:
            from src.utils.message import Log
            Log.error(f"[{self.__class__.__name__}] Undo error: {e}", exc_info=True)
            return CommandResult.fail(str(e))
    
    @abstractmethod
    def _do_undo(self) -> CommandResult:
        """
        Perform the actual undo operation.
        
        Should restore state from self._undo_data.
        
        Returns:
            CommandResult with success status
        """
        pass
    
    @abstractmethod
    def to_osc(self) -> Tuple[str, List[Any]]:
        """
        Convert command to OSC message format.
        
        Returns:
            Tuple of (osc_address, [args...])
            Example: ("/echozero/timecode/add_event", [1, 0, 0, 1.5, "cmd", "{}"])
        """
        pass
    
    def _log_info(self, message: str):
        """Log info message."""
        from src.utils.message import Log
        Log.info(f"[{self.__class__.__name__}] {message}")
    
    def _log_error(self, message: str):
        """Log error message."""
        from src.utils.message import Log
        Log.error(f"[{self.__class__.__name__}] {message}")
