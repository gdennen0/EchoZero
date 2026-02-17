"""
Result Types for Application Commands

Structured return types for all application operations.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any, TypeVar, Generic
from enum import Enum


class ResultStatus(Enum):
    """Status of a command execution"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"


# Type variable for generic CommandResult
T = TypeVar('T')


@dataclass
class CommandResult(Generic[T]):
    """
    Structured result from application commands.
    
    Provides rich information about command execution:
    - status: Success, error, or warning
    - message: Human-readable result message
    - data: Structured data (entities, lists, etc.)
    - errors: List of error messages
    - warnings: List of warning messages
    
    Type Parameters:
        T: Type of data returned (Block, List[Block], Connection, etc.)
    
    Examples:
        CommandResult[Block] - Returns a single Block entity
        CommandResult[List[Block]] - Returns a list of Block entities
        CommandResult[None] - Returns no data (for operations like delete)
    """
    status: ResultStatus
    message: str
    data: Optional[T] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Check if command was successful"""
        return self.status == ResultStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        """Check if command failed"""
        return self.status == ResultStatus.ERROR
    
    @classmethod
    def success_result(cls, message: str, data: T = None) -> 'CommandResult[T]':
        """
        Create a success result.
        
        Args:
            message: Human-readable success message
            data: Result data (entity, list of entities, etc.)
            
        Returns:
            CommandResult[T] with SUCCESS status
        """
        return cls(
            status=ResultStatus.SUCCESS,
            message=message,
            data=data
        )
    
    @classmethod
    def error_result(cls, message: str, errors: List[str] = None) -> 'CommandResult[T]':
        """
        Create an error result.
        
        Args:
            message: Human-readable error message
            errors: List of detailed error messages
            
        Returns:
            CommandResult[T] with ERROR status and no data
        """
        return cls(
            status=ResultStatus.ERROR,
            message=message,
            errors=errors or []
        )
    
    @classmethod
    def warning_result(cls, message: str, data: T = None, warnings: List[str] = None) -> 'CommandResult[T]':
        """
        Create a warning result.
        
        Args:
            message: Human-readable warning message
            data: Result data (may be partial)
            warnings: List of warning messages
            
        Returns:
            CommandResult[T] with WARNING status
        """
        return cls(
            status=ResultStatus.WARNING,
            message=message,
            data=data,
            warnings=warnings or []
        )


