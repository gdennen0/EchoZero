"""
Error hierarchy: Structured exception types for EchoZero.
Exists because silent failures are banned (STYLE.md) — every error path uses typed exceptions.
Domain errors vs infrastructure errors enable different handling strategies at the pipeline boundary.
"""


class EchoZeroError(Exception):
    """Base exception for all EchoZero errors."""


class DomainError(EchoZeroError):
    """An operation violates a domain rule or invariant."""


class ValidationError(DomainError):
    """Input data or state fails validation before processing."""


class PersistenceError(EchoZeroError):
    """Raised by persistence layer operations."""


class EngineError(EchoZeroError):
    """Raised by pipeline engine operations."""


class ConfigurationError(EchoZeroError):
    """Raised for invalid configuration or missing dependencies."""


class InfrastructureError(EchoZeroError):
    """A persistence, IO, or external system operation failed."""


class ExecutionError(EchoZeroError):
    """A block execution or pipeline run failed."""


class OperationCancelledError(EchoZeroError):
    """An operation was cancelled via cancellation token."""
