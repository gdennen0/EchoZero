"""
Remote control package: Thin private wrapper over the EchoZero automation bridge.
Exists because phone access needs session-scoped controls without widening the raw bridge surface.
Connects localhost automation transport to a small mobile-facing HTTP layer.
"""

from .server import RemoteControlServer

__all__ = ["RemoteControlServer"]
