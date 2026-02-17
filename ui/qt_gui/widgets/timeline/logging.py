"""
Timeline Logging

Optional logging wrapper for the timeline widget.
Makes the package standalone by not requiring external logging infrastructure.

If you have an existing logging system, you can redirect timeline logs:

    from ui.qt_gui.widgets.timeline.logging import TimelineLog
    TimelineLog.set_handler(my_log_function)

Or disable logging entirely:

    TimelineLog.enabled = False
"""

import sys
from typing import Callable, Optional


class TimelineLog:
    """
    Simple logging wrapper for timeline widgets.
    
    By default, logs to stderr. Can be customized or disabled.
    """
    
    # Enable/disable all logging
    enabled: bool = True
    
    # Log levels
    DEBUG: int = 10
    INFO: int = 20
    WARNING: int = 30
    ERROR: int = 40
    
    # Current log level (DEBUG and above)
    level: int = DEBUG
    
    # Custom handler (if set, overrides default behavior)
    _handler: Optional[Callable[[int, str], None]] = None
    
    @classmethod
    def set_handler(cls, handler: Callable[[int, str], None]) -> None:
        """
        Set a custom log handler.
        
        Args:
            handler: Function that takes (level: int, message: str)
        """
        cls._handler = handler
    
    @classmethod
    def _log(cls, level: int, message: str) -> None:
        """Internal log method"""
        if not cls.enabled or level < cls.level:
            return
        
        if cls._handler:
            cls._handler(level, message)
        else:
            # Default: print to stderr
            level_names = {
                cls.DEBUG: "DEBUG",
                cls.INFO: "INFO",
                cls.WARNING: "WARNING",
                cls.ERROR: "ERROR"
            }
            level_name = level_names.get(level, "LOG")
            print(f"[Timeline:{level_name}] {message}", file=sys.stderr)
    
    @classmethod
    def debug(cls, message: str) -> None:
        """Log debug message"""
        cls._log(cls.DEBUG, message)
    
    @classmethod
    def info(cls, message: str) -> None:
        """Log info message"""
        cls._log(cls.INFO, message)
    
    @classmethod
    def warning(cls, message: str) -> None:
        """Log warning message"""
        cls._log(cls.WARNING, message)
    
    @classmethod
    def error(cls, message: str) -> None:
        """Log error message"""
        cls._log(cls.ERROR, message)


# Convenience alias
Log = TimelineLog


