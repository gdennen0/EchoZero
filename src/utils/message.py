import logging
import os
import sys
from datetime import datetime
from logging import Logger, Handler
from colorama import init, Fore, Style
init(autoreset=True)

class GUIConsoleHandler(Handler):
    """
    Custom logging handler that writes to the GUI console widget.
    This handler can be added to the logger to ensure all log messages
    appear in the GUI console.
    """
    def __init__(self, console_widget=None):
        super().__init__()
        self.console_widget = console_widget
        self.level_map = {
            logging.DEBUG: "DEBUG",
            logging.INFO: "INFO", 
            logging.WARNING: "WARNING",
            logging.ERROR: "ERROR",
            logging.CRITICAL: "ERROR"
        }
    
    def set_console_widget(self, console_widget):
        """Set the console widget to write to"""
        self.console_widget = console_widget
    
    def emit(self, record):
        """Emit a log record to the GUI console"""
        if self.console_widget and hasattr(self.console_widget, 'add_output'):
            try:
                # Get the level name
                level = self.level_map.get(record.levelno, "INFO")
                
                # Format the message
                message = self.format(record)
                
                # Send to console widget
                self.console_widget.add_output(level, message)
            except Exception:
                # If there's an error writing to GUI console, don't crash
                pass

def create_log_directory(log_folder: str = None):
    """
    Ensures that the log directory exists. If not, it creates it.
    
    Args:
        log_folder: Optional path to log folder. If None, uses platform-specific location.
    """
    if log_folder is None:
        from src.utils.paths import get_logs_dir
        log_folder = str(get_logs_dir())
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    return log_folder

def get_log_file_path(log_folder: str = None) -> str:
    """
    Returns a log file path with a timestamp in the name.
    Format: logs/echozero_YYYY-mm-dd_HHMMSS.log
    
    Args:
        log_folder: Optional path to log folder. If None, uses platform-specific location.
    """
    if log_folder is None:
        from src.utils.paths import get_logs_dir
        log_folder = str(get_logs_dir())
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(log_folder, f"echozero_{timestamp}.log")

def purge_old_logs(log_folder: str = None, keep: int = 10):
    """
    Removes older log files, keeping only the most recent 'keep' files.
    Logs are assumed to be named in the format echozero_YYYY-mm-dd_HHMMSS.log,
    so lexicographical sort will match chronological order.
    
    Args:
        log_folder: Optional path to log folder. If None, uses platform-specific location.
        keep: Number of most recent log files to keep.
    """
    if log_folder is None:
        from src.utils.paths import get_logs_dir
        log_folder = str(get_logs_dir())
    
    # Gather all log files that match our naming pattern:
    all_logs = [f for f in os.listdir(log_folder)
                if f.startswith("echozero_") and f.endswith(".log")]
    
    # Sort them oldest first (lexicographically works due to our timestamp naming):
    all_logs.sort()
    
    # Keep only the last 'keep' logs:
    logs_to_remove = all_logs[:-keep]  # all but the last 'keep' logs
    for old_file in logs_to_remove:
        os.remove(os.path.join(log_folder, old_file))
class ColorFormatter(logging.Formatter):
    """
    A formatter that colorizes log level names using colorama.
    """
    color_map = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.LIGHTRED_EX,
    }

    def format(self, record):
        # Avoid using dir() which can trigger recursion
        try:
            color = self.color_map.get(record.levelno, Fore.WHITE)
            # Make a copy to avoid modifying the original record
            record = logging.makeLogRecord(record.__dict__)
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        except:
            # If colorization fails, just continue without it
            pass
        return super().format(record)
    
def init_logger(
    name: str = "primary logger",
    log_folder: str = None,
    console_logging: bool = True,
    file_logging: bool = True,
    level: int = logging.DEBUG
) -> Logger:
    """
    Initializes and configures the logger with the specified settings.
    :param name: The logger's name.
    :param log_folder: The folder where log files should go. If None, uses platform-specific location.
    :param console_logging: Whether to log to the console.
    :param file_logging: Whether to log to a file.
    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :return: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs if root logger also logs

    # Ensure logger doesn't get duplicate handlers if init_logger is called multiple times:
    if not logger.handlers:
        # File Handler
        if file_logging:
            log_folder = create_log_directory(log_folder)
            purge_old_logs(log_folder, keep=10)
            log_file_path = get_log_file_path(log_folder)
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(level)
            
            # Configure file format
            file_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Console Handler
        if console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            # Optional: You can colorize the console messages by integrating colorama or a specialized formatter
            console_formatter = ColorFormatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger

class RepetitiveMessageFilter(logging.Filter):
    """
    Filter to suppress repetitive DEBUG messages that clutter the console.
    Filters out common noisy patterns like cache hits, status checks, etc.
    """
    
    # Patterns to filter out (case-insensitive substring matching)
    FILTER_PATTERNS = [
        "Cache HIT:",
        "Cache MISS:",
        "BlockItem: Received BlockStatusChanged event",
        "BlockItem: Scheduling status update",
        "BlockItem: Data state unchanged",
        "BlockItem: Refreshed block entity",
        "DataStateService: Checking port",
        "DataStateService: Block",
        "DataStateService: port '",
        "[DELETE DEBUG]",
    ]
    
    # Logger names to filter (if you want to filter entire loggers)
    FILTER_LOGGERS = set()
    
    def filter(self, record):
        """Return False to filter out the message, True to allow it"""
        # Always allow non-DEBUG messages
        if record.levelno > logging.DEBUG:
            return True
        
        # Check if logger name should be filtered
        if record.name in self.FILTER_LOGGERS:
            return False
        
        # Check message content against filter patterns
        message = record.getMessage()
        for pattern in self.FILTER_PATTERNS:
            if pattern.lower() in message.lower():
                return False
        
        return True


class Log:
    """
    Wrapper class to keep your existing interface (e.g., Log.info(...)),
    but behind the scenes we'll use Python's logging.
    """
    # A module-level logger. In practice, you might want to pass the logger object around,
    # or store it in a place that is easy to import.
    _logger: Logger = init_logger(name="EchoZeroLogger", console_logging=True, file_logging=True, level=logging.DEBUG)
    _gui_handler: GUIConsoleHandler | None = None
    _repetitive_filter: RepetitiveMessageFilter | None = None

    @classmethod
    def set_logger(cls, logger: Logger):
        """
        If you need to replace the logger at runtime or reinitialize with different settings,
        call this method.
        """
        cls._logger = logger
    
    @classmethod
    def set_level(cls, level: str | int):
        """
        Set the logging level dynamically.
        
        Args:
            level: Log level as string ("DEBUG", "INFO", "WARNING", "ERROR") or int
        """
        if isinstance(level, str):
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            level = level_map.get(level.upper(), logging.INFO)
        
        cls._logger.setLevel(level)
        # Update all handlers
        for handler in cls._logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def enable_repetitive_filter(cls, enable: bool = True):
        """
        Enable or disable filtering of repetitive DEBUG messages.
        
        Args:
            enable: If True, filter out repetitive messages. If False, show all messages.
        """
        if enable:
            if cls._repetitive_filter is None:
                cls._repetitive_filter = RepetitiveMessageFilter()
            # Add filter to all handlers
            for handler in cls._logger.handlers:
                handler.addFilter(cls._repetitive_filter)
        else:
            if cls._repetitive_filter is not None:
                # Remove filter from all handlers
                for handler in cls._logger.handlers:
                    handler.removeFilter(cls._repetitive_filter)

    @classmethod
    def message(cls, text: str):
        cls._logger.info(text)

    @classmethod
    def command(cls, text: str):
        cls._logger.info(f"[COMMAND] {text}")

    @classmethod
    def debug(cls, text: str):
        cls._logger.debug(text)

    @classmethod
    def info(cls, text: str):
        cls._logger.info(text)

    @classmethod
    def warning(cls, text: str, exc_info: bool = False):
        if exc_info:
            cls._logger.warning(text, exc_info=True)
        else:
            cls._logger.warning(text)

    @classmethod
    def error(cls, text: str):
        cls._logger.error(text)

    @classmethod
    def unknown(cls, text: str):
        cls._logger.warning(f"[UNKNOWN] {text}")

    @classmethod
    def parser(cls, text: str):
        cls._logger.debug(f"[PARSER] {text}")

    @classmethod
    def special(cls, text: str):
        # You could treat 'special' logs with a different log level or tagging
        cls._logger.info(f"[SPECIAL] {text}")

    @classmethod
    def prompt(cls, text: str):
        # For user prompts, you might even want a separate severity or just treat as info
        cls._logger.info(f"[PROMPT] {text}")

    @classmethod
    def list(cls, title, list_items, atrib=None):
        header_length = 60
        header = title.center(header_length, '*')
        cls._logger.info(header)
        for index, item in enumerate(list_items):
            if not atrib:
                cls._logger.info(f"[LIST] Index: {index}: Item: {str(item)[:60]}")
            else:
                cls._logger.info(f"[LIST] Index: {index}: Item.{atrib}: {str(getattr(item, atrib))[:60]}")
        cls._logger.info('*' * header_length)

    @classmethod
    def set_gui_console(cls, console_widget):
        """
        Set the GUI console widget to receive all log messages.
        This ensures all log messages go to the GUI console.
        """
        if cls._gui_handler is None:
            cls._gui_handler = GUIConsoleHandler(console_widget)
            # Add handler to our logger
            cls._logger.addHandler(cls._gui_handler)
            # Also add handler to root logger to capture all external library logs
            logging.getLogger().addHandler(cls._gui_handler)
        else:
            cls._gui_handler.set_console_widget(console_widget)

    @classmethod
    def remove_gui_console(cls):
        """
        Remove the GUI console handler from the logger.
        """
        if cls._gui_handler is not None:
            cls._logger.removeHandler(cls._gui_handler)
            # Also remove from root logger
            logging.getLogger().removeHandler(cls._gui_handler)
            cls._gui_handler = None