import logging
import os
import sys
from datetime import datetime
from logging import Logger
from colorama import init, Fore, Style
init(autoreset=True)

# You can still import and call get_current_time from your tools if desired,
# or just rely on standard logging formatting capabilities.

def create_log_directory(log_folder: str = "logs"):
    """
    Ensures that the log directory exists. If not, it creates it.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

def get_log_file_path(log_folder: str = "logs") -> str:
    """
    Returns a log file path with a timestamp in the name.
    Format: logs/myapp_YYYY-mm-dd_HHMMSS.log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return os.path.join(log_folder, f"myapp_{timestamp}.log")

def purge_old_logs(log_folder: str = "logs", keep: int = 10):
    """
    Removes older log files, keeping only the most recent 'keep' files.
    Logs are assumed to be named in the format myapp_YYYY-mm-dd_HHMMSS.log,
    so lexicographical sort will match chronological order.
    """
    # Gather all log files that match our naming pattern:
    all_logs = [f for f in os.listdir(log_folder)
                if f.startswith("myapp_") and f.endswith(".log")]
    
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
        if 'color_map' in dir(self):
            color = self.color_map.get(record.levelno, Fore.WHITE)
            record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)
    
def init_logger(
    name: str = "primary logger",
    log_folder: str = "logs",
    console_logging: bool = True,
    file_logging: bool = True,
    level: int = logging.DEBUG
) -> Logger:
    """
    Initializes and configures the logger with the specified settings.
    :param name: The logger's name.
    :param log_folder: The folder where log files should go.
    :param console_logging: Whether to log to the console.
    :param file_logging: Whether to log to a file.
    :param level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :return: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent duplicate logs if root logger also logs

    # Ensure logger doesn’t get duplicate handlers if init_logger is called multiple times:
    if not logger.handlers:
        # File Handler
        if file_logging:
            create_log_directory(log_folder)
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

class Log:
    """
    Wrapper class to keep your existing interface (e.g., Log.info(...)),
    but behind the scenes we’ll use Python’s logging.
    """
    # A module-level logger. In practice, you might want to pass the logger object around,
    # or store it in a place that is easy to import.
    _logger: Logger = init_logger(name="EchoZeroLogger", console_logging=True, file_logging=True, level=logging.DEBUG)

    @classmethod
    def set_logger(cls, logger: Logger):
        """
        If you need to replace the logger at runtime or reinitialize with different settings,
        call this method.
        """
        cls._logger = logger

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
    def warning(cls, text: str):
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