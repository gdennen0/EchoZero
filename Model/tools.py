import os
import datetime

# ---------------------
# Logging Functionality
# ---------------------

import datetime

class Log:
    
    def time_decorator(func):
        def wrapper(*args, **kwargs):
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {current_time}")
            return func(*args, **kwargs)
        return wrapper
    
    @time_decorator
    def debug(message):
        print(f"DEBUG: {message}")
    @time_decorator
    def info(message):
        print(f"INFO: {message}")

    @time_decorator
    def warning(message):
        print(f"WARNING: {message}")

    @time_decorator
    def error(message):
        print(f"ERROR: {message}")

    @time_decorator
    def unknown(message, level):
        print(f"UNKNOWN '{level}': {message}")  # Default if unknown level is passed


def prompt(prompt_message):
    # Prompt user in terminal and return the response
    response = input(prompt_message)
    return response


def path_exists(path):
    # Checks if the path exists
    return os.path.exists(path)

def file_exists(path):
    # Checks if the path is a file
    return os.path.isfile(path)

def is_valid_audio_format(path):
    # Checks if the file is valid format
    valid_extensions = ['.wav', '.mp3', '.flac', '.aac']
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in valid_extensions
