import datetime
import os
from message import Log


@staticmethod
def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time
    
def prompt(prompt_message):
    # Prompt user in terminal and return the response
    response = input(prompt_message)
    return response

def yes_no_prompt(prompt_message):
    # Prompt user with a yes/no question and return True for yes and False for no
    valid_yes = {'yes', 'y', 'ye', 'YES', 'Y', 'YE'}
    valid_no = {'no', 'n', 'NO', 'N'}
    
    while True:
        response = input(prompt_message).strip().lower()
        if response in valid_yes:
            return True
        elif response in valid_no:
            return False
        else:
            Log.prompt("Please respond with 'yes' or 'no' (or 'y' or 'n').")

def check_audio_path(path):
    abs_path = os.path.abspath(path)
    if not path_exists(abs_path): # Check if the path is valid
        Log.error(f"Invalid Path: '{abs_path}'")
        return
    if not file_exists(abs_path):   # Check if the file exists at specified path
        Log.error(f"File does not exist at specified path")
        return
    if not is_valid_audio_format(abs_path): # Check if audio is in a usable format
        Log.error(f"Invalid audio format")
        return
    
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

def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time

