import datetime
import os
from message import Log
import librosa
import torchaudio


@staticmethod
def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time
    
def prompt(prompt_message):
    # Prompt user in terminal and return the response
    response = input(prompt_message)
    return response

def prompt_selection(prompt_text, options):
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, object in enumerate(options_list):
        Log.info(f"{i}: {object}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection: ")
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index], index
        elif selection in options_list:
            return options[selection], selection
        Log.error("Invalid selection. Please enter a valid key or index.")

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

def check_project_path(path):
    abs_path = os.path.abspath(path)
    if not path_exists(abs_path): # Check if the path is valid
        Log.error(f"Invalid Path: '{abs_path}'")
        return False
    if not file_exists(abs_path):   # Check if the file exists at specified path
        Log.error(f"File does not exist at specified path")
        return False
    if not is_valid_project_format(abs_path): # Check if project is in a usable format
        Log.error(f"Invalid project format")
        return False
    return True

def check_audio_path(path):
    abs_path = os.path.abspath(path)
    if not path_exists(abs_path): # Check if the path is valid
        Log.error(f"Invalid Path: '{abs_path}'")
        return False
    if not file_exists(abs_path):   # Check if the file exists at specified path
        Log.error(f"File does not exist at specified path")
        return False
    if not is_valid_audio_format(abs_path): # Check if audio is in a usable format
        Log.error(f"Invalid audio format")
        return False
    return True
    
def is_valid_project_format(path):
    # Check if the path has a .json extension
    _, file_extension = os.path.splitext(path)
    if file_extension.lower() != '.json':
        Log.error(f"Invalid project format: Expected a .json file, got '{file_extension}'")
        return False
    return True

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

def create_audio_data(audio_file_path, target_sr):
    # creates audio data array using librosa
    # standardise sample rate??
    data, sr = librosa.load(audio_file_path, sr=target_sr)
    return data, sr

def create_audio_tensor(audio_file_path, target_sr):
    # creates a tensor object with the audio file
    # standardise sample rate??
    audio_tensor, sr = torchaudio.load(audio_file_path, target_sr)
    return audio_tensor, sr


