import datetime
import os

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
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').")


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