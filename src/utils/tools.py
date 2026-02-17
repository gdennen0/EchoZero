import datetime
import os
from src.utils.message import Log
import librosa
import torchaudio
import time

@staticmethod
def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time
    
def prompt(prompt_message):
    """Prompt user for input via terminal"""
    response = input(prompt_message)
    if response.lower() in ['e', 'exit']:
        Log.info("Selection exited by user.")
        return
    return response

def prompt_selection(prompt_text, options, allow_multiple=False):
    """Prompt user for selection via terminal"""
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    elif isinstance(options, set):
        options_list = list(options)
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        if hasattr(obj, 'name'):
            Log.info(f"{i}: {obj.name}")
        else:
            Log.info(f"{i}: {obj}")
    
    if allow_multiple:
        Log.info("Enter multiple selections separated by commas (e.g., 0,2,4)")
    
    while True:
        selection_prompt = f"Please enter the key or index for your selection (or 'e' to exit): "
        if allow_multiple:
            selection_prompt = f"Please enter indices/keys separated by commas (or 'e' to exit): "
        selection = prompt(selection_prompt)
        if not selection:
            Log.info("Selection exited by user.")
            return None
        
        if allow_multiple:
            # Handle multiple selections
            selections = [s.strip() for s in selection.split(',')]
            results = []
            valid = True
            for sel in selections:
                if sel.isdigit():
                    index = int(sel)
                    if 0 <= index < len(options_list):
                        results.append(options_list[index])
                    else:
                        Log.error(f"Invalid index: {sel}")
                        valid = False
                        break
                elif sel in options_list:
                    results.append(options[sel])
                else:
                    Log.error(f"Invalid selection: {sel}")
                    valid = False
                    break
        if valid:
            return results
        else:
            # Handle single selection
            if selection.isdigit():
                index = int(selection)
                if 0 <= index < len(options_list):
                    return options_list[index]
            elif selection in options_list:
                return options[selection]
            Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def prompt_yes_no(prompt_text):
    """Prompt user for yes/no via terminal"""
    Log.info(prompt_text)
    response = input("Please enter 'y' for yes or 'n' for no: ")
    if response.lower() in ['y', 'yes']:
        return True
    elif response.lower() in ['n', 'no']:
        return False
    else:
        Log.error("Invalid selection. Please enter 'y' for yes or 'n' for no.")
        return prompt_yes_no(prompt_text)

def prompt_selection_with_type(prompt_text, options):
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        Log.info(f"{i}: {obj.type}>{obj.name}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection: 
            Log.info("Selection exited by user.")
            return None, None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index], index
        elif selection in options_list:
            return options[selection], selection
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def prompt_selection_by_name(prompt_text, options):
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        Log.info(f"{i}: {obj.name}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection: 
            Log.info("Selection exited by user.")
            return None, None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index], options_list[index].name
        elif selection in options_list:
            return options[selection], options[selection].name
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def prompt_selection_with_type_and_parent_block(prompt_text, options): #imsorryfortheshitename
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        Log.info(f"{i}: {obj.parent_block.name}:{obj.type}>{obj.name}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection:
            Log.info("Selection exited by user.")
            return None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index]
        elif selection in options_list:
            return options[selection]
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

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
    try:
        abs_path = os.path.abspath(path)
        if not path_exists(abs_path): # Check if the path is valid
            Log.error(f"Invalid Path: '{abs_path}'")
            return False
        if not file_exists(abs_path):   # Check if the file exists at specified path
            Log.error(f"File does not exist at specified path")
            return False

        return True
    except Exception as e:
        Log.error(f"An error occurred while checking project path: {e}")
        return False
    

def generate_unique_name(base_name, existing_items):
    existing_names = {item.name for item in existing_items}
    if base_name not in existing_names:
        return base_name
    numbers = [
        int(name[len(base_name) :])
        for name in existing_names
        if name.startswith(base_name) and name[len(base_name) :].isdigit()
    ]
    max_num = max(numbers) if numbers else 1
    new_num = max_num + 1
    return f"{base_name}{new_num}"

def check_audio_path(path):
    try:
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
    except Exception as e:
        Log.error(f"An error occurred while checking audio path: {e}")
        return False


def validate_audio_data_item(audio_item):
    """
    Validate that an AudioDataItem has a valid, accessible audio file.
    
    Args:
        audio_item: AudioDataItem to validate
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        from src.shared.domain.entities import AudioDataItem
        
        if not isinstance(audio_item, AudioDataItem):
            return False, "Not an AudioDataItem"
        
        # Check file_path attribute
        file_path = audio_item.file_path
        if not file_path:
            # Check metadata as fallback
            file_path = audio_item.metadata.get('file_path')
        
        if not file_path:
            return False, "No file path specified"
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"Audio file not found: {file_path}"
        
        # Check if it's a valid audio format
        if not is_valid_audio_format(file_path):
            return False, f"Invalid audio format: {os.path.splitext(file_path)[1]}"
        
        return True, None
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_audio_items(audio_items):
    """
    Validate multiple AudioDataItems and return validation results.
    
    Args:
        audio_items: List of AudioDataItem objects to validate
        
    Returns:
        dict: {
            'valid': list of valid AudioDataItems,
            'invalid': list of (AudioDataItem, error_message) tuples,
            'all_valid': bool indicating if all items are valid
        }
    """
    valid_items = []
    invalid_items = []
    
    if not audio_items:
        return {'valid': [], 'invalid': [], 'all_valid': True}
    
    for item in audio_items:
        is_valid, error_msg = validate_audio_data_item(item)
        if is_valid:
            valid_items.append(item)
        else:
            invalid_items.append((item, error_msg))
    
    return {
        'valid': valid_items,
        'invalid': invalid_items,
        'all_valid': len(invalid_items) == 0
    }

def path_exists(path):
    try:
        # Checks if the path exists
        return os.path.exists(path)
    except Exception as e:
        Log.error(f"An error occurred while checking if path exists: {e}")
        return False

def file_exists(path):
    try:
        # Checks if the path is a file
        return os.path.isfile(path)
    except Exception as e:
        Log.error(f"An error occurred while checking if file exists: {e}")
        return False

def is_valid_audio_format(path):
    try:
        # Checks if the file is valid format
        valid_extensions = ['.wav', '.mp3', '.flac', '.aac']
        _, file_extension = os.path.splitext(path)
        return file_extension.lower() in valid_extensions
    except Exception as e:
        Log.error(f"An error occurred while validating audio format: {e}")
        return False

def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def create_audio_data(audio_file_path, target_sr):
    try:
        # creates audio data array using librosa
        # standardise sample rate??
        data, sr = librosa.load(audio_file_path, sr=target_sr)
        return data
    except Exception as e:
        Log.error(f"An error occurred while creating audio data: {e}")
        return None

def create_audio_tensor(audio_file_path, target_sr):
    try:
        # creates a tensor object with the audio file
        # standardise sample rate??
        audio_tensor, sr = torchaudio.load(audio_file_path, target_sr)
        return audio_tensor, sr
    except Exception as e:
        Log.error(f"An error occurred while creating audio tensor: {e}")
        return None, None
    
def get_object_by_name(object_list, object_name):
    for object in object_list:
        if object.name == object_name:
            return object
    return None


class gtimer():
    def __init__(self):
        self.start_time = 0.00
        self.elapsed_time = 0.00

    def start(self):
        self.start_time = time.time()

    def pause(self):
        self.elapsed_time = time.time() - self.start_time

    def stop(self):
        self.elapsed_time = time.time() - self.start_time
        return self.elapsed_time

def prompt_file_path(prompt_message, file_ext=None, directory=None):
    """
    Prompt the user to enter a file path with optional file extension filtering.
    
    Args:
        prompt_message (str): The message to display to the user
        file_ext (str, optional): File extension to filter by (e.g., 'json', 'wav')
        directory (str, optional): Starting directory to look in
    
    Returns:
        str: The selected file path or None if selection was cancelled
    """
    import os
    import glob
    
    Log.info(prompt_message)
    
    # Set up directory
    if directory is None:
        directory = os.getcwd()
    
    # Format file extension
    if file_ext:
        if not file_ext.startswith('.'):
            file_ext = f".{file_ext}"
    
    # Allow user to directly enter a path
    user_input = prompt("Enter file path:")
    
    # Return if user entered a direct path
    if user_input and user_input.lower() not in ['e', 'exit']:
        if file_ext and not user_input.lower().endswith(file_ext.lower()):
            Log.warning(f"File does not have the expected extension: {file_ext}")
            if not prompt_yes_no("Continue with this file anyway?"):
                return prompt_file_path(prompt_message, file_ext, directory)
        
        # Validate the file exists
        if not os.path.exists(user_input):
            Log.error(f"File not found: {user_input}")
            return prompt_file_path(prompt_message, file_ext, directory)
            
        return user_input
    
    # Exit if requested
    if user_input and user_input.lower() in ['e', 'exit']:
        Log.info("File selection cancelled")
        return None
    
    # Browse files in the directory
    while True:
        Log.info(f"Current directory: {directory}")
        
        # Get files and directories
        contents = []
        
        # Add parent directory option
        contents.append("..")
        
        # Add directories
        for item in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                contents.append(f"{item}/")
        
        # Add files (with optional extension filtering)
        for item in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                if file_ext is None or item.lower().endswith(file_ext.lower()):
                    contents.append(item)
        
        # Display directory contents
        for i, item in enumerate(contents):
            Log.info(f"{i}: {item}")
        
        # Get user selection
        selection = prompt("Enter number to select, or type a new path (or 'e' to exit): ")
        
        # Exit if requested
        if not selection or selection.lower() in ['e', 'exit']:
            Log.info("File selection cancelled")
            return None
        
        # Handle numeric selection
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(contents):
                selected_item = contents[index]
                
                # Handle parent directory
                if selected_item == "..":
                    directory = os.path.dirname(directory)
                    continue
                
                # Handle directory
                if selected_item.endswith("/"):
                    directory = os.path.join(directory, selected_item[:-1])
                    continue
                
                # Handle file selection
                return os.path.join(directory, selected_item)
        
        # Handle direct path entry
        elif os.path.exists(selection):
            if os.path.isdir(selection):
                directory = selection
                continue
            else:
                if file_ext and not selection.lower().endswith(file_ext.lower()):
                    Log.warning(f"File does not have the expected extension: {file_ext}")
                    if not prompt_yes_no("Continue with this file anyway?"):
                        continue
                return selection
        
        Log.error("Invalid selection. Please enter a valid number or path.")
