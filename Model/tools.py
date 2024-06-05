import os
# Select aduio file and check that it is a valid format

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
