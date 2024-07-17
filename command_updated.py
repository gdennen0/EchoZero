import os
from message import Log
from tools import prompt, yes_no_prompt, path_exists, file_exists, is_valid_audio_format

"""
Responsible for Validating and directing input streams to execute the proper control actions

"""
# the command_tools class is responsible for the decorators used in the command class
class command_tools:
    def __init__(self, table):
        self.table = {}

    def function_command(self, func):
        # Add the function to the table with its name as the key
        self.table[func.__name__] = func
        Log.special(f"Registered {func.__name__} in command table.")
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    

class Command:
    command_table = {}
    tools = command_tools(command_table)
    def __init__(self, model, control):
        self.model = model
        self.control = control
        Log.info("Initialized Command Module")
        self.stems = None

    @tools.function_command
    def list_audio_objects(self):
        objects = self.model.audio.objects
        for index, a in enumerate(objects):
            Log.info(f"[{index}] {a.name}")

    @tools.function_command
    def delete_audio_object(self, index):
        self.model.audio.delete(index)

    @tools.function_command
    def ingest(self, path=None, opath=None):
        # BEGIN INPUT VALIDATION
        if not path:
            # Get user input if path is not specified
            str_path = str(prompt("Please Enter Path: "))
            abs_path = os.path.abspath(str_path)
        if path:
            abs_path = os.path.abspath(path)
        # validity check
        if not path_exists(abs_path): # Check if the path is valid
            Log.error(f"Invalid Path: '{abs_path}'")
            return
        if not file_exists(abs_path):   # Check if the file exists at specified path
            Log.error(f"File does not exist at specified path")
            return
        if not is_valid_audio_format(abs_path): # Check if audio is in a usable format
            Log.error(f"Invalid audio format")
            return
        
        # BEGIN LOAD INTO PROGRAM
        # add the audio into the model
        self.control.load_audio(abs_path)
        # CALL THE ingest_to_stems function here...

        # initialize loading audio into the audio_separator module
        if yes_no_prompt("Generate stems from song?"):
            Log.info(f"Initializing Stems")
        if not opath:
            #get user input if path is not specified
            str_path = str(prompt("Please Enter Output Path"))
            abs_path = os.path.abspath(str_path)
        if opath:
            abs_path = os.path.abspath(opath)
        # Validity check
        if not path_exists(abs_path):
            Log.error(f"Invalid Path: {abs_path}")
        
        self.control.generate_stems(abs_path)

        self.ingest_to_stems(abs_path)


    # ingests audio file and sets output files ready for stem generation
    @tools.function_command
    def ingest_to_stems(self, abs_path, opath=None):
        if yes_no_prompt("Generate stems from song?"):
            Log.info(f"Initializing stems")
        if not opath:
            # get user input for output path
            str_path = str(prompt("Please Enter Output Path: "))
            o_abs_path = os.path.abspath(str_path)
        if opath:
            o_abs_path = os.path.abspath(opath)
        # Validity Check
        if not path_exists(o_abs_path):
            Log.error(f"Invalid Path: {o_abs_path}")
            return
        self.control.generate_stems(abs_path, o_abs_path)

    @tools.function_command
    def digest(self, a=None):
        # apply the pre transformation
        if yes_no_prompt("Apply pre transformation?"):
            Log.info(f"Pre transformation applied to {a.name}")
        # run onset detection
        if yes_no_prompt("run offset transformation?"):
            Log.info("Run offset detection")
        # apply post transformation
        if yes_no_prompt("run post transformation?"):
            Log.info("Apply post transformation")
    


















