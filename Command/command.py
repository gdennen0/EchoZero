import os
from message import Log
from tools import prompt, yes_no_prompt, check_audio_path
from Model.transformations import stem_generation
import librosa
import torchaudio
import shutil
from Model.audio import audio
from .digest import Digest

TARGET_SAMPLERATE = 44100
FRAMERATE = 30

"""
Responsible for Validating and directing input streams to execute the proper control actions

"""
class Command:
    def __init__(self, model,):
        self.model = model
        Log.info("Initialized Command Module")
        self.commands = {
            "ingest" : self.ingest,
            "digest" : Digest,
            "list_audio_objects" : self.list_audio_objects,
            "delete_audio_object" : self.delete_audio_object,
            "select_audio": self.select_audio,
            "generate_stems": self.generate_stems,
        }
        self.stems = None
        
    def list_audio_objects(self):
        self.model.audio.list()
        Log.command("list_audio_objects")

    def delete_audio_object(self, index):
        self.model.audio.delete(index)
        Log.command(f"delete_audio_object at index {index}")
                    
    def select_audio(self):
        while True:
            try:
                index = int(prompt("Please enter the index for the audio object you'd like to select: "))
                break
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")
        self.model.audio.select(index)

    def ingest(self, path=None, opath=None):
        # BEGIN INPUT VALIDATION
        while True:
            if not path:
                # Get user input if path is not specified
                path = str(prompt("Please Enter Path: "))
            
            if check_audio_path(path):
                abs_path = os.path.abspath(path)
                # add the audio into the model
                self.load_audio(abs_path)
                # CALL THE ingest_to_stems function here..
                break
            else: 
                Log.error(f"Invalid audio path {path}")
                path = None  # Reset path to prompt again

    def load_audio(self, audio_file_path, target_sr=TARGET_SAMPLERATE):
        # Loads all necessary audio data
        audio_data, _ = create_audio_data(audio_file_path, target_sr)   # creates audio data array using librosa
        t, _ = create_audio_tensor(audio_file_path, target_sr)  # creates a tensor object with the audio file
        # Name Selection Loop
        while True:
            name = prompt("Please enter audio object name: ")
            # Check if the name already exists in the audio_model objects
            existing_names = [obj.name for obj in self.model.audio.objects]
            if name in existing_names:
                Log.error(f"Name '{name}' already exists in audio objects, please use a unique name")
            else:
                break
        # Path Selection 
        # Check the base path
        base_path = os.path.join(os.getcwd(), 'Data', 'Audio')  # the dir location for internal audio object storage
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # check the path for the audio object
        # Start of  Selection
        a_path = os.path.join(base_path, name)
        stems_path = os.path.join(a_path, "stems")
        if os.path.exists(a_path):
            if yes_no_prompt(f"Folder '{a_path}' already exists. Do you want to overwrite it?: "):
                shutil.rmtree(a_path)
                os.makedirs(a_path)
                os.makedirs(stems_path)
            else:
                Log.error(f"Folder '{a_path}' already exists and was not overwritten.")
                return
        else:
            os.makedirs(a_path)
            os.makedirs(stems_path)

        # Create the audio object
        a = create_audio_object(audio_file_path, audio_data, t, target_sr, FRAMERATE, None, name, a_path)  # creates an audio object and updates the necessary data
        self.model.audio.add_audio(a)

    def generate_stems(self, a_index=None):
        generate_stems = True
        Log.command(f"Command initiated: 'generate_stems'")
        Log.message(f"Enter 'help' for list of commands")
        if a_index is None:
            self.list_audio_objects()
            a_index = int(prompt("Please enter index for audio object you would like to generate stems for: "))

        tensor = self.model.audio.get_tensor(a_index)
        sr = self.model.audio.get_sr(a_index)
        audio_file_path = self.model.audio.get_audio_file_path(a_index)
        stems_path = self.model.audio.get_stems_file_path(a_index)


        self.stem_init = stem_generation(tensor, sr, audio_file_path, stems_path, "Demucs") # now refs the Model>transformation.py file...
        # self.model.audio.add_stems(stems)
        
        # Print all the first layer of keys in the model_dict dictionary
        model_keys = self.stem_init.model_dict.keys()
        Log.info("Available models:")
        for key in model_keys:
            Log.info(key)

        # Prompts user for model selection
        self.ai_model = str(prompt("Please enter model for selection: "))

        # Prints next layer of selected layer of the model_dict
        model_categories = self.stem_init.model_dict[self.ai_model].keys()

        Log.info("Available Categories:")
        for key in model_categories:
            Log.info(key)

        # Prompts user for model category selection
        self.ai_category = str(prompt("Please enter model category for selection: "))

        model_trainings = self.stem_init.model_dict[self.ai_model][self.ai_category].keys()
        
        Log.info("Available Models: ")
        for key in model_trainings:
            Log.info(key)

        # Prompts user to select specific model training
        self.ai_training = str(prompt("Please enter specific trained model: "))

        #assigns selection to model value
        self.model_value = self.stem_init.model_dict[self.ai_model][self.ai_category][self.ai_training]
        Log.info(f"Loaded Model: {self.ai_model}, Loaded Training: {self.model_value}")
        # Loads Model
        self.stem_init.load_model(self.model_value)
        
        # Generates Stems
        self.stem_init.separate_stems()




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

def create_audio_object(audio_file_path, data, tensor, sr, fps, type, name, path):
    Log.info("create_audio_object")
    # creates an audio object and updates the necessary data
    a = audio()
    a.set_audio_file_path(audio_file_path)
    a.set_original_data(data)
    a.set_tensor(tensor)
    a.set_sample_rate(sr)
    a.set_frame_rate(fps)
    a.set_type(type)
    a.set_name(name)
    a.set_path(path)
    return a
    


















