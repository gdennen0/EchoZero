import os
from message import Log
from tools import prompt, yes_no_prompt, check_audio_path, prompt_selection, create_audio_tensor, create_audio_data
from Model.transformations import stem_generation
import shutil
from Model.audio import audio
from .digest import Digest
import json




"""
Responsible for Validating and directing input streams to execute the proper control actions

"""
class Command:
    def __init__(self, project, settings):
        self.model = project.model
        self.settings = settings
        self.project_dir = project.dir
        self.project = project
        Log.info("Initialized Command Module")
        self.digest = Digest(self.model)
        self.commands = {
            "ingest" : self.ingest,
            "digest" : self.digest,
            "list_audio_objects" : self.list_audio_objects,
            "delete_audio_object" : self.delete_audio_object,
            "select_audio": self.select_audio,
            "generate_stems": self.generate_stems,
            "save":self.project.save,
            "save_as": self.project.save_as,
            "dataview": self.dataview,
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

    def dataview(self):
        divider = Log.info("-"*100)
        Log.info("Starting to log project data...")
        
        # Deserialize the main model
        model_data = self.model.serialize()
        
        # Log the main model data
        Log.info("Main Model Data:")
        divider
        # Log each audio object and its sub-objects
        for index, audio_obj in enumerate(self.model.audio.objects):
            # Log.info(f"Audio Object {index}:")
            audio_data = audio_obj.serialize()
            self.log_data(audio_data)
            divider

            # Log stems if they exist
            if audio_obj.stems:
                for stem_index, stem in enumerate(audio_obj.stems):
                    # Log.info(f"  Stem {stem_index}:")
                    stem_data = stem.serialize()
                    self.log_data(stem_data, indent=2)
                    divider

                    # Log event pools if they exist
                    if stem.event_pools:
                        for pool_index, pool in enumerate(stem.event_pools):
                            # Log.info(f"    Event Pool {pool_index}:")
                            pool_data = pool.serialize()
                            self.log_data(pool_data, indent=4)
                            divider

                            # Log events if they exist
                            if pool.objects:
                                for event_index, event in enumerate(pool.objects):
                                    # Log.info(f"      Event {event_index}:")
                                    if event:
                                        event_data = event.serialize()
                                        self.log_data(event_data, indent=6)
                                        divider
        
        Log.info("Finished logging project data.")

    def log_data(self, data, indent=0, key=None):
        if isinstance(data, dict):
            for key, value in data.items():
                self.log_data(value, indent + 1, key=key)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                self.log_data(item, indent + 1, key=index)
        else:
            Log.info(f"{'  ' * indent}{key}: {data}")
            pass
    
    Log.info("Finished logging project data.")

    def ingest(self, path=None, opath=None):
        while True:
            if not path:
                # Get user input if path is not specified
                path = str(prompt("Please Enter Path: "))
            
            if check_audio_path(path):
                abs_path = os.path.abspath(path)
                # add the audio into the model
                self.load_audio(abs_path)
                break
            else: 
                Log.error(f"Invalid audio path {path}")
                path = None  # Reset path to prompt again

    def load_audio(self, audio_file_path, target_sr=None, framerate=None,):
        if target_sr is None:
            target_sr = self.settings['AUDIO_TARGET_SAMPLERATE']
        if framerate is None:
            framerate = self.settings['AUDIO_FRAMERATE']

        audio_data, _ = create_audio_data(audio_file_path, target_sr)
        t, _ = create_audio_tensor(audio_file_path, target_sr)

        while True:    # Name Selection Loop
            name = prompt("Please enter audio object name: ")
            # Check if the name already exists in the audio_model objects
            existing_names = [obj.name for obj in self.model.audio.objects]
            if name in existing_names:
                Log.error(f"Name '{name}' already exists in audio objects, please use a unique name")
            else:
                break
        
        # check the path for the audio object
        a_path = os.path.join(self.project.dir, "audio", name)   # join the name with base path
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

        extension = os.path.splitext(audio_file_path)[1]
        shutil.copy(audio_file_path, os.path.join(a_path, name + extension))
        Log.info(f"Copied original audio file to: {a_path}")

        # Create the audio object
        a = self.model.audio.generate_audio_object(self.project_dir, extension, audio_data, t, target_sr, framerate, None, name)  # creates an audio object and updates the necessary data
        self.model.audio.add(a)
        self.update_audio_metadata(a)


    def update_audio_metadata(self, a):
        metadata_file_path = os.path.join(a.directory, f"{a.name}_metadata.json")
        metadata = a.get_audio_metadata()
        with open(metadata_file_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)
        Log.info(f"Audio metadata written to: {metadata_file_path}")
        return metadata

    def generate_stems(self, a_index=None):
        Log.command(f"Command initiated: 'generate_stems'")
        if a_index is None:
            self.list_audio_objects()
            a_index = int(prompt("Please enter index for audio object you would like to generate stems for: "))

        a = self.model.audio.get_object(a_index)
        audio_file_path = self.model.audio.get_audio_file_path(a_index)
        stems_dir = self.model.audio.get_stems_file_path(a_index)

        splitter = stem_generation(a.tensor, a.sample_rate, audio_file_path, stems_dir, "Demucs") # now refs the Model>transformation.py file...
        
        # Print and prompt for model selection
        model_keys = splitter.models
        ai_model, _ = prompt_selection("Available models:", model_keys)

        # Print and prompt for model category selection
        model_categories = splitter.models[ai_model]
        ai_category, _ = prompt_selection("Available Categories:", model_categories)

        # Print and prompt for specific model training selection
        model_trainings = splitter.models[ai_model][ai_category]
        ai_training, _  = prompt_selection("Available Models:", model_trainings)

        # Assigns selection to model value
        model_value = splitter.models[ai_model][ai_category][ai_training]

        # Loads Model into stem generator
        splitter.load_model(model_value)
        
        Log.info(f"Loaded Model: {ai_model}, Loaded Training: {model_value}")

        # Generates Stems
        stem_metadata = {
            "Name":a.name,
            "Model": ai_model,
            "Category": ai_category,
            "Training": ai_training,
        }

        stem_filenames = splitter.separate_stems() # generates the files at initialized paths and returns the
        Log.info(f"Generated stem filenames: '{stem_filenames}'")

        for filename in stem_filenames:
            path = f"{stems_dir}/{filename}"
            Log.info(f"Adding stem with path {filename}")
            a.add_stem(path)

        self.update_audio_metadata(a) # refreshes audio metadata


    def generate_stem_metadata(self, metadata, audio_file_path):
        
        metadata_file_path = os.path.join(audio_file_path, "_metadata.json")
        with open(metadata_file_path, 'w') as f:
            json.dump(metadata, f)
        Log.info(f"Metadata saved to {metadata_file_path}")
        pass


    


















