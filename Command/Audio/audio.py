import os
import shutil
import json
from message import Log
from Command.command_module import CommandModule
from tools import prompt_selection


class Audio(CommandModule):
    def __init__(self, model, settings):
        super().__init__(model=model,settings=settings)
        self.name = "Audio"

        self.add_command("list_audio_objects", self.list_audio_objects)
        self.add_command("show_features", self.show_audio_object_features)

    def get_commands(self):
        return self.commands

    def list_audio_objects(self):
        self.model.audio.list()
        Log.command("list_audio_objects")

    def delete_audio_object(self):
        Log.error(f"Delete Audio: Not yet implemented")
        # self.model.audio.delete(index)
        # Log.command(f"delete_audio_object at index {index}")

    def _update_audio_metadata(self, a):
        metadata_file_path = os.path.join(a.directory, f"{a.name}_metadata.json")
        metadata = a.get_audio_metadata()
        with open(metadata_file_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)
        Log.info(f"Audio metadata written to: {metadata_file_path}")
        return metadata
    
    def show_audio_object_features(self):
        audio_selections = []
        a, _  = prompt_selection("Select an audio object to operate on: ", self.model.audio.objects)
        audio_selections.append("Self")
        if a.stems:
            for stem in a.stems:
                audio_selections.append(stem.name)
        sel_obj, selection = prompt_selection("Select audio to analyze", audio_selections)
        if isinstance(selection, int):
            if selection == 0:
                Log.info(f"Selected original audio from audio object {a.name}")
                return a
            elif selection > 0:
                s = a.stems[selection - 1]  # Corrected indexing
                Log.info(f"Selected stem {s.name} from audio object {a.name}")
                return s
            
        elif isinstance(selection, str):
            if selection == "Self":
                Log.info(f"Selected original audio from audio object {a.name}")
                return a
            else:
                for stem in a.stems:
                    if stem.name == selection:
                        Log.info(f"Selected stem {stem.name} from audio object {a.name}")
                        return stem


        for feature in a.features:
            Log.info(f"Feature {feature}")





