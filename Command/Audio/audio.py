import os
import shutil
import json
from message import Log

class Audio:
    def __init__(self, model, settings, project_dir):
        self.model = model
        self.settings = settings
        self.project_dir = project_dir

    def get_commands(self):
        return {
            "list_audio_objects": self.list_audio_objects,
            "delete_audio_object": self.delete_audio_object
        }

    def list_audio_objects(self):
        self.model.audio.list()
        Log.command("list_audio_objects")

    def delete_audio_object(self, index):
        self.model.audio.delete(index)
        Log.command(f"delete_audio_object at index {index}")

    def update_audio_metadata(self, a):
        metadata_file_path = os.path.join(a.directory, f"{a.name}_metadata.json")
        metadata = a.get_audio_metadata()
        with open(metadata_file_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)
        Log.info(f"Audio metadata written to: {metadata_file_path}")
        return metadata