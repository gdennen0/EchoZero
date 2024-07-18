import os
import shutil
import json
from message import Log
from Command.command_item import CommandItem


class Audio:
    def __init__(self, model, settings, project_dir):
        self.model = model
        self.settings = settings
        self.project_dir = project_dir
        self.commands = []
        self.name = "Audio"
        self.sub_modules = []

        self.add_command("list_audio_objects", self.list_audio_objects)

    def add_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def add_sub_module(self, sub_module):
        self.sub_modules.append(sub_module)

    def get_commands(self):
        return self.commands

    def list_audio_objects(self):
        self.model.audio.list()
        Log.command("list_audio_objects")

    def delete_audio_object(self):
        Log.info(f"Not yet implemented")
        # self.model.audio.delete(index)
        # Log.command(f"delete_audio_object at index {index}")

    def update_audio_metadata(self, a):
        metadata_file_path = os.path.join(a.directory, f"{a.name}_metadata.json")
        metadata = a.get_audio_metadata()
        with open(metadata_file_path, 'w') as meta_file:
            json.dump(metadata, meta_file, indent=4)
        Log.info(f"Audio metadata written to: {metadata_file_path}")
        return metadata


