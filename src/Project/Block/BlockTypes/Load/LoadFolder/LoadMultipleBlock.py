from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Output.Types.audio_output import AudioOutput

from src.Utils.tools import prompt_selection, prompt_file_path, prompt
import librosa
from src.Utils.message import Log
import os
from pathlib import Path
import json
import numpy as np

class LoadMultipleBlock(Block):
    name = "LoadMultiple"
    type = "LoadMultiple"
    
    def __init__(self):
        super().__init__()
        self.name = "LoadMultiple" 
        self.type = "LoadMultiple"

        self.audio_source_dir = str(Path(__file__).resolve().parents[6] / "sources" / "audio")
        self.selected_folder_path = None
        self.audio_file_paths = []
        self.current_file_index = -1  # -1 means no file selected yet
        self.current_file = None

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.command.add("select_folder", self.select_folder)
        self.command.add("execute", self.execute)
        self.command.add("list_audio_files", self.list_audio_files)

    def select_folder(self, selected_folder_path=None):
        """Select a folder containing audio files"""
        if not selected_folder_path:
            selected_folder_path = prompt("Select a folder containing audio files: ")
      
        if not os.path.isdir(selected_folder_path):
            Log.error(f"Selected path is not a directory: {selected_folder_path}")
            return False
            
        self.selected_folder_path = selected_folder_path
        self.audio_file_paths = []
        for file in os.listdir(selected_folder_path):
            if os.path.isfile(os.path.join(selected_folder_path, file)):
                self.audio_file_paths.append(os.path.join(selected_folder_path, file))

        Log.info(f"Selected folder: {selected_folder_path}")
        Log.info(f"Audio files: {self.audio_file_paths}")
        return True
    
    def execute(self):
        """Execute the block"""
        for filepath in self.audio_file_paths:
            # Check if the file is an audio file
            audio_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']
            file_extension = os.path.splitext(filepath)[1].lower()
            
            if file_extension not in audio_extensions:
                Log.info(f"Skipping non-audio file: {filepath}")
                continue
                
            # Get the filename from the path  when passing multiple args= right now you just have to seperate args
            filename = os.path.basename(filepath)
            Log.info(f"Extracted filename: {filename}")
            Log.info(f"Processing audio file: {filepath}")
            self.send_command(f"commandsequencer edit_command command_type=block, parent_name=loadaudio, command_name=select_file, file_path={filepath}")
            self.send_command(f"commandsequencer edit_command command_type=project, command_name=save_as, file_name={filename}, save_directory=project")
            self.send_command(f"commandsequencer execute")
        return True
    
    def list_audio_files(self):
        """List all audio files in the selected folder"""
        if not self.audio_file_paths:
            Log.info("No audio files found")
            return False
        
        Log.info(f"Audio files in {self.selected_folder_path}:")
        for filepath in self.audio_file_paths:
            Log.info(filepath)
        
        return True
    
    def reset_file_index(self):
        """Reset the current file index to -1 (no file selected)"""
        self.current_file_index = -1
        self.current_file = None
        Log.info("File index reset")
        return True
    
    def create_command_sequence(self):
        """Create a command sequence for CommandSequencer to process all files in the folder"""
        if not self.audio_files:
            Log.error("No audio files available. Please select a folder first.")
            return False

    def process(self, input_data):
        return input_data

    def get_metadata(self):
        """Get metadata for the block"""
        return {
            "name": self.name,
            "type": self.type,
            "selected_folder_path": self.selected_folder_path,
            "audio_file_paths": self.audio_file_paths,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        """Save the block data"""
        self.data.save(save_dir)

    def load(self, block_dir):
        """Load the block from saved data"""
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.select_folder(selected_folder_path=block_metadata.get("selected_folder_path"))
        self.current_file_index = block_metadata.get("current_file_index", -1)
        self.current_file = block_metadata.get("current_file")

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())



