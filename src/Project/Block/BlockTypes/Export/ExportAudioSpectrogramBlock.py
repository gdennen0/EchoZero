from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from src.Project.Block.Input.Types.audio_input import AudioInput
import json

DEFAULT_EXPORT_SPECTROGRAM_PATH = os.path.join(os.getcwd(), "tmp")
DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE = "png"
DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE = 44100

class ExportAudioSpectrogramBlock(Block):
    name = "ExportAudioSpectrogram"
    def __init__(self):
        super().__init__()
        # Set attributes
        self.name = "ExportAudioSpectrogram"
        self.type = "ExportAudioSpectrogram"

        # Add attributes
        self.file_type = DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE
        self.audio_settings = {"sample_rate": DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE}
        self.destination_path = DEFAULT_EXPORT_SPECTROGRAM_PATH
        self.supported_file_types = ["png", "jpg", "jpeg"]

        # Add port types and ports
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        # Add commands
        self.command.add("select_file_type", self.select_file_type)
        self.command.add("set_audio_settings", self.set_audio_settings)
        self.command.add("set_destination_path", self.set_destination_path)
        self.command.add("export", self.export)
        self.command.add("reload", self.reload)

        Log.info(f"{self.name} initialized with supported file types: {self.supported_file_types}")

    def select_file_type(self, file_type=None):
        """Command to select the file type for exporting spectrogram."""
        if file_type:
            if file_type in self.supported_file_types:
                self.file_type = file_type
                Log.info(f"Selected file type: {self.file_type}")
            else:
                Log.error(f"Unsupported file type passed as an argument: {file_type}")
        else:
            file_type = prompt_selection("Select the export spectrogram file type:", self.supported_file_types)
            self.file_type = file_type
            Log.info(f"Selected file type: {self.file_type}")

    def set_audio_settings(self, sample_rate=None):
        """Command to set audio file settings for spectrogram."""
        if sample_rate:
            self.audio_settings["sample_rate"] = sample_rate
        else:
            self.audio_settings["sample_rate"] = prompt_selection("Select the export sample rate:", ["44100", "48000", "96000"])

        Log.info(f"Set audio settings: {self.audio_settings}")

    def set_destination_path(self, path=None):
        """Command to set the destination path for the exported spectrogram."""
        if path:
            if not os.path.exists(path):
                try:        
                    os.makedirs(path)
                    Log.info(f"Created destination directory: {path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
                return
        else:
            self.destination_path = prompt("Enter destination path: ")
            if not os.path.exists(self.destination_path):
                try:
                    os.makedirs(self.destination_path)
                    Log.info(f"Created destination directory: {self.destination_path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
        Log.info(f"Set destination path: {self.destination_path}")

    def set_file_type(self, file_type):
        self.file_type = file_type
        Log.info(f"Set file type: {self.file_type}")

    def export(self):
        """Command to export the audio spectrogram based on settings."""
        if not self.data:
            Log.error("No audio data available to export.")
            return
        if not self.file_type:
            Log.error("File type not selected.")
            return
        if not self.destination_path:
            Log.error("Destination path not set.")
            return

        for item in self.data.get_all():
            spectrogram_file_name = f"{item.name}_spectrogram.{self.file_type}"
            spectrogram_file_path = os.path.join(self.destination_path, spectrogram_file_name)
            self.export_spectrogram(item, spectrogram_file_path)
            Log.info(f"Exported spectrogram to {spectrogram_file_path}")


    def export_spectrogram(self, audio_item, path):
        """Export audio data as a spectrogram image."""
        y = audio_item.data
        Log.debug(f"Audio data shape: {np.shape(y)}")  # Add this debug line
        Log.debug(f"Audio data type: {type(y)}")       # Add this debug line
        
        # Ensure y is a 1D numpy array of float32
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        if len(y.shape) > 1:
            y = y.flatten()
        y = y.astype(np.float32)
        
        Log.debug(f"Processed audio shape: {np.shape(y)}")  # Add this debug line
        
        sr = audio_item.sample_rate
        stft_y = librosa.stft(y)
        sr = audio_item.sample_rate
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(stft_y), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def process(self, input_data):
        # No need to process data, just export it
        processed_data = input_data
        return processed_data
    
    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "destination_path": self.destination_path,
            "audio_settings": self.audio_settings,            
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def save(self, save_dir):
        # does not save any data, just metadata
        pass

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_file_type(block_metadata.get("file_type"))
        self.set_destination_path(block_metadata.get("destination_path"))
        self.set_audio_settings(block_metadata.get("audio_settings"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())