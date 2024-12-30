from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from src.Project.Block.Input.Types.event_input import EventInput

DEFAULT_EXPORT_SPECTROGRAM_PATH = os.path.join(os.getcwd(), "tmp", "event_spectrograms")
DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE = "png"
DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE = 44100

class ExportEventSpectrogramsBlock(Block):
    name = "ExportEventSpectrogram"
    def __init__(self):
        super().__init__()
        # Set attributes
        self.name = "ExportEventSpectrogram"
        self.type = "ExportEventSpectrogram"

        # Add attributes
        self.file_type = DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE
        self.audio_settings = {"sample_rate": DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE}
        self.destination_path = DEFAULT_EXPORT_SPECTROGRAM_PATH
        self.supported_file_types = ["png", "jpg", "jpeg"]

        # Add port types and ports
        self.input.add_type(EventInput)
        self.input.add("EventInput")

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

        for event_data in self.data.get_all():
            for event in event_data.items:
                spectrogram_file_name = f"{event.name}_spectrogram.{self.file_type}"
                spectrogram_file_path = os.path.join(self.destination_path, spectrogram_file_name)
                self.export_spectrogram(event, spectrogram_file_path)
                Log.info(f"Exported spectrogram to {spectrogram_file_path}")


    def export_spectrogram(self, event, path):
        """Export audio data as a spectrogram image."""
        y = event.data.data
        Log.debug(f"Audio data shape: {np.shape(y)}")  # Add this debug line
        Log.debug(f"Audio data type: {type(y)}")       # Add this debug line
        
        # Ensure y is a 1D numpy array of float32
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        if len(y.shape) > 1:
            y = y.flatten()
        y = y.astype(np.float32)
        
        Log.debug(f"Processed audio shape: {np.shape(y)}")  # Add this debug line
        
        sr = event.data.sample_rate
        stft_y = librosa.stft(y)
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
    
    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "destination_path": self.destination_path,
            "audio_settings": self.audio_settings,            
            "data": self.data.save(),
            "input": self.input.save(),
            "output": self.output.save()
        }

    def load(self, data):
        self.file_type = data.get("file_type")
        self.destination_path = data.get("destination_path")
        self.audio_settings = data.get("audio_settings")
        self.input.load(data.get("input")) # just need to reconnect the inputs

        self.reload()