from Block.block import Block
from Connections.port_types.audio_port import AudioPort
from message import Log
from tools import prompt_selection, prompt_selection_with_type, prompt
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

DEFAULT_EXPORT_SPECTROGRAM_PATH = os.path.join(os.getcwd(), "tmp")
DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE = "png"
DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE = 44100

class ExportAudioSpectrogramBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "ExportAudioSpectrogram"
        self.type = "ExportAudioSpectrogram"

        # Initialize settings
        self.file_type = DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE
        self.audio_settings = {
            "sample_rate": DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE
        }
        self.destination_path = DEFAULT_EXPORT_SPECTROGRAM_PATH

        # Add supported file types
        self.supported_file_types = ["png", "jpg", "jpeg"]

        # Add commands
        self.add_command("select_file_type", self.select_file_type)
        self.add_command("set_audio_settings", self.set_audio_settings)
        self.add_command("set_destination_path", self.set_destination_path)
        self.add_command("export", self.export)
        self.add_command("reload", self.reload)

        # Add port types and ports
        self.add_port_type(AudioPort)
        self.add_input_port("AudioPort")
        # self.add_output_port("AudioPort")

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
            file_type, _ = prompt_selection("Select the export spectrogram file type:", self.supported_file_types)
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

        try:
            for item in self.data:
                spectrogram_file_name = f"{item.name}_spectrogram.{self.file_type}"
                spectrogram_file_path = os.path.join(self.destination_path, spectrogram_file_name)
                self.export_spectrogram(item, spectrogram_file_path)
                Log.info(f"Exported spectrogram to {spectrogram_file_path}")
        except Exception as e:
            Log.error(f"Failed to export spectrogram: {e}")

    def export_spectrogram(self, audio_item, path):
        """Export audio data as a spectrogram image."""
        try:
            y = audio_item.data
            sr = self.audio_settings.get("sample_rate", DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE)
            plt.figure(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
        except Exception as e:
            Log.error(f"Error exporting spectrogram: {e}")

    def reload(self):
        """Reload the block's data."""
        super().reload()
        Log.info(f"{self.name} reloaded successfully.")