from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from src.Project.Block.Input.Types.event_input import EventInput
import json

DEFAULT_EXPORT_SPECTROGRAM_PATH = os.path.join(os.getcwd(), "tmp", "event_spectrograms")
DEFAULT_EXPORT_SPECTROGRAM_FILE_TYPE = "png"
DEFAULT_EXPORT_SPECTROGRAM_SAMPLE_RATE = 44100

class ExportEventSpectrogramsBlock(Block):
    name = "ExportEventSpectrogram"
    type = "ExportEventSpectrogram"
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
        self.mode = "individual"
        self.supported_modes = ["individual", "multi"]

        # Add port types and ports
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        # Add commands
        self.command.add("set_mode", self.set_mode)
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

    def set_mode(self, mode=None):
        """Command to set the mode for exporting spectrogram."""
        if mode:
            if mode in self.supported_modes:
                self.mode = mode
                Log.info(f"Selected mode: {self.mode}")
            else:
                Log.error(f"Unsupported mode passed as an argument: {mode}")
        else:
            self.mode = prompt_selection("Select the export mode:", self.supported_modes)

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

        if self.mode == "individual":
            for event_data in self.data.get_all():
                for event in event_data.items:
                    spectrogram_file_name = f"{event.name}_spectrogram.{self.file_type}"
                    spectrogram_file_path = os.path.join(self.destination_path, spectrogram_file_name)
                    self.export_individual_spectrogram(event, spectrogram_file_path)
                    Log.info(f"Exported spectrogram to {spectrogram_file_path}")
        
        elif self.mode == "multi":
            spectrogram_file_name = f"all_spectrogram.{self.file_type}"
            spectrogram_file_path = os.path.join(self.destination_path, spectrogram_file_name)
            self.export_all_spectrogram(self.data, spectrogram_file_path)
            Log.info(f"Exported spectrogram to {spectrogram_file_path}")

    def export_all_spectrogram(self, data, path):
        """
        Export a multi-plot spectrogram image where each event in 'data' gets its
        own individual spectrogram in a grid.
        """
        all_event_data = data.get_all() if data else []
        if not all_event_data:
            Log.error("No event data available to export spectrograms.")
            return

        # Prepare a list of (event, waveform, sample_rate)
        events_to_plot = []
        for event_data in all_event_data:
            for event in event_data.items:
                wave = event.data.data
                sr = event.data.sample_rate if event.data else None

                if wave is not None and len(wave) > 0 and sr is not None:
                    # Ensure wave is a 1D float32 numpy array
                    if isinstance(wave, (list, tuple)):
                        wave = np.array(wave)
                    if len(wave.shape) > 1:
                        wave = wave.flatten()
                    wave = wave.astype(np.float32)
                    events_to_plot.append((event, wave, sr))
                else:
                    Log.debug(f"Skipping an event with no valid waveform or sample rate: {event.name}")

        if not events_to_plot:
            Log.error("No valid event waveforms found to plot spectrograms.")
            return

        import math
        count = len(events_to_plot)
        columns = 3
        rows = math.ceil(count / columns)
        
        # Limit the maximum figure size (in inches) to avoid ValueError with large images
        # Both width and height will not exceed 100 inches, which is typically safe.
        max_inches = 100
        fig_width_in = min(columns * 4, max_inches)
        fig_height_in = min(rows * 4, max_inches)

        # Create figure and axes
        fig, axes = plt.subplots(rows, columns, figsize=(fig_width_in, fig_height_in))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Create each spectrogram in its own subplot
        for idx, (event, wave, sr) in enumerate(events_to_plot):
            Log.debug(f"Creating spectrogram for event: {event.name}")
            stft_data = librosa.stft(wave)
            D = librosa.amplitude_to_db(np.abs(stft_data), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[idx])
            axes[idx].set_title(f"Spectrogram: {event.name}")
            axes[idx].set_xlabel("Time")
            axes[idx].set_ylabel("Frequency")

        # Remove any unused subplots
        for ax in axes[len(events_to_plot):]:
            ax.axis('off')

        # Save and close
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def export_individual_spectrogram(self, event, path):
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
    
    def save(self, save_dir):
        # does not save any data, just metadata
        pass   

    def get_metadata(self):
        metadata = {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "destination_path": self.destination_path,
            "audio_settings": self.audio_settings, 
            "mode": self.mode,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
        return metadata

    # def load(self, metadata, block_dir):
    #     self.file_type = data.get("file_type")
    #     self.destination_path = data.get("destination_path")
    #     self.audio_settings = data.get("audio_settings")
    #     self.mode = data.get("mode")
    #     self.input.load(data.get("input")) # just need to reconnect the inputs

    #     self.reload()



    def load(self, block_dir):
        # get block metadata
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_file_type(block_metadata.get("file_type"))
        self.set_destination_path(block_metadata.get("destination_path"))
        self.set_audio_settings(block_metadata.get("audio_settings"))
        self.set_mode(block_metadata.get("mode"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())