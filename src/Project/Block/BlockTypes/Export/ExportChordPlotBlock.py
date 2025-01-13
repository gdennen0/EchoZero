import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Input.Types.event_input import EventInput  # Assume EventInput exists
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Data.Types.event_data import EventData
import json

DEFAULT_EXPORT_PATH = os.path.join(os.getcwd(), "tmp")
DEFAULT_EXPORT_FILE_TYPE = "png"
DEFAULT_EXPORT_SAMPLE_RATE = 44100

class ExportChordPlotBlock(Block):
    """
    A block to export a combined plot of AudioData and EventData.
    The audio data is plotted as the base spectrogram, with event time sections overlayed.
    """
    name = "ExportChordPlot"
    type = "ExportChordPlot"
    
    def __init__(self):
        super().__init__()
        # Set attributes
        self.name = "ExportChordPlot"
        self.type = "ExportChordPlot"

        # Add attributes
        self.file_type = DEFAULT_EXPORT_FILE_TYPE
        self.sample_rate = DEFAULT_EXPORT_SAMPLE_RATE
        self.destination_path = DEFAULT_EXPORT_PATH
        self.supported_file_types = ["png", "jpg", "jpeg"]
        self.audio_data = None
        self.event_data = None

        # Add port types and ports
        self.input.add_type(AudioInput)
        self.input.add_type(EventInput)
        self.input.add("AudioInput")
        self.input.add("EventInput")

        # Add commands
        self.command.add("select_file_type", self.select_file_type)
        self.command.add("set_sample_rate", self.set_sample_rate)
        self.command.add("set_destination_path", self.set_destination_path)
        self.command.add("export", self.export)
        self.command.add("reload", self.reload)

        Log.info(f"{self.name} initialized with supported file types: {self.supported_file_types}")

    def select_file_type(self, file_type=None):
        """Command to select the file type for exporting the plot."""
        if file_type:
            if file_type in self.supported_file_types:
                self.file_type = file_type
                Log.info(f"Selected file type: {self.file_type}")
            else:
                Log.error(f"Unsupported file type passed as an argument: {file_type}")
        else:
            file_type = prompt_selection("Select the export file type:", self.supported_file_types)
            self.file_type = file_type
            Log.info(f"Selected file type: {self.file_type}")

    def set_sample_rate(self, sample_rate=None):
        """Command to set the sample rate for processing audio."""
        if sample_rate:
            self.sample_rate = int(sample_rate)
        else:
            self.sample_rate = int(prompt_selection("Select the sample rate:", ["44100", "48000", "96000"]))
        Log.info(f"Set sample rate: {self.sample_rate}")

    def set_destination_path(self, path=None):
        """Command to set the destination path for the exported plot."""
        if path:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                    Log.info(f"Created destination directory: {path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
                    return
            self.destination_path = path
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
        """Command to export the combined audio and event plot based on settings."""
        if not self.audio_data:
            Log.error("No audio data available to export.")
            return
        if not self.event_data:
            Log.error("No event data available to overlay.")
            return
        if not self.file_type:
            Log.error("File type not selected.")
            return
        if not self.destination_path:
            Log.error("Destination path not set.")
            return

        plot_file_name = f"{self.audio_data.name}_with_events.{self.file_type}"
        plot_file_path = os.path.join(self.destination_path, plot_file_name)
        self.export_combined_plot(self.audio_data, self.event_data, plot_file_path)
        Log.info(f"Exported combined plot to {plot_file_path}")


    def export_combined_plot(self, audio_item, events, path):
        """Export a combined plot of audio waveform and event overlays with colored highlights."""
        y = audio_item.data
        sr = audio_item.sample_rate

        Log.debug(f"Audio data shape: {np.shape(y)}")
        Log.debug(f"Audio data type: {type(y)}")

        # Ensure y is a 1D numpy array of float32
        if isinstance(y, (list, tuple)):
            y = np.array(y)
        if len(y.shape) > 1:
            y = y.flatten()
        y = y.astype(np.float32)

        Log.debug(f"Processed audio shape: {np.shape(y)}")

        # Create a color map for chord classifications
        chord_types = sorted(set(event.classification for event in events.get_all()))
        color_palette = plt.get_cmap('tab20')
        colors = {chord: color_palette(i % 20) for i, chord in enumerate(chord_types)}

        plt.figure(figsize=(14, 6))
        ax = plt.gca()

        # Plot waveform
        librosa.display.waveshow(y, sr=sr, alpha=0.6, color='gray')
        plt.title(f"Waveform with Chords for {audio_item.name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Overlay chord highlights
        for event_item in events.get_all():
            time_range = event_item.time.split('-')
            start_time = float(time_range[0])
            end_time = float(time_range[1])
            chord_type = event_item.classification
            color = colors.get(chord_type, 'blue')  # Default to blue if not found
            ax.axvspan(start_time, end_time, color=color, alpha=0.3)
            # Optional: Annotate chord type at the start of the span
            #  ß ax.text(start_time, 0.95, chord_type, color='black', fontsize=8, ha='left', va='top', transform=ax.get_xaxis_transform())

        # Create custom legends
        handles = [plt.Line2D([0], [0], color=color, lw=4, alpha=0.3) for color in colors.values()]
        labels = list(colors.keys())
        plt.legend(handles, labels, title="Chords", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def process(self, input_data):
        """
        Ingest audio and event data.

        Args:
            input_data (dict): Dictionary containing 'AudioInput' and 'EventInput' data.

        Returns:
            dict: Processed data (unchanged in this case).
        """
        for item in input_data:
            if item.type == "AudioData":
                self.audio_data = item
            elif item.type == "EventData":
                self.event_data = item

        return input_data

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "sample_rate": self.sample_rate,
            "destination_path": self.destination_path,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)
        pass

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # Load attributes
        self.set_name(block_metadata.get("name", "ExportAudioWithEvents"))
        self.set_type(block_metadata.get("type", "ExportAudioWithEvents"))
        self.select_file_type(block_metadata.get("file_type", DEFAULT_EXPORT_FILE_TYPE))
        self.set_sample_rate(block_metadata.get("sample_rate", DEFAULT_EXPORT_SAMPLE_RATE))
        self.set_destination_path(block_metadata.get("destination_path", DEFAULT_EXPORT_PATH))

        # Load inputs
        self.input.load(block_metadata.get("input", {}))
        self.output.load(block_metadata.get("output", {}))
        self.data.load(block_metadata.get("metadata", {}), block_dir)

        self.output.push_all(self.data.get_all())