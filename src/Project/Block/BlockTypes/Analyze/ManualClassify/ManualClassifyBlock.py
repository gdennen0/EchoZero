from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
import numpy as np
import librosa
import io
import base64
import plotly.graph_objs as go
import dash
from dash import callback_context
import plotly.graph_objs as go
import scipy.io.wavfile as wavfile
from flask import request
from src.Utils.tools import prompt_selection

# Import the separated layout and callbacks
from src.Project.Block.BlockTypes.Analyze.ManualClassify.UI import layout
from src.Project.Block.BlockTypes.Analyze.ManualClassify.UI import callbacks

DEFAULT_VISUALIZATION = "spectrogram_db"
DEFAULT_TRANSFORMATION = "none"
DEFAULT_FIGURE_HEIGHT = 400
DEFAULT_COLOR_SCALE = "viridis"

class ManualClassifyBlock(Block):
    """
    A user-driven classification block that holds audio events and provides
    commands for manual classification. The UI layer (e.g., Dash web app) should
    call these commands rather than referencing the block’s internal state directly.
    """

    name = "ManualClassify"
    type = "ManualClassify"

    def __init__(self):
        super().__init__()
        self.name = "ManualClassify"
        self.type = "ManualClassify"

        self.layout_registered = False

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.current_index = 0
        self.selected_event = None
        self.available_classifications = set()

        self.visualization = DEFAULT_VISUALIZATION
        self.visualization_types = ["spectrogram_db", "spectrogram_cqt", "spectrogram_cens", "waveform"]
        
        self.transformation = DEFAULT_TRANSFORMATION
        self.transformation_types = ["none","stft", "mfcc", "chroma", "onset_strength"]

        self.figure_height = DEFAULT_FIGURE_HEIGHT
        self.figure_height_types = [400, 600, 800, 1000]
        self.color_scale = DEFAULT_COLOR_SCALE
        self.color_scale_types = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']
        
        self.command.add("previous_event", self.previous_event)
        self.command.add("next_event", self.next_event)
        self.command.add("jump_to_event", self.jump_to_event)
        self.command.add("classify_event", self.set_classification)

        self.command.add("register_ui", self.register_ui)
        self.command.add("register", self.register_ui)
        self.command.add("get_page_url", self.get_page_url)
        self.command.add("add_visualization", self.set_visualization)
        self.command.add("add_transformation", self.set_transformation)
        self.command.add("set_figure_height", self.set_figure_height)
        self.command.add("set_color_scale", self.set_color_scale)

# UI SETTERS
    def set_name(self, name):
        self.name = name
        if not self.layout_registered:
            self.register_ui()
        Log.info(f"Updated Blocks name to: '{name}'")   

    def set_figure_height(self, figure_height=None):
        """
        Set the height of the figure.
        """
        if figure_height is None:
            figure_height = prompt_selection(f"Enter new figure height (current: {self.figure_height}): ", self.figure_height_types)
        if figure_height in self.figure_height_types:
            self.figure_height = figure_height
        else:
            Log.error(f"Invalid figure height: {figure_height}, must be one of {self.figure_height_types}")

    def set_color_scale(self, color_scale=None):
        """
        Set the color scale of the figure.
        """
        if color_scale is None:
            color_scale = prompt_selection(f"Enter new color scale (current: {self.color_scale}): ", self.color_scale_types)
        if color_scale in self.color_scale_types:
            self.color_scale = color_scale
        else:
            Log.error(f"Invalid color scale: {color_scale}, must be one of {self.color_scale_types}")

# DATA OUTPUT TRANSFORMATIONS
    def set_transformation(self, transformation=None):
        """
        Add a transformation type to the block.
        """
        if transformation is None:
            transformation = prompt_selection(f"Enter new transformation type (current: {self.transformation_types}): ", self.transformation_types)
        if transformation in self.transformation_types:
            self.transformation = transformation
        else:
            Log.error(f"Invalid transformation type: {transformation}, must be one of {self.transformation_types}")

    def set_visualization(self, visualization=None):
        """
        Add a visualization type to the block.
        """
        if visualization is None:
            visualization = prompt_selection(f"Enter new visualization type (current: {self.visualization_types}): ", self.visualization_types)
        if visualization in self.visualization_types:
            self.visualization = visualization
        else:
            Log.error(f"Invalid visualization type: {visualization}, must be one of {self.visualization_types}")

    def process(self, input_data):
        """
        Collect incoming data and store it in self.data.
        data passes without any transformation, transformation is handled in the UI and applied directly to the data
        """
        return input_data

    def get_audio_events(self):
        """
        Returns all audio events in the blocks data.
        """
        all_events = []
        for event_data in self.data.get_all():
            for single_event in event_data.items:
                if isinstance(single_event.data, AudioData) and single_event.data.data is not None:
                    all_events.append(single_event)
        return all_events

    def set_selected_event(self, event_index):
        """
        Sets the selected event to the event at the given index.
        """
        audio_events = self.get_audio_events()
        if 0 <= event_index < len(audio_events):
            self.selected_event = audio_events[event_index]
            Log.info(f"Selected event: {self.selected_event.name}")
        else:
            self.selected_event = None

    def previous_event(self):
        """
        Moves to the previous event.
        """
        new_index = max(self.current_index - 1, 0)
        self.current_index = new_index
        Log.info(f"Moved to previous event: Index {self.current_index}")
        self.set_selected_event(self.current_index)

    def next_event(self):
        """
        Moves to the next event.
        """
        audio_events = self.get_audio_events()
        new_index = min(self.current_index + 1, len(audio_events) - 1)
        self.current_index = new_index
        Log.info(f"Moved to next event: Index {self.current_index}")
        self.set_selected_event(self.current_index)

    def jump_to_event(self, event_index):
        """
        Jumps to the event at the given index.
        """
        audio_events = self.get_audio_events()
        if 0 <= event_index < len(audio_events):
            self.current_index = event_index
            Log.info(f"Jumped to event: Index {self.current_index}")
            self.set_selected_event(self.current_index)
        else:
            Log.warning(f"Invalid jump event index: {event_index}")
            self.set_selected_event(self.current_index)

    def set_classification(self, classification_value):
        """
        Sets the classification for the current event.
        """
        audio_events = self.get_audio_events()
        if audio_events:
            audio_events[self.current_index].set_classification(classification_value)
            self.available_classifications.add(classification_value)
            Log.info(f"Event {audio_events[self.current_index].name} classified as: {classification_value}")

    def get_page_url(self): 
        """
        Returns the URL for the current page.
        """
        for page_id, page_data in dash.page_registry.items():
            if page_id == self.name:
                return page_data['path']
        return None

    def build_figure(self):
        """
        Build spectrogram data from the event at self.current_index.
        """
        audio_events = self.get_audio_events()
        if not audio_events:
            return go.Figure()

        audio_event = audio_events[self.current_index]
        audio_data = audio_event.data  #TODO replace with proper get methods here
        audio_samples = audio_data.data
        sample_rate = audio_data.sample_rate

        if audio_samples.ndim > 1:
            audio_samples = np.mean(audio_samples, axis=-1)

        # Update FFT to base its size relative to the length of each audio sample
        nfft = int(len(audio_samples) / 10)
        hop_length = int(nfft / 2)

        if self.transformation == "stft":
            transformed_data = librosa.stft(audio_samples, n_fft=nfft, hop_length=hop_length)
        elif self.transformation == "mfcc":
            transformed_data = librosa.feature.mfcc(y=audio_samples, sr=sample_rate, n_mfcc=128)
        elif self.transformation == "chroma":
            transformed_data = librosa.feature.chroma_cqt(y=audio_samples, sr=sample_rate)
        elif self.transformation == "onset_strength":
            transformed_data = librosa.onset.onset_strength(y=audio_samples, sr=sample_rate)
        elif self.transformation == "none":
            transformed_data = audio_samples
        else:
            Log.error(f"Transformation type {self.transformation} not found in {self.name}, bypassing transformation")

        # Check if transformed_data is 1D
        if transformed_data.ndim == 1 and self.visualization.startswith("spectrogram"):
            Log.error(f"Cannot generate spectrogram with 1D transformed_data for visualization type {self.visualization}.")
            return go.Figure()

        if self.visualization == "spectrogram_db":
            spectrogram = librosa.amplitude_to_db(np.abs(transformed_data), ref=np.max)
            x = np.linspace(0, len(audio_samples) / sample_rate, spectrogram.shape[1])
            y = np.linspace(0, sample_rate / 2, spectrogram.shape[0])
            x_label = "Time (s)"
            y_label = "Frequency (Hz)"
        elif self.visualization == "spectrogram_cqt":
            spectrogram = librosa.feature.chroma_cqt(y=audio_samples, sr=sample_rate)
            x = np.linspace(0, len(audio_samples) / sample_rate, spectrogram.shape[1])
            y = np.linspace(0, sample_rate / 2, spectrogram.shape[0])
            x_label = "Time (s)"
            y_label = "Frequency (Hz)"
        elif self.visualization == "spectrogram_cens":
            spectrogram = librosa.feature.chroma_cens(y=audio_samples, sr=sample_rate)
            x = np.linspace(0, len(audio_samples) / sample_rate, spectrogram.shape[1])
            y = np.linspace(0, sample_rate / 2, spectrogram.shape[0])
            x_label = "Time (s)"
            y_label = "Frequency (Hz)"
        elif self.visualization == "waveform":
            waveform = audio_samples
            x = np.linspace(0, len(audio_samples) / sample_rate, len(waveform))
            y = np.linspace(0, sample_rate / 2, 1)
            x_label = "Time (s)"
            y_label = "Amplitude"
        else:
            Log.error(f"Visualization type {self.visualization} not found in {self.name}, using default spectrogram_db")

        # Create Figure
        if self.visualization.startswith("spectrogram"):
            figure = go.Figure(
                data=[go.Heatmap(x=x, y=y, z=spectrogram, colorscale=self.color_scale)]
            )
        elif self.visualization == "waveform":
            figure = go.Figure(
                data=[go.Scatter(x=x, y=waveform, mode="lines")]
            )
        else:
            figure = go.Figure()

        figure.update_layout(
            title=f"{self.visualization.capitalize()} - {self.transformation.capitalize()}",
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=self.figure_height,
        )
        return figure

    def build_audio_src(self):
        """
        Convert the current audio event to a WAV in memory then return a base64-encoded URI.
        """
        audio_events = self.get_audio_events()
        if not audio_events:
            return ""

        audio_event = audio_events[self.current_index]
        audio_samples = audio_event.data.data
        sample_rate = getattr(audio_event.data, "sample_rate", 44100)

        if audio_samples.ndim > 1:
            audio_samples = np.mean(audio_samples, axis=-1)

        scaled_samples = (audio_samples * 32767).astype(np.int16)

        with io.BytesIO() as buf:
            wavfile.write(buf, sample_rate, scaled_samples)
            encoded_audio = base64.b64encode(buf.getvalue()).decode("ascii")

        return f"data:audio/wav;base64,{encoded_audio}"

    def generate_event_label(self):
        """
        Returns a label for the current event.
        """
        audio_events = self.get_audio_events()

        total_events = len(audio_events)
        current_event_name = audio_events[self.current_index].name
        return f"Event: {current_event_name} ({self.current_index + 1}/{total_events})"
    
    def register_ui(self):
        """
        Uses the parent's Dash app (self.parent._app) to define the layout and register callbacks.
        """
        if not self.layout_registered:
            if not self.parent or not hasattr(self.parent, '_app'):
                Log.error("Parent app not found. Ensure that the block is properly connected to a Dash app.")
                return

            Log.info(f"Registering page: /{self.name}")

            dash.register_page(
                self.name,
                layout = layout.build_layout(self),
                name = self.name,
            )

            # Register callbacks
            callbacks.register_callbacks(self, self.parent._app)

            self.layout_registered = True

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "visualization": self.visualization,
            "transformation": self.transformation,
            "color_scale": self.color_scale,
            "figure_height": self.figure_height,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }  
    
    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_visualization(visualization=block_metadata.get("visualization", DEFAULT_VISUALIZATION))
        self.set_transformation(transformation=block_metadata.get("transformation", DEFAULT_TRANSFORMATION))
        self.set_color_scale(color_scale=block_metadata.get("color_scale", DEFAULT_COLOR_SCALE))
        self.set_figure_height(figure_height=block_metadata.get("figure_height", DEFAULT_FIGURE_HEIGHT))
        
        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push loaded data to output 
        self.output.push_all(self.data.get_all())      

        self.register_ui()