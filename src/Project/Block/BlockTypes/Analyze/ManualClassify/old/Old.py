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
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.graph_objs as go
import scipy.io.wavfile as wavfile
from flask import request
from dash_extensions import EventListener
from src.Utils.tools import prompt_selection

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

        self.visualization = "spectrogram_db" # default visualization
        self.visualization_types = ["spectrogram_db", "spectrogram_cqt", "spectrogram_cens", "waveform"]
        
        self.transformation = "none"
        self.transformation_types = ["none","stft", "mfcc", "chroma", "onset_strength"]

        self.figure_height = 400
        self.figure_height_types = [400, 600, 800, 1000]
        self.color_scale = "viridis"
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

        self.command.add("register_layout", self.register_layout)
        self.command.add("get_page_url", self.get_page_url)
        self.command.add("add_visualization", self.set_visualization)
        self.command.add("add_transformation", self.set_transformation)
        self.command.add("set_figure_height", self.set_figure_height)
        self.command.add("set_color_scale", self.set_color_scale)

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

    # Transformation settings
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

    # Visualization settings
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
        data passes without any transormation, transformation is handled in the UI and applied directly to the data
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

        #update NFTT to base its size relative to the length of each audio sample, obviously a large sample should have a larger size nfft/hop length
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
        
    def register_layout(self):
        """
        Uses the parent's Dash app (self.parent._app) to define
        """
        app = self.parent._app

        # 2) Create an EventListener to capture arrow key events in the browser window
        keyboard_listener = EventListener(
            id=f"{self.name}-keyboard-listener",
            events=[
                {
                    "event": "keydown",
                    "props": ["key", "code", "n_events"],
                    "target": "window"
                }
            ]
        )

        layout_content = html.Div([
            html.Div([
    
                html.H1(f"{self.name.capitalize()} EventData Viewer"),
                # 3) Insert the EventListener into your layout (can go anywhere in the DOM)
                keyboard_listener,
                # Optional: A small div to show which arrow was pressed (like a debug print)
                html.Div(id=f"{self.name}-arrow-key-output"),

                html.Div(
                    id=f"{self.name}-current-event-name",
                    style={"fontWeight": "bold", "marginBottom": "10px"}
                ),
                html.Div([
                    html.Button("Previous", id=f"{self.name}-prev-button", n_clicks=0, style={"marginRight": "5px"}),
                    html.Button("Next", id=f"{self.name}-next-button", n_clicks=0, style={"marginRight": "5px"}),
                ]),

                html.Div([
                    # New Dropdown for Visualization Type
                    html.Div([
                        html.Label("Select Visualization:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-visualization-dropdown",
                            options=[{"label": vis, "value": vis} for vis in self.visualization_types],
                            value=self.visualization,
                            placeholder="Select visualization type",
                            style={"width": "100%",},
                            clearable=False,
                        ),
                    ], style={"flex": 1, "marginRight": "10px"}),

                    html.Div([
                        # New Dropdown for Transformation Type
                        html.Label("Select Transformation:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-transformation-dropdown",
                            options=[{"label": trans, "value": trans} for trans in self.transformation_types],
                            value=self.transformation,
                            placeholder="Select transformation type",
                            style={"width": "100%",},
                            clearable=False,
                        ),
                    ], style={"flex": 1, "marginRight": "10px"}),

                    html.Div([
                        # New Dropdown for Figure Height
                        html.Label("Select Figure Height:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-figure-height-dropdown",
                            options=[{"label": height, "value": height} for height in self.figure_height_types],
                            value=self.figure_height,
                            placeholder="Select figure height",
                            style={"width": "100%",},
                            clearable=False,
                        ),
                    ], style={"flex": 1, "marginRight": "10px"}),

                    html.Div([
                        # New Dropdown for Color Scale
                        html.Label("Select Color Scale:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-color-scale-dropdown",
                            options=[{"label": color_scale, "value": color_scale} for color_scale in self.color_scale_types],
                            value=self.color_scale,
                            placeholder="Select color scale",
                            style={"width": "100%",},
                            clearable=False,
                        ),
                    ], style={"flex": 1,}),
                ], style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "flexWrap": "wrap",
                    "marginBottom": "20px"
                }
                ),

                # event selection
                html.Label("Jump to Event:", style={"marginRight": "10px"}),
                dcc.Dropdown(
                    id=f"{self.name}-event-jump-dropdown",
                    options=[],
                    placeholder="Select an event",
                    style={"width": "50%", "marginBottom": "10px"},
                    clearable=True,
                ),

                # event visualization
                dcc.Graph(id=f"{self.name}-spectrogram-plot"),

                # event audio player
                html.Audio(
                    id=f"{self.name}-audio-player",
                    controls=True,
                    style={"marginTop": "20px", "display": "block"},
                ),
                #audio player controls
                html.Button(
                    "Play / Pause",
                    id=f"{self.name}-play-toggle-btn"
                ),

                # classification dropdown
                html.Div([
                    html.Div([
                        html.Label("Classification Suggestions:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-classification-suggestions-dropdown",
                        options=[],
                        value="",
                        placeholder="Select classification",
                        clearable=True,
                        style={"width": "200px",}
                    ),
                    ], style={"flex": 1,}),
                    dcc.Input(
                        id=f"{self.name}-classification-input",
                        type="text",
                        value="",
                        placeholder="Or type custom",
                        style={"width": "200px"}
                    ),
                    html.Button(
                        "Save Classification",
                        id=f"{self.name}-save-class-button",
                        n_clicks=0,
                        style={"marginRight": "5px"}
                        ),

                ], style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "flexWrap": "wrap",
                    "marginBottom": "20px"
                }
                ),

            ], style={"padding": "10px"}),
        ])

        Log.info(f"Registering page: /{self.name}")
        dash.register_page(
            self.name,
            layout=layout_content,
            name=self.name
        )
        app.clientside_callback(
            """
            function(nClicks, audioElementId) {
                if (!audioElementId) {
                    return window.dash_clientside.no_update;
                }
                const audioTag = document.getElementById(audioElementId);
                if (audioTag) {
                    if (audioTag.paused) {
                        audioTag.play();
                    } else {
                        audioTag.pause();
                    }
                }
                return window.dash_clientside.no_update;
            }
            """,
            # Output(f"{self.name}-audio-player", "src"),  # Dummy output to satisfy Dash
            Input(f"{self.name}-play-toggle-btn", "n_clicks"),
            State(f"{self.name}-audio-player", "id"),  # This is where audioElementId comes from
        )
        @app.callback(
        [
            Output(f"{self.name}-spectrogram-plot", "figure"),
            Output(f"{self.name}-current-event-name", "children"),
            Output(f"{self.name}-classification-input", "value"),
            Output(f"{self.name}-classification-suggestions-dropdown", "options"),
            Output(f"{self.name}-classification-suggestions-dropdown", "value"),
            Output(f"{self.name}-audio-player", "src"),
            Output(f"{self.name}-event-jump-dropdown", "options"),  
            Output(f"{self.name}-event-jump-dropdown", "value"),
            Output(f"{self.name}-visualization-dropdown", "value"),
            Output(f"{self.name}-transformation-dropdown", "value"),
            Output(f"{self.name}-figure-height-dropdown", "value"),
            Output(f"{self.name}-color-scale-dropdown", "value"),
            Output(f"{self.name}-arrow-key-output", "children")
        ],
        [
            Input(f"{self.name}-prev-button", "n_clicks"),
            Input(f"{self.name}-next-button", "n_clicks"),
            Input(f"{self.name}-save-class-button", "n_clicks"),
            Input(f"{self.name}-classification-suggestions-dropdown", "value"),
            Input(f"{self.name}-event-jump-dropdown", "value"),
            Input(f"{self.name}-visualization-dropdown", "value"),
            Input(f"{self.name}-transformation-dropdown", "value"),
            Input(f"{self.name}-figure-height-dropdown", "value"),
            Input(f"{self.name}-color-scale-dropdown", "value"),
            Input(f"{self.name}-keyboard-listener", "n_events"),
        ],
        [
            State(f"{self.name}-keyboard-listener", "event"),
            State(f"{self.name}-classification-input", "value")
        ]
    )
        def update_view(
            prev_clicks,
            next_clicks,
            save_clicks,
            selected_suggestion,
            jump_index,
            selected_visualization,
            selected_transformation,
            selected_figure_height,
            selected_color_scale,
            n_keydown,
            keyboad_event,
            typed_class
        ):
            changed_component = [p["prop_id"] for p in callback_context.triggered][0]
            Log.info(f"Callback triggered by: {changed_component}")

            classification_saved = False  # Flag to track if classification was saved

            if f"{self.name}-visualization-dropdown" in changed_component:
                self.set_visualization(visualization=selected_visualization)
                Log.info(f"Set visualization to: {selected_visualization}")

            if f"{self.name}-transformation-dropdown" in changed_component:
                Log.info(f"Setting transformation to: {selected_transformation}")
                self.set_transformation(transformation=selected_transformation)
                Log.info(f"Set transformation to: {selected_transformation}")

            if f"{self.name}-figure-height-dropdown" in changed_component:
                self.set_figure_height(figure_height=selected_figure_height)
                Log.info(f"Set figure height to: {selected_figure_height}")

            if f"{self.name}-color-scale-dropdown" in changed_component:
                self.set_color_scale(color_scale=selected_color_scale)
                Log.info(f"Set color scale to: {selected_color_scale}")

            # Decide if we need to classify
            new_classification = typed_class or selected_suggestion

            # Ensure current classification is in available_classifications
            if self.selected_event and self.selected_event.classification:
                self.available_classifications.add(self.selected_event.classification)

            # If anything triggered classification, set classification:
            if self.get_audio_events() and any(comp in changed_component for comp in [
                f"{self.name}-prev-button",
                f"{self.name}-next-button",
                f"{self.name}-save-class-button",
                f"{self.name}-classification-suggestions-dropdown",
                f"{self.name}-event-jump-dropdown",
                f"{self.name}-visualization-dropdown",
                f"{self.name}-transformation-dropdown",
                f"{self.name}-figure-height-dropdown",
                f"{self.name}-color-scale-dropdown  "
            ]):
                if f"{self.name}-save-class-button" in changed_component and typed_class:
                    self.set_classification(typed_class)
                    classification_saved = True
                    Log.info(f"Set classification to: {typed_class}")
                elif selected_suggestion:
                    self.set_classification(selected_suggestion)
                    Log.info(f"Set classification to: {selected_suggestion}")

            # 6) Handle ArrowKey event
            arrow_pressed = ""
            if keyboad_event:
                pressed_key = keyboad_event.get("key", "")
                if pressed_key == "ArrowLeft":
                    self.previous_event()
                    Log.info("ArrowLeft => previous_event()")
                elif pressed_key == "ArrowRight":
                    self.next_event()
                    Log.info("ArrowRight => next_event()")
                elif pressed_key == "ArrowUp":
                    Log.info("ArrowUp pressed (not assigned to event navigation)")
                elif pressed_key == "ArrowDown":
                    Log.info("ArrowDown pressed (not assigned to event navigation)")
                elif pressed_key == "Enter":
                    Log.info("Enter pressed")
                elif pressed_key == "Space":
                    Log.info("Space pressed")

            # Also handle manual clicking of Prev/Next/Jump
            if self.get_audio_events():
                if f"{self.name}-prev-button" in changed_component:
                    self.previous_event()
                elif f"{self.name}-next-button" in changed_component:
                    self.next_event()
                elif f"{self.name}-event-jump-dropdown" in changed_component:
                    if jump_index is not None:
                        self.jump_to_event(jump_index)
                    else:
                        Log.warning("No event index provided for jump.")
                elif f"{self.name}-save-class-button" in changed_component:
                    if typed_class:
                        self.set_classification(typed_class)
                        classification_saved = True
                    else:
                        Log.warning("No classification provided for save.")

            # Classification suggestions
            sorted_dropdown_options = (
                [{"label": c, "value": c} for c in sorted(self.available_classifications)]
                if self.available_classifications
                else []
            )

            # Current classification
            if self.selected_event and self.selected_event.classification in self.available_classifications:
                dropdown_value = self.selected_event.classification
                text_input_value = "" if classification_saved else typed_class
            elif self.selected_event and self.selected_event.classification:
                # Add current classification to available_classifications if missing
                self.available_classifications.add(self.selected_event.classification)
                dropdown_value = self.selected_event.classification
                text_input_value = "" if classification_saved else typed_class
            else:
                dropdown_value = ""
                text_input_value = typed_class  # Preserve user input

            # Event jump options
            jump_to_event_options = (
                [
                    {"label": f"{event.name}", "value": idx}
                    for idx, event in enumerate(self.get_audio_events())
                ]
                if self.get_audio_events()
                else []
            )

            figure = self.build_figure() if self.get_audio_events() else go.Figure()
            event_label = self.generate_event_label() if self.get_audio_events() else "No Events Available"
            audio_src = self.build_audio_src() if self.get_audio_events() else ""

            try: 

                return (
                    figure,
                    event_label,
                    text_input_value,
                    sorted_dropdown_options,
                    dropdown_value,
                    audio_src,
                    jump_to_event_options,
                    self.current_index if self.get_audio_events() else None,
                    self.visualization,
                    self.transformation,
                    self.figure_height,
                    self.color_scale,
                    arrow_pressed  # optional debug text
                )
            except Exception as e:
                Log.error(f"Error in callback: {e}")
                # Return default or empty values to prevent Dash from breaking
                return (
                    go.Figure(),
                    "Error loading event",
                    typed_class,  # Preserve user input even on error
                    [],
                    "",
                    "",
                    [],
                    None,
                    self.visualization,
                    self.transformation,
                    self.figure_height,
                    self.color_scale,
                    str(e)
                )
            
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

        self.register_layout()