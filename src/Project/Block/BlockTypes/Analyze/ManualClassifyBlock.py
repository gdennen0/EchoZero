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

        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.current_index = 0
        self.selected_event = None
        self.available_classifications = set()

        self.command.add("previous_event", self.previous_event)
        self.command.add("next_event", self.next_event)
        self.command.add("jump_to_event", self.jump_to_event)
        self.command.add("classify_event", self.set_classification)

        self.command.add("register_layout", self.register_layout)
        self.command.add("get_page_url", self.get_page_url)

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
        self.set_selected_event(self.current_index)

    def next_event(self):
        """
        Moves to the next event.
        """
        audio_events = self.get_audio_events()
        new_index = min(self.current_index + 1, len(audio_events) - 1)
        self.current_index = new_index
        self.set_selected_event(self.current_index)

    def jump_to_event(self, event_index):
        """
        Jumps to the event at the given index.
        """
        audio_events = self.get_audio_events()
        if 0 <= event_index < len(audio_events):
            self.current_index = event_index
            self.set_selected_event(self.current_index)
        else:
            Log.warning("Invalid jump event index.")
            self.set_selected_event(self.current_index)

    def set_classification(self, classification_value):
        """
        Sets the classification for the current event.
        """
        audio_events = self.get_audio_events()
        if audio_events:
            audio_events[self.current_index].set_classification(classification_value)
            self.available_classifications.add(classification_value)
            Log.info(f"Event {audio_events[self.current_index].name} classified: {classification_value}")

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
        audio_data = audio_event.data
        audio_samples = audio_data.data
        sample_rate = audio_data.sample_rate

        if audio_samples.ndim > 1:
            audio_samples = np.mean(audio_samples, axis=-1)

        stft_transform = librosa.stft(audio_samples, n_fft=1024, hop_length=512)
        spectrogram_db = librosa.amplitude_to_db(np.abs(stft_transform), ref=np.max)

        times = np.linspace(0, len(audio_samples) / sample_rate, spectrogram_db.shape[1])
        freqs = np.linspace(0, sample_rate / 2, spectrogram_db.shape[0])

        figure = go.Figure(
            data=[go.Heatmap(x=times, y=freqs, z=spectrogram_db, colorscale="Viridis")]
        )
        figure.update_layout(
            title="Spectrogram",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            height=400
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
        # Build & store the entire layout

        layout_content = dcc.Tabs([
            dcc.Tab(label="Classification", children=[
                html.Div([
                    html.H2("Manual Audio Classification Web UI"),

                    html.Label("Jump to Event:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{self.name}-event-jump-dropdown",
                        options=[],
                        placeholder="Select an event",
                        style={"width": "50%", "marginBottom": "10px"}
                    ),

                    html.Div(
                        id=f"{self.name}-current-event-name",
                        style={"fontWeight": "bold", "marginBottom": "10px"}
                    ),

                    dcc.Graph(id=f"{self.name}-spectrogram-plot"),

                    html.Audio(
                        id=f"{self.name}-audio-player",
                        controls=True,
                        style={"marginTop": "20px", "display": "block"},
                    ),

                    html.Div([
                        html.Label("Classification Suggestions:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id=f"{self.name}-classification-suggestions-dropdown",
                            options=[],
                            value="",
                            placeholder="Select classification",
                            clearable=True,
                            style={"width": "200px", "display": "inline-block", "marginRight": "10px"}
                        ),
                        dcc.Input(
                            id=f"{self.name}-classification-input",
                            type="text",
                            value="",
                            placeholder="Or type custom",
                            style={"display": "inline-block"}
                        )
                    ], style={"marginBottom": "10px", "marginTop": "20px"}),

                    html.Div([
                        html.Button("Previous", id=f"{self.name}-prev-button", n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("Next", id=f"{self.name}-next-button", n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("Save Classification", id=f"{self.name}-save-class-button", n_clicks=0, style={"marginRight": "5px"}),
                    ])
                ], style={"padding": "10px"})
            ]),
            dcc.Tab(label="Logs", children=[
                dcc.Interval(id=f"{self.name}-log-interval", interval=1000, n_intervals=0),
                html.H2("Server Logs"),
                html.Div(
                    id=f"{self.name}-log-output",
                    style={
                        "whiteSpace": "pre-wrap",
                        "backgroundColor": "#f9f9f9",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                        "maxHeight": "400px",
                        "overflowY": "auto"
                    }
                )
            ])
        ])

        # layout_content = html.Div([
        #     html.H1(f"{self.name}"),
        #     html.P("This is a test page for the Manual Classify Block."),
        # ])
        
        Log.info(f"Registering page: /{self.name}")
        dash.register_page(
            self.name,
            layout=layout_content,
            # path=f"/{self.name}",
            name=self.name
        )

        @app.callback(
            [
                Output(f"{self.name}-spectrogram-plot", "figure"),
                Output(f"{self.name}-current-event-name", "children"),                 # was event-name
                Output(f"{self.name}-classification-input", "value"),
                Output(f"{self.name}-classification-suggestions-dropdown", "options"), # was classification-dropdown
                Output(f"{self.name}-classification-suggestions-dropdown", "value"),   # was classification-dropdown
                Output(f"{self.name}-audio-player", "src"),
                Output(f"{self.name}-event-jump-dropdown", "options"),                 # was jump-dropdown
                Output(f"{self.name}-event-jump-dropdown", "value")                    # was jump-dropdown
            ],
            [
                Input(f"{self.name}-prev-button", "n_clicks"),
                Input(f"{self.name}-next-button", "n_clicks"),
                Input(f"{self.name}-save-class-button", "n_clicks"),                   # was save-button
                Input(f"{self.name}-classification-suggestions-dropdown", "value"),    # was classification-dropdown
                Input(f"{self.name}-event-jump-dropdown", "value"),                    # was jump-dropdown
            ],
            [State(f"{self.name}-classification-input", "value")]
        )
        def update_view(
            prev_clicks,
            next_clicks,
            save_clicks,
            selected_suggestion,
            jump_index,
            typed_class
        ):
            
            changed_component = [p["prop_id"] for p in callback_context.triggered][0]
            new_classification = typed_class or selected_suggestion

            # If anything triggered classification, set classification:
            if self.get_audio_events() and any(comp in changed_component for comp in [
                f"{self.name}-prev-button",
                f"{self.name}-next-button",
                f"{self.name}-save-class-button",
                f"{self.name}-classification-suggestions-dropdown",
                f"{self.name}-event-jump-dropdown"
            ]):
                self.set_classification(new_classification)

            # Navigation
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
                    else:
                        Log.warning("No classification provided for save.")

            # Build figure or empty fallback
            figure = self.build_figure() if self.get_audio_events() else go.Figure()
            event_label = self.generate_event_label() if self.get_audio_events() else "No Events Available"
            audio_src = self.build_audio_src() if self.get_audio_events() else ""

            # Classification suggestions
            sorted_dropdown_options = [
                {"label": c, "value": c} for c in sorted(self.available_classifications)
            ] if self.available_classifications else []

            # Current classification
            if self.selected_event and self.selected_event.classification in self.available_classifications:
                dropdown_value = self.selected_event.classification
                text_input_value = ""
            else:
                dropdown_value = ""
                text_input_value = self.selected_event.classification if (self.selected_event and self.selected_event.classification) else ""

            # Event jump options
            jump_to_event_options = [
                {"label": f"{event.name}", "value": idx}
                for idx, event in enumerate(self.get_audio_events())
            ] if self.get_audio_events() else []

            return (
                figure,
                event_label,
                text_input_value,
                sorted_dropdown_options,
                dropdown_value,
                audio_src,
                jump_to_event_options,
                self.current_index if self.get_audio_events() else None
            )

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
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
        
        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push loaded data to output 
        self.output.push_all(self.data.get_all())      

        self.register_layout()