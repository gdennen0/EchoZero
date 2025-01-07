import logging

from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
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
import threading
import scipy.io.wavfile as wavfile
import socket
from flask import request
import requests

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
        self.host="127.0.0.1"
        self.port=8050


        # COMMANDS: these provide all the necessary functionality
        self.command.add("previous_event", self.previous_event)
        self.command.add("next_event", self.next_event)
        self.command.add("jump_to_event", self.jump_to_event)
        self.command.add("classify_event", self.set_classification)
        self.command.add("start_ui", self.start_ui)
        self.command.add("get_url", self.get_url)
        self.command.add("stop_ui", self.stop_ui)
        self.command.add("reload_ui", self.reload_ui)

        self.start_ui()

    def process(self, input_data):
        """
        Collect incoming data and store it in self.data.
        """
        return input_data
    
    def get_url(self):
        Log.info(f"URL: http://{self.host}:{self.port}")

    def get_audio_events(self):
        all_events = []
        for event_data in self.data.get_all():
            for single_event in event_data.items:
                if isinstance(single_event.data, AudioData) and single_event.data.data is not None:
                    all_events.append(single_event)
        return all_events

    # COMMAND HANDLERS
    # if data is changed then the indexes will need to be updated

    def set_selected_event(self, event_index):
        """
        Updates the selected event based on self.current_index and
        the events available in self.data.
        """
        audio_events = self.get_audio_events()
        if self.current_index < 0 or self.current_index >= len(audio_events):
            self.selected_event = None
            return

        self.selected_event = audio_events[self.current_index]
        Log.info(f"Selected event: {self.selected_event.name}")

    def previous_event(self):
        """
        Move to the previous event in the list.
        """
        old_index = self.current_index
        self.current_index = max(self.current_index - 1, 0)
        Log.info(f"Moved from event {old_index + 1} to {self.current_index + 1} (previous).")
        
        self.set_selected_event(self.current_index)


    def next_event(self):
        """
        Move to the next event in the list.
        """
        audio_events = self.get_audio_events()
        old_index = self.current_index
        self.current_index = min(self.current_index + 1, len(audio_events) - 1)
        Log.info(f"Moved from event {old_index + 1} to {self.current_index + 1} (next).")
        self.set_selected_event(self.current_index)

    def jump_to_event(self, event_index):
        """
        Jump to the specified event index.
        """
        audio_events = self.get_audio_events()
        if event_index is not None and 0 <= event_index < len(audio_events):
            old_index = self.current_index
            self.current_index = event_index
            Log.info(f"Jumped from event {old_index + 1} to {event_index + 1}.")
            self.set_selected_event(self.current_index)
        else:
            Log.warning(f"Invalid jump event index specified.")
            self.set_selected_event(self.current_index)

    def set_classification(self, classification_value):
        """
        Save the classification to the currently selected event
        """
        audio_events = self.get_audio_events()
        audio_events[self.current_index].set_classification(classification_value)
        self.available_classifications.add(classification_value)
        Log.info(f"Set classification for event {audio_events[self.current_index].name} to {classification_value}.")

    # INTERNAL / UTILITY METHODS

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


    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }  
    
    # TODO make this a utility function
    def find_available_port(self, host, start_port=8050, max_port=8100):
        import time
        time.sleep(.2) # wait for any other processes to finish
        port = start_port
        while port <= max_port:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex((host, port))
                if result != 0:
                    return port
                Log.warning(f"Port {port} is in use. Trying port {port + 1}...")
                port += 1
        raise RuntimeError("No available ports found.")
    
    def start_ui(self):
        """
        Spin up the Dash UI. 
        """
        
        self.port = self.find_available_port(self.host)

        if not self.get_audio_events():
            Log.warning("No audio events found. Starting UI with blank placeholders.")
            self.selected_event = None

        if not self.selected_event and self.get_audio_events():
            self.selected_event = self.get_audio_events()[0]

        self._app = dash.Dash(__name__)
        self._app.title = self.name

        # Define Shutdown Route
        @self._app.server.route('/shutdown', methods=['POST'])
        def shutdown():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                Log.error("Not running with the Werkzeug Server")
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            Log.info("Shutdown function called.")
            return 'Server shutting down...'


        # UI Layout (always shows classification / logs):
        self._app.layout = dcc.Tabs([
            dcc.Tab(label="Classification", children=[
                html.Div([
                    html.H2("Manual Audio Classification Web UI"),

                    html.Label("Jump to Event:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id="event-jump-dropdown",
                        options=[],  # updated dynamically
                        placeholder="Select an event",
                        style={"width": "50%", "marginBottom": "10px"}
                    ),

                    html.Div(
                        id="current-event-name",
                        style={"fontWeight": "bold", "marginBottom": "10px"}
                    ),

                    dcc.Graph(id="spectrogram-plot"),

                    html.Audio(
                        id="audio-player",
                        controls=True,
                        style={"marginTop": "20px", "display": "block"},
                    ),

                    html.Div([
                        html.Label("Classification Suggestions:", style={"marginRight": "10px"}),
                        dcc.Dropdown(
                            id="classification-suggestions-dropdown",
                            options=[],
                            value="",
                            placeholder="Select classification",
                            clearable=True,
                            style={"width": "200px", "display": "inline-block", "marginRight": "10px"}
                        ),
                        dcc.Input(
                            id="classification-input",
                            type="text",
                            value="",
                            placeholder="Or type custom",
                            style={"display": "inline-block"}
                        )
                    ], style={"marginBottom": "10px", "marginTop": "20px"}),

                    html.Div([
                        html.Button("Previous", id="prev-button", n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("Next", id="next-button", n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("Save Classification", id="save-class-button", n_clicks=0, style={"marginRight": "5px"}),
                    ])
                ], style={"padding": "10px"})
            ]),
            dcc.Tab(label="Logs", children=[
                dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
                html.H2("Server Logs"),
                html.Div(
                    id="log-output",
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

        @self._app.callback(
            [
                Output("spectrogram-plot", "figure"),
                Output("current-event-name", "children"),
                Output("classification-input", "value"),
                Output("classification-suggestions-dropdown", "options"),
                Output("classification-suggestions-dropdown", "value"),
                Output("audio-player", "src"),
                Output("event-jump-dropdown", "options"),
                Output("event-jump-dropdown", "value")
            ],
            [
                Input("prev-button", "n_clicks"),
                Input("next-button", "n_clicks"),
                Input("save-class-button", "n_clicks"),
                Input("classification-suggestions-dropdown", "value"),
                Input("event-jump-dropdown", "value"),
            ],
            [State("classification-input", "value")]
        )
        def update_classification_view(
            prev_clicks,
            next_clicks,
            save_clicks,
            selected_suggestion,
            jump_index,
            typed_class
        ):
            changed_component = [p["prop_id"] for p in callback_context.triggered][0]

            # Priority: if user typed something, use that. Otherwise use the selected suggestion
            new_classification = typed_class or selected_suggestion


            # If something triggered a classification change, save it:
            if self.get_audio_events():
                if any(comp in changed_component for comp in [
                    "prev-button", 
                    "next-button", 
                    "save-class-button",
                    "classification-suggestions-dropdown",
                    "event-jump-dropdown"
                ]):
                    self.set_classification(new_classification)

            # Navigation (only if we have audio events)
            if self.get_audio_events():
                if "prev-button" in changed_component:
                    self.previous_event()
                elif "next-button" in changed_component:
                    self.next_event()
                elif "event-jump-dropdown" in changed_component:
                    if jump_index is not None:
                        self.jump_to_event(jump_index)
                    else:
                        Log.warning("No event index provided for jump.")
                elif "save-class-button" in changed_component:
                    if typed_class:
                        self.set_classification(typed_class)
                    else:
                        Log.warning("No classification provided for save.") 

            # Fallback (empty / safe defaults if no events exist)
            figure = self.build_figure() if self.get_audio_events() else go.Figure()
            event_label = self.generate_event_label() if self.get_audio_events() else "No Events Available"
            audio_src = self.build_audio_src() if self.get_audio_events() else ""

            # For the suggestions dropdown
            sorted_dropdown_options = [
                {"label": c, "value": c} for c in sorted(self.available_classifications)
            ] if self.available_classifications else []

            # Hold the current classification in either dropdown or text, if we have selected_event
            if self.selected_event and self.selected_event.classification in self.available_classifications:
                dropdown_value = self.selected_event.classification
                text_input_value = ""              
            else:
                if self.selected_event and self.selected_event.classification:
                    dropdown_value = ""
                    text_input_value = self.selected_event.classification
                else:
                    dropdown_value = ""
                    text_input_value = ""

            # for the jump to event dropdown
            jump_to_event_options = [
                {"label": f"{event.name}", "value": idx} for idx, event in enumerate(self.get_audio_events())
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

        def _run_server():
            Log.info(f"Starting Dash server at http://{self.host}:{self.port}")
            self._app.run_server(host=self.host, port=self.port, debug=False)
            Log.info("Dash server shutdown complete.")

        self._server_thread = threading.Thread(target=_run_server, daemon=True)
        self._server_thread.start()
        Log.info(f"Dash UI started at http://{self.host}:{self.port}")   

    def stop_ui(self):
        """
        Stops the Dash UI and its server thread.
        """
        try:
            shutdown_url = f"http://{self.host}:{self.port}/shutdown"
            response = requests.post(shutdown_url)
            if response.status_code == 200:
                Log.info("Dash UI shutdown successfully.")
            else:
                Log.error(f"Failed to shutdown Dash UI. Status Code: {response.status_code}")
        except Exception as e:
            Log.error(f"Error shutting down Dash UI: {e}")

    def reload_ui(self):
        """
        Reloads the Dash UI.
        """
        self.stop_ui()
        self.start_ui()

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