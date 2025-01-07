import threading
import base64
import io
import logging
import numpy as np
import dash
import librosa
import plotly.graph_objs as go
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State

from src.Project.Block.block import Block
from src.Project.Block.Input.Types.ui_input import UIInput
from src.Project.Block.Output.Types.ui_output import UIOutput

class ManualClassifyUIBlock(Block):
    """
    A Block that starts and manages a Dash web application
    for manual audio classification. It can connect to the
    ManualClassifyBlock using UIInput/UIOutput for commands
    and status messages, just like other blocks.
    """

    name = "ManualClassifyUI"

    def __init__(self):
        super().__init__()
        self.name = "ManualClassifyUI"
        self.type = "ManualClassifyUI"

        # Add specialized inputs/outputs for UI flow
        self.input.add_type(UIInput)
        self.input.add("UIInput")

        self.output.add_type(UIOutput)
        self.output.add("UIOutput")

        self._server_thread = None
        self._app = None
        self._in_memory_logger = None

        # Register command(s)
        self.command.add("start_ui", self.start_ui)
        # ... add more commands as needed

    def process(self, input_data):
        """
        Since this block primarily handles UI, it may not
        transform data in the typical sense. We simply store
        or rely on the data that arrives through connections.
        """
        return input_data

    def start_ui(self, host="127.0.0.1", port=8050):
        """
        Spin up the Dash UI. All data is exchanged via UIInput/UIOutput,
        so there's no direct reference to the ManualClassifyBlock here.
        """
        from wip.ManualClassifyBlock import InMemoryLogger
        self._in_memory_logger = InMemoryLogger(enabled=True)

        # Setup Werkzeug logging -> InMemory
        import logging
        class WerkzeugLogToInMemory(logging.Handler):
            def __init__(self, in_memory_logger):
                super().__init__()
                self.in_memory_logger = in_memory_logger
            def emit(self, record):
                log_entry = self.format(record)
                self.in_memory_logger.log_info(log_entry)

        werkzeug_logger = logging.getLogger("werkzeug")
        werkzeug_logger.setLevel(logging.INFO)
        werkzeug_logger.handlers = []
        werkzeug_logger.propagate = False
        werkzeug_logger.addHandler(WerkzeugLogToInMemory(self._in_memory_logger))

        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
        from dash import callback_context
        import plotly.graph_objs as go

        self._app = dash.Dash(__name__)
        self._app.title = "Manual Audio Classification"

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
                        html.Button("Save & Quit", id="save-quit-button", n_clicks=0)
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

        # Grab our UIInput
        ui_input = self.input.get("UIInput")

        @self._app.callback(
            [
                Output("spectrogram-plot", "figure"),
                Output("current-event-name", "children"),
                Output("classification-input", "value"),
                Output("classification-suggestions-dropdown", "options"),
                Output("classification-suggestions-dropdown", "value"),
                Output("audio-player", "src")
            ],
            [
                Input("prev-button", "n_clicks"),
                Input("next-button", "n_clicks"),
                Input("save-class-button", "n_clicks"),
                Input("classification-suggestions-dropdown", "value"),
                Input("event-jump-dropdown", "value")
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
            new_classification = typed_class or selected_suggestion

            # Push relevant commands through UIInput
            if ui_input:
                # If something triggered a classification change, save it:
                if any(comp in changed_component for comp in [
                    "prev-button", 
                    "next-button", 
                    "save-class-button",
                    "classification-suggestions-dropdown",
                    "event-jump-dropdown"
                ]):
                    ui_input.push_command("save_classification", new_classification)

                # Navigation
                if "prev-button" in changed_component:
                    ui_input.push_command("previous_event")
                elif "next-button" in changed_component:
                    ui_input.push_command("next_event")
                elif "event-jump-dropdown" in changed_component and jump_index is not None:
                    ui_input.push_command("jump_to_event", jump_index)

                # Always request new UI state
                ui_input.push_command("get_current_ui_state")

            # Now retrieve the updated state from the block
            current_state = {}
            if ui_input:
                current_state = ui_input.read_data("CURRENT_UI_STATE") or {}

            # Fallbacks if no data:
            figure = current_state.get("figure", go.Figure())
            event_label = current_state.get("event_label", "No events available.")
            audio_src = current_state.get("audio_src", "")
            updated_suggestions = current_state.get("available_classifications", [])
            current_event_class = current_state.get("current_classification", "")

            # For the suggestions dropdown
            sorted_dropdown_options = [
                {"label": c, "value": c} for c in updated_suggestions
            ]
            dropdown_value = (
                current_event_class if current_event_class in updated_suggestions else ""
            )

            # If the classification isn't in suggestions, place it in text box
            text_input_value = ""
            if not dropdown_value:
                text_input_value = current_event_class

            return (
                figure, 
                event_label, 
                text_input_value,
                sorted_dropdown_options,
                dropdown_value,
                audio_src
            )

        @self._app.callback(
            Output("save-quit-button", "disabled"),
            [Input("save-quit-button", "n_clicks")],
            [State("classification-input", "value"),
             State("classification-suggestions-dropdown", "value")]
        )
        def handle_save_and_quit(n_clicks, text_value, dropdown_value):
            if n_clicks > 0 and ui_input:
                final_class = text_value or dropdown_value
                ui_input.push_command("quit_classification", final_class)
                ui_input.push_command("get_current_ui_state")
            return False

        @self._app.callback(
            Output("log-output", "children"),
            [Input("log-interval", "n_intervals")]
        )
        def update_logs(_):
            return self._in_memory_logger.get_logs()

        def _run_server():
            self._in_memory_logger.log_info(f"Starting Dash server at http://{host}:{port}")
            self._app.run_server(host=host, port=port, debug=False)
            self._in_memory_logger.log_info("Dash server shutdown complete.")

        self._server_thread = threading.Thread(target=_run_server, daemon=True)
        self._server_thread.start()
        self._in_memory_logger.log_info(f"Dash UI started at http://{host}:{port}")

    def poll_incoming_logs(self):
        """
        Retrieve logs from the ManualClassifyBlock’s UIOutput,
        which is connected to this block’s UIInput.
        """
        ui_input = self.input.get("UIInput")
        if not ui_input:
            return []

        # If your UIInput is storing logs in e.g. ui_logs, you might do:
        new_logs = ui_input.read_logs()
        # Then, if you want to store them locally or display them, you can do so.
        return new_logs 
    
    def save(self):
        pass

    def load(self, path):
        pass