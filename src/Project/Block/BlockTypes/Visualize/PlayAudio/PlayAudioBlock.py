from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.audio_output import AudioOutput
import io
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import callback_context
import scipy.io.wavfile as wavfile
import numpy as np

class PlayAudioBlock(Block):
    """
    A block that provides a simple audio playback interface using Dash.
    Users can navigate through audio items and play them directly from the UI.
    """

    name = "PlayAudio"
    type = "PlayAudio"

    def __init__(self):
        super().__init__()
        self.name = "PlayAudio"
        self.type = "PlayAudio"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.current_index = 0

        self.command.add("previous_audio", self.previous_audio)
        self.command.add("next_audio", self.next_audio)
        self.command.add("jump_to_audio", self.jump_to_audio)
        self.command.add("register_layout", self.register_layout)

    def process(self, input_data):
        """
        Collect incoming AudioData and store it in self.data.
        Passes data without any transformation.
        """
        return input_data

    def get_audio_items(self):
        """
        Returns all AudioData items in the block's data.
        """
        return [audio for audio in self.data.get_all() if isinstance(audio, AudioData) and audio.data is not None]

    def set_selected_audio(self, audio_index):
        """
        Sets the selected audio to the audio at the given index.
        """
        audio_items = self.get_audio_items()
        if 0 <= audio_index < len(audio_items):
            self.current_index = audio_index
            Log.info(f"Selected audio: {audio_items[self.current_index].name}")
        else:
            Log.warning(f"Invalid audio index: {audio_index}")

    def previous_audio(self):
        """
        Moves to the previous audio item.
        """
        audio_items = self.get_audio_items()
        new_index = max(self.current_index - 1, 0)
        if new_index != self.current_index:
            self.current_index = new_index
            Log.info(f"Moved to previous audio: Index {self.current_index}")
            self.set_selected_audio(self.current_index)

    def next_audio(self):
        """
        Moves to the next audio item.
        """
        audio_items = self.get_audio_items()
        new_index = min(self.current_index + 1, len(audio_items) - 1)
        if new_index != self.current_index:
            self.current_index = new_index
            Log.info(f"Moved to next audio: Index {self.current_index}")
            self.set_selected_audio(self.current_index)

    def jump_to_audio(self, audio_index):
        """
        Jumps to the audio item at the given index.
        """
        audio_items = self.get_audio_items()
        if 0 <= audio_index < len(audio_items):
            self.current_index = audio_index
            Log.info(f"Jumped to audio: Index {self.current_index}")
            self.set_selected_audio(self.current_index)
        else:
            Log.warning(f"Invalid jump audio index: {audio_index}")

    def build_audio_src(self):
        """
        Convert the current audio item to a WAV in memory and return a base64-encoded URI.
        """
        audio_items = self.get_audio_items()
        if not audio_items:
            return ""

        audio_item = audio_items[self.current_index]
        audio_samples = audio_item.data
        sample_rate = getattr(audio_item, "sample_rate", 44100)

        if audio_samples.ndim > 1:
            audio_samples = audio_samples.mean(axis=-1)

        scaled_samples = (audio_samples * 32767).astype(np.int16)

        with io.BytesIO() as buf:
            wavfile.write(buf, sample_rate, scaled_samples)
            encoded_audio = base64.b64encode(buf.getvalue()).decode("ascii")

        return f"data:audio/wav;base64,{encoded_audio}"

    def generate_audio_label(self):
        """
        Returns a label for the current audio item.
        """
        audio_items = self.get_audio_items()
        total_audios = len(audio_items)
        if total_audios == 0:
            return "No Audio Available"
        current_audio_name = audio_items[self.current_index].name
        return f"Audio: {current_audio_name} ({self.current_index + 1}/{total_audios})"

    def register_layout(self):
        """
        Defines the Dash layout for the PlayAudioBlock.
        """
        app = self.parent._app

        layout_content = html.Div([
            html.Div([
                html.H2("Audio Playback Interface"),

                html.Div(
                    id=f"{self.name}-current-audio-name",
                    style={"fontWeight": "bold", "marginBottom": "10px"}
                ),

                html.Audio(
                    id=f"{self.name}-audio-player",
                    controls=True,
                    style={"marginTop": "20px", "display": "block"},
                ),

                html.Div([
                    html.Button("Previous", id=f"{self.name}-prev-button", n_clicks=0, style={"marginRight": "5px"}),
                    html.Button("Next", id=f"{self.name}-next-button", n_clicks=0, style={"marginRight": "5px"}),
                ], style={"marginTop": "10px"}),

                html.Label("Jump to Audio:", style={"marginRight": "10px", "marginTop": "20px"}),
                dcc.Dropdown(
                    id=f"{self.name}-audio-jump-dropdown",
                    options=[],
                    placeholder="Select an audio file",
                    style={"width": "50%", "marginBottom": "10px"}
                ),
            ], style={"padding": "10px"})
        ])

        Log.info(f"Registering page: /{self.name}")
        dash.register_page(
            self.name,
            layout=layout_content,
            name=self.name
        )

        @app.callback(
            [
                Output(f"{self.name}-audio-player", "src"),
                Output(f"{self.name}-current-audio-name", "children"),
                Output(f"{self.name}-audio-jump-dropdown", "options"),
                Output(f"{self.name}-audio-jump-dropdown", "value"),
            ],
            [
                Input(f"{self.name}-prev-button", "n_clicks"),
                Input(f"{self.name}-next-button", "n_clicks"),
                Input(f"{self.name}-audio-jump-dropdown", "value"),
            ],
            [
                State(f"{self.name}-audio-jump-dropdown", "options")
            ]
        )
        def update_audio_view(prev_clicks, next_clicks, jump_index, current_options):
            changed_component = [p["prop_id"] for p in callback_context.triggered][0]
            Log.info(f"Callback triggered by: {changed_component}")

            if f"{self.name}-prev-button" in changed_component:
                self.previous_audio()
            elif f"{self.name}-next-button" in changed_component:
                self.next_audio()
            elif f"{self.name}-audio-jump-dropdown" in changed_component:
                if jump_index is not None:
                    self.jump_to_audio(jump_index)
                else:
                    Log.warning("No audio index provided for jump.")

            figure = ""  # No figure needed for audio playback
            audio_label = self.generate_audio_label()
            audio_src = self.build_audio_src()

            # Audio jump options
            jump_to_audio_options = (
                [
                    {"label": f"{audio.name}", "value": idx}
                    for idx, audio in enumerate(self.get_audio_items())
                ]
                if self.get_audio_items()
                else []
            )

            return (
                audio_src,
                audio_label,
                jump_to_audio_options,
                self.current_index if self.get_audio_items() else None,
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

        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        # Load sub-components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # Push the results to the output ports
        self.output.push_all(self.data.get_all())

        # Register the Dash layout
        self.register_layout()