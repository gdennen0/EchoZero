from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Block.Input.Types.event_input import EventInput

import dash
from dash import html

# Import the separated layout and callbacks
from src.Project.Block.BlockTypes.Visualize.DashAudioEditor.UI import layout
from src.Project.Block.BlockTypes.Visualize.DashAudioEditor.UI import callbacks

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import base64
import io

class DashAudioEditorBlock(Block):
    """
    An audio editor block that provides an audio player, waveform visualization,
    event markers, and a synchronized playhead within a Dash multipage application.
    """
    name = "DashAudioEditor"
    type = "DashAudioEditor"

    def __init__(self):
        super().__init__()
        self.name = "DashAudioEditor"
        self.type = "DashAudioEditor"

        self.layout_registered = False
        # Define input and output types
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.command.add("register_ui", self.register_ui)

    def register_ui(self):
        """
        Registers the UI layout and callbacks for the AudioEditorBlock.
        """
        if not self.layout_registered:
            if not self.parent or not hasattr(self.parent, '_app'):
                Log.error("Parent app not found. Ensure that the block is properly connected to a Dash app.")
                return

            Log.info(f"Registering AudioEditor page: /{self.name}")

            dash.register_page(
                self.name,
                layout=layout.build_layout(self),
                name=self.name,
            )

            # Register callbacks
            callbacks.register_callbacks(self, self.parent._app)

            self.layout_registered = True
    
    def build_combined_figure(self):
        """
        Builds a combined figure with waveform and event data for the AudioEditorBlock.
        """
        Log.info(f"Building combined figure with {len(self.data.get_all())} data objects")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        # Add waveform data
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                def reduce_resolution(y, factor):
                    return y[::factor]

                y = data_object.get()
                reduced_y = reduce_resolution(y, 3000)
                sr = data_object.get_sample_rate()
                fig.add_trace(go.Scatter(
                    x=np.linspace(0, len(y) / sr, num=len(reduced_y)),
                    y=reduced_y,
                    mode='lines',
                    name='Waveform'
                ), row=1, col=1)

        # Add event data
        y_axis_value = 1
        x_values = []
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                for event_item in data_object.get():
                    x_values.append(event_item.get_time())
                    fig.add_trace(go.Scatter(
                        x=[event_item.get_time()],
                        y=[y_axis_value],
                        mode='markers',
                        # marker=dict(size=10),
                        name=f"Event {event_item.get_name()}"
                    ), row=2, col=1)
                y_axis_value += 1

        # Add playhead line
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=2,  # Spans both subplots
            line=dict(color="RoyalBlue", width=2),
            name = f"{self.name}-playhead"
        )

        if len(y) > 0 and sr > 0:
            # Update layout
            fig.update_layout(
                title="Combined Waveform and Event Figure",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            margin=dict(l=40, r=40, t=40, b=0),
            clickmode='event+select',
            xaxis=dict(
                range=[0, len(y) / sr],  # Lock x-axis range for combined figure
                constrain='domain',
                rangeslider=dict(visible=True)  # Prevent zooming out beyond the data range
            ),
            config={'displayModeBar': False},
            )
        else:
            fig.update_layout(
                title="Combined Waveform and Event Figure",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            margin=dict(l=40, r=40, t=40, b=0),
            clickmode='event+select',
            config={'displayModeBar': False},
        )
        return fig

    def set_name(self, name):
        self.name = name
        if not self.layout_registered:
            self.register_ui()
        Log.info(f"Updated Blocks name to: '{name}'")

    def process(self, input_data):
        """
        Process the input data and update the output.
        """
        return input_data

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

        # Push loaded data to output
        self.output.push_all(self.data.get_all())

        self.register_ui()