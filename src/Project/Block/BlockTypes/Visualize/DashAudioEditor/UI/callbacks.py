from dash.dependencies import Input, Output, State
from dash import callback_context
import plotly.graph_objs as go
import base64
import io
import librosa
import numpy as np
from src.Utils.message import Log
from dash import dcc

def register_callbacks(block, app):
    """
    Registers all the necessary Dash callbacks for the AudioEditorBlock.

    Args:
        block (AudioEditorBlock): The instance of AudioEditorBlock.
        app (dash.Dash): The Dash app instance.
    """
    @app.callback(
        Output(f'{block.name}-combined-figure', 'figure'),
        Input(f'{block.name}-reload', 'n_clicks'),
    )
    def update_combined_figure(n_clicks):
        return block.build_combined_figure()