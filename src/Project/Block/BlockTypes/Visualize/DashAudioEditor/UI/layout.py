from dash_extensions import EventListener
from dash import dcc, html
import dash_player

def build_layout(block):
    """a
    Builds the Dash layout for the AudioEditorBlock.

    Args:
        block (AudioEditorBlock): The instance of AudioEditorBlock.

    Returns:
        html.Div: The layout component.
    """
    keyboard_listener = EventListener(
        id=f"{block.name}-keyboard-listener",
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
            html.H1(f"{block.name} Audio Editor"),
            keyboard_listener,
            # Full Source Waveform
            html.Button("Reload", id=f"{block.name}-reload", n_clicks=0),

            html.Div([
                dcc.Graph(
                    id=f"{block.name}-combined-figure",
                    figure={},
                    style={"height": "200px"}
                )
            ], style={"marginBottom": "20px"}),

        ], style={"padding": "20px"}),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto"})
    
    return layout_content