from dash_extensions import EventListener
from dash import dcc, html


def build_layout(block):
    """
    Builds the Dash layout for the ManualClassifyBlock.
    
    Args:
        block (ManualClassifyBlock): The instance of ManualClassifyBlock.
    
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
            html.H1(f"{block.name.capitalize()} EventData Viewer"),
            keyboard_listener,
            html.Div(id=f"{block.name}-arrow-key-output"),

            html.Div(
                id=f"{block.name}-current-event-name",
                style={"fontWeight": "bold", "marginBottom": "10px"}
            ),
            html.Div([
                html.Button("Previous", id=f"{block.name}-prev-button", n_clicks=0, style={"marginRight": "5px"}),
                html.Button("Next", id=f"{block.name}-next-button", n_clicks=0, style={"marginRight": "5px"}),
            ]),

            html.Div([
                html.Div([
                    html.Label("Select Visualization:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{block.name}-visualization-dropdown",
                        options=[{"label": vis, "value": vis} for vis in block.visualization_types],
                        value=block.visualization,
                        placeholder="Select visualization type",
                        style={"width": "100%"},
                        clearable=False,
                    ),
                ], style={"flex": 1, "marginRight": "10px"}),

                html.Div([
                    html.Label("Select Transformation:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{block.name}-transformation-dropdown",
                        options=[{"label": trans, "value": trans} for trans in block.transformation_types],
                        value=block.transformation,
                        placeholder="Select transformation type",
                        style={"width": "100%"},
                        clearable=False,
                    ),
                ], style={"flex": 1, "marginRight": "10px"}),

                html.Div([
                    html.Label("Select Figure Height:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{block.name}-figure-height-dropdown",
                        options=[{"label": height, "value": height} for height in block.figure_height_types],
                        value=block.figure_height,
                        placeholder="Select figure height",
                        style={"width": "100%"},
                        clearable=False,
                    ),
                ], style={"flex": 1, "marginRight": "10px"}),

                html.Div([
                    html.Label("Select Color Scale:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{block.name}-color-scale-dropdown",
                        options=[{"label": color_scale, "value": color_scale} for color_scale in block.color_scale_types],
                        value=block.color_scale,
                        placeholder="Select color scale",
                        style={"width": "100%"},
                        clearable=False,
                    ),
                ], style={"flex": 1}),
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "flexWrap": "wrap",
                "marginBottom": "20px"
            }),

            html.Label("Jump to Event:", style={"marginRight": "10px"}),
            dcc.Dropdown(
                id=f"{block.name}-event-jump-dropdown",
                options=[],
                placeholder="Select an event",
                style={"width": "50%", "marginBottom": "10px"},
                clearable=True,
            ),

            dcc.Graph(id=f"{block.name}-spectrogram-plot"),

            html.Audio(
                id=f"{block.name}-audio-player",
                controls=True,
                style={"marginTop": "20px", "display": "block"},
            ),
            html.Button(
                "Play / Pause",
                id=f"{block.name}-play-toggle-btn"
            ),

            html.Div([
                html.Div([
                    html.Label("Classification Suggestions:", style={"marginRight": "10px"}),
                    dcc.Dropdown(
                        id=f"{block.name}-classification-suggestions-dropdown",
                        options=[],
                        value="",
                        placeholder="Select classification",
                        clearable=True,
                        style={"width": "200px"},
                    ),
                ], style={"flex": 1}),
                dcc.Input(
                    id=f"{block.name}-classification-input",
                    type="text",
                    value="",
                    placeholder="Or type custom",
                    style={"width": "200px"}
                ),
                html.Button(
                    "Save Classification",
                    id=f"{block.name}-save-class-button",
                    n_clicks=0,
                    style={"marginRight": "5px"}
                ),
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "flexWrap": "wrap",
                "marginBottom": "20px"
            }),
        ], style={"padding": "10px"}),
    ])

    return layout_content