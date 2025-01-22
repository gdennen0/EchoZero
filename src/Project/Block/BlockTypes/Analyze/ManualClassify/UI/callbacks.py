from dash.dependencies import Input, Output, State
from flask import session
import plotly.graph_objs as go
from dash import callback_context
from src.Utils.message import Log


def register_callbacks(block, app):
    """
    Registers all the necessary Dash callbacks for the ManualClassifyBlock.
    
    Args:
        block (ManualClassifyBlock): The instance of ManualClassifyBlock.
        app (dash.Dash): The Dash app instance.
    """

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
        Input(f"{block.name}-play-toggle-btn", "n_clicks"),
        State(f"{block.name}-audio-player", "id"),  # This is where audioElementId comes from
    )

    @app.callback(
        [
            Output(f"{block.name}-spectrogram-plot", "figure"),
            Output(f"{block.name}-current-event-name", "children"),
            Output(f"{block.name}-classification-input", "value"),
            Output(f"{block.name}-classification-suggestions-dropdown", "options"),
            Output(f"{block.name}-classification-suggestions-dropdown", "value"),
            Output(f"{block.name}-audio-player", "src"),
            Output(f"{block.name}-event-jump-dropdown", "options"),
            Output(f"{block.name}-event-jump-dropdown", "value"),
            Output(f"{block.name}-visualization-dropdown", "value"),
            Output(f"{block.name}-transformation-dropdown", "value"),
            Output(f"{block.name}-figure-height-dropdown", "value"),
            Output(f"{block.name}-color-scale-dropdown", "value"),
            Output(f"{block.name}-arrow-key-output", "children")
        ],
        [
            Input(f"{block.name}-prev-button", "n_clicks"),
            Input(f"{block.name}-next-button", "n_clicks"),
            Input(f"{block.name}-save-class-button", "n_clicks"),
            Input(f"{block.name}-classification-suggestions-dropdown", "value"),
            Input(f"{block.name}-event-jump-dropdown", "value"),
            Input(f"{block.name}-visualization-dropdown", "value"),
            Input(f"{block.name}-transformation-dropdown", "value"),
            Input(f"{block.name}-figure-height-dropdown", "value"),
            Input(f"{block.name}-color-scale-dropdown", "value"),
            Input(f"{block.name}-keyboard-listener", "n_events"),
        ],
        [
            State(f"{block.name}-keyboard-listener", "event"),
            State(f"{block.name}-classification-input", "value")
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
        keyboard_event,
        typed_class
    ):
        changed_component = [p["prop_id"] for p in callback_context.triggered][0]
        Log.info(f"Callback triggered by: {changed_component}")

        classification_saved = False  # Flag to track if classification was saved

        if f"{block.name}-visualization-dropdown" in changed_component:
            block.set_visualization(visualization=selected_visualization)
            Log.info(f"Set visualization to: {selected_visualization}")

        if f"{block.name}-transformation-dropdown" in changed_component:
            Log.info(f"Setting transformation to: {selected_transformation}")
            block.set_transformation(transformation=selected_transformation)
            Log.info(f"Set transformation to: {selected_transformation}")

        if f"{block.name}-figure-height-dropdown" in changed_component:
            block.set_figure_height(figure_height=selected_figure_height)
            Log.info(f"Set figure height to: {selected_figure_height}")

        if f"{block.name}-color-scale-dropdown" in changed_component:
            block.set_color_scale(color_scale=selected_color_scale)
            Log.info(f"Set color scale to: {selected_color_scale}")


        # Ensure current classification is in available_classifications
        if block.selected_event and block.selected_event.classification:
            block.available_classifications.add(block.selected_event.classification)

        # If anything triggered classification, set classification:
        if block.get_audio_events() and any(comp in changed_component for comp in [
            f"{block.name}-prev-button",
            f"{block.name}-next-button",
            f"{block.name}-save-class-button",
            f"{block.name}-classification-suggestions-dropdown",
            f"{block.name}-event-jump-dropdown",
            f"{block.name}-visualization-dropdown",
            f"{block.name}-transformation-dropdown",
            f"{block.name}-figure-height-dropdown",
            f"{block.name}-color-scale-dropdown"
        ]):
            if f"{block.name}-save-class-button" in changed_component and typed_class:
                block.set_classification(typed_class)
                classification_saved = True
                Log.info(f"Set classification to: {typed_class}")
            elif selected_suggestion:
                block.set_classification(selected_suggestion)
                Log.info(f"Set classification to: {selected_suggestion}")

        # Handle ArrowKey event
        arrow_pressed = ""
        if keyboard_event:
            pressed_key = keyboard_event.get("key", "")
            if pressed_key == "ArrowLeft":
                block.previous_event()
                Log.info("ArrowLeft => previous_event()")
            elif pressed_key == "ArrowRight":
                block.next_event()
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
        if block.get_audio_events():
            if f"{block.name}-prev-button" in changed_component:
                block.previous_event()
            elif f"{block.name}-next-button" in changed_component:
                block.next_event()
            elif f"{block.name}-event-jump-dropdown" in changed_component:
                if jump_index is not None:
                    block.jump_to_event(jump_index)
                else:
                    Log.warning("No event index provided for jump.")
            elif f"{block.name}-save-class-button" in changed_component:
                if typed_class:
                    block.set_classification(typed_class)
                    classification_saved = True
                else:
                    Log.warning("No classification provided for save.")

        # Classification suggestions
        sorted_dropdown_options = (
            [{"label": c, "value": c} for c in sorted(block.available_classifications)]
            if block.available_classifications
            else []
        )

        # Current classification
        if block.selected_event and block.selected_event.classification in block.available_classifications:
            dropdown_value = block.selected_event.classification
            text_input_value = "" if classification_saved else typed_class
        elif block.selected_event and block.selected_event.classification:
            # Add current classification to available_classifications if missing
            block.available_classifications.add(block.selected_event.classification)
            dropdown_value = block.selected_event.classification
            text_input_value = "" if classification_saved else typed_class
        else:
            dropdown_value = ""
            text_input_value = typed_class  # Preserve user input

        # Event jump options
        jump_to_event_options = (
            [
                {"label": f"{event.name}", "value": idx}
                for idx, event in enumerate(block.get_audio_events())
            ]
            if block.get_audio_events()
            else []
        )

        figure = block.build_figure() if block.get_audio_events() else go.Figure()
        event_label = block.generate_event_label() if block.get_audio_events() else "No Events Available"
        audio_src = block.build_audio_src() if block.get_audio_events() else ""

        try:
            return (
                figure,
                event_label,
                text_input_value,
                sorted_dropdown_options,
                dropdown_value,
                audio_src,
                jump_to_event_options,
                block.current_index if block.get_audio_events() else None,
                block.visualization,
                block.transformation,
                block.figure_height,
                block.color_scale,
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
                block.visualization,
                block.transformation,
                block.figure_height,
                block.color_scale,
                str(e)
            )