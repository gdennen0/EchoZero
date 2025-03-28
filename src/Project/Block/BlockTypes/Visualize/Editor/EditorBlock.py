from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication

from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.BlockTypes.Visualize.Editor.UI.EditorUI import EditorUI
import tempfile
from scipy.io.wavfile import write
from src.Utils.tools import prompt_selection
import time
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Data.Types.audio_data import AudioData
import uuid

class EditorBlock(Block):
    """
    A PyQt-based audio editor block that displays waveform (top), events (bottom),
    and manages audio playback with a unified playhead.

    Usage:
        1. Load or receive AudioData and EventData into this block (via 'process' or load).
        2. Call 'build_combined_plot()' to update the waveform and event plots.
        3. Call 'show_ui()' to display the editor window.
        4. Optionally use 'load_audio_file' to load an audio file directly for playback.
    """

    name = "Editor"
    type = "Editor"

    def __init__(self):
        super().__init__()
        self.name = "Editor"
        self.type = "Editor"

        # Add input & output definitions
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Add a command to show the UI from an external trigger
        self.command.add("show_ui", self.show_ui)
        self.command.add("refresh", self.refresh)
        self.command.add("load_audio", self.load_audio)

        # Create UI and media player
        self.ui = EditorUI()

        # Connect UI signals to playback logic
        self.ui.connect_signals(
            play_callback=self.play_audio,
            stop_callback=self.stop_audio,
            reset_callback=self.reset_audio,

            play_event_callback=self.play_event_audio,
            stop_event_callback=self.stop_event_audio,
            next_event_callback=self.next_event,
            previous_event_callback=self.previous_event,

            create_event_callback=self.create_event_from_roi,

        )

        # Connect shortcut signals to methods
        self.ui.next_event_shortcut_activated.connect(self.next_event)
        self.ui.previous_event_shortcut_activated.connect(self.previous_event)
        self.ui.toggle_play_stop_shortcut_activated.connect(self.toggle_play_pause)
        self.ui.toggle_event_play_stop_shortcut_activated.connect(self.toggle_event_play_pause)
        self.ui.up_layer_shortcut_activated.connect(self.up_layer)
        self.ui.down_layer_shortcut_activated.connect(self.down_layer)
        self.ui.plot_clicked.connect(self.on_plot_click)

        # Connect the delete event button signal to the delete method
        self.ui.delete_event_button_clicked.connect(self.delete_selected_events)

        self.playback_state = "unloaded"

        self.audio_roi_start = 0
        self.audio_roi_end = 0
        # Connect ROI change signal
        self.ui.roi_changed.connect(self.on_roi_changed)

        # Initialize QMediaPlayers
        self.player = QMediaPlayer()
        self.player.setNotifyInterval(30)  # Set notify interval to 30ms for more frequent updates
        self.player.positionChanged.connect(self.on_player_position_changed)
        self.player.stateChanged.connect(self.on_player_state_changed)

        self.event_player = QMediaPlayer()
        self.event_player.setNotifyInterval(30)  # Set notify interval to 30ms for more frequent updates
        self.event_player.positionChanged.connect(self.on_event_player_position_changed)
        self.event_player.stateChanged.connect(self.on_event_player_state_changed)

        self._playback_position = 0
        self._playback_duration = 0
        self._event_playback_position = 0
        self._event_playback_duration = 0

        self.selected_events = []  # List to store selected events
        self.selected_layer = None

        self.event_data_dict = {}  # Dictionary to store event data

        self.classifications = []

        self.ui.classification_changed.connect(self.handle_classification_change)
        # self.ui.event_moved.connect(self.handle_event_moved)
        self.ui.event_time_edit.editingFinished.connect(self.on_event_time_edited)

        # Connect shortcut signals to methods
        self.ui.move_roi_to_playhead_shortcut_activated.connect(self.move_roi_to_playhead)
        self.ui.create_event_shortcut_activated.connect(self.create_event_from_roi)
        self.ui.delete_event_shortcut_activated.connect(self.delete_selected_events)

    def on_roi_changed(self, start, end):
        """Update stored ROI bounds when changed"""
        self.audio_roi_start = start
        self.audio_roi_end = end
        Log.info(f"ROI updated: {start:.3f}s to {end:.3f}s")

    def create_event_from_roi(self):
        """Create a new event from the current ROI selection"""
        # Find AudioData in block's data
        audio_data = None
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object
                break

        if audio_data is None:
            Log.error("No AudioData found to create event from")
            return

        if self.audio_roi_start >= self.audio_roi_end:
            Log.error("Invalid ROI selection")
            return
        
        if not self.selected_layer:
            Log.error("No layer selected")
            return

        # Calculate sample indices
        sample_rate = audio_data.get_sample_rate()
        start_sample = int(self.audio_roi_start * sample_rate)
        end_sample = int(self.audio_roi_end * sample_rate)

        # Extract audio segment
        full_audio = audio_data.get()
        if start_sample >= len(full_audio) or end_sample > len(full_audio):
            Log.error("ROI selection out of bounds")
            return

        audio_segment = full_audio[start_sample:end_sample]
        classification = self.selected_layer.replace("Events", "")

        # Create new AudioData for the event
        event_audio = AudioData()
        event_audio.set(audio_segment)
        event_audio.set_sample_rate(sample_rate)

        # Create new EventItem
        event_name = f"Event_{uuid.uuid4().hex[:8]}"  # Generate unique name
        new_event = EventItem()
        new_event.set_name(event_name)
        new_event.set_time(self.audio_roi_start)  # Use ROI start as event time
        new_event.set(event_audio)
        new_event.set_classification(classification)

        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                if data_object.get_name() == self.selected_layer:
                    data_object.add_item(new_event)
                    Log.info(f"Created new event '{event_name}' at {self.audio_roi_start:.3f}s")
                    # self.refresh()
                    self.update_event_plot(new_event, self.selected_layer, self.selected_layer)
                    break

        Log.error("ERROR ADDING EVENT TO AN EVENT DATA LAYER")

    def on_plot_click(self, x_value):
        """
        Update the playhead position based on the clicked x-value.
        Ensures accurate time positioning by rounding to nearest sample.
        """
        # Get the current audio data and sample rate
        audio_data = None
        sample_rate = 0
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object.get()
                sample_rate = data_object.get_sample_rate()
                break

        if sample_rate > 0:
            # Convert x_value (seconds) to sample index
            sample_index = int(round(x_value * sample_rate))
            # Convert back to precise time in seconds
            precise_time = sample_index / sample_rate
            # Convert to milliseconds for position setting
            position_ms = int(precise_time * 1000)
            
            self.set_current_time(position_ms)
            self.ui.update_playhead(position_ms)
            self.ui.update_playback_clock(precise_time)
            Log.info(f"Plot clicked - Time: {precise_time:.3f}s, Position: {position_ms}ms")
        else:
            Log.error("No valid sample rate found for time conversion")

    def on_event_time_edited(self):
        """
        Handle the event time edit box changes.
        """
        if self.selected_events:
            try:
                new_time = float(self.ui.event_time_edit.text())
                for event in self.selected_events:
                    event.set_time(new_time)
                self.jump_to_event_time(self.selected_events[-1])
                Log.info(f"Event time updated to: {new_time}")
            except ValueError:
                Log.error("Invalid time entered. Please enter a valid number.")

    def jump_to_event_time(self, event):
        """
        Jump the audio play location and clock time to the event's time on the main waveform plot.
        """
        event_time_seconds = event.get_time()
        event_time_ms = int(event_time_seconds * 1000)
        self.set_current_time(event_time_ms)  # Set the audio play location
        self.ui.update_playhead(event_time_ms)  # Update the playhead position
        self.ui.update_playback_clock(event_time_seconds)  # Update the playback clock

    def next_event(self):
        """
        Move to the next event in the timeline and update the playhead.
        """
        events = self.get_sorted_events()
        if not self.selected_events:
            Log.info("No selected event to move from.")
            return

        current_index = events.index(self.selected_events[-1])
        if current_index < len(events) - 1:
            next_event = events[current_index + 1]
            self.selected_events = [next_event]
            self.jump_to_event_time(next_event)
            self.ui.update_event_info(next_event)
            self.ui.highlight_event_points(self.selected_events, self.selected_layer)

    def previous_event(self):
        """
        Move to the previous event in the timeline and update the playhead.
        """
        if not self.selected_events:
            Log.info("No selected event to move from.")
            return

        events = self.get_sorted_events()
        current_index = events.index(self.selected_events[0])
        if current_index > 0:
            prev_event = events[current_index - 1]
            self.selected_events = [prev_event]
            self.jump_to_event_time(prev_event)
            self.ui.update_event_info(prev_event)
            self.ui.highlight_event_points(self.selected_events, self.selected_layer)

    def get_sorted_events(self):
        """
        Retrieve and sort all events by their time, and return a list of event objects.
        """
        all_events = []
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                layer_name = data_object.get_name()
                if layer_name == self.selected_layer:
                    events = data_object.get()
                    for event in events:
                        all_events.append(event)  # Append the event object directly
        sorted_events = sorted(all_events, key=lambda event: event.get_time())  # Sort by event time
        return sorted_events

    def update_playhead_to_event(self, event):
        """
        Update the playhead position to the event's time.
        """
        event_time = event.get_time()
        self.ui.update_playhead(event_time * 1000)  # Convert to milliseconds
        self.ui.update_event_info(event)
        Log.info(f"updated playhead to event time ms:{event_time * 1000}, s:{event_time}")

    def handle_classification_change(self, new_classification):
        """
        Handle the classification text from the UI.
        """
        Log.info(f"New classification entered: {new_classification}")
        if new_classification:
            if len(self.classifications) > 0:
                # Log.info(f"updated classification: {new_classification}")
                for data_object in self.data.get_all():
                    if data_object.type == "EventData":
                        eventdata_name = data_object.get_name().replace('Events', '')
                        if eventdata_name.lower() == new_classification.lower():
                            # Found matching EventData layer
                            self.set_selected_events_classification(new_classification)
                            self.move_selected_events(self.selected_layer, new_classification)
                            return

                # If we get here, no matching EventData was found
                # Create new EventData for this classification
                new_event_data = EventData()
                new_event_data.set_name(f"{new_classification}Events")
                new_event_data.set_type("EventData")
                self.data.add(new_event_data)
                
                # Move events to new layer
                self.set_selected_events_classification(new_classification)
                self.move_selected_events(self.selected_layer, new_classification)
                Log.info(f"Created new EventData layer for classification: {new_classification}")

    def set_selected_events_classification(self, new_classification):
        """
        Set the classification of all selected events.
        """
        for event in self.selected_events:
            event.set_classification(new_classification)

    def move_selected_events(self, from_layer, to_layer):
        """
        Move all selected events from one layer to another and update the plot in place.
        """

        to_layer = f"{to_layer}Events"
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                if data_object.get_name() == from_layer:
                    for event in self.selected_events:
                        Log.info(f"Moving event {event.get_name()} from {from_layer} to {to_layer}")
                        data_object.remove_item(event)
                if data_object.get_name() == to_layer:
                    for event in self.selected_events:
                        # event.data['layer'] = to_layer # need to update the layer data for the spots as well
                        data_object.add_item(event)

        # Update the plot in place for all selected events
        for event in self.selected_events:
            self.update_event_plot(event, from_layer, to_layer)

        # Update the selected layer to match the new layer
        self.selected_layer = to_layer

        # Update the highlights
        self.ui.highlight_event_points(self.selected_events, to_layer)
        self.ui.highlight_layer_title(to_layer)

        Log.info(f"Moved events from {from_layer} to {to_layer}")

    def update_event_plot(self, event, from_layer, to_layer):
        """
        Update the event's position in the plot without rebuilding the entire plot.
        """
        for item in self.ui.event_plot.items():
            if isinstance(item, pg.ScatterPlotItem):
                # Create a new list of points, excluding the one we want to move
                new_spots = []
                for spot in item.points():
                    if not (spot.data()['name'] == event.get_name() and spot.data()['layer'] == from_layer):
                        # Keep all other points
                        new_spots.append({
                            'pos': spot.pos(),
                            'data': spot.data(),
                            'size': 15,
                            'brush': pg.mkBrush(150, 150, 150, 255),
                            'symbol': 'd'
                        })
                
                if to_layer is not None:
                    # Add the moved point to the new spots list
                    new_spots.append({
                        'pos': (event.get_time(), self.get_layer_index(to_layer)),
                    'data': {'name': event.get_name(), 'layer': to_layer},
                    'size': 15,
                    'brush': pg.mkBrush(150, 150, 150, 255),
                    'symbol': 'd'
                    })
                
                # Clear existing points and add all points back
                item.clear()
                item.addPoints(new_spots)
                break

    def get_layer_index(self, layer_name):
        """
        Get the y-index for a given layer name.
        """
        y_index = 1
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                if data_object.get_name() == layer_name:
                    return y_index
                y_index += 1
        return y_index

    def update_classification_dropdown(self):
       """
       Updates the classification dropdown in the UI with the current classifications.
       """
       self.ui.update_classification_dropdown(self.classifications)
       Log.info(f"Updated classification dropdown options")

    def collect_local_classifications(self):
        """
        Collects all classifications from the local event data
        """
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                for event_item in data_object.get():        
                    classification = event_item.get_classification()
                    if classification is not None:  # Ensure classification is not None
                        self.add_classification(classification)

    def add_classification(self, classification):
        """
        Adds a classification to the local event data
        """
        if classification is not None:
            if isinstance(classification, str):
                classification = classification.lower()
                if self.classifications is None:
                    self.classifications = []
                if classification not in self.classifications:
                    self.classifications.append(classification)
                    self.update_classification_dropdown()
                    Log.info(f"Added classification: {classification}")

    def remove_classification(self, classification):
        """
        Removes a classification from the local event data
        """
        if classification in self.classifications:
            self.classifications.remove(classification)

    def find_event(self, name, layer):
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                if data_object.get_name() == layer:
                    events = data_object.get()
                    for event in events:
                        if event.get_name() == name:
                            Log.info(f"Match found: {event}")
                            return event
        Log.info("No match found")
        return None

    def on_event_click(self, points, ev):
        modifiers = QApplication.keyboardModifiers()
        shift_pressed = modifiers == QtCore.Qt.ShiftModifier

        for i, event_point in enumerate(ev):
            event_data = event_point.data()
            event_name = event_data['name']
            event_layer = event_data['layer']
            Log.info(f"clicked event {event_name} in layer {event_layer}")
            found_event = self.find_event(event_name, event_layer)

            if found_event:
                if shift_pressed:
                    # Add to selected_events if shift is pressed
                    if found_event not in self.selected_events:
                        self.selected_events.append(found_event)
                else:
                    # Normal click behavior
                    self.selected_events = [found_event]
                    self.selected_layer = event_layer
                    self.jump_to_event_time(found_event)

        if self.selected_events:
            self.ui.update_event_info(self.selected_events[-1])
            self.ui.highlight_event_points(self.selected_events, self.selected_layer)
            self.ui.highlight_layer_title(self.selected_layer)
        else:
            Log.info("No event data found")

    def play_event_audio(self):
        """
        Loads and plays the currently selected event's audio data using a QMediaPlayer.
        """
        if not self.selected_events:
            Log.info("No selected event available to play.")
            return

        event_audio_data = self.selected_events[-1].get()
        if (not event_audio_data) or (event_audio_data.get_data() is None) or len(event_audio_data.get_data()) == 0:
            Log.error("No valid audio data found in the selected event.")
            return

        sample_rate = event_audio_data.get_sample_rate()
        snippet_data = event_audio_data.get_data()
        # Convert snippet_data to int16 if needed, then write to a temp .wav as before
        if snippet_data.dtype != np.int16:
            snippet_data = (snippet_data * 32767).astype(np.int16)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_name = temp_file.name
        temp_file.close()

        write(temp_file_name, sample_rate, snippet_data)
        
        # Load into event_player
        self.event_player.stop()
        self.event_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_name)))
        self._event_playback_position = 0
        self._event_playback_duration = (len(snippet_data) / sample_rate) * 1000.0

        self.event_player.play()

    def stop_event_audio(self):
        """
        Stops the currently playing event snippet audio and resets event snippet playhead.
        """
        self.event_player.stop()
        self._event_playback_position = 0
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_playhead", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, 0)
        )
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_clock", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, 0.0)
        )

    def on_player_position_changed(self, position):
        """
        Gets called whenever the main audio player's position changes.
        """
        self._playback_position = position
        current_time_seconds = float(position) / 1000.0
        # Log.info(f"Main audio position: {position} ms, {current_time_seconds} s")  # Debugging log
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_playhead", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, position)
        )
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_playback_clock", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, current_time_seconds)
        )

    def on_player_state_changed(self, state):
        """
        Updates self.playback_state based on QMediaPlayer state changes.
        """
        if state == QMediaPlayer.StoppedState:
            if self.playback_state == "playing":
                self.playback_state = "stopped"
                Log.info("Audio playback ended or stopped.")

    def on_event_player_position_changed(self, position):
        """
        Event snippet player position changed signal.
        """
        self._event_playback_position = position
        position_s = float(position) / 1000.0
        # Log.info(f"Event audio position: {position} ms, {position_s} s")  # Debugging log
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_playhead", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, position)
        )
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_clock", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, position_s)
        )

    def on_event_player_state_changed(self, state):
        """
        Handles snippet (event) player state changes.
        Resets position if it has finished playing.
        """
        if state == QMediaPlayer.StoppedState:
            # Reset the snippet playback position 
            self._event_playback_position = 0 
            QtCore.QMetaObject.invokeMethod(
                self.ui, "update_event_info_playhead", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, 0)
            )
            QtCore.QMetaObject.invokeMethod(
                self.ui, "update_event_info_clock", QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(float, 0.0)
            )

    def show_ui(self):
        Log.info("Showing UI")
        self.ui.show()

    def process(self, input_data):
        self.stop_audio()
        self.set_current_time(0)


        return input_data
    
    def refresh(self):
        """
        Refreshes the UI by rebuilding the plot and reloading audio.
        """
        # Store current ROI state
        roi_start, roi_end = self.ui.audio_roi.getRegion()
        
        # Clear all plot widgets to remove stale data before rebuilding
        self.ui.waveform_plot.clear()
        self.ui.event_plot.clear()
        self.ui.waveform_title_plot.clear()
        self.ui.event_title_plot.clear()

        # Re-add playhead indicators
        self.ui.waveform_plot.addItem(self.ui.playhead_waveform_line)
        self.ui.event_plot.addItem(self.ui.playhead_event_line)

        # Add a vertical line at x = 0
        vertical_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('w', width=1))
        self.ui.waveform_plot.addItem(vertical_line)
        self.ui.event_plot.addItem(vertical_line)

        # Rebuild the plot with current data
        self.build_plot()

        # Reload audio data for playback
        self.load_audio()

        # Restore ROI state and ensure it is added to the waveform plot
        self.ui.waveform_plot.addItem(self.ui.audio_roi)
        self.ui.audio_roi.setRegion((roi_start, roi_end))
        self.audio_roi_start = roi_start
        self.audio_roi_end = roi_end

        # Update classification dropdown
        self.ui.update_classification_dropdown(self.classifications)

        # Update event info if any event is selected
        self.ui.update_event_info(self.selected_events[-1] if self.selected_events else None)

    def build_plot(self):
        Log.info(f"Building combined plot with {len(self.data.get_all())} data objects.")

        # Retrieve audio + event data
        audio_data = None
        sample_rate = 0
        audio_data_name = "Audio"  # default if none found
        events = {}
        y_index = 1  # Start y_index at 1 for the first row
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object.get()
                sample_rate = data_object.get_sample_rate()
                audio_data_name = data_object.get_name()  # <--- If your AudioData has a "name"
            elif data_object.type == "EventData":
                event_category = data_object.get_name()
                if event_category not in events:
                    events[event_category] = []
                events[event_category].extend(data_object.get())
                y_index += 1  # Increment y_index for each EventData type

        if audio_data is not None and len(audio_data) > 0 and sample_rate > 0:
            self.ui.set_event_plot_limits(x_max=len(audio_data) / sample_rate, y_max=y_index)
            self.ui.set_waveform_plot_limits(x_max=len(audio_data) / sample_rate, y_max=1)
        else:
            Log.error("No valid audio data found for plotting.")

        # Plot waveform
        if audio_data is not None and len(audio_data) > 0 and sample_rate > 0:
            detail_ratio = 0.05  # Adjust to control detail level
            if len(audio_data) > 1 / detail_ratio:
                step = int(1 / detail_ratio)
                reduced_y = audio_data[::step]
                x_vals = np.linspace(0, len(audio_data) / sample_rate, num=len(reduced_y))
            else:
                reduced_y = audio_data
                x_vals = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

            self.ui.waveform_plot.plot(x_vals, reduced_y, pen='w')
            self.ui.waveform_plot.setXRange(0, len(audio_data) / sample_rate)

            # Add waveform title text to the waveform_title_plot
            waveform_title_plot_view_box = self.ui.waveform_title_plot.getViewBox()
            waveform_title_plot_width = waveform_title_plot_view_box.viewRect().width()
            waveform_title_text = pg.TextItem(text=audio_data_name, anchor=(0.5, 0.5), color='w')
            waveform_title_text.setPos(waveform_title_plot_width, 1)
            self.ui.waveform_title_plot.addItem(waveform_title_text)
        else:
            Log.info("No valid audio data found for plotting.")

        # Plot EventData and update event label plots
        if len(events) > 0:
            spots = []
            y_index = 1  # Reset y_index for plotting, start at 1
            for layer, events_list in events.items():
                for i, event in enumerate(events_list):
                    x = event.get_time()
                    name = event.get_name()
                    spot = {
                        'pos': (x, y_index),
                        'size': 15,
                        'brush': pg.mkBrush(150, 150, 150, 255),
                        'data': {'name': name, 'layer': layer},
                    }
                    spots.append(spot)

                # Add a horizontal line in the event plot for row separation
                self.ui.event_plot.addItem(pg.InfiniteLine(pos=y_index + 0.5, angle=0, pen=pg.mkPen('w', width=1)))

                # Add a text item for the row title in the event_title_plot
                view_box = self.ui.event_title_plot.getViewBox()
                plot_width = view_box.viewRect().width()
                text_item = pg.TextItem(text=layer, anchor=(0.5, 0.5), color='w')
                text_item.setPos(plot_width, y_index)
                self.ui.event_title_plot.addItem(text_item)

                y_index += 1  # Increment y_index for each layer

            event_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen(color='w', width=2),
                symbol='d',
            )
            event_scatter.addPoints(spots)
            event_scatter.sigClicked.connect(self.on_event_click)
            self.ui.event_plot.addItem(event_scatter)
        else:
            Log.info("No event data found for plotting.")

    # ------------------------------
    # Below are playback-related methods
    # ------------------------------

    def match_event_by_name(self, name, y_value):
        # Match event by name and y-axis value
        for event_name, event_data in self.event_data_dict.items():
            if event_name == name and event_data.get("y_index") == y_value:
                Log.info(f"Match found: {event_data}")
                return event_data
        Log.info("No match found")
        return None

    def load_audio(self):
        """
        Loads the existing block's AudioData into the QMediaPlayer for full playback.
        """
        audio_data = None
        sample_rate = 0
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object.get()
                sample_rate = data_object.get_sample_rate()

        if audio_data is None or len(audio_data) == 0 or sample_rate <= 0:
            Log.error("No valid audio data found for playback.")
            return

        self._playback_duration = (len(audio_data) / sample_rate) * 1000
        Log.info(f"Playback duration set to {self._playback_duration} ms")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_name = temp_file.name
        temp_file.close()

        write(temp_file_name, sample_rate, audio_data)
        # Use QMediaPlayer for main audio
        self.player.stop()
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file_name)))
        self.playback_state = "stopped"

    def play_audio(self):
        """
        Plays audio from current position using QMediaPlayer.
        """
        start_time_ms = self._playback_position
        self.playback_state = "playing"
        # Set the QMediaPlayer's position before playing
        self.player.setPosition(int(start_time_ms))
        self.player.play()
        Log.info(f"Playing audio from position {start_time_ms} ms")

    def pause_audio(self):
        """
        Pauses the playback. 
        """
        self.player.pause()
        self.playback_state = "paused"

    def stop_audio(self):
        """
        Stops the audio completely (and reset the internal state).
        """
        if self.playback_state == "playing":
            self.player.stop()
            self.playback_state = "stopped"
            Log.info("Audio stopped.")
        elif self.playback_state == "stopped":
            Log.info("Audio already stopped.")

    def reset_audio(self):
        """
        Resets audio back to time zero and restarts if audio is playing.
        """
        Log.error("Restarting audio")
        if self.playback_state == "playing":
            self.player.stop()
            self.playback_state = "stopped"
            self.set_current_time(0)
            self.player.play()
            self.playback_state = "playing"
        elif self.playback_state == "stopped":
            self.set_current_time(0)

    def get_current_time(self):
        """
        Retrieves the player's current position in seconds.
        """
        current_time_ms = self.player.position()
        return float(current_time_ms) / 1000.0

    def set_current_time(self, jump_time_ms):
        """
        Manually sets the QMediaPlayer position in milliseconds, 
        and updates UI accordingly.
        """
        self._playback_position = jump_time_ms
        self.player.setPosition(int(jump_time_ms))
        Log.info(f"Setting current time to {jump_time_ms} ms")
        self.refresh_playhead_location()
        self.refresh_playback_clock()

    def refresh_playhead_location(self):
        self.ui.update_playhead(self._playback_position)

    def refresh_playback_clock(self):
        self.ui.update_playback_clock(self._playback_position / 1000.0)

    def toggle_play_pause(self):
        """
        Toggles between play and pause states.
        """
        if self.playback_state == "playing":
            self.pause_audio()
        else:
            self.play_audio()

    def toggle_event_play_pause(self):
        """
        Toggles between play and pause states for event audio.
        """
        if self.event_player.state() == QMediaPlayer.PlayingState:
            self.stop_event_audio()
        else:
            self.play_event_audio()

    # ------------------------------
    # Block base methods
    # ------------------------------
    def set_name(self, name):
        self.name = name
        Log.info(f"Block name updated to: {name}")

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "classifications": self.classifications,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))
        self.output.push_all(self.data.get_all())
        self.classifications = block_metadata.get("classifications") or []  # Ensure it's a list

        # --- NEW CODE: Reset user selections so stale selections do not affect the new data ---
        self.selected_events = []
        self.selected_layer = None
        # --- END NEW CODE ---

        # Rebuild the plot after loading data
        self.build_plot()
        self.collect_local_classifications()
        self.update_classification_dropdown()
        self.load_audio()
        self.show_ui()

    def get_layers(self):
        """
        Get a list of all event layers ordered by their y-axis position.
        """
        layer_positions = []
        for data_object in self.data.get_all():
            if data_object.type == "EventData":
                layer_name = data_object.get_name()
                # Get y-index for the layer (1-based index)
                y_index = self.get_layer_index(layer_name)
                layer_positions.append((y_index, layer_name))
        
        # Sort by y-index and return just the layer names
        return [layer for _, layer in sorted(layer_positions)]

    def find_closest_event(self, target_time, layer):
        """
        Find the event closest in time to the target time in the given layer.
        """
        closest_event = None
        min_time_diff = float('inf')
        
        for data_object in self.data.get_all():
            if data_object.type == "EventData" and data_object.get_name() == layer:
                events = data_object.get()
                for event in events:
                    time_diff = abs(event.get_time() - target_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_event = event
        
        return closest_event

    def up_layer(self):
        """
        Move to the layer above (higher y-index) and select the closest event to the current time.
        """
        if not self.selected_layer:
            # If no layer is selected, select the bottom-most layer
            layers = self.get_layers()
            if layers:
                self.selected_layer = layers[0]  # Start from bottom layer
                closest_event = self.find_closest_event(self.get_current_time(), layers[0])
                if closest_event:
                    self.selected_events = [closest_event]
                    self.jump_to_event_time(closest_event)
                    self.ui.update_event_info(closest_event)
                    # Add highlights for initial selection
                    self.ui.highlight_event_points(self.selected_events, self.selected_layer)
                    self.ui.highlight_layer_title(self.selected_layer)
                return

        layers = self.get_layers()
        try:
            current_index = layers.index(self.selected_layer)
            
            if current_index < len(layers) - 1:  # Can move up
                # Get current time from selected event or playhead
                current_time = self.selected_events[-1].get_time() if self.selected_events else self.get_current_time()
                
                # Move to layer above (higher y-index)
                next_layer = layers[current_index + 1]
                self.selected_layer = next_layer
                
                # Find closest event in new layer
                closest_event = self.find_closest_event(current_time, next_layer)
                if closest_event:
                    self.selected_events = [closest_event]
                    self.jump_to_event_time(closest_event)
                    self.ui.update_event_info(closest_event)
                    # Add highlights for new selection
                    self.ui.highlight_event_points(self.selected_events, self.selected_layer)
                    self.ui.highlight_layer_title(self.selected_layer)
                    Log.info(f"Moved up to layer: {next_layer}")
                else:
                    Log.info(f"No events found in layer: {next_layer}")
        except ValueError:
            Log.error(f"Current layer {self.selected_layer} not found in layers list")

    def down_layer(self):
        """
        Move to the layer below (lower y-index) and select the closest event to the current time.
        """
        if not self.selected_layer:
            # If no layer is selected, select the top-most layer
            layers = self.get_layers()
            if layers:
                self.selected_layer = layers[-1]  # Start from top layer
                closest_event = self.find_closest_event(self.get_current_time(), layers[-1])
                if closest_event:
                    self.selected_events = [closest_event]
                    self.jump_to_event_time(closest_event)
                    self.ui.update_event_info(closest_event)
                    # Add highlights for initial selection
                    self.ui.highlight_event_points(self.selected_events, self.selected_layer)
                    self.ui.highlight_layer_title(self.selected_layer)
                return

        layers = self.get_layers()
        try:
            current_index = layers.index(self.selected_layer)
            
            if current_index > 0:  # Can move down
                # Get current time from selected event or playhead
                current_time = self.selected_events[-1].get_time() if self.selected_events else self.get_current_time()
                
                # Move to layer below (lower y-index)
                prev_layer = layers[current_index - 1]
                self.selected_layer = prev_layer
                
                # Find closest event in new layer
                closest_event = self.find_closest_event(current_time, prev_layer)
                if closest_event:
                    self.selected_events = [closest_event]
                    self.jump_to_event_time(closest_event)
                    self.ui.update_event_info(closest_event)
                    # Add highlights for new selection
                    self.ui.highlight_event_points(self.selected_events, self.selected_layer)
                    self.ui.highlight_layer_title(self.selected_layer)
                    Log.info(f"Moved down to layer: {prev_layer}")
                else:
                    Log.info(f"No events found in layer: {prev_layer}")
        except ValueError:
            Log.error(f"Current layer {self.selected_layer} not found in layers list")

    def delete_selected_events(self):
        """
        Deletes the selected events from the data and updates the plot.
        """
        if not self.selected_events:
            Log.info("No selected events to delete.")
            return

        for event in self.selected_events:
            # Remove the event from the data
            for data_object in self.data.get_all():
                if data_object.type == "EventData" and data_object.get_name() == self.selected_layer:
                    data_object.remove_item(event)
                    Log.info(f"Deleted event {event.get_name()} from layer {self.selected_layer}")
            # Update the event plot to remove the deleted event
            self.update_event_plot(event, self.selected_layer, None)
            # Clear the selected events list
        self.selected_events = []

        # Refresh the plot to remove the deleted events
        # self.refresh()

    def move_roi_to_playhead(self):
        """
        Moves the ROI to be centered on the current playhead position.
        """
        current_time = self.get_current_time()
        self.ui.move_roi_to_playhead(current_time)
        # Update stored ROI bounds
        self.audio_roi_start, self.audio_roi_end = self.ui.audio_roi.getRegion()
        Log.info(f"Moved ROI to playhead position: {current_time:.3f}s")
