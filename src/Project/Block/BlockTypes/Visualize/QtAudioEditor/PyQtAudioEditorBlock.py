from PyQt5 import QtCore, QtMultimedia
import numpy as np
import pyqtgraph as pg
import os

from src.Utils.message import Log
from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.BlockTypes.Visualize.QtAudioEditor.UI.PyQtAudioEditorUI import PyQtAudioEditorUI
import pygame
import tempfile
from scipy.io.wavfile import write
from PyQt5 import QtWidgets
import time

class PyQtAudioEditorBlock(Block):
    """
    A PyQt-based audio editor block that displays waveform (top), events (bottom),
    and manages audio playback with a unified playhead.

    Usage:
        1. Load or receive AudioData and EventData into this block (via 'process' or load).
        2. Call 'build_combined_plot()' to update the waveform and event plots.
        3. Call 'show_ui()' to display the editor window.
        4. Optionally use 'load_audio_file' to load an audio file directly for playback.
    """

    name = "PyQtAudioEditor"
    type = "PyQtAudioEditor"

    def __init__(self):
        super().__init__()
        self.name = "PyQtAudioEditor"
        self.type = "PyQtAudioEditor"

        # Add input & output definitions
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Add a command to show the UI from an external trigger
        self.command.add("show_ui", self.show_ui)
        self.command.add("refresh", self.build_plot)

        # Create UI and media player
        self.ui = PyQtAudioEditorUI()

        # Connect UI signals to playback logic
        self.ui.connect_signals(
            play_callback=self.play_audio,
            stop_callback=self.stop_audio,

            play_event_callback=self.play_event_audio,
            stop_event_callback=self.stop_event_audio
        )
        self.playback_state = "stopped"
        pygame.mixer.init()

        self._is_playing = False
        self._playback_position = 0
        self._playback_duration = 0  # Will be set when audio is loaded

        # Event snippet playback state
        self._event_is_playing = False
        self._event_playback_position = 0
        self._event_playback_duration = 0

        self.selected_event = None
        self.event_sound = None

        # Create a thread for playback
        self.playback_thread = QtCore.QThread()
        self.playback_thread.run = self.playback_loop
        self.playback_thread.start()

        # Separate thread for event snippet playback
        self.event_playback_thread = QtCore.QThread()
        self.event_playback_thread.run = self.event_playback_loop
        self.event_playback_thread.start()

        self.event_data_dict = {}  # Dictionary to store event data



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
        for i, event_point in enumerate(ev):
            event_data = event_point.data()
            event_name = event_data['name']
            event_layer = event_data['layer']
            Log.info(f"clicked event {event_name} in layer {event_layer}")
            found_event = self.find_event(event_name, event_layer)

            # Store the found event as the current selection
            if found_event:
                self.selected_event = found_event

        if event_data:
            self.ui.update_event_info(found_event)
        else:
            Log.info("No event data found")

    def play_event_audio(self):
        """
        Loads and plays the currently selected event's audio data, if any,
        with a separate snippet playback loop that drives a playhead
        on the 'event info waveform plot'.
        """
        if not self.selected_event:
            Log.info("No selected event available to play.")
            return

        event_audio_data = self.selected_event.get()
        if (not event_audio_data) or (event_audio_data.get_data() is None) or (len(event_audio_data.get_data()) == 0):
            Log.error("No valid audio data found in the selected event.")
            return

        sample_rate = event_audio_data.get_sample_rate()
        snippet_data = event_audio_data.get_data()

        # Convert to 16-bit PCM format if needed
        if snippet_data.dtype != np.int16:
            snippet_data = (snippet_data * 32767).astype(np.int16)

        # Temporary file for playback
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_name = temp_file.name
        temp_file.close()

        write(temp_file_name, sample_rate, snippet_data)
        self.event_sound = pygame.mixer.Sound(temp_file_name)

        # Reset snippet playback tracking
        snippet_length_ms = (len(snippet_data) / sample_rate) * 1000.0
        self._event_playback_duration = snippet_length_ms
        self._event_playback_position = 0
        self._event_is_playing = True

        # Start playback
        self.event_sound.play()

    def stop_event_audio(self):
        """
        Stops the currently playing event snippet audio and resets event snippet playhead.
        """
        if self.event_sound:
            self.event_sound.stop()
            self.event_sound = None
        
        self._event_is_playing = False
        self._event_playback_position = 0
        # Optionally reset the playhead visually
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_playhead", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(int, 0)
        )
        QtCore.QMetaObject.invokeMethod(
            self.ui, "update_event_info_clock", QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(float, 0.0)
        )   

    def event_playback_loop(self):
        """
        Continuously checks if an event snippet is playing and
        updates the playhead on the event info waveform plot.
        """
        while True:
            if self._event_is_playing:
                self._event_playback_position += 1  # Update every 1ms
                # If we reached the end, reset
                if self._event_playback_position >= self._event_playback_duration:
                    self._event_playback_position = 0
                    self._event_is_playing = False

                # Update any "event info" playhead & time label in the UI
                QtCore.QMetaObject.invokeMethod(
                    self.ui, "update_event_info_playhead", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, self._event_playback_position)
                )
                current_time_seconds = self._event_playback_position / 1000.0
                QtCore.QMetaObject.invokeMethod(
                    self.ui, "update_event_info_clock", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(float, current_time_seconds)
                )
            time.sleep(0.001)

    def playback_loop(self):
        while True:
            if self._is_playing:
                self._playback_position += 1  # Update every 10ms
                if self._playback_position >= self._playback_duration:
                    self._playback_position = 0
                    self._is_playing = False

                # Emit signal to update UI
                QtCore.QMetaObject.invokeMethod(
                    self.ui, "update_playhead", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, self._playback_position)
                )
                current_time_seconds = self._playback_position / 1000.0
                QtCore.QMetaObject.invokeMethod(
                    self.ui, "update_playback_clock", QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(float, current_time_seconds)
                )

            time.sleep(0.001)  # Sleep for 1ms

    def show_ui(self):
        Log.info("Showing UI")
        self.ui.show()

    def process(self, input_data):
        return input_data

    def build_plot(self):
        Log.info(f"Building combined plot with {len(self.data.get_all())} data objects.")

        # Clear existing items
        self.ui.waveform_plot.clear()
        self.ui.event_plot.clear()

        # Re-add playhead indicators
        self.ui.waveform_plot.addItem(self.ui.playhead_waveform_line)
        self.ui.event_plot.addItem(self.ui.playhead_event_line)

        # Retrieve audio + event data
        audio_data = None
        sample_rate = 0
        events = {}
        y_index = 0
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object.get()
                sample_rate = data_object.get_sample_rate()
            elif data_object.type == "EventData":
                # Append to our events list
                event_category = data_object.get_name()
                if event_category not in events:
                    events[event_category] = []
                events[event_category].extend(data_object.get())
                y_index += 1

        # Set the y range for the event plot
        self.ui.set_event_plot_limits(x_max=len(audio_data) / sample_rate, y_max=y_index)
        self.ui.set_waveform_plot_limits(x_max=len(audio_data) / sample_rate, y_max=1)

        # Plot waveform
        if audio_data is not None and len(audio_data) > 0 and sample_rate > 0:
            # Downsample if it's very large
            if len(audio_data) > 3000:
                reduced_y = audio_data[::3000]
                x_vals = np.linspace(0, len(audio_data) / sample_rate, num=len(reduced_y))
            else:
                reduced_y = audio_data
                x_vals = np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data))

            self.ui.waveform_plot.plot(x_vals, reduced_y, pen='b')
            self.ui.waveform_plot.setXRange(0, len(audio_data) / sample_rate)
        else:
            Log.info("No valid audio data found for plotting.")

        # Plot EventData
        if len(events) > 0:
            spots = []
            for layer, events in events.items():
                for i, event in enumerate(events):
                    x = event.get_time()
                    name = event.get_name()

                    spot = {
                        'pos': (x, y_index), 
                        'size': 10,
                        'brush': pg.mkBrush(100, 100, 255, 150),
                        'data': {'name': name, 'layer':layer},
                        }
                    spots.append(spot)

            event_scatter = pg.ScatterPlotItem(pen=pg.mkPen(None), symbol='o')

            # Add all points to the scatter plot with their metadata
            event_scatter.addPoints(spots)
            
            # Connect the click signal to the handler
            # Place each event at an increasing y index to stack them
            # for i, evt in enumerate(events):
            #     t = evt.get_time()   # event time (in seconds presumably)
            #     nm = evt.get_name()  # event label
            #     scatter_data_x.append(t)
            #     scatter_data_y.append(y_index)
            #     scatter_symbols.append('o')
            #     scatter_names.append(nm)
            #     # Add metadata for each point
            #     scatter_data.append({"name": nm})


                # Store event data in the dictionary
                # spots.append({
                #     "pos": (t, y_index),
                #     "size": 10,
                #     "brush": pg.mkBrush(100, 100, 255, 150),
                #     "pen": pg.mkPen(None),
                #     "name": nm,
            #     # })
            # event_scatter = pg.ScatterPlotItem(spots=spots)
            # event_scatter = pg.ScatterPlotItem(
            #     x=scatter_data_x,
            #     y=scatter_data_y,
            #     symbol=scatter_symbols,
            #     size=10,
            #     pen=pg.mkPen(None),
            #     brush=pg.mkBrush(100, 100, 255, 150),
            #     data=scatter_data,
            #     name=scatter_names,
            #     spots=scatter_names,
            # )
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
        audio_data = None
        sample_rate = 0
        for data_object in self.data.get_all():
            if data_object.type == "AudioData":
                audio_data = data_object.get()
                sample_rate = data_object.get_sample_rate()

        if audio_data is None or len(audio_data) == 0 or sample_rate <= 0:
            Log.error("No valid audio data found for playback.")
            return

        # Set playback duration based on audio length
        self._playback_duration = (len(audio_data) / sample_rate) * 1000
        Log.info(f"Playback duration set to {self._playback_duration} ms")

        # Write the numpy array to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file_name = temp_file.name
        temp_file.close()

        write(temp_file_name, sample_rate, audio_data)
        # Load the temporary file into pygame mixer
        pygame.mixer.music.load(temp_file_name)
        Log.info(f"Loaded audio data into Audio Editor")

    def play_audio(self):
        self._is_playing = True
        pygame.mixer.music.play()

    def stop_audio(self):
        self._is_playing = False
        pygame.mixer.music.stop()

    def get_current_time(self):
        current_time_ms = pygame.mixer.music.get_pos()
        current_time_seconds = current_time_ms / 1000.0
        return current_time_seconds
        
    def set_current_time(self, jump_time):
        pygame.mixer.music.play(start=jump_time)


    def restart_audio(self):
        Log.error("Restarting audio")
        pygame.mixer.music.rewind()
        pygame.mixer.music.play()


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

        # Rebuild the plot after loading data
        self.build_plot()
        self.load_audio()
        self.show_ui()