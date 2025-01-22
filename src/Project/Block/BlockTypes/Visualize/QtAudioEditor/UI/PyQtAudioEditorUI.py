from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import time
from src.Utils.message import Log
import numpy as np

class PyQtAudioEditorUI(QtWidgets.QWidget):
    """
    The UI portion of the PyQtAudioEditorBlock, responsible for displaying
    waveform and event plots, as well as playback controls (Play, Stop, Position Slider).
    """


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Editor")

        # Main layout
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.main_layout)

        # Left layout for controls and plots
        self.left_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        # Controls layout (top)
        self.controls_layout = QtWidgets.QHBoxLayout()
        self.left_layout.addLayout(self.controls_layout)

        # Playback controls
        self.play_button = QtWidgets.QPushButton("Play")
        self.stop_button = QtWidgets.QPushButton("Stop")

        # Playback clock label
        self.playback_clock = QtWidgets.QLabel("00:00:00.000")

        # Add controls to the controls layout
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.stop_button)
        self.controls_layout.addWidget(self.playback_clock)

        # Waveform plot (top)
        self.waveform_plot = pg.PlotWidget()
        self.left_layout.addWidget(self.waveform_plot)

        # Event plot (bottom)
        self.event_plot = pg.PlotWidget()
        self.left_layout.addWidget(self.event_plot)

        self.waveform_plot.setXLink(self.event_plot)
        # self.waveform_plot.setYLink(self.event_plot)

        # Create a vertical playhead line for each plot
        self.playhead_waveform_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color='r', width=2)
        )
        self.playhead_event_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color='r', width=2)
        )

        # Create a vertical playhead line for the event info waveform plot
        self.playhead_event_info_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(color='r', width=2)
        )

        # Event info playback clock label
        self.event_info_playback_clock = QtWidgets.QLabel("00:00:00.000")


        # Add playhead lines to the plots
        self.waveform_plot.addItem(self.playhead_waveform_line)
        self.event_plot.addItem(self.playhead_event_line)


        # Disable auto-scaling for the Y-axis
        self.waveform_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.waveform_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.event_plot.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        self.event_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        self.right_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

       # Right layout for event info
        self.event_info_layout = QtWidgets.QVBoxLayout()
        self.right_layout.addLayout(self.event_info_layout)

        self.event_info_waveform_plot = pg.PlotWidget()
        self.right_layout.addWidget(self.event_info_waveform_plot)

        self.event_info_waveform_plot.addItem(self.playhead_event_info_line)
        self.event_info_layout.addWidget(self.event_info_playback_clock)

        # Event info label
        self.event_info_label = QtWidgets.QLabel("Event Info:")
        self.event_info_layout.addWidget(self.event_info_label)

        # Event info text area
        self.event_info_text = QtWidgets.QTextEdit()
        self.event_info_text.setReadOnly(True)
        self.event_info_layout.addWidget(self.event_info_text)


        # Add buttons to play/stop the event snippet.
        self.play_event_button = QtWidgets.QPushButton("Play Event")
        self.stop_event_button = QtWidgets.QPushButton("Stop Event")
        self.event_info_layout.addWidget(self.play_event_button)
        self.event_info_layout.addWidget(self.stop_event_button)

    @QtCore.pyqtSlot(int)
    def update_playhead(self, position_ms):
        # Update the playhead position and any other UI elements
        position_s = position_ms / 1000   

        self.playhead_waveform_line.setPos(position_s)
        self.playhead_event_line.setPos(position_s)
        # Update playhead lines if necessary

    @QtCore.pyqtSlot(float)
    def update_playback_clock(self, time_seconds):
        # Format time in minutes:seconds
        hours = int(time_seconds // 3600)
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds - int(time_seconds)) * 1000)
        self.playback_clock.setText(f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")

    @QtCore.pyqtSlot(dict)
    def update_event_info(self, event):
        event_name = event.get_name()
        event_time = event.get_time()
        classification = event.get_classification()
        info_text = (
            f"Name: {event_name}\n"
            f"Time: {event_time}\n"
            f"Classification: {classification}\n"
        )
        self.event_info_text.setText(info_text)

        audio_data = event.get()  # Assuming this method exists
        if audio_data.get_data() is not None and len(audio_data.get_data()) > 0:
            x_vals = np.linspace(0, len(audio_data.get_data()) / audio_data.get_sample_rate(), num=len(audio_data.get_data()))
            self.event_info_waveform_plot.clear()
            self.event_info_waveform_plot.plot(x_vals, audio_data.get_data())
        else:
            Log.info("No valid waveform data found for plotting.")

    @QtCore.pyqtSlot(int)
    def update_event_info_playhead(self, position_ms):
        """
        Update the playhead position on the event info waveform plot.
        """
        position_s = position_ms / 1000
        self.playhead_event_info_line.setPos(position_s)

    @QtCore.pyqtSlot(float)
    def update_event_info_clock(self, time_seconds):
        """
        Update the event info playback clock display.
        """
        hours = int(time_seconds // 3600)
        minutes = int(time_seconds // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int((time_seconds - int(time_seconds)) * 1000)
        self.event_info_playback_clock.setText(f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}")


    def start_playback(self):
        self._is_playing = True

    def stop_playback(self):
        self._is_playing = False

    def set_event_plot_limits(self, x_max, y_max):
        self.event_plot.getViewBox().setLimits(
            xMin=0,
            xMax=x_max,
            yMin=0,
            yMax=(y_max + 1)
        ) 
    def set_waveform_plot_limits(self, x_max, y_max):
        self.waveform_plot.getViewBox().setLimits(
            xMin=0,
            xMax=x_max,
            yMin=-1,
            yMax=y_max
        ) 

    def connect_signals(self, play_callback, stop_callback,play_event_callback=None, stop_event_callback=None):
        """
        Connects UI signals to callbacks provided by the editor block.
        """
        self.play_button.clicked.connect(play_callback)
        self.stop_button.clicked.connect(stop_callback)
        self.play_button.clicked.connect(self.start_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        # self.event_plot.scene().sigMouseClicked.connect(event_click_callback)

        # New event callbacks (guard for None if not provided)
        if play_event_callback:
            self.play_event_button.clicked.connect(play_event_callback)
        if stop_event_callback:
            self.stop_event_button.clicked.connect(stop_event_callback)