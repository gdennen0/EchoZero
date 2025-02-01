from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from src.Utils.message import Log
import numpy as np
from PyQt5.QtWidgets import QDesktopWidget, QShortcut
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QKeySequence

class EditorUI(QtWidgets.QWidget):
    """
    The UI portion of the PyQtAudioEditorBlock, responsible for displaying
    waveform and event plots, as well as playback controls (Play, Stop, Position Slider).

    This is all of the QT code for the editor.
    """

    classification_changed = pyqtSignal(str)

    # Define signals for shortcut actions
    next_event_shortcut_activated = pyqtSignal()
    previous_event_shortcut_activated = pyqtSignal()
    toggle_play_stop_shortcut_activated = pyqtSignal()
    toggle_event_play_stop_shortcut_activated = pyqtSignal()  # New signal for event audio toggle
    up_layer_shortcut_activated = pyqtSignal()  # Renamed from next_layer_shortcut_activated
    down_layer_shortcut_activated = pyqtSignal()  # Renamed from previous_layer_shortcut_activated

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Editor")

        # Get the screen size
        screen_size = QDesktopWidget().screenGeometry()
        
        # Define initial width for title plots
        self.title_plot_initial_width = int(screen_size.width() * 0.040)

        # Initial plot height proportion
        plot_height_proportion = 0.075  # 7.5% of the screen height

        # Calculate the initial maximum height for the plots
        self.base_plot_height = int(screen_size.height() * plot_height_proportion)

        button_min_width = int(screen_size.width() * 0.05)
        button_max_width = int(screen_size.width() * 0.1)

        # Main layout
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(self.main_layout)

        # Left layout for controls and plots
        self.left_layout = QtWidgets.QVBoxLayout()
        self.left_layout.setAlignment(QtCore.Qt.AlignTop)  # Align elements to the top
        self.main_layout.addLayout(self.left_layout)
        self.left_layout.setSpacing(0)

        # Controls layout (top)
        self.controls_layout = QtWidgets.QHBoxLayout()
        self.left_layout.addLayout(self.controls_layout)

        # Playback controls
        self.play_button = QtWidgets.QPushButton("Play")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.reset_button = QtWidgets.QPushButton("Reset")

        # Set minimum width for playback buttons
        self.play_button.setMinimumWidth(button_min_width)
        self.play_button.setMaximumWidth(button_max_width)
        self.stop_button.setMinimumWidth(button_min_width)
        self.stop_button.setMaximumWidth(button_max_width)

        self.reset_button.setMinimumWidth(button_min_width)
        self.reset_button.setMaximumWidth(button_max_width)

        # Playback clock label
        self.playback_clock = QtWidgets.QLabel("00:00:00.000")
        self.playback_clock.setStyleSheet("""
            font-size: 24px;  /* Increase font size */
            font-family: 'Courier New', Courier, monospace;  /* Use a monospaced font */
            font-weight: bold;  /* Make the font bolder */
        """)

        # Add controls to the controls layout
        self.controls_layout.addWidget(self.play_button)
        self.controls_layout.addWidget(self.stop_button)
        self.controls_layout.addWidget(self.reset_button)
        # Add padding after stop button
        self.controls_layout.addSpacing(10)

        self.controls_layout.addWidget(self.playback_clock)
        self.controls_layout.addSpacerItem(QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        ))

        # Waveform plot (top)
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setMaximumHeight(self.base_plot_height)
        self.waveform_plot.getViewBox().setMouseEnabled(y=False)  # Disable y-axis zoom
        self.waveform_plot.hideAxis('left')  # Hide Y-axis

        # -- NEW CODE: Create a title plot for the waveform and a horizontal layout
        self.waveform_title_plot = pg.PlotWidget()
        self.waveform_title_plot.setMaximumHeight(self.base_plot_height)
        self.waveform_title_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.waveform_title_plot.hideAxis('bottom')
        self.waveform_title_plot.hideAxis('left')
        self.waveform_title_plot.setFixedWidth(self.title_plot_initial_width)  # Set initial width
        
        waveform_layout = QtWidgets.QHBoxLayout()
        waveform_layout.setSpacing(0)  # Remove space between plots
        waveform_layout.addWidget(self.waveform_title_plot)
        waveform_layout.addWidget(self.waveform_plot)
        self.left_layout.addLayout(waveform_layout)
        # -- END NEW CODE

        # Event plot (bottom)
        self.event_plot = pg.PlotWidget()
        self.event_plot.setMaximumHeight(self.base_plot_height)
        self.waveform_plot.setXLink(self.event_plot)
        self.event_plot.getViewBox().setMouseEnabled(y=False)  # Disable y-axis zoom
        self.event_plot.hideAxis('left')  # Hide Y-axis
        self.event_plot.hideAxis('bottom')  # Hide x-axis

        # -- NEW CODE: Create a title plot for event rows and a horizontal layout
        self.event_title_plot = pg.PlotWidget()
        self.event_title_plot.setMaximumHeight(self.base_plot_height)  # Match initial height
        self.event_title_plot.getViewBox().setMouseEnabled(x=False, y=False)
        self.event_title_plot.hideAxis('bottom')
        self.event_title_plot.hideAxis('left')
        self.event_title_plot.setFixedWidth(self.title_plot_initial_width)  # Set initial width

        # Link the y-axis of event_plot and event_title_plot
        self.event_plot.setYLink(self.event_title_plot)

        event_layout = QtWidgets.QHBoxLayout()
        event_layout.setSpacing(0)  # Remove space between plots
        event_layout.addWidget(self.event_title_plot)
        event_layout.addWidget(self.event_plot)
        self.left_layout.addLayout(event_layout)
        # -- END NEW CODE

        # Enable panning and zooming on the x-axis
        self.waveform_plot.getViewBox().setMouseEnabled(x=True, y=False)
        self.event_plot.getViewBox().setMouseEnabled(x=True, y=False)

        # Create a vertical playhead line for each plot
        self.playhead_waveform_line = pg.InfiniteLine(
            angle=90,
            movable=False, 
            pen=pg.mkPen(color='r', width=2)
        )
        self.playhead_event_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color='r', width=2)
        )
        self.playhead_event_info_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color='r', width=2),
        )

        # Add playhead lines to the plots
        self.waveform_plot.addItem(self.playhead_waveform_line)
        self.event_plot.addItem(self.playhead_event_line)

        self.left_layout.addSpacerItem(QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        ))

        # Navigation buttons
        self.next_event_button = QtWidgets.QPushButton("Next Event")
        self.previous_event_button = QtWidgets.QPushButton("Previous Event")

        # Set minimum and maximum width for navigation buttons
        self.next_event_button.setMinimumWidth(button_min_width)
        self.previous_event_button.setMinimumWidth(button_min_width)
        self.next_event_button.setMaximumWidth(button_max_width)
        self.previous_event_button.setMaximumWidth(button_max_width)

        # Add navigation buttons to the controls layout
        self.controls_layout.addWidget(self.previous_event_button)
        self.controls_layout.addWidget(self.next_event_button)

        # Build the right layout
        self.right_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

        # EVENT INFO LAYOUT
        # Right layout for event info
        self.event_info_layout = QtWidgets.QVBoxLayout()
        self.right_layout.addLayout(self.event_info_layout)

        self.event_info_waveform_plot = pg.PlotWidget()
        self.event_info_waveform_plot.getViewBox().setMouseEnabled(y=False)  # Disable y-axis zoom

        self.right_layout.addWidget(self.event_info_waveform_plot)
        self.event_info_waveform_plot.setMaximumHeight(self.base_plot_height)
        self.event_info_waveform_plot.addItem(self.playhead_event_info_line)

        # Classification dropdown
        self.classification_dropdown = QtWidgets.QComboBox(self)
        self.classification_dropdown.setEditable(True)  # Allow text input
        self.classification_dropdown.currentIndexChanged.connect(self.on_classification_selected)

        self.classification_header = QtWidgets.QHBoxLayout()
        self.save_classification_button = QtWidgets.QPushButton("Save Classification")
        self.save_classification_button.clicked.connect(self.on_save_classification)

        self.classification_header.addWidget(QtWidgets.QLabel("Classification:"))
        self.classification_header.addWidget(self.save_classification_button)

        self.event_info_layout.addLayout(self.classification_header)
        self.event_info_layout.addWidget(self.classification_dropdown)

        # Event time layout
        self.event_time_layout = QtWidgets.QHBoxLayout()
        self.event_time_label = QtWidgets.QLabel("Event Time (s):")
        self.event_time_edit = QtWidgets.QLineEdit()

        self.event_time_layout.addWidget(self.event_time_label)
        self.event_time_layout.addWidget(self.event_time_edit)

        self.event_info_layout.addLayout(self.event_time_layout)

        # Create a horizontal layout for the event playback controller
        self.event_buttons_layout = QtWidgets.QHBoxLayout()

        # Add buttons to play/stop the event snippet.
        self.play_event_button = QtWidgets.QPushButton("Play Event")
        self.stop_event_button = QtWidgets.QPushButton("Stop Event")

        # Set minimum width for playback buttons
        self.play_event_button.setMinimumWidth(button_min_width)
        self.stop_event_button.setMinimumWidth(button_min_width)
        self.play_event_button.setMaximumWidth(button_max_width)
        self.stop_event_button.setMaximumWidth(button_max_width)

        self.event_info_playback_clock = QtWidgets.QLabel("00:00:00.000")
        self.event_info_playback_clock.setStyleSheet("""
            font-size: 18px;  /* Increase font size */
            font-family: 'Courier New', Courier, monospace;  /* Use a monospaced font */
            font-weight: bold;  /* Make the font bolder */
        """)

        # Add the elements to the layout
        self.event_buttons_layout.addWidget(self.play_event_button)
        self.event_buttons_layout.addWidget(self.stop_event_button)
        self.event_buttons_layout.addWidget(self.event_info_playback_clock)

        # add the sub layout to the event info layout
        self.event_info_layout.addLayout(self.event_buttons_layout)

        # Add a spacer item to push everything to the top
        self.right_layout.addSpacerItem(QtWidgets.QSpacerItem(
            0, 0, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        ))

        self.main_layout.setStretch(0, 3)  # Left layout takes 75%
        self.main_layout.setStretch(1, 1)  # Right layout takes 25%
        
        # Add keyboard shortcuts
        self.add_shortcuts()

        # Add tracking for the currently highlighted point
        self.current_highlighted_point = None
        self.default_point_brush = pg.mkBrush(150, 150, 150, 255)  # Default gray color
        self.highlight_point_brush = pg.mkBrush(255, 165, 0, 255)  # Orange color

        # Add tracking for highlighted layer title
        self.current_highlighted_title = None
        self.default_title_color = 'w'  # Default white color
        self.highlight_title_color = '#FFA500'  # Orange color to match point highlight

        self.selected_event_points = []  # Track multiple highlighted points

    def add_shortcuts(self):
        """
        Adds keyboard shortcuts for navigation and playback controls.
        """
        # Keep existing event navigation shortcuts
        next_event_shortcut = QShortcut(QKeySequence(QtCore.Qt.Key_Right), self)
        next_event_shortcut.activated.connect(self.next_event_shortcut_activated)

        previous_event_shortcut = QShortcut(QKeySequence(QtCore.Qt.Key_Left), self)
        previous_event_shortcut.activated.connect(self.previous_event_shortcut_activated)

        # Keep existing playback shortcuts
        play_stop_shortcut = QShortcut(QKeySequence(QtCore.Qt.Key_Space), self)
        play_stop_shortcut.activated.connect(self.toggle_play_stop_shortcut_activated)

        event_play_stop_shortcut = QShortcut(QKeySequence("Shift+Space"), self)
        event_play_stop_shortcut.activated.connect(self.toggle_event_play_stop_shortcut_activated)

        # Update layer navigation shortcuts to be more intuitive
        up_layer_shortcut = QShortcut(QKeySequence(QtCore.Qt.Key_Up), self)
        up_layer_shortcut.activated.connect(self.up_layer_shortcut_activated)

        down_layer_shortcut = QShortcut(QKeySequence(QtCore.Qt.Key_Down), self)
        down_layer_shortcut.activated.connect(self.down_layer_shortcut_activated)

    def toggle_play_stop(self):
        """
        Toggles between play and pause states.
        """
        # Check if the play button is enabled, indicating that playback is stopped or paused
        if self.play_button.isEnabled():
            self.play_button.click()  # Start playback
        else:
            self.stop_button.click()  # Pause playback

    @QtCore.pyqtSlot(int)
    def update_playhead(self, position_ms):
        # Update the playhead position and any other UI elements
        # Log.info(f"UPDATE_PLAYHEAD CALLBACK: updated to {position_ms}")
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
        # Update the fields with event data
        classification = event.get_classification()
        if classification is None:
            classification = "None"  # or any default value you prefer
        self.classification_dropdown.setCurrentText(classification)

        # update the event time edit box
        self.event_time_edit.setText(str(event.get_time()))

        # Update the waveform plot
        audio_data = event.get()
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
        # Log.info(f"Updating event info playhead to {position_s} seconds")  # Debugging log
        self.playhead_event_info_line.setPos(position_ms)

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

    def on_classification_selected(self, index):
        # Emit the custom signal with the selected text
        selected_text = self.classification_dropdown.itemText(index)
        self.classification_changed.emit(selected_text)

    def on_save_classification(self):
        # Emit the custom signal with the new text
        self.classification_changed.emit(self.classification_dropdown.currentText())

    def update_classification_dropdown(self, classifications):
        """
        Updates the classification dropdown with the provided classifications.
        """
        self.classification_dropdown.clear()  # Clear existing items
        self.classification_dropdown.addItems(classifications)  # Add new items

    def set_event_plot_limits(self, x_max, y_max):
        self.event_plot.getViewBox().setLimits(
            xMin=0,
            xMax=x_max,
            yMin=0,
            yMax=(y_max + 1)
        ) 
        # Adjust the maximum height of the event plot based on the number of rows
        new_max_height = self.base_plot_height + y_max * 25
        self.event_plot.setMaximumHeight(new_max_height)
        self.event_title_plot.setMaximumHeight(new_max_height)  # Ensure the title plot matches the event plot height

        # Set the same y-axis limits for event_title_plot
        self.event_title_plot.getViewBox().setLimits(
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

    def connect_signals(self, play_callback, stop_callback, reset_callback, play_event_callback=None, stop_event_callback=None, next_event_callback=None, previous_event_callback=None):
        """
        Connects UI signals to callbacks provided by the editor block.
        """
        self.play_button.clicked.connect(play_callback)
        self.stop_button.clicked.connect(stop_callback)
        self.reset_button.clicked.connect(reset_callback)
        # self.event_plot.scene().sigMouseClicked.connect(event_click_callback)

        # New event callbacks (guard for None if not provided)
        if play_event_callback:
            self.play_event_button.clicked.connect(play_event_callback)
        if stop_event_callback:
            self.stop_event_button.clicked.connect(stop_event_callback)
                # Connect next and previous event buttons to callbacks
        if next_event_callback:
            self.next_event_button.clicked.connect(next_event_callback)
        if previous_event_callback:
            self.previous_event_button.clicked.connect(previous_event_callback)

    def highlight_event_points(self, events, event_layer):
        """
        Highlights the specified event points in orange and reverts the previous highlights.
        """
        scatter_item = None
        for item in self.event_plot.items():
            if isinstance(item, pg.ScatterPlotItem):
                scatter_item = item
                break

        if scatter_item is None:
            return

        # Reset previous highlights if they exist
        for point_info in self.selected_event_points:
            prev_point = scatter_item.points()[point_info['index']]
            prev_point.setBrush(self.default_point_brush)

        # Clear the list of selected event points
        self.selected_event_points.clear()

        # Find and highlight the new points
        for i, point in enumerate(scatter_item.points()):
            point_data = point.data()
            for event in events:
                if (point_data['name'] == event.get_name() and 
                    point_data['layer'] == event_layer):
                    point.setBrush(self.highlight_point_brush)
                    self.selected_event_points.append({
                        'index': i,
                        'name': event.get_name(),
                        'layer': event_layer
                    })

    def highlight_layer_title(self, layer_name):
        """
        Highlights the title of the selected layer and reverts the previous highlight.
        """
        # Reset previous highlight if it exists
        if self.current_highlighted_title:
            prev_title = self.current_highlighted_title['text_item']
            prev_title.setColor(self.default_title_color)

        # Find and highlight the new title
        for item in self.event_title_plot.items():
            if isinstance(item, pg.TextItem):
                if item.textItem.toPlainText() == layer_name:
                    item.setColor(self.highlight_title_color)
                    self.current_highlighted_title = {
                        'text_item': item,
                        'layer': layer_name
                    }
                    break

    # def update_waveform_title(self, title):
    #     """
    #     Clear and display a text item in the waveform title plot.
    #     """
    #     self.waveform_title_plot.clear()
    #     text_item = pg.TextItem(text=title, anchor=(0, 0.5), color='w')
    #     text_item.setPos(0, 0)  # Position the text at (0,0)
    #     self.waveform_title_plot.addItem(text_item)

    # def update_event_titles(self, events):
    #     """
    #     Clear and display row labels (layer names) in the event title plot.
    #     Align with how we place horizontal lines in the main event_plot.
    #     """
    #     self.event_title_plot.clear()

    #     y_index = 1
    #     for event in events:
    #         # Draw a horizontal line to match the row separation
    #         line = pg.InfiniteLine(pos=y_index + 0.5, angle=0, pen=pg.mkPen('w', width=1))
    #         self.event_title_plot.addItem(line)

    #         # Place a text item for the layer name
    #         text_item = pg.TextItem(text=layer_name, anchor=(1, 0.5), color='w')
    #         text_item.setPos(0, y_index)
    #         self.event_title_plot.addItem(text_item)

    #         y_index += 1