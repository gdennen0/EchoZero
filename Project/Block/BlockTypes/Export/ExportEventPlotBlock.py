from Project.Block.block import Block
from Utils.message import Log
from Utils.tools import prompt_selection, prompt
import os
import matplotlib.pyplot as plt
from Project.Data.Types.event_data import EventData
from Project.Block.Input.Types.event_input import EventInput

class ExportEventPlotBlock(Block):
    name = "ExportEventPlot"
    def __init__(self):
        super().__init__()
        # Set attributes
        self.name = "ExportEventPlot"
        self.type = "ExportEventPlot"

        # Add attributes
        self.file_type = "png"
        self.destination_path = "/tmp"
        self.supported_file_types = ["png", "jpg", "jpeg"]

        # Add port types and ports
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        # Add commands
        self.command.add("select_file_type", self.select_file_type)
        self.command.add("set_destination_path", self.set_destination_path)
        self.command.add("export", self.export)

        Log.info(f"{self.name} initialized with supported file types: {self.supported_file_types}")

    def select_file_type(self, file_type=None):
        """Command to select the file type for exporting event plot."""
        if file_type:
            if file_type in self.supported_file_types:
                self.file_type = file_type
                Log.info(f"Selected file type: {self.file_type}")
            else:
                Log.error(f"Unsupported file type passed as an argument: {file_type}")
        else:
            file_type = prompt_selection("Select the export event plot file type:", self.supported_file_types)
            self.file_type = file_type
            Log.info(f"Selected file type: {self.file_type}")

    def set_destination_path(self, path=None):
        """Command to set the destination path for the exported event plot."""
        if path:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                    Log.info(f"Created destination directory: {path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
                    return
            self.destination_path = path
            Log.info(f"Set destination path: {self.destination_path}")
        else:
            path = prompt("Enter destination path for the event plot: ")
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                    Log.info(f"Created destination directory: {path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
                    return
            self.destination_path = path
            Log.info(f"Set destination path: {self.destination_path}")

    def export(self):
        """Command to export the event plot based on settings."""
        if not self.data:
            Log.error("No event data available to export.")
            return
        if not self.file_type:
            Log.error("File type not selected.")
            return
        if not self.destination_path:
            Log.error("Destination path not set.")
            return

        for event_output in self.data.get_all():
            event_data = event_output.data
            plot_file_name = f"{event_output.name}_event_plot.{self.file_type}"
            plot_file_path = os.path.join(self.destination_path, plot_file_name)
            self.export_event_plot(event_data, plot_file_path)
            Log.info(f"Exported event plot to {plot_file_path}")

    def export_event_plot(self, event_data, path):
        """Export event data as a plot image."""
        if not isinstance(event_data, EventData):
            Log.error("Input data is not of type EventData.")
            return

        event_times = [item.time for item in event_data.items if item.time is not None]

        if not event_times:
            Log.error("No event times available to plot.")
            return

        plt.figure(figsize=(10, 4))
        plt.eventplot(event_times, lineoffsets=1, colors='b')
        plt.xlabel('Time (s)')
        plt.yticks([])
        plt.title('Event Plot')
        plt.tight_layout()
        try:
            plt.savefig(path)
            Log.debug(f"Event plot saved to {path}")
        except Exception as e:
            Log.error(f"Failed to save event plot: {e}")
        finally:
            plt.close()

    def process(self, input_data):
        # Pass through data without modification
        return input_data

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "destination_path": self.destination_path,
            "supported_file_types": self.supported_file_types,
            "input": self.input.save(),
            "output": self.output.save()
        }

    def load(self, data):
        self.file_type = data.get("file_type", "png")
        self.destination_path = data.get("destination_path", "/tmp")
        self.supported_file_types = data.get("supported_file_types", ["png", "jpg", "jpeg"])
        self.input.load(data.get("input"))
        self.output.load(data.get("output"))

