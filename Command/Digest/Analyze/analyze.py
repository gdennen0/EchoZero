from Command.Digest.PointTypes.onset import Onset
from Command.command_item import CommandItem
from message import Log
from tools import prompt

class Analyze:
    def __init__(self):
        self.points = []
        self.point_types = [
            Onset(),
        ]
        self.commands = []
        self.sub_modules = []
        self.name = "Analyze"

        self.add_command("start", self.start)
        self.add_command("add", self.add)
        self.add_command("list_point_types", self.list_point_types)
        self.add_command("list_points", self.list_points)

    def add_command(self, name, command):
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        self.commands.append(cmd_item)

    def add_sub_module(self, sub_module):
        self.sub_modules.append(sub_module)

    def add(self, point_object_type_index=None):
        while True:
            try:
                self.list_point_types()
                point_object_type_index = int(prompt("What is the index of the Analyze point type you'd like to add? "))
                if point_object_type_index < len(self.point_types):
                    break
                else:
                    Log.error("Invalid index. Please try again.")
            except ValueError:
                Log.error("Invalid input. Please enter a valid integer.")

        point_type = self.point_types[point_object_type_index]
        self.points.append(point_type)
        self.list_point_types()
        Log.info(f"Added point object: '{point_type.type}' into Analyze")


    def list_point_types(self):
        Log.list("Available point types: ", self.point_types, atrib="type")

    def list_points(self):
        Log.list("Active point objects: ", self.points, atrib="type")

    def start(self, d): # pass start the audio data
        result_table = []
        Log.info("Initializing analysis of data")
        if len(self.points) >= 1:
            for index, point in enumerate(self.points):
                Log.info(f"Point Index: {index}, Name: {point.name}")
                data = point.apply(d)
                result_object = Result(data, point.type) # Standardizing the result object so it can always be understood properly
                result_table.append(result_object)
            Log.info("Analysis complete")
            return result_table
        else: 
            Log.error("There are no points in the analyze instance")
            result_object = Result(d, "Audio") # Standardizing the result object so it can always be understood properly
            result_table.append(result_object)
            return result_table
        
class Result:
    def __init__(self, data, type):
        self.data = data
        self.type = type