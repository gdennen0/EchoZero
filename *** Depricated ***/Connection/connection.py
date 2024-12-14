from Utils.message import Log
from Utils.tools import prompt
from Save.save_controller import SaveController

class Connection:
    def __init__(self):
        self.input_port = None
        self.output_port = None
        self.name = "Connection"

        self.save = SaveController(self)
        self.save.set_recursive_depth(1) # only save the ports, not the ports' connections which would feedback 5ever
        self.save.add_attribute("name")
        self.save.add_sub_module("input_port")
        self.save.add_sub_module("output_port")

    def set_name(self, name=None):
        if name:
            self.name = name
        else:
            self.name = prompt(f"Please enter a name for the connection: ")
            Log.info(f"Connection name set to: {self.name}")
        
    def set_parent_block(self, parent):
        self.parent = parent
        Log.info(f"Connection {self.name} parent set to: {parent.name}")


    def set_description(self, description=None):
        if description:
            self.description = description
        else:
            self.description = prompt(f"Please enter a description for the connection: ")
            Log.info(f"Connection description set to: {self.description}")

    def set_input_port(self, input):
        if input.type == "input":
            self.input_port = input
            Log.info(f" '{self.name}' Connection input port set to: {input.name}")
        else:
            Log.error(f"Invalid input port type '{input.type}' for '{input.name}' connection")

    def set_output_port(self, output ):
        if output.type == "output": 
            self.output_port = output
            Log.info(f"'{self.name}' Connection output port set to: {output.name}")
        else:
            Log.error(f"Invalid output port  type '{output.type}' for '{output.name}' connection")

    

