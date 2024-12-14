from Project.Block.block import Block
from Utils.message import Log
from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread
from Project.Block.Input.Types.osc_input import OSCInput
from Project.Block.Output.Types.osc_output import OSCOutput

class OSCBlock(Block):
    name = "OSCBlock"
    def __init__(self):
        super().__init__()
        self.name = "OSCBlock"
        self.type = "OSCBlock"

        self.input.add_type(OSCInput)
        self.input.add("OSCInput")
        self.output.add_type(OSCOutput)
        self.output.add("OSCOutput")

        # Initialize OSC Client
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # Modify IP and port as needed

        # Initialize OSC Dispatcher and Server
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.set_default_handler(self.osc_message_handler)

        self.osc_server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9000), self.dispatcher)  # Modify IP and port as needed

        # Start OSC server in a separate thread
        self.server_thread = Thread(target=self.osc_server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        Log.info(f"OSC Server started on {self.osc_server.server_address}")

        # Register OSC-related commands
        self.command.add("send_osc", self.send_osc_message)

    def osc_message_handler(self, address, *args):
        Log.info(f"Received OSC message - Address: {address}, Arguments: {args}")
        # Implement custom handling based on OSC address
        if address == "/reload":
            self.reload()
        elif address == "/connect":
            if len(args) >= 2:
                input_name, output_name = args[:2]
                self.connect_ports(input_name, output_name)
        # Add more handlers as needed

    def send_osc_message(self, address, *args):
        Log.info(f"Sending OSC message - Address: {address}, Arguments: {args}")
        self.osc_client.send_message(address, args)

    def connect_ports(self, input_name, output_name):
        input_port = self.input.get(input_name)
        output_port = self.output.get(output_name)
        if input_port and output_port:
            input_port.connect(output_port)
            Log.info(f"Connected input '{input_name}' to output '{output_name}' via OSCBlock.")
        else:
            Log.error(f"Failed to connect. Input: {input_name}, Output: {output_name}")

    def process(self, input_data):
        # OSCBlock may not need to process input data, but implement if necessary
        processed_data = input_data
        return processed_data

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "data": self.data.save(),
            "input": self.input.save(),
            "output": self.output.save(),
            # Add any OSC-specific settings here
        }

    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.data.load(data.get("data"))
        self.input.load(data.get("input"))
        self.output.load(data.get("output"))
        # Load any OSC-specific settings here
        Log.info(f"OSCBlock '{self.name}' loaded successfully.")