from src.Project.Block.block import Block
from src.Utils.message import Log
from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread
from src.Project.Block.Input.Types.osc_input import OSCInput
from src.Project.Block.Output.Types.osc_output import OSCOutput

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

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())