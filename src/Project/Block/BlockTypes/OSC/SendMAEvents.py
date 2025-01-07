from src.Project.Block.block import Block
from src.Utils.message import Log
from pythonosc import dispatcher, osc_server, udp_client
import time
from threading import Thread
from src.Project.Block.Input.Types.osc_input import OSCInput
from src.Project.Block.Output.Types.osc_output import OSCOutput
from src.Project.Block.Input.Types.event_input import EventInput

DEFAULT_FRAME_RATE = 30
DEFAULT_SEQ_INT = 1

class SendMAEvents(Block):
    name = "SendMAEvents"
    type = "SendMAEvents"
    
    def __init__(self):
        super().__init__()
        self.name = "SendMAEvents"
        self.type = "SendMAEvents"
        self.frame_rate = DEFAULT_FRAME_RATE

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(OSCOutput)
        self.output.add("OSCOutput")

        # Initialize OSC Client
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 8000)  # Modify IP and port as needed

        self.send_osc_message("/cmd", "test")

        # Initialize OSC Dispatcher and Server
        #self.dispatcher = dispatcher.Dispatcher()
        #self.dispatcher.set_default_handler(self.osc_message_handler)

        #self.osc_server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 9000), self.dispatcher)  # Modify IP and port as needed

        # Start OSC server in a separate thread
        #self.server_thread = Thread(target=self.osc_server.serve_forever)
        #self.server_thread.daemon = True
        #self.server_thread.start()
        #Log.info(f"OSC Server started on {self.osc_server.server_address}")

        # Register OSC-related commands
        self.command.add("send_osc", self.send_osc_message)
        self.command.add("batch_send_events", self.batch_send_events)
        self.command.add("send_event_classes", self.send_event_classes)

        

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
        time.sleep(0.008)
        Log.info(f"Sending OSC message - Address: {address}, Arguments: {args}")
        self.osc_client.send_message(address, args)
        

    def connect_ports(self, input_name, output_name):
        input_port = self.input.get(input_name)
        output_port = self.output.get(output_name)
        if input_port and output_port:
            input_port.connect(output_port)
            Log.info(f"Connected input '{input_name}' to output '{output_name}' via SendMAEvents.")
        else:
            Log.error(f"Failed to connect. Input: {input_name}, Output: {output_name}")

    def process(self, input_data):
        # SendMAEvents may not need to process input data, but implement if necessary
        return input_data 

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def send_event_classes(self):
        event_counter = 1
        self.send_osc_message(f"/cmd", f"Store Timecode {DEFAULT_SEQ_INT}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Set TC {DEFAULT_SEQ_INT} Property \"Cursor\" 0 /nc")
        self.send_osc_message(f"/cmd", f"Set TC {DEFAULT_SEQ_INT} Property TCSlot 1")

        sequence_classes = []
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if event_item.classification not in sequence_classes:
                    sequence_classes.append(event_item.classification)

        for classes in sequence_classes:
            self.send_osc_message(f"/cmd", f"Store Sequence \"{classes}\" /o /nc")

        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if event_item.classification in sequence_classes:
                    self.send_osc_message(f"/cmd", f"Set TC {DEFAULT_SEQ_INT} Property \"Cursor\" \"{event_item.time}\" /nc")
                    self.send_osc_message(f"/cmd", f"Record Timecode {DEFAULT_SEQ_INT} /o /nc")
                    self.send_osc_message(f"/cmd", f"Go Sequence \"{event_item.classification}\" /nc")
                    self.send_osc_message(f"/cmd", f"Off TC {DEFAULT_SEQ_INT} /nc")
                    self.send_osc_message(f"/cmd", f"Off Sequence \"{event_item.classification}\" /nc")

                    
    def batch_send_events(self):
        """
        Function will send all events connected to the input port, to MA in one sequence, each event being a cue in the sequence.
        """
        event_counter = 1
        self.send_osc_message(f"/cmd", f"Store Sequence {DEFAULT_SEQ_INT} Cue 0 \"MARK\"/o /nc")
        self.send_osc_message(f"/cmd", f"Store Timecode {DEFAULT_SEQ_INT}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Assign Sequence {DEFAULT_SEQ_INT} at TC {DEFAULT_SEQ_INT}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Set TC {DEFAULT_SEQ_INT} Property \"Cursor\" 0 /nc")
        self.send_osc_message(f"/cmd", f"Set TC {DEFAULT_SEQ_INT} Property TCSlot 1")
        self.send_osc_message(f"/cmd", f"Go Sequence {DEFAULT_SEQ_INT} Cue 0 /nc")
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                self.time = '%.2f'%(event_item.time)
                Log.info(f"TIME: {self.time}")
                cmd_str: str = f"Store Sequence {DEFAULT_SEQ_INT} Cue {event_counter} \"{event_item.classification}\" /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Set Timecode {DEFAULT_SEQ_INT} Property \"Cursor\" \"{self.time}\" /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Record TC {DEFAULT_SEQ_INT} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Go Sequence {DEFAULT_SEQ_INT} Cue {event_counter} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Off TC {DEFAULT_SEQ_INT} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                event_counter += 1
       
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

    