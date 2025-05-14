from src.Project.Block.block import Block
from src.Utils.message import Log
from pythonosc import udp_client
import time
from src.Project.Block.Output.Types.osc_output import OSCOutput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Utils.tools import prompt

DEFAULT_FRAME_RATE: int = 30
DEFAULT_SEQUENCE_POOL: int = 1102
DEFAULT_TC_POOL: int = 101
DEFAULT_IP_ADDRESS: str = "10.0.0.174"
DEFAULT_PORT: int = 8080

class ExportMA3Block(Block):
    name = "ExportMA3"
    type = "ExportMA3"
    
    def __init__(self):
        super().__init__()
        self.name = "ExportMA3"
        self.type = "ExportMA3"
        self.frame_rate: int = DEFAULT_FRAME_RATE
        self.ip_address: str = DEFAULT_IP_ADDRESS
        self.ip_port: int = DEFAULT_PORT
        self.tc_pool: int = DEFAULT_TC_POOL
        self.sequence_pool: int = DEFAULT_SEQUENCE_POOL

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(OSCOutput)
        self.output.add("OSCOutput")

        

        # Initialize OSC Client
        self.osc_client = udp_client.SimpleUDPClient(self.ip_address, self.ip_port)  # Modify IP and port as needed

        # Register OSC-related commands
        self.command.add("send_ma_cuestack", self.batch_send_events)
        self.command.add("send_ma_hit_buttons", self.send_event_classes)
        self.command.add("test_connection", self.test_connection)
        self.command.add("set_ip", self.set_ip)
        self.command.add("set_port", self.set_port)
        self.command.add("set_tc_pool", self.set_tc_pool)
        self.command.add("set_sequence_pool", self.set_sequence_pool)


    def test_connection(self):
        self.send_osc_message(f"{self.ip_address}", "Test")

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

    def set_ip(self):
        self.ip_address = prompt("Enter MA3 OSC Server IP:")
        self.ip_address = str(self.ip_address)
        Log.info(f"Set IP: {self.ip_address}")

    def set_port(self):
        self.ip_port = prompt("Enter MA3 OSC Server Port:")
        self.ip_port = int(self.ip_port)
        Log.info(f"Set Port: {self.ip_port}")

    def set_tc_pool(self):
        self.tc_pool = prompt("Set Destination Timecode Pool: ")
        self.tc_pool = int(self.tc_pool)
        Log.info(f"Set TC Pool: {self.tc_pool}")

    def set_sequence_pool(self):
        self.sequence_pool = prompt("Set Destination Timecode Pool: ")
        self.sequence_pool = int(self.sequence_pool)
        Log.info(f"Set TC Pool: {self.sequence_pool}")

    def set_tc_framerate(self):
        self.frame_rate = prompt("Set Destination Timecode Pool: ")
        self.frame_rate = int(self.frame_rate)
        Log.info(f"Set TC Pool: {self.frame_rate}")

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
    
    def _ma_complete_message(self):
        self.send_osc_message(f"/cmd", f"(---------------COMPLETE---------------)")
    
    def send_event_classes(self):
        event_counter = 1
        self.send_osc_message(f"/cmd", f"Store Timecode {self.tc_pool}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property \"Cursor\" 0 /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property TCSlot 1")

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
                    self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property \"Cursor\" \"{event_item.time}\" /nc")
                    self.send_osc_message(f"/cmd", f"Record Timecode {self.tc_pool} /o /nc")
                    self.send_osc_message(f"/cmd", f"Go Sequence \"{event_item.classification}\" /nc")
                    self.send_osc_message(f"/cmd", f"Off TC {self.tc_pool} /nc")
                    self.send_osc_message(f"/cmd", f"Off Sequence \"{event_item.classification}\" /nc")
        self._ma_complete_message()
                    
    def batch_send_events(self):
        """
        Function will send all events connected to the input port, to MA in one sequence, each event being a cue in the sequence.
        """
        event_counter = 1
        self.send_osc_message(f"/cmd", f"Store Sequence {self.sequence_pool} Cue 0 \"MARK\"/o /nc")
        self.send_osc_message(f"/cmd", f"Store Timecode {self.tc_pool}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Assign Sequence {self.sequence_pool} at TC {self.tc_pool}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property \"Cursor\" 0 /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property TCSlot 1")
        self.send_osc_message(f"/cmd", f"Go Sequence {self.sequence_pool} Cue 0 /nc")
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                self.time = '%.2f'%(event_item.time)
                Log.info(f"TIME: {self.time}")
                cmd_str: str = f"Store Sequence {self.sequence_pool} Cue {event_counter} \"{event_item.classification}\" /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Set Timecode {self.tc_pool} Property \"Cursor\" \"{self.time}\" /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Record TC {self.tc_pool} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Go Sequence {self.sequence_pool} Cue {event_counter} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                cmd_str: str = f"Off TC {self.tc_pool} /nc"
                self.send_osc_message(f"/cmd", cmd_str)
                event_counter += 1
        self._ma_complete_message()
       
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

    