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
DEFAULT_IP_ADDRESS: str = "192.168.1.138"
DEFAULT_PORT: int = 8080
DEFAULT_TC_POOL_NAME: str = "Song"

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
        self.tc_pool_name: str = DEFAULT_TC_POOL_NAME

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(OSCOutput)
        self.output.add("OSCOutput")

        

        # Initialize OSC Client
        self.start_osc_client()

        # Register OSC-related commands
        self.command.add("send_ma_cuestack", self.batch_send_events)
        self.command.add("send_ma_hit_buttons", self.send_event_classes)
        self.command.add("test_connection", self.test_connection)
        self.command.add("set_ip", self.set_ip)
        self.command.add("set_port", self.set_port)
        self.command.add("set_tc_pool", self.set_tc_pool)
        self.command.add("set_tc_pool_name", self.set_tc_pool_name)
        self.command.add("set_sequence_pool", self.set_sequence_pool)

    def start_osc_client(self):
        self.osc_client = udp_client.SimpleUDPClient(self.ip_address, self.ip_port)  # Modify IP and port as needed

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

    def set_ip(self, ip_address=None):
        if ip_address is None:
            self.ip_address = prompt(f"Enter new MA3 OSC Server IP, current ({self.ip_address})")
            self.ip_address = str(self.ip_address)
        else:
            self.ip_address = ip_address
        Log.info(f"Set IP: {self.ip_address}")
        self.start_osc_client()

    def set_port(self, ip_port=None):
        if ip_port is None:
            self.ip_port = prompt(f"Enter new MA3 OSC Server Port, current ({self.ip_port})")
            self.ip_port = int(self.ip_port)
        else:
            self.ip_port = ip_port
        Log.info(f"Set Port: {self.ip_port}")
        self.start_osc_client()

    def set_tc_pool(self, tc_pool=None):
        if tc_pool is None:
            self.tc_pool = prompt(f"Set Destination Timecode Pool, current ({self.tc_pool})")
            self.tc_pool = int(self.tc_pool)
        else:
            self.tc_pool = tc_pool
        Log.info(f"Set TC Pool: {self.tc_pool}")

    def set_tc_pool_name(self, tc_pool_name=None):
        if tc_pool_name is None:
            self.tc_pool_name = prompt(f"Set Destination Timecode Pool Name, current ({self.tc_pool_name})")
        else:
            self.tc_pool_name = tc_pool_name
        Log.info(f"Set TC Pool Name: {self.tc_pool_name}")

    def set_sequence_pool(self, sequence_pool=None):
        if sequence_pool is None:
            self.sequence_pool = prompt(f"Set Destination Sequence Pool, current ({self.sequence_pool})")
            self.sequence_pool = int(self.sequence_pool)
        else:
            self.sequence_pool = sequence_pool
        Log.info(f"Set sequence pool: {self.sequence_pool}")

    def set_tc_framerate(self, frame_rate=None):
        if frame_rate is None:
            self.frame_rate = prompt(f"Set Destination Timecode Frame Rate, current ({self.frame_rate})")
            self.frame_rate = int(self.frame_rate)
        else:
            self.frame_rate = frame_rate
        Log.info(f"Set frame rate: {self.frame_rate}")

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
            "ip_address": self.ip_address,
            "ip_port": self.ip_port,
            "tc_pool": self.tc_pool,
            "tc_pool_name": self.tc_pool_name,
            "sequence_pool": self.sequence_pool,
            "frame_rate": self.frame_rate,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def _ma_complete_message(self):
        # self.send_osc_message(f"/cmd", f"(---------------COMPLETE---------------)")
        Log.info(f"(---------------COMPLETE---------------)")
    
    def send_event_classes(self):
        event_counter = 1
        self.send_osc_message(f"/cmd", f"Store Timecode {self.tc_pool}.1.1 /o /nc")
        self.send_osc_message(f"/cmd", f"Label Timecode {self.tc_pool} \"{self.tc_pool_name}\" /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property \"Cursor\" 0 /nc")
        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property TCSlot 1")

        sequence_classes = []
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if event_item.classification not in sequence_classes:
                    sequence_classes.append(event_item.classification)

        sub_sequence_counter = int(self.sequence_pool)
        sub_sequence_classes = {}
        for classes in sequence_classes:
            sub_sequence_classes[classes] = sub_sequence_counter 
            self.send_osc_message(f"/cmd", f"Store Sequence {sub_sequence_counter} /o /nc")
            self.send_osc_message(f"/cmd", f"Label Sequence {sub_sequence_counter} \"{classes}_{self.tc_pool_name}\"")
            sub_sequence_counter = sub_sequence_counter + 1

        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                for sub_sequence_class_name, sequence_number in sub_sequence_classes.items():
                    if sub_sequence_class_name == event_item.classification:
                        self.send_osc_message(f"/cmd", f"Set TC {self.tc_pool} Property \"Cursor\" \"{event_item.time}\" /nc")
                        self.send_osc_message(f"/cmd", f"Record Timecode {self.tc_pool} /o /nc")
                        self.send_osc_message(f"/cmd", f"Go Sequence {sequence_number} /nc")
                        self.send_osc_message(f"/cmd", f"Off TC {self.tc_pool} /nc")
                        self.send_osc_message(f"/cmd", f"Off Sequence {sequence_number} /nc")
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
        self.set_ip(block_metadata.get("ip_address"))
        self.set_port(block_metadata.get("ip_port"))
        self.set_tc_pool(block_metadata.get("tc_pool"))
        self.set_tc_pool_name(block_metadata.get("tc_pool_name"))
        self.set_sequence_pool(block_metadata.get("sequence_pool"))
        self.set_tc_framerate(block_metadata.get("frame_rate"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())

    
