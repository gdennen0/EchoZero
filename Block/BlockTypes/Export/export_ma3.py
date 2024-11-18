from Block.block import Block
from Connections.port_types.event_port import EventPort
from message import Log
from tools import prompt_selection, prompt_selection_with_type, prompt
import os
from pydub import AudioSegment
import soundfile as sf
from argparse import ArgumentParser
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


DEFAULT_EXPORT_IP = "127.0.0.1"
DEFAULT_EXPORT_IP_PORT = 8000

class ExportMA3Block(Block):
    def __init__(self):
        super().__init__()
        self.name = "ExportMa3"
        self.type = "ExportMA3"

        self.tc_pool = None
        self.seq_pool = None
        
        self.framerate = None # MA TC Framerate

        self.ip = DEFAULT_EXPORT_IP
        self.ip_port = DEFAULT_EXPORT_IP_PORT
        self.client = None
        self.cmd = None

        # Add commands
        self.add_command("select_timecode_pool", self.select_timecode_pool)
        self.add_command("select_sequence_pool", self.select_sequence_pool)
        self.add_command("export", self.export)
        self.add_command("reload", self.reload)

        # Add port types and ports
        self.add_port_type(EventPort)
        self.add_input_port("EventPort")
        # self.add_output_port("EventPort")

        #Log.info(f"{self.name} initialized with supported file types:")

    def select_timecode_pool(self, pool_int=None):
        """Command to select TC Pool"""
        if pool_int:
            self.tc_pool = pool_int
        else:
            self.tc_pool = prompt("Enter a Timecode Pool as Whole Integer: ")

    def select_sequence_pool(self, pool_int=None):
        """Command to select Sequence Pool"""
        if pool_int:
            self.seq_pool = pool_int
        else:
            self.seq_pool = prompt("Enter a Sequence Pool as a Whole Integer: ")

    def export(self):
        """Command to export events to MA3."""
        if not self.data:
            Log.error("No audio data available to export.")
            return
        if not self.file_type:
            Log.error("File type not selected.")
            return
        if not self.destination_path:
            Log.error("Destination path not set.")
            return

        # Construct the export file name
        export_file_name = f"{self.name}_export.{self.file_type}"
        export_file_path = os.path.join(self.destination_path, export_file_name)

        try:
            # Export logic based on file type
            if self.file_type == "wav":
                self.export_wav(export_file_path)
            elif self.file_type == "mp3":
                self.export_mp3(export_file_path)
            elif self.file_type == "flac":
                self.export_flac(export_file_path)
            elif self.file_type == "aac":
                self.export_aac(export_file_path)
            else:
                Log.error(f"Unsupported file type: {self.file_type}")
                return

            Log.info(f"Exported audio to {export_file_path}")
        except Exception as e:
            Log.error(f"Failed to export audio: {e}")

    def reload(self):
        """Reload the block's data."""
        super().reload()
        Log.info(f"{self.name} reloaded successfully.")

    def establish_connection(self):
        # Code to establish initial connection of data to MA... export func is used to update data from EZ -> MA
        pass

    def sec_to_events(self, framerate=None):
        # Convert seconds to min, sec, frames
        pass

    def get_ma_framerate(self):
        self.send_lua()
        
        def handle_framerate(unused_addr, args, framerate):
            try:
                # Attempt to convert framerate to an integer
                int_framerate = int(framerate)
                self.framerate = int_framerate
                print(f"Framerate set to {self.framerate}")
                server.server_close()  # Close the server if the conversion is successful
                print("Server closed successfully.")
            except ValueError:
                # Handle the case where framerate is not an integer
                print("Received framerate is not an integer.")

        parser = ArgumentParser()
        parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
        parser.add_argument("--port", type=int, default=8000, help="The port to listen on")
        args = parser.parse_args()

        dispatcher = Dispatcher()
        dispatcher.map("/framerate", handle_framerate, "Framerate")

        global server
        server = ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
        print(f"Serving on {server.server_address}")
        server.serve_forever()
    
    def send_lua(self):
        luacmd:str = """return function() DataPool().Timecodes[101]:Dump(); local tc = DataPool().Timecodes[101]; local tc_framerate = tc.FrameReadOut; Printf("TC: "..tc_framerate); local framerate_number = tonumber(tc_framerate:match("%d+")); Cmd("SetUserVariable \"dp_framerate\" "..framerate_number); Cmd("SendOSC 1 \"/framerate,i," .. framerate_number.."\"") end"""
        cmd:str = f"Lua \"{luacmd}\""
        print(cmd)
        self.send_osc_cmd(cmd)

    def send_osc_cmd(self, cmd=None):
        self.cmd = cmd
        self.client = SimpleUDPClient(self.ip, self.ip_port)
        Log.info(f"OSC Client initialized at {self.ip}:{self.ip_port}")
        self.client.send_message("/cmd", self.cmd)





    def establish_lua_connection(self):
        pass



class osc_interaction():
    def __init__(self):
        self.tc_pool = None
        self.seq_pool = None
        self.framerate = None
        self.ip = DEFAULT_EXPORT_IP
        self.ip_port = DEFAULT_EXPORT_IP_PORT
        self.client = None
        self.cmd = None
    def sec_to_events(self, framerate=None):
        # Convert seconds to min, sec, frames
        pass

    def get_ma_framerate(self):
        self.send_lua()
        
        def handle_framerate(unused_addr, args, framerate):
            try:
                # Attempt to convert framerate to an integer
                int_framerate = int(framerate)
                self.framerate = int_framerate
                print(f"Framerate set to {self.framerate}")
                server.server_close()  # Close the server if the conversion is successful
                print("Server closed successfully.")
            except ValueError:
                # Handle the case where framerate is not an integer
                print("Received framerate is not an integer.")

        parser = ArgumentParser()
        parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
        parser.add_argument("--port", type=int, default=8000, help="The port to listen on")
        args = parser.parse_args()

        dispatcher = Dispatcher()
        dispatcher.map("/framerate", handle_framerate, "Framerate")

        global server
        server = ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
        print(f"Serving on {server.server_address}")
        server.serve_forever()
    
    def send_lua(self):
        luacmd = """
\"return function() local tc = DataPool().Timecodes[101]; local tc_framerate = tc.FrameReadOut; Printf('TC: ' .. tc_framerate); local framerate_number = tonumber(tc_framerate:match('%d+')); Cmd('SendOSC 1 \'/framerate,i,' .. framerate_number .. '\'') end\"
        """
        cmd:str = f"Lua {luacmd}"
        print(cmd)
        self.send_osc_cmd(cmd)

    def send_osc_cmd(self, cmd=None):
        self.cmd = cmd
        self.client = SimpleUDPClient(self.ip, self.ip_port)
        Log.info(f"OSC Client initialized at {self.ip}:{self.ip_port}")
        self.client.send_message("/cmd", self.cmd)


t_server = osc_interaction()

ma_framework = t_server.get_ma_framerate()