from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from argparse import ArgumentParser


from message import Log
from tools import prompt

class OSC_Connection():
    def __init__(self, ip, ip_port):
        self.osc_ready:bool = False
        self.ip:str = ip
        self.ip_port:str = ip_port

        self.tc_pool:str = None
        self.seq_pool:int = None
        self.framerate:int = None

        self.client:str = None
        self.cmd:str = None

        self.handlers:str = self._setup_osc_handlers()
        self.server:str = None
        self.osc_parser:str = None
        self.osc_args:str = None

        self.keep_running = True  # Flag to control the server running state

    def check_variables(self):
        while True:
            if self.client:
                if self.ip:
                    if self.ip_port:
                        if self.tc_pool:
                            if self.seq_pool != None:
                                if self.framerate:
                                    self.osc_ready = True
                                    break
                                else:
                                    self.get_framerate()
                                    continue
                            else:
                                self.seq_pool = int(prompt("Please set target Sequence Pool: "))
                                continue
                        else:
                            self.tc_pool = str(prompt("Please set target Timecode Pool.TrackGroup.Track: "))
                            continue
                    else:
                        self.ip_port = str(prompt("Please set IP Port: "))
                        continue
                else:
                    self.ip = str(prompt("Please Set IP Address: "))
                    continue
            else:
                self.establish_osc_client()
                continue

    def get_framerate(self):
        # Import Plugin to send data
        self.cmd_via_osc("Import Library \"template.xml\" at Plugin 999 /o")
        self.cmd_via_osc("Call Plugin 999")
        self.establish_osc_server(self.handlers)


    def establish_osc_client(self):
        self.client = SimpleUDPClient(self.ip, self.ip_port)
        Log.info(f"OSC Client initialized at {self.ip}:{self.ip_port}")

    def cmd_via_osc(self, cmd:str):
        Log.info(f"cmd_via_osc: {cmd}")
        self.cmd = cmd
        self.client.send_message("/cmd", self.cmd)
        self.cmd = None

    def establish_osc_server(self, handlers):
        Log.info("Establishing OSC Server: ")
        self.osc_parser = ArgumentParser()
        self.osc_parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
        self.osc_parser.add_argument("--port", type=int, default=8000, help="The port to listen on")
        self.osc_args = self.osc_parser.parse_args()

        dispatcher = Dispatcher()
        for address, handler in handlers.items():
            # Wrap the original handler to include a stop mechanism
            def stop_handler(address, *args):
                handler(address, *args)  # Call the original handler
                if self._is_valid_osc_message(args):  # Assuming this method checks message validity
                    self.keep_running = False  # Set the flag to False to stop the server

            dispatcher.map(address, stop_handler)

        self.server = ThreadingOSCUDPServer((self.osc_args.ip, self.osc_args.port), dispatcher)
        Log.info(f"Serving on {self.server.server_address}")
        self._run_server()

    def _run_server(self):
        while self.keep_running:
            self.server.handle_request()  # Handle one request at a time

    def _is_valid_osc_message(self, *args):
        if args[0]:
            return True  # Placeholder for validation logic
        self.keep_running = False
        self.server.close()

    def _setup_osc_handlers(self):
        Log.info(f"Indexing OSC Handlers")
        self.handlers = {
            "/framerate": self.handle_framerate,
            "/print": self.cmdl_print
        }
        return self.handlers

    def handle_framerate(self, unused_addr, *args):
        self.framerate = args[0]
        return self.framerate

    def cmdl_print(self, unused_addr, *args):
        # Assuming the command is the first element in args
        if args:
            cmd = args[0]
            Log.info(f"CONSOLE OSC MESSAGE: {cmd}")
        else:
            Log.error("No command provided to cmdl_print")

