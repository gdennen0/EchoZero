from Block.block import Block
from Connections.port_types.event_port import EventPort
from message import Log
from tools import prompt_selection, prompt_selection_with_type, prompt
from Model.Export.Ma_File_Transfer import MA_File_Transfer
from Model.Export.MA_OSC_Connection import OSC_Connection


DEFAULT_EXPORT_IP:str = "127.0.0.1"
DEFAULT_EXPORT_IP_PORT:int = 8000

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
        self.console_os = None

        self.ma_file_transfer = MA_File_Transfer()
        self.osc_connection = OSC_Connection(self.ip, self.ip_port)

        # Add commands
        self.add_command("select_timecode_pool", self.select_timecode_pool)
        self.add_command("select_sequence_pool", self.select_sequence_pool)
        self.add_command("export", self.export)
        self.add_command("reload", self.reload)
        self.add_command("establish_connection", self.establish_connection)

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
            Log.error("No data available to export.")
            return

    def reload(self):
        """Reload the block's data."""
        super().reload()
        Log.info(f"{self.name} reloaded successfully.")

    def establish_connection(self):
        while True:
            # Check if OS is set:
            if self.ma_file_transfer.os:
                # method in osc connection that checks if all variables are valid before continuing
                self.osc_connection.check_variables(self.osc_connection.MA_import_template_xml())
                if self.osc_connection.osc_ready:
                    # rest of the prog.
                    self.osc_connection.check_variables(self.osc_connection.MA_set_tc_events(self.data))
                    break
                else:
                    continue
                # start server listener...
                #self.osc_connection.establish_osc_server(self.osc_connection.handlers)
                # Exit listener when variable is passed
                #self.osc_connection.cmd_via_osc("Delete Plugin 999")
                break
            else:
                self.ma_file_transfer.set_console_type()
                continue


    def get_tc_framerate():
        pass
        # Send Lua file
        # Load Lua file via osc
        # Start listener then execute Lua file 

