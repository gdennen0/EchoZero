import os
from Utils.tools import prompt, prompt_selection
import shutil
from Utils.message import Log


DEFAULT_CONSOLE_OS:str = ["ConsoleHardware", "Windows", "Mac", "Linux"]
DEFAULT_TEMPLATE_SOURCE:str = "Block/BlockTypes/Export/Parts/template.xml"


class MA_File_Transfer():
    def __init__(self):
        self.os:str = None
        self.ip:str = None
        self.port:int = 22
        self.user:str = "madata"
        self.pw:str = "madata"
        self.fixed_filepath:str = None
        self.system_path:str = None
        self.ma_directory = None
        self.source_path = DEFAULT_TEMPLATE_SOURCE
        self.console_os = DEFAULT_CONSOLE_OS

    def connection_os(self):
        self.os = prompt_selection("Please select MA3 instance type: ", DEFAULT_CONSOLE_OS)
        self.os = self.os[0]
        Log.info(f"User Selected MA3 Instance of: {self.os}")


    def locate_ma_windows(self):
        self.fixed_filepath:str = "/ProgramData/MALightingTechnology/gma3_library/datapools/plugins"
        self.system_path:str = os.environ.get("SYSTEMDRIVE", "C:")  # Default to C: if SYSTEMDRIVE is not set
        self.ma_directory:str = self.get_program_directory(self.system_path)
        Log.info(f"Windows Directory: {self.ma_directory}")
        self.copy_template()

    def locate_ma_mac(self):
        Log.info("Need to write the ability to locate the plugins folder on Mac")

    def locate_ma_linux(self):
        Log.info("unfortunately MA Lighting hasn't released MA3 software for Linux, unless you're installing directly on console hardware.")

    def ftp_ma_console(self):
        Log.info("Need to write the ftp module for connecting to an MA")

    def set_console_type(self):
        if self.os == self.console_os[0]:
            self.ftp_ma_console()
        elif self.os == self.console_os[1]:
            self.locate_ma_windows()
            self.copy_template()
        elif self.os == self.console_os[2]:
            self.locate_ma_mac()
            self.copy_template()
        elif self.os == self.console_os[3]:
            self.locate_ma_linux()
            self.connection_os()
        else:
            self.connection_os()

    def get_program_directory(self, base_drive):
        self.base_path = base_drive
        # Get the base directory depending on the operating syste
        program_path = os.path.join(self.base_path, self.fixed_filepath)
        return program_path
    
    def copy_template(self):
        Log.info(f"copy_template_init")
        while not self.ma_directory or not os.path.isdir(self.ma_directory):
            Log.error("Invalid or unset directory. Please set a valid directory.")
            self.connection_os()  # This will prompt the user to set the directory
        try:
            shutil.copy(self.source_path, self.ma_directory)
            Log.info(f"Copied {self.source_path} to {self.ma_directory}")
        except Exception as e:
            Log.error(f"Failed to copy file: {e}")



##FOR TESTING PURPOSES>>>
#test_connection = MA_File_Transfer(ip="127.0.0.1")
#console_type = test_connection.set_console_type()