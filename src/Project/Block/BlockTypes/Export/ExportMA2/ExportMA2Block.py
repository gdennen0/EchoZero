from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Utils.tools import prompt, prompt_selection
from lxml import etree
from lupa import LuaRuntime
import os


DEFAULT_FRAME_RATE = "30 FPS"
DEFAULT_TC_POOL = 101
DEFAULT_SEQUENCE_POOL = 101
DEFAULT_EXEC_PAGE = 101
DEFAULT_EXEC_INT = 115
DEFAULT_MA_FOLDER = "C:\ProgramData\MA Lighting Technologies\grandma\gma2_V_3.9.60"

class ExportMA2Block(Block):
    name = "ExportMA2"
    type = "ExportMA2"
    
    def __init__(self):
        super().__init__()
        self.name = "ExportMA2"
        self.type = "ExportMA2"

        self.frame_rate: str = DEFAULT_FRAME_RATE
        self.tc_pool: int = DEFAULT_TC_POOL
        self.sequence_pool: int = DEFAULT_SEQUENCE_POOL
        self.exec_page: int = DEFAULT_EXEC_PAGE
        self.exec_int: int = DEFAULT_EXEC_INT
        self.ma_folder: str = DEFAULT_MA_FOLDER
        self.xml_root: str = ""
        self.sequence_root: str = ""
        self.tc_export_name: str = ""
        self.sequence_export_name: str = "EZ_SEQUENCE_"
        self.exec_export_name: str = "EZ_EXEC_"
        self.sequence_classes: dict = None
        self.exported_objects: dict = {
            "sequence" : [],
            "exec" : [],
            "timecode" : []
        }
        self.lua_import: str = ""


        self.lua_instr_set: str = ""

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.command.add("set_frame_rate", self.set_frame_rate)
        self.command.add("set_tc_pool", self.set_tc_pool)
        self.command.add("set_sequence_pool", self.set_sequence_pool)
        self.command.add("set_export_exec", self.set_export_exec)
        self.command.add("set_export_type", self.set_export_type)

        self.command.add("export_template", self._format_data)
        self.command.add("data_xml", self._data_xml)
        
        self.tc_xml: str = ""
        self.export_preface: str = f"EZ"
        self.sequence_xml: dict = {}
        self.exec_xml: dict = {}
        self.ns_map: dict = {}
        self.export_type: int = None # 0 = hit, 1 = sequence
        self.hit_counter: int = 0
        self.sequence_counter: int = self.sequence_pool
        self.hit_dictionary: dict = {}
        self.stack_dictionary: dict = {}
        self.xml_dictionary: dict = {
            "sequence" : {},
            "exec" : {},
            "timecode" : {}
        }
        self.lua_string: str = ""
        self.event_dictionary: dict = {
            "sequence" : {
                0 : {
                    "name" : "",
                    "pool" : "",
                    "time" : [],
                    "cue" : {
                        0 : { # template cue entry
                            "name" : "Mark",
                            "pool" : "0",
                            "time" : "0",
                        }
                    }
                },
                "MainCuestack" : {
                    "name" : "MainCuestack",
                    "pool" : DEFAULT_SEQUENCE_POOL,
                    "time" : [],
                    "cue" : {
                        0 : { # template cue entry
                            "name" : "Mark",
                            "pool" : "0",
                            "time" : "0",
                        }
                    }
                }
            },
            "timecode" : { 
                "name" : "",
                "pool" : "",
                "fps" : "",
                "event" : {
                    0 : { # template event entry
                        "sequence_name" : "",
                        "exec_pool" : "",
                        "cue_pool" : "",
                        "time" : "",
                    }
                }
            }
        }

  # Find if there's a thumbdrive connected
  # If there is, set the ma_folder to the thumbdrive
  # If there is no thumbdrive, set the ma_folder to the default folder
  # If there is no thumbdrive or default folder, raise an error


    """
    USER INPUT: 
    """
    def set_frame_rate(self):
        self.frame_rate = int(prompt_selection("Enter the frame rate: ", {"1/100 Seconds": "1/100 Seconds", "30 FPS": "30 FPS", "25 FPS": "25 FPS", "24 FPS": "24 FPS"}))

    def set_tc_pool(self):
        self.tc_pool = int(prompt("Enter the TC pool: "))

    def set_sequence_pool(self):
        self.sequence_pool = int(prompt("Enter the sequence pool: "))
    
    def set_export_exec(self):
        self.page_pool = float(prompt("Enter the export page: "))

    def set_ma_folder(self):
        self.ma_folder = prompt("Enter the MA folder: ")
    
    def set_export_type(self):
        self.export_type = prompt_selection("Enter the export type: ", { "hit" : "hit", "sequence" : "sequence"})

    """
    FUNCTIONS:
    """
    def _xml_root(self, pool: int, name: str, type: str):
        """
        This function creates the root element for the XML file
        """

        ns_map = {
            None: "http://schemas.malighting.de/grandma2/xml/MA",
            'xsi': "http://www.w3.org/2001/XMLSchema-instance"
        }
        xml_root = etree.Element("MA", nsmap=self.ns_map, major_vers="3", minor_vers="9", stream_vers="60", 
                        schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd", 
                        attrib={"{" + ns_map['xsi'] + "}schemaLocation": "http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd"})
        info = etree.SubElement(xml_root, "Info", datetime="2025-01-19T14:56:58", showfile="AI_GENERATED_SHOWFILE")
        #Log.info(f"fn:_xml_root: {xml_root}")
        return xml_root

    def _stack_data(self, event_item, iteration):
        # Add data to the dictionary
        self.stack_dictionary['name'][iteration] = event_item.classification
        self.stack_dictionary['time'][iteration] = event_item.time

        # Create a list of tuples from the dictionary data
        combined_list = [(key, self.stack_dictionary['name'][key], self.stack_dictionary['time'][key]) for key in self.stack_dictionary['name']]

        # Sort the list by time (index 2 of the tuple)
        combined_list.sort(key=lambda x: x[2])

        # Clear the existing dictionary
        self.stack_dictionary['name'].clear()
        self.stack_dictionary['time'].clear()

        # Rebuild the dictionary with sorted data
        for index, (key, name, time) in enumerate(combined_list):
            self.stack_dictionary['name'][index] = name
            self.stack_dictionary['time'][index] = time

    def _hit_data(self, event_item):
        if event_item.classification in self.hit_dictionary:
            self.hit_dictionary[event_item.classification].append(event_item.time)
        else:
            self.hit_dictionary[event_item.classification] = [event_item.time]

    def _format_data(self):
        """
        This function formats the data into two dicionaries, one for stack data and one for hit data...
        1st part of the main process
        """
        self.stack_dictionary = {'name': {}, 'time': {}}
        self.hit_dictionary = {}
        iteration = 0
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                self._stack_data(event_item, iteration)
                self._hit_data(event_item)
                iteration += 1
        #Log.info(f"\nHit Dictionary: {self.hit_dictionary}")
        #Log.info(f"\nStack Dictionary: {self.stack_dictionary}")

    """
    Data Export:
    """
    def _data_xml(self):
        """

        This function creates the xml structure for the data
        """
        loop_index = 0
        timecode_index = None
        # Initialize the first timecode XML root only if it hasn't been initialized yet
        if loop_index not in self.xml_dictionary['timecode']:
            self.xml_dictionary['timecode'] = self._xml_root(pool=1, name="timecode", type="timecode")
            timecode_index = etree.SubElement(self.xml_dictionary['timecode'], "Timecode", name=f"songname", index=f"{(self.tc_pool - 1)}", lenght="6500", offset="0", play_mode="Play", slot="TC Slot 1", frame_format=f"30 FPS")

        while True:
            if self.export_type:
                if self.export_type == "hit":
                    for key, time in self.hit_dictionary.items():
                        self._if_hit(key, time, loop_index, timecode_index)

                        #Log.info(f"passing in hit loop_index: {loop_index}, key: {key}, time: {time}")
                        loop_index += 1
                        self.sequence_pool += 1
                        # Ensure the next index is initialized only if it doesn't exist
                    #self._save_xml()
                    break
                elif self.export_type == "sequence":
                    #Log.info(f"Pre-loop_index: {loop_index}")
                    loop_index = self._if_stack(self.stack_dictionary['name'][0], self.stack_dictionary['time'][0], loop_index, timecode_index)
                    #Log.info(f"Post-loop_index: {loop_index}")
                    for key, value in self.stack_dictionary['time'].items():
                        #Log.info(f"Passing Name: {self.stack_dictionary['name'][key]} and Time: {value}")
                        #Log.info(f"passing in sequence loop_index: {loop_index}, key: {key}, value: {value}")
                        loop_index = self._if_stack((self.stack_dictionary['name'][key]), value, loop_index, timecode_index)
                        loop_index += 1
                    self.sequence_pool += 1
                    #self._save_xml()
                    break
            else:
                self.set_export_type()
                continue

        self._save_xml()
        #self._write_lua_files()
        #self._check_export_list()
        self._new_lua_file()

    def _get_xml_dict_size(self, dict_name):
        # Check if the dictionary exists and is not None

        if self.xml_dictionary.get(dict_name) is not None:

            # Check if the dictionary is empty
            if not self.xml_dictionary[dict_name]:
                return 0  # Return 0 or another appropriate value for empty dictionary
            try:
                max_index = max(self.xml_dictionary[dict_name].keys())
                return max_index
            except KeyError:
                return 0
        else:
            return 0

    def _if_stack(self, event_name, event_time, loop_index, timecode_index):
        """
        This function connects the data to the sequence, executor and timecode xmls
        """
        #Log.info(f"fn:_if_stack: event_name: {event_name} event_time: {event_time} loop_index: {loop_index}")
        # Ensure the dictionary for 'sequence' is initialized
        if 'sequence' not in self.xml_dictionary:
            self.xml_dictionary['sequence'] = {}
        
        if self.xml_dictionary['sequence'] is None:
            self.xml_dictionary['sequence'] = self._xml_root(pool=1, name=event_name, type="sequence")


        # Assign the new XML root to a new index in the dictionary
        self._xml_sequence(type="stack", class_name=event_name, time=event_time, index_number=loop_index)
        Log.info(f"fn:_if_stack: event_name: {event_name}, event_time: {event_time}, loop_index: {loop_index}")
        self._xml_exec(event_name, event_time, loop_index)
        return loop_index

    def _if_hit(self, event_name, event_time, loop_index, timecode_index):
        """
        This function checks if the event is a hit event
        """
        #Log.info(f"fn:_if_hit: name: {event_name}, time: {event_time}, loop_index: {loop_index}")
        self.xml_dictionary['sequence'][loop_index] = self._xml_root(pool=1, name=event_name, type="sequence")
        self.xml_dictionary['exec'][loop_index] = self._xml_root(pool=1, name=event_name, type="exec")
        self._xml_sequence(type="hit", class_name=event_name, time=event_time, index_number=loop_index)
        self._xml_exec(event_name, event_time, loop_index)
        self._xml_timecode_(event_name, event_time, loop_index, timecode_index)
        self.exec_int += 1

    def _xml_sequence(self, type, class_name, time, index_number):
        """
        This function creates the sequence xml
        """
        if type == "stack":
            max_index = self._get_xml_dict_size('sequence')
            Log.info(f"fn:_xml_sequence: max_index: {max_index}")
            # Ensure the dictionary is initialized and has the key `max_index`
            if max_index not in self.xml_dictionary['sequence']:
                self.xml_dictionary['sequence'][max_index] = None
                if self.xml_dictionary['sequence'][max_index] is None:
                    self.xml_dictionary['sequence'][max_index] = self._xml_root(pool=1, name="Initial", type="sequence")
                    Log.info(f"fn:_xml_sequence: Added entry at {max_index} at xml_dict: {self.xml_dictionary['sequence'][max_index]}")
                    sequ = etree.SubElement(self.xml_dictionary['sequence'][max_index], "Sequ", index=f"{(self.sequence_pool)}", name=f"{class_name}", timecode_slot="255", forced_position_mode="0")
                    cue_xsi = etree.SubElement(sequ, "Cue", { "{http://www.w3.org/2001/XMLSchema-instance}nil" : "true" })
                    cue_index = etree.SubElement(sequ, "Cue", index=f"{index_number}")
                    cue_number = etree.SubElement(cue_index, "Number", number=f"1", sub_number="0")
                else:
                    cue = etree.SubElement(self.xml_dictionary['sequence'][max_index], "Cue", index=f"{index_number}")
                    cue_number = etree.SubElement(cue, "Number", number=f"{max_index}", sub_number="0")
                    cue_part = etree.SubElement(cue, "CuePart", index="0", name=class_name)
        elif type == "hit":
            max_index = self._get_xml_dict_size('sequence')
            # Ensure the dictionary is initialized and has the key `max_index`
            if max_index not in self.xml_dictionary['sequence']:
                # Initialize it here if it's not present
                self.xml_dictionary['sequence'][max_index] = None
            if self.xml_dictionary['sequence'][max_index] is None:
                # Initialize it here if it's None
                self.xml_dictionary['sequence'][max_index] = self._xml_root(pool=1, name="Initial", type="sequence")
                #Log.info(f"Added entry at {max_index} at xml_dict: {self.xml_dictionary['sequence'][max_index]}")
            sequ = etree.SubElement(self.xml_dictionary['sequence'][max_index], "Sequ", index=f"{(self.sequence_pool)}", name=f"{class_name}", timecode_slot="255", forced_position_mode="0")
            cue_xsi = etree.SubElement(sequ, "Cue", { "{http://www.w3.org/2001/XMLSchema-instance}nil" : "true" })
            cue_index = etree.SubElement(sequ, "Cue", index=f"{index_number}")
            cue_number = etree.SubElement(cue_index, "Number", number=f"1", sub_number="0")
            cue_part = etree.SubElement(cue_index, "CuePart", index="0", name=class_name)
            cue_part_preset_timing = etree.SubElement(cue_part, "CuePartPresetTiming")


    def _xml_exec(self, class_name, event_time, loop_index):
        """
        This function creates the executor xml
        """
        max_index = self._get_xml_dict_size('exec')
        if max_index not in self.xml_dictionary['exec']:
            self.xml_dictionary['exec'][max_index] = self._xml_root(pool=1, name="Initial", type="exec")

        exec_root = self.xml_dictionary['exec'][max_index]
        if exec_root is None:
            exec_root = self._xml_root(pool=1, name="Initial", type="exec")
            self.xml_dictionary['exec'][max_index] = exec_root

        exec_offset = etree.SubElement(exec_root, "Exec", offset_from_first="0")
        exec_assignment = etree.SubElement(exec_offset, "Assignment", name=f"{class_name} {self.sequence_pool}")
        etree.SubElement(exec_assignment, "No").text = "25"
        etree.SubElement(exec_assignment, "No").text = "1"
        etree.SubElement(exec_assignment, "No").text = f"{self.sequence_pool}"

        assignment_exec = etree.SubElement(exec_offset, "AssignmentExec", fader="16384")
        buttons = ["12299", "0", "0", "0"]
        for button in buttons:
            etree.SubElement(assignment_exec, "Button").text = button

        exec_playback = {
            "auto_start": "true", "auto_stop": "true", "auto_stop_off_time": "true", "auto_fix": "false",
            "loop_breaking_go": "false", "priority": "Normal", "soft_ltp": "true", "playback_master": "0",
            "wrap_around": "true", "restart_mode": "0", "trigger_is_go": "false", "cmd_disable": "false",
            "tracking": "true", "release_on_first_step": "true", "auto_stomp": "false", "speed_scale": "norm",
            "speed_master": "0", "rate_master": "0", "stepped_rate": "false", "swop_protect": "false",
            "kill_protect": "false", "ignore_exec_time": "false", "off_on_overwritten": "true",
            "MIB_always": "false", "MIB_never": "false", "chaser": "false", "cross_fader_mode": "false",
            "auto_black_move": "false", "scale_effect_rate": "true", "auto_master_go": "0"
        }
        etree.SubElement(exec_offset, "Playback", **exec_playback)


    def _xml_timecode_(self, class_name, event_time, loop_index, timecode_index):
        """
        This function creates the timecode xml for a hit event
        """
        #Log.info(f"fn:_xml_timecode_ dict check: {self.xml_dictionary['timecode']}")
        if self.export_type == "hit":
            subtrack = self._xml_timecode_subtrack(timecode_index, class_name, event_time, loop_index)
            self._xml_timecode_event(subtrack, class_name, event_time, loop_index)
        elif self.export_type == "sequence":
            if loop_index == 0:
                subtrack = self._xml_timecode_subtrack(timecode_index, class_name, event_time, loop_index)
                self._xml_timecode_event(subtrack, class_name, event_time, loop_index)
            elif loop_index > 0:
                self._xml_timecode_event(subtrack, class_name, event_time, loop_index)
        #Log.info(f"fn:_xml_timecode: {self.xml_dictionary['timecode'][loop_index]} \n class_name: {class_name} \n event_time: {event_time} \n loop_index: {loop_index} \n -----------------------------------")


    def _xml_timecode_event(self, subtrack, class_name, event_time, loop_index):
        """
        This function adds the timecode events
        """
        #Log.info(f"fn:_xml_timecode_event: subtrack: {subtrack}, class_name: {class_name}, event_time: {event_time}, loop_index: {loop_index}")
        if type(event_time) is list:
            for event in event_time:
                # Add event with event as time
                event_index = etree.SubElement(subtrack, "Event", index=f"{loop_index}", time=f"{event}", command="Goto", pressed="true", step="1")
                cue = etree.SubElement(event_index, "Cue", name="Cue 1")
                etree.SubElement(cue, "No").text = "1"
                etree.SubElement(cue, "No").text = f"{self.exec_page}"
                etree.SubElement(cue, "No").text = "1"
        elif type(event_time) is not list:
            # Add event with event_time as time
            event_index = etree.SubElement(subtrack, "Event", index="0", time=f"{event_time}", command="Goto", pressed="true", step=f"{loop_index + 1}")
            cue = etree.SubElement(event_index, "Cue", name=f"{class_name}")
            etree.SubElement(cue, "No").text = "1"
            etree.SubElement(cue, "No").text = f"{self.exec_page}"
            etree.SubElement(cue, "No").text = f"{loop_index + 1}"


    def _xml_timecode_subtrack(self, timecode_index, class_name, event_time, loop_index):
        """
        This function adds the timecode subtrack
        """
        #Log.info(f"fn:_xml_timecode_subtrack: timecode_index: {timecode_index}, class_name: {class_name}, event_time: {event_time}, loop_index: {loop_index}")
        track_index = etree.SubElement(timecode_index, "Track", index="0", active="true", expanded="true")
        track_object = etree.SubElement(track_index, "Object", name=f"{class_name} {self.exec_page}.{self.exec_int}")
        etree.SubElement(track_object, "No").text = "30"
        etree.SubElement(track_object, "No").text = "1"
        etree.SubElement(track_object, "No").text = f"{self.exec_page}"
        etree.SubElement(track_object, "No").text = f"{self.exec_int}"
        subtrack = etree.SubElement(track_index, "SubTrack", index=f"{loop_index}")
        self.tc_pool += 1
        return subtrack
    
    # DLETE THIS
    def _write_xml_file(self, xml_element, class_type, integer, integer_secondary):
        if class_type == "exec":
            export_name = f"{self.ma_folder}/importexport/{self.export_preface}_{class_type}_{integer}.{integer_secondary}.xml"
            self._write_xml(xml_element, export_name)
            self._update_export_list(class_type, (f"{self.export_preface}_{class_type}_{integer}.{integer_secondary}.xml"))
        elif class_type != "exec":
            export_name = f"{self.ma_folder}/importexport/{self.export_preface}_{class_type}_{integer}.xml"
            self._write_xml(xml_element, export_name)
            self._update_export_list(class_type, (f"{self.export_preface}_{class_type}_{integer}.xml"))


    def _save_xml(self):
        # Reset the pools to default
        self.tc_pool = DEFAULT_TC_POOL
        self.sequence_pool = DEFAULT_SEQUENCE_POOL
        self.exec_page = DEFAULT_EXEC_PAGE
        self.exec_int = DEFAULT_EXEC_INT
        tc_exported: bool = False
        # Iterate through the xml_dictionary and save each element to a file
        for index, xml_element in self.xml_dictionary.items():
            if index == 'timecode':
                if tc_exported == False:
                    export_name = f"{self.ma_folder}/importexport/{self.export_preface}_{index}_{self.tc_pool}.xml"
                    self._write_xml(xml_element, export_name)
                    self._update_export_list(index, f"{self.export_preface}_{index}_{self.tc_pool}.xml")
                    tc_exported = True
                else:
                    pass
            elif index == 'sequence':
                for key, value in xml_element.items():
                    export_name = f"{self.ma_folder}/importexport/{self.export_preface}_{index}_{self.sequence_pool}.xml"
                    self._write_xml(value, export_name)
                    self._update_export_list(index, f"{self.export_preface}_{index}_{self.sequence_pool}.xml")
                    self.sequence_pool += 1
            elif index == 'exec':
                for key, value in xml_element.items():
                    export_name = f"{self.ma_folder}/importexport/{self.export_preface}_{index}_{self.exec_page}.{self.exec_int}.xml"
                    self._write_xml(value, export_name)
                    self._update_export_list(index, f"{self.export_preface}_{index}_{self.exec_page}.{self.exec_int}.xml")
                    self.exec_int += 1        


    def _write_lua_files(self):
        print(f"fn:_write_lua_files: {self.exported_objects}")
        # Create the Lua XML
        lua_xml = self._xml_root(pool=1, name="EZ_import", type="lua")
        plugin = etree.SubElement(lua_xml, "Plugin", index="1", execute_on_load="0", name="EZ_import", luafile=f"EZ_import.lua")
        self._write_xml(lua_xml, f"{self.ma_folder}/plugins/EZ_import.xml")

        lua_script: str = "local function main()"
        for index, filename in self.exported_objects.items():
            for file in filename:
                result = file.rsplit('/', 1)[-1]
                integer, integer_secondary = self._parse_filename(result)
                Log.info(f"fn:_write_lua_files: result: {result} index: {index} integer: {integer} integer_secondary: {integer_secondary}")
                if integer is not None:
                    if integer_secondary is None:
                        lua_script += f'\n gma.cmd("Import {result} at {index} {integer} /m /nc")'
                    elif integer_secondary is not None:
                        lua_script += f'\n gma.cmd("Store Page {integer} /m /nc")'
                        lua_script += f'\n gma.cmd("Import {result} at {index} {integer}.{integer_secondary} /m /nc")'
                elif integer is None:

                    #Log.info(f"fn:_write_lua_files: not writing to XML")
                    pass
        lua_script += f'\nend \nmain()'
        #with open(self.ma_folder + "/plugins/EZ_import.lua", "w") as file:
        #    file.write(lua_script)
        #Log.info(f"fn:_write_lua_files: LUA: {lua_script}")

    def _python_dict_to_lua_dict(self):  
        """
        This function converts the python dictionary to a lua dictionary... simplified exported_objects dictionary
        """
        Log.info(f"fn:_python_dict_to_lua_dict: NEW RUN: \n")
        Log.info(f"fn:_python_dict_to_lua_dict: exported_objects: {self.exported_objects}")
        lua_dictionary: str = " {"
        for object_type, file_list in self.exported_objects.items():
            lua_dictionary += "\n"
            iteration = 0
            for filename in file_list:
                Log.info(f"fn:_python_dict_to_lua_dict: filename: {filename}, iteration: {iteration}, object_type: {object_type}")
                if len(object_type) == 1:
                    lua_dictionary += f"{object_type} = "
                    lua_dictionary += "{"
                    lua_dictionary += f"\"{filename}\""
                    lua_dictionary += "} \n"
                elif len(object_type) > 1:
                    if iteration == 0:
                        # START OF DICTIONARY ENTRY
                        # class_type = {"filename1", 
                        lua_dictionary += f"{object_type} = "
                        lua_dictionary += "{"
                        lua_dictionary += f"\"{filename}\", "
                    elif iteration > 0:
                        if object_type == "timecode":
                            lua_dictionary += "}, \n"
                        if iteration == len(file_list)-1:
                            # END OF DICTIONARY ENTRY
                            # "filename3"},
                            lua_dictionary += f"\"{filename}\""
                            lua_dictionary += "}, \n"
                        elif iteration < len(file_list) - 1:
                            # MIDDLE OF DICTIONARY ENTRY
                            # "filename2",
                            lua_dictionary += f"\"{filename}\", "
                            if object_type == "timecode":
                                lua_dictionary += "}, \n"
                iteration += 1
        lua_dictionary += "\n}"
        return lua_dictionary

    def _new_lua_file(self):
        lua_script = "exec = nil\nsequence = nil\ntimecode= nil\n"
        export_dictionary = self._python_dict_to_lua_dict()
        lua_script += "\nfunction prompt(Title, Message)"
        lua_script += "\n    input = gma.textinput(Title, Message)"
        lua_script += "\n    return input"
        lua_script += "\nend"
        lua_script += f"\nexport_dictionary = {export_dictionary}"
        lua_script += "\nfunction remove_xml_extension(filename)"
        lua_script += "\n    if filename:sub(-4) == '.xml' then"
        lua_script += "\n        return filename:sub(1, -5)"
        lua_script += "\n    else"
        lua_script += "\n        return filename"
        lua_script += "\n    end"
        lua_script += "\nend"
        lua_script += "\nfunction parse_filename(filename)"
        lua_script += "\n    filename = remove_xml_extension(filename)"
        lua_script += "\n    integer = filename:match(\"%d+$\")"
        lua_script += "\n    return tonumber(integer)"
        lua_script += "\nend"
        lua_script += "\nfunction import_prompt(object_type, filename)"
        lua_script += "\n    Title = 'Import ' .. filename .. ' at ' .. object_type"
        lua_script += "\n    Message = parse_filename(filename)"
        lua_script += "\n    input = prompt(Title, Message)"
        lua_script += "\n    local cmd = Title .. ' ' .. input .. ' /m /nc'"
        lua_script += "\n    gma.cmd(cmd)"
        lua_script += "\n    gma.feedback(filename)"
        lua_script += "\n    gma.feedback(object_type)"
        lua_script += "\nend\n"
        lua_script += "\nfunction main()"
        lua_script += "\n    for object_type, file_list in pairs(export_dictionary) do"
        lua_script += "\n        for _, filename in pairs(file_list) do"
        lua_script += "\n            gma.feedback(object_type)"
        lua_script += "\n            gma.feedback(filename)"
        lua_script += "\n            import_prompt(object_type, filename)"
        lua_script += "\n        end"
        lua_script += "\n    end"
        lua_script += "\nend"

        lua_script += "\nmain()"
        with open(self.ma_folder + "/plugins/EZ_import.lua", "w") as file:
            file.write(lua_script)


    def _parse_filename(self, filename):

        # Remove the '.xml' extension if present
        if filename.endswith('.xml'):
            filename = filename[:-4]
        # Remove the 'EZ_' prefix
        if filename.startswith('EZ_'):
            filename = filename[3:]
        # Identify and remove the next part which could be 'sequence', 'exec', or 'timecode'
        for keyword in ['sequence_', 'exec_', 'timecode_']:
            if keyword in filename:
                start_index = filename.index(keyword)
                end_index = start_index + len(keyword)
                filename = filename[:start_index] + filename[end_index:]
                break
        # Split the remaining string by underscores and remove empty strings
        parts = [part for part in filename.split('.') if part]
        # Extract numbers and convert them to integers
        numbers = [int(part) for part in parts if part.isdigit()]
        # Assign the first number to integer
        integer = numbers[0] if numbers else None
        # Assign the second number to integer_secondary if it exists
        integer_secondary = numbers[1] if len(numbers) > 1 else None
        return integer, integer_secondary


    def _write_xml(self, xml_dictionary, filename):
        et = etree.ElementTree(xml_dictionary)
        #Log.info(f"fn:_write_xml: file: {xml_dictionary}")
        et.write(filename, pretty_print=True, xml_declaration=True, encoding="UTF-8")


    def _update_export_list(self, type, filename):
        #Log.info(f"fn:_update_export_list: type: {type} filename: {filename}")
        self.exported_objects[type].append(filename)


    def _check_export_list(self):
        print(f"\n")
        for index, filename in self.exported_objects.items():
            Log.info(f"fn:_check_export_list: {index} : {filename}")
    

    def _float_to_fps(self, time: float, fps: int) -> float:
        time = float(time)
        return round(time * fps, 2)


    def process(self, input_data):
        # SendMAEvents may not need to process input data, but implement if necessary
        self._format_data()
        self._data_xml()
        return input_data 


    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
    
    def connect_ports(self, input_name, output_name):
            input_port = self.input.get(input_name)
            output_port = self.output.get(output_name)
            if input_port and output_port:
                input_port.connect(output_port)
                Log.info(f"Connected input '{input_name}' to output '{output_name}' via SendMAEvents.")
            else:
                Log.error(f"Failed to connect. Input: {input_name}, Output: {output_name}") 

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

    