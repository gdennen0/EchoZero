from src.Project.Block.block import Block
from src.Utils.message import Log
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Utils.tools import prompt, prompt_selection
from lxml import etree
from lupa import LuaRuntime


DEFAULT_FRAME_RATE = "30 FPS"
DEFAULT_TC_POOL = 31
DEFAULT_SEQUENCE_POOL = 94
DEFAULT_EXEC_PAGE = 100
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
        self.tc_root: str = ""
        self.sequence_root: str = ""
        self.tc_export_name: str = ""
        self.sequence_export_name: str = "EZ_SEQUENCE_"
        self.exec_export_name: str = "EZ_EXEC_"
        self.sequence_classes: dict = {}
        self.exported_objects: list = []
        self.lua_import: str = ""

        self.lua_instr_set: str = ""

        self.input.add_type(EventInput)
        self.input.add("EventInput")
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.command.add("export", self._tc_xml_header)
        self.command.add("set_frame_rate", self.set_frame_rate)
        self.command.add("set_tc_pool", self.set_tc_pool)
        self.command.add("set_sequence_pool", self.set_sequence_pool)
        self.command.add("set_export_exec", self.set_export_exec)
        self.command.add("process_events", self._process_events)
        self.command.add("DEBUG_print_classes", self.DEBUG_print_classes)
        self.command.add("DEBUG_print_data", self.DEBUG_print_data)

        # Process data
        # Batch send events... 
        # for event item... create a sequence XML + export it (with the name of the event)... save class name in dictionary with ez data as key and class name as value
        # if type in type classes.. create 

# V2 IDEA:
# Create a dictionary of dictionaries, each sub dictionary will contain the event item and it's time, 
# The main dictionary will use the event type (kick, snare, hat, etc) as the key and the sub dictionary as the value

    









    
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
    
    def _xml_root(self):
        """
        This function creates the root element for the XML file
        """
        ns_map = {
            None: "http://schemas.malighting.de/grandma2/xml/MA",
            'xsi': "http://www.w3.org/2001/XMLSchema-instance"
        }
        self.tc_root = etree.Element("MA", nsmap=ns_map, major_vers="3", minor_vers="9", stream_vers="60", 
                        schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd", 
                        attrib={"{" + ns_map['xsi'] + "}schemaLocation": "http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd"})
        return self.tc_root
    
    def _xml_end(self, xml_filename: str, tree_root: str):
        """
        This function creates the end element for the XML file
        """
        tree = etree.ElementTree(tree_root)
        tree.write(self.ma_folder + f"/importexport/EZ_tc_{xml_filename}.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
        self.exported_objects.append(f'EZ_tc_{xml_filename}.xml')
    
    def _xml_function(self, func: str):
        """
        This function is a template for creating XML files, creating the header saving a file, with the meat of the xml passed in as a function
        function: str = the function that will be passed in to to create the xml
        """
        def wrapper(self, *args, **kwargs):
            root = self._xml_root()
            result = func(root)
            self._xml_end(self.result.name, root)
            return root
        return wrapper

    

    def _tc_xml_header(self):
        self.tc_root = self._xml_root()
        # Add child elements
        info = etree.SubElement(self.tc_root, "Info", datetime="2025-01-15T02:53:44", showfile="AI_GENERATED_SHOWFILE")
        timecode = etree.SubElement(self.tc_root, "Timecode", name=f"{self.tc_export_name}", index=f"{(self.tc_pool)-1}", slot="TC Slot 1", frame_format="30 FPS")
    
        self._set_classes()

        for classes in self.sequence_classes:
            Log.info(f"Creating sequence for {classes} : {self.sequence_classes[classes]}, sequence int: {self.exec_int}")
            track = etree.SubElement(timecode, "Track", index="0", active="true", expanded="true")
            object_elem = etree.SubElement(track, "Object", name=f"{self.sequence_export_name} {self.exec_page}.{self.exec_int}")
            etree.SubElement(object_elem, "No").text = "30"
            etree.SubElement(object_elem, "No").text = "1"
            etree.SubElement(object_elem, "No").text = f"{self.exec_page}"
            etree.SubElement(object_elem, "No").text = f"{self.exec_int}"
            subtrack = etree.SubElement(track, "SubTrack", index="0")
            self._xml_sequence_exec(classes, self.exec_page, self.exec_int)
            self._process_events(subtrack, self.exec_page, self.exec_int, classes)
            self.exec_int += 1
        
        # Create an ElementTree object from the root element
        tree = etree.ElementTree(self.tc_root)
        # Write the XML to a file
        tree.write(self.ma_folder + f"/importexport/EZ_tc_{self.tc_pool}_{self.tc_export_name}.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
        self.exported_objects.append(f'EZ_tc_{self.tc_pool}_{self.tc_export_name}.xml')
        self._exported_to_lua()

    def _process_events(self, subtrack, exec_page, exec_int, classes):
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if event_item.classification == classes:
                    event_item.time = self._float_to_fps(event_item.time, 30)
                    Log.info(f"Event Time: {event_item.time} frames")
                    self._event_to_xml(event_item, subtrack, exec_page)
                elif event_item.classification != classes:
                    Log.error(f"Skipping: Event {event_item.classification} not in class {classes}")

        for classes in self.sequence_classes:
            Log.info(f"Key: {classes} Value: {self.sequence_classes[classes]}")
            

        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if event_item.classification in self.sequence_classes:
                    Log.info(f"Key: {event_item.classification} Value: {self.sequence_classes[event_item.classification]}")

    def _float_to_fps(self, time: float, fps: int) -> float:
        time = float(time)
        return round(time * fps, 2)
    
    def _event_to_xml(self, event_item, timecode_xml, exec_page):
        """
        This function will convert the event item to an xml element
        """
        event = etree.SubElement(timecode_xml, "Event", index="0", time=f"{event_item.time}", command="Goto", pressed="true", step="1")
        cue = etree.SubElement(event, "Cue", name="Cue 1")
        etree.SubElement(cue, "No").text = "1"
        etree.SubElement(cue, "No").text = f"{exec_page}"
        etree.SubElement(cue, "No").text = "1"

    def _set_classes(self):
        for event_data in self.data.get_all():
            for event_item in event_data.get_all():
                if self.sequence_classes.get(event_item.classification) is None:
                    self.sequence_classes[event_item.classification] = self.sequence_export_name + event_item.classification
                    #Log.info(f"Key: {event_item.classification}, Value: {self.sequence_classes[event_item.classification]}")

    def DEBUG_print_classes(self):
        for classes in self.sequence_classes:
            Log.info(f"Key: {classes} Value: {self.sequence_classes[classes]}")

    def DEBUG_print_data(self):
        for event_data in self.data.get_all():
            for classes in self.sequence_classes:
                Log.info(f"Key: {classes} Value: {self.sequence_classes[classes]}")
                for event_item in event_data.get_all():
                    if event_item.classification in self.sequence_classes.values():
                        Log.info(f"Key: {event_item.classification} Value: {event_item.time}")
                    else:
                        break


    def _xml_sequence_exec(self, event_class, exec_page, exec_int):
        """
        This function creates xmls for both the sequence and exec page of the hit button, that can be imported in MA
        """
        # Define namespaces for sequence
        ns_map = {
            None: "http://schemas.malighting.de/grandma2/xml/MA",  # default namespace
            'xsi': "http://www.w3.org/2001/XMLSchema-instance"
        }

        # Create the root element
        sequence_root = etree.Element("MA", nsmap=ns_map, major_vers="3", minor_vers="9", stream_vers="60",
                            schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd")

        # Add child elements
        sequence_info = etree.SubElement(sequence_root, "Info", datetime="2025-01-19T14:56:58", showfile="AI_GENERATED_SHOWFILE")
        sequence = etree.SubElement(sequence_root, "Sequ", index=f"{(self.sequence_pool-1)}", name=f"{self.sequence_export_name}{event_class}", timecode_slot="255", forced_position_mode="0")
        sequence_cue = etree.SubElement(sequence_info, "Cue", index="1")
        sequence_number = etree.SubElement(sequence_cue, "Number", number="1", sub_number="0")
        cue_part = etree.SubElement(sequence_cue, "CuePart", index="0")
        cue_part_preset_timing = etree.SubElement(cue_part, "CuePartPresetTiming")

        # Add multiple PresetTiming elements
        for _ in range(10):
            etree.SubElement(cue_part_preset_timing, "PresetTiming")
            
        #EXPORT EXEC

        # Create the root element with namespaces and attributes
        exec_root = etree.Element("MA", nsmap=ns_map, major_vers="3", minor_vers="9", stream_vers="60",
                                  schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd")

        # Add child elements
        exec_info = etree.SubElement(exec_root, "Info", datetime="2025-01-19T15:17:53", showfile="songpart_1")
        exec_elem = etree.SubElement(exec_root, "Exec", offset_from_first="0")

        # Assignment element with nested No elements
        exec_assignment = etree.SubElement(exec_elem, "Assignment", name=f"{self.exec_export_name}{event_class}")
        etree.SubElement(exec_assignment, "No").text = "25"
        etree.SubElement(exec_assignment, "No").text = "1"
        etree.SubElement(exec_assignment, "No").text = f"{self.sequence_pool}"

        # AssignmentExec element with multiple Button elements
        assignment_exec = etree.SubElement(exec_elem, "AssignmentExec", fader="16384")
        buttons = ["12302", "12311", "12290", "0"]
        for button in buttons:
            etree.SubElement(assignment_exec, "Button").text = button

        # Playback element with attributes
        playback_attrs = {
            "auto_start": "true", "auto_stop": "true", "auto_stop_off_time": "true", "auto_fix": "false",
            "loop_breaking_go": "false", "priority": "Normal", "soft_ltp": "true", "playback_master": "0",
            "wrap_around": "true", "restart_mode": "0", "trigger_is_go": "false", "cmd_disable": "false",
            "tracking": "true", "release_on_first_step": "true", "auto_stomp": "false", "speed_scale": "norm",
            "speed_master": "0", "rate_master": "0", "stepped_rate": "false", "swop_protect": "false",
            "kill_protect": "false", "ignore_exec_time": "false", "off_on_overwritten": "true",
            "MIB_always": "false", "MIB_never": "false", "chaser": "false", "cross_fader_mode": "false",
            "auto_black_move": "false", "scale_effect_rate": "true", "auto_master_go": "0"
        }
        etree.SubElement(exec_elem, "Playback", **playback_attrs)

        seq_tree = etree.ElementTree(sequence_root)
        seq_tree.write(self.ma_folder + f"/importexport/EZ_seq_{self.sequence_pool}_{event_class}.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
        self.exported_objects.append(f'EZ_seq_{self.sequence_pool}_{event_class}.xml')

        exec_tree = etree.ElementTree(exec_root)
        exec_tree.write(self.ma_folder + f"/importexport/EZ_exec_{self.exec_page}.{self.exec_int}_{event_class}.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")
        self.exported_objects.append(f'EZ_exec_{self.exec_page}.{self.exec_int}_{event_class}.xml')
        self.sequence_pool += 1
        self.exec_int += 1

    def _exported_to_lua(self):
        Log.info(f"fn:_exported_to_lua")
        lua_script: str = "local function main()"
        for filename in self.exported_objects:
            Log.info(f"fn:_exported_to_lua: {filename}")
            # Remove the '.xml' extension and split the filename by '_'
            parts = filename.replace('.xml', '').split('_')
            # Create a dictionary with specific keys
            filename_dict = {
                "SourceApplication": parts[0] if len(parts) > 0 else "",
                "Type": parts[1] if len(parts) > 1 else "",
                "Pool": parts[2] if len(parts) > 2 else "",
                "Label": parts[3] if len(parts) > 3 else ""
            }
            
            # Log the initial Pool value
            Log.info(f" Pool Value: {filename_dict['Pool']}")

            # Check if 'Pool' contains a hyphen and correctly handle it
            if "-" in filename_dict["Pool"]:
                pool_parts = filename_dict["Pool"].split('-')
                if len(pool_parts) == 2:
                    filename_dict["Pool"] = f"{pool_parts[0]}.{pool_parts[1]}"
                    filename_dict["Label"] = parts[3] if len(parts) > 3 else ""  # Ensure label is correctly assigned
                    Log.info(f"Converted Pool Value: {filename_dict['Pool']}")

            # Use the dictionary to format the Lua command
            lua_script += f'\n gma.cmd("Import {filename} at {filename_dict["Type"]} {filename_dict["Pool"]}")'
        lua_script += "\n end \n main()"
        with open(self.ma_folder + "/plugins/EZ_import.lua", "w") as file:
            file.write(lua_script)

        # Write the XML to load the lua script
        ns_map = {
            None: "http://schemas.malighting.de/grandma2/xml/MA",  # default namespace
            'xsi': "http://www.w3.org/2001/XMLSchema-instance"
        }

        # Create the root element
        plugin_root = etree.Element("MA", nsmap=ns_map, major_vers="3", minor_vers="9", stream_vers="60",
                        schemaLocation="http://schemas.malighting.de/grandma2/xml/MA http://schemas.malighting.de/grandma2/xml/3.9.60/MA.xsd")

        plugin_info = etree.SubElement(plugin_root, "Info", datetime="2025-01-19T15:17:53", showfile="songpart_1")
        plugin_elem = etree.SubElement(plugin_root, "Plugin", index="1", execute_on_load="0", name="EZ_import", luafile="EZ_import.lua")

        plugin_tree = etree.ElementTree(plugin_root)
        plugin_tree.write(self.ma_folder + "/plugins/EZ_import.xml", pretty_print=True, xml_declaration=True, encoding="utf-8")


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

    