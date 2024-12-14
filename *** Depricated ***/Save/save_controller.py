import json
from Data.data import Data
from Utils.message import Log

class SaveController:
    def __init__(self, parent):
        self.parent = parent
        self.attributes = []
        self.sub_modules = []
        self.recursive_depth_limit = None
        self.current_depth = 0

    def set_recursive_depth(self, depth):
        """Set the maximum recursive depth for saving."""
        if isinstance(depth, int) and depth >= 0:
            self.recursive_depth_limit = depth
            Log.info(f"Set recursive depth to {depth}")
        else:
            Log.error("Invalid depth value. It must be a non-negative integer.")

    def add_attribute(self, name):
        self.attributes.append(name)

    def add_sub_module(self, name):
        """ This just adds a reference to the passed sub module and this instance of save controller will run the save methods on the sub module(s)"""
        self.sub_modules.append(name)


    def to_dict(self):
        """Save the attributes to a file."""
        if self.recursive_depth_limit is not None and self.current_depth > self.recursive_depth_limit:
            Log.info("Reached maximum recursive depth.")
            return {}

        self.current_depth += 1  # Increment current depth

        savable_data = {}
        primary_attributes = {}

        if len(self.attributes) > 0:
            for attribute_name in self.attributes:
               if "." in attribute_name:
                    attribute_name_parts = attribute_name.split(".")
                    if len(attribute_name_parts) == 2:
                        attribute = getattr(self.parent, attribute_name_parts[0])
                        for part in attribute_name_parts[1:]:
                            connected_attribute = getattr(attribute, part)
                            if connected_attribute:
                                attribute = connected_attribute
                            else:
                                Log.error(f"Attribute '{part}' not found in '{attribute_name}'")
                                break
                        primary_attributes[attribute_name] = attribute
                    else:
                        Log.error(f"Invalid connected attribute: {attribute_name} ({len(attribute_name_parts)} parts) only 2 parts allowed")
               elif attribute_name == "data" and getattr(self.parent, attribute_name):
                   Log.info(f"Saving Data: {attribute_name} with data type: {type(getattr(self.parent, attribute_name))}")
                   primary_attributes[attribute_name] = getattr(self.parent, attribute_name).save.to_dict()
               elif getattr(self.parent, attribute_name):
                    Log.info(f"Saving attribute: {attribute_name} with data type: {type(getattr(self.parent, attribute_name))}")
                    primary_attributes[attribute_name] = getattr(self.parent, attribute_name)
               elif getattr(self.parent, attribute_name) is None:
                   Log.error(f"{self.parent.name} Attribute '{attribute_name}' is None")
               elif isinstance(getattr(self.parent, attribute_name), list):
                    Log.info(f"Saving list: {attribute_name}")
                    for item in getattr(self.parent, attribute_name):
                        if hasattr(item, 'save'):
                            primary_attributes[attribute_name] = item.save.to_dict()
                        else:
                            Log.error(f"Item in list {attribute_name} does not have a save method")
        sub_modules = {}
        for sub_module_name in self.sub_modules:
            Log.info(f"{self.parent.name} | Checking sub module: {sub_module_name}")
            sub_module = getattr(self.parent, sub_module_name)
            if sub_module and hasattr(sub_module, 'save'):
                sub_modules[sub_module_name] = sub_module.save.to_dict()
            elif isinstance(sub_module, list):
                for item in sub_module:
                    Log.info(f"{self.parent.name} | Checking item in list {sub_module_name}: {item}")
                    if hasattr(item, 'save'):
                        sub_modules[sub_module_name] = item.save.to_dict()
                    else:
                        Log.error(f"Item in list {sub_module_name} does not have a save method")
            else:
                Log.error(f"Sub module {sub_module_name} does not have a save method")

        savable_data["attribute"] = primary_attributes
        if len(sub_modules) > 0:
            savable_data["module"] = sub_modules

        self.current_depth -= 1  # Decrement current depth after processing

        return savable_data
