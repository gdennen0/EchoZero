from Utils.message import Log
from Attribute.attribute import Attribute

class AttributeController:
    def __init__(self):
        self.attributes = []

    def add(self, name, value):
        attribute = self.match_attribute(name)
        if attribute:
            Log.error(f"Attribute '{name}' already exists!!!")
        else:
            attribute = Attribute()
            attribute.set_name(name)
            attribute.set_value(value)
            self.attributes.append(attribute)

    def append(self, attribute_name, value):
        attribute = self.match_attribute(attribute_name)
        if attribute is None:
            Log.error(f"Attribute '{attribute_name}' does not exist.")
            return

        current_value = attribute.get_value()
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            Log.error(f"Attribute '{attribute_name}' is not a list and cannot append values.")

    def remove(self, name):
        for attribute in self.attributes:
            if attribute.name == name:
                self.attributes.remove(attribute)
                break

    def set(self, name, value):
        attribute = self.match_attribute(name)
        if attribute:
            attribute.set_value(value)
        else:
            Log.error(f"Attribute '{name}' does not exist. Please add it first.")

    def get(self, name):
        """returns the value of the attribute"""
        attribute = self.match_attribute(name)
        if attribute:
            return attribute.get_value()
        else:
            Log.info(f"Attribute '{name}' does not exist.")

    def list(self):
        return list(self.attributes.keys())
    
    def to_dict(self):
        return [attribute.to_dict() for attribute in self.attributes]
    
    def from_dict(self, dict):
        for attribute in dict:
            attribute = Attribute()
            attribute.from_dict(attribute)
            self.attributes.append(attribute)
            Log.info(f"Loaded attribute '{attribute.name}'.")

    def items(self):
        return self.attributes

    def match_attribute(self, name):
        """
        Matches an attribute by name from a list of attributes.

        :param attributes: List of Attribute objects.
        :param name: The name of the attribute to match.
        :return: The matched Attribute object or None if not found.
        """
        normalized_name = name.strip().lower()
        for attribute in self.attributes:
            if attribute.name.strip().lower() == normalized_name:
                return attribute
        return None