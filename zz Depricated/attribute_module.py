from message import Log

class Attribute:
    def __init__(self, name):
        self.name = None
        self.value = None
        self.save = False

    def set_name(self, name):
        self.name = name

    def to_dict(self):
        return {"name": self.name, "value": self.value}
    
    def from_dict(self, dict):
        self.name = dict["name"]
        self.value = dict["value"]

    def set_save(self, save):
        self.save = save

    def set_value(self, value):
        self.value = value

class AttributeModule:
    def __init__(self):
        self.attributes = []

    def add(self, name, value=None, save=True):
        if name in self.attributes:
            raise ValueError(f"Attribute '{name}' already exists.")
        else:
            attribute = Attribute()
            attribute.set_name(name)
            attribute.set_value(value)
            attribute.set_save(save)
            self.attributes.append(attribute)

    def remove(self, name):
        for attribute in self.attributes:
            if attribute.name == name:
                self.attributes.remove(attribute)
                break

    def set(self, name, value):
        if name not in self.attributes:
            raise KeyError(f"Attribute '{name}' does not exist.")
        for attribute in self.attributes:
            if attribute.name == name:
                attribute.set_value(value)
                break

    def get(self, name):
        if name not in self.attributes:
            raise KeyError(f"Attribute '{name}' does not exist.")
        return self.attributes[name]['value']

    def list(self):
        return list(self.attributes.keys())
    
    def to_dict(self):
        return [attribute.to_dict() for attribute in self.attributes if attribute.save]
    
    def from_dict(self, dict):
        for attribute in dict:
            attribute = Attribute()
            attribute.from_dict(attribute)
            self.attributes.append(attribute)
            Log.info(f"Loaded attribute '{attribute.name}'.")