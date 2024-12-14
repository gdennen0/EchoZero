class Attribute:
    def __init__(self):
        self.name = None
        self.value = None

    def set_name(self, name):
        self.name = name

    def to_dict(self):
        if isinstance(self.value, (int, float, str, bool)):
            value = self.value
        elif hasattr(self.value, 'attribute'):
            value = self.value.attribute.to_dict()
        elif isinstance(self.value, list):
            value = [
                item.to_dict() if hasattr(item, 'to_dict') and callable(getattr(item, 'to_dict')) else item
                for item in self.value
            ]
        else:
            value = str(f"{self.value} ERROR COULD NOT CONVERT VALUE")  # Fallback for unsupported types
        return {"name": self.name, "value": value}
    
    def from_dict(self, dict):
        self.name = dict["name"]
        self.value = dict["value"]

    def set_save(self, save):
        self.save = save

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value