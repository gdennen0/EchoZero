from message import Log

class Event:
    """
    Event Object
    """
    def __init__(self):
        self.frame = None
        self.name = None
        self.category = None

    def set_frame(self, frame):
        """
        Sets the frame of the event.
        """
        try:
            self.frame = int(frame)
            Log.info(f"Frame set to {self.frame}")
        except ValueError:
            Log.error(f"Invalid frame value: {frame}. Could not convert to int.")

    def set_name(self, name):
        """
        Sets the name of the event.
        """
        if isinstance(name, str):
            self.name = name.strip()
            Log.info(f"Name set to {self.name}")
        else:
            Log.error(f"Invalid name value: {name}. Name must be a string.")

    def set_category(self, category):
        """
        Sets the category of the event.
        """
        if isinstance(category, str):
            self.category = category.strip()
            Log.info(f"Category set to {self.category}")
        else:
            Log.error(f"Invalid category value: {category}. Category must be a string.")

    def serialize(self):
        """
        to a JSON format.
        """
        try:
            return {
                'event': {
                    "frame": self.frame,
                    "name": self.name,
                    "category": self.category,
                }
            }
        except (AttributeError, TypeError) as e:
            Log.error(f"Error during event serialization: {e}")
            return {}

    def deserialize(self, data):
        """
        Deserializes the event data from a JSON format.
        """
        try:
            if 'event' in data:
                event_data = data['event']
                self.frame = event_data.get('frame', None)
                self.name = event_data.get('name', None)
                self.category = event_data.get('category', None)
            return self
        except (KeyError, TypeError) as e:
            Log.error(f"Error during event deserialization: {e}")
            return self
