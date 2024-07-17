from message import Log
from .event import Event

class EventPool:
    """
    Event Pool Object
    """
    def __init__(self):
        self.name = None
        self.objects = []

    def set_name(self, name):
        """
        Sets the name of the event pool.
        """
        self.name = name
        Log.info(f"Event pool name set to {name}")

    def add_event(self, event):
        """
        Adds an event to the event pool.
        """
        self.objects.append(event)
        Log.info(f"Event added: {event}")

    def set_event_list(self, event_list):
        """
        Sets the event list of the event pool.
        """
        self.objects = event_list
        Log.info(f"Event list set with {len(event_list)} events")

    def serialize(self):
        """
        Serializes the event pool data into a JSON format.
        """
        try:
            return {
                'event_pool': {
                    "name": self.name,
                    "events": [event.serialize() for event in self.objects if event is not None] if self.objects else []
                }
            }
        except (AttributeError, TypeError) as e:
            Log.error(f"Error during event_pool serialization: {e}")
            return {}

    def deserialize(self, data):
        """
        Deserializes the event pool data from a JSON format.
        """
        try:
            if 'event_pool' in data:
                event_pool_data = data['event_pool']
                self.set_name(event_pool_data.get('name', ''))
                if event_pool_data.get('events'):
                    for event_data in event_pool_data['events']:
                        event = Event()
                        event.deserialize(event_data)
                        self.objects.append(event)
                else:
                    self.objects = []
                    Log.warning("No events found")
            else:
                self.objects = []
                Log.warning("Event pool data is null")
            return self
        except (KeyError, TypeError) as e:
            Log.error(f"Error during event_pool deserialization: {e}")
            return self