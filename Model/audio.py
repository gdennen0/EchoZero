from message import Log
from tools import prompt, path_exists, yes_no_prompt, create_audio_tensor, create_audio_data
import os
import json

# ==================
# Audio Model Class
# ==================
class audio_model:
    # Model to store instances of audio in
    def __init__(self):
        self.objects = []
        self.selected_audio = None
        Log.info("Initialized Audio Model")

    def serialize(self):
        try:
            serialized_objects = []
            for obj in self.objects:
                serialized_objects.append(obj.serialize())
            return serialized_objects
        except Exception as e:
            Log.error(f"Error during audio_model serialization: {e}")
    
    def deserialize(self, data):
        try:
            self.objects = []
            self.selected_audio = None
            for obj_data in data:
                new_audio = audio()
                new_audio.deserialize(obj_data)
                self.objects.append(new_audio)
            Log.info("Audio model deserialized successfully")
        except Exception as e:
            Log.error(f"Error during audio_model deserialization: {e}")
    
    def reset(self):
        self.objects = []
        self.selected_audio = None
        Log.info("Reset Audio Model")

    def select(self, index):
        Log.info(f"Attempting to select audio object with index {index}")
        if index is not None:
            if index < len(self.objects):
                self.selected_audio = self.objects[index]
                Log.info(f"Selected audio {self.selected_audio.name} at index: {index}")
                return
            else:
                Log.error("Index out of range")
        else:
            Log.error(f"Passed index has a None value '{index}' to an audio object")
                    
    def list(self):
        Log.info('*' * 20 + "AUDIO OBJECTS" + '*' * 20)
        for index, object in enumerate(self.objects):
            Log.info(f"Index: {index}, Audio Name: {object.name}")
        Log.info('*' * 53)

    def generate_audio_object(self, dir, extension, data, tensor, sr, fps, type, name):
        Log.info("create_audio_object")
        # creates an audio object and updates the necessary data
        a = audio()
        a.set_dir(dir + f"/audio/{name}")
        a.set_extension(extension)
        a.set_audio(data)
        a.set_tensor(tensor)
        a.set_sample_rate(sr)
        a.set_frame_rate(fps)
        a.set_type(type)
        a.set_name(name)
        a.set_path()

        Log.info(f"Generated audio {a.name} to object")
        return a
    
    def add(self, a):
        if isinstance(a, audio):
            self.objects.append(a)
            Log.info("Added audio object to model objects")
        else:
            Log.error("Attempted to add a non-audio object.")

    def add_stems(self, stems):
        # adds passed stems to the selected audio objects stems list
        if self.selected_audio: # check to see if audio object is selected
            for stem in stems:
                self.selected_audio.stems.append(stem)
                print(f"added stem '{stem.name} to {self.selected_audio.name}")
        else: 
            Log.error("No audio object selected, please ")

    def delete(self, a_index):
        name = self.objects[a_index].name
        del self.objects[a_index]
        Log.warning(f"Deleted audio object '{name}' at index: {a_index}")

    def rename(self, a_index, new_name):
        old_name = self.objects[a_index].name
        self.objects[a_index].name = new_name
        Log.info(f"Renamed {old_name} to {new_name}")
    
    def get_audio_file_path(self, index):
        return self.objects[index].directory + f"/{self.objects[index].name}.mp3" # FIX THIS TO HAVE DYNAMIC EXTENSIONS

    def get_stems_file_path(self, index):
        stems_path = self.objects[index].directory + "/Stems"
        return stems_path
    
    def get_tensor(self, index):
        return self.objects[index].tensor
    
    def get_sr(self, index):
        return self.objects[index].sample_rate
    
    def get_audio(self, index):
        return self.objects[index].audio
    
    def get_object(self, index):
        return self.objects[index]
    
# ===================
# Audio Object Class
# ===================

class audio:
    # Audio Object
    def __init__(self):
        self.directory = None
        self.path = None
        self.audio = None
        self.sample_rate = None
        self.frame_rate = None
        self.type = None
        self.name = None
        self.length_ms = None
        self.processed_status = None
        self.event_pools = []
        self.tensor = None
        self.stems = []
    
    def serialize(self):
        try:
            data = {
                "directory": self.directory,
                "path": self.path,
                "sample_rate": self.sample_rate,
                "frame_rate": self.frame_rate,
                "type": self.type,
                "name": self.name,
                "length_ms": self.length_ms,
                "processed_status": self.processed_status,
                "event_pools": [pool.serialize() for pool in self.event_pools],
                "tensor": self.tensor is not None,
                "stems": [stem.serialize() for stem in self.stems],
            }
            # Log.info(f"{self.name} Completed audio serialization")
            return data
        except Exception as e:
            Log.error(f"Error during audio serialization: {e}")
    
    def deserialize(self, data):
        try:
            data_dict = data
            self.directory = data_dict["directory"]
            self.path = data_dict['path']
            self.sample_rate = data_dict["sample_rate"]
            self.frame_rate = data_dict["frame_rate"]
            self.type = data_dict["type"]
            self.name = data_dict["name"]
            self.length_ms = data_dict["length_ms"]
            self.processed_status = data_dict["processed_status"]
            if 'event_pools' in data:
                for pool_data in data['event_pools']:
                    event_pool_object = event_pool()
                    event_pool_object.deserialize(pool_data)
                    self.add_event_pool(event_pool_object)
                    # Log.info(f"Deserialized event pool {event_pool_object.name}")

            if data_dict['tensor']:
                self.tensor = create_audio_tensor(self.path, self.sample_rate)
            
            for stem in data_dict['stems']:
                s = Stem()
                s.deserialize(stem)
                self.stems.append(s)
                # Log.info(f"Appended stem {s.name}")

            # Log summary of all deserialized information
            # Log.info(f"Audio object deserialized with directory: {self.directory}, sample rate: {self.sample_rate}, "
            #         f"frame rate: {self.frame_rate}, type: {self.type}, name: {self.name}, length (ms): {self.length_ms}, "
            #         f"processed status: {self.processed_status}, tensor exists: {self.tensor is not None}, "
            #         f"stems exist: {self.stems is not None}")
        except Exception as e:
            Log.error(f"Error during audio deserialization: {e}")

    def set_dir(self, dir):
        self.directory = dir
        Log.info(f"Audio directory set to '{self.directory}'")

    def set_extension(self, extension):
        self.extension = extension
        Log.info(f"Extension set to '{extension}'")

    def set_path(self):
        if self.extension:
            self.path = self.directory + f"/{self.name}{self.extension}"
            Log.info(f"set audio path to {self.path}")
        else:
            Log.error(f"No extension set for file, cannot set path for audio obj")

    def set_audio(self, data):
        self.audio = data
        Log.info(f"Audio data loaded")

    def set_tensor(self, t):
        self.tensor = t
        Log.info(f"Set tensor data")
        
    def set_sample_rate(self, rate):
        # Sets the sample_rate variable 
        self.sample_rate = rate
        Log.info(f"Sample rate set to {rate}")

    def set_frame_rate(self, rate):
        # Sets the frame_rate variable 
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_type(self, type):
        # Sets the type variable 
        self.type = type
        Log.info(f"Type set to {type}")

    def set_name(self, name):
        # Sets the name variable
        self.name = name
        Log.info(f"Name set to {name}")

    def set_length_ms(self, length):
        # Sets the length_ms variable 
        self.length_ms = length
        Log.info(f"Length in ms set to {length}")

    def set_processed_status(self, status):
        # Sets the processed_status variable
        self.processed_status = status
        Log.info(f"Processed status set to {status}")

    def get_audio_metadata(self):
        metadata = self.serialize()
        return metadata
    
    def get_audio_file_path(self):
        return self.path 
    
    def add_stem(self, path):
        stem_name = os.path.basename(path).split('.')[0]
        audio_data, _ = create_audio_data(path, self.sample_rate)  # Load with native sampling rate
        t, _ = create_audio_tensor(path, self.sample_rate)
        s = Stem()
        s.set_name(stem_name)
        s.set_path(path)
        s.set_audio(audio_data)
        s.set_tensor(t)
        s.set_sample_rate(self.sample_rate)
        self.stems.append(s)
        generate_metadata(s)

    def add_event_pool(self, event_pool_object):
        self.event_pools.append(event_pool_object)
        Log.info(f"added event pool object to stem object named '{self.name}'")

def generate_metadata(s):
    stem_metadata = {
        "name": s.name,
        "path": s.path,
    }
    metadata_path = os.path.splitext(s.path)[0] + "_metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(stem_metadata, f, indent=4)
            Log.info(f"Metadata for stem '{s.name}' saved to {metadata_path}")
    except Exception as e:
        Log.error(f"Error during metadata generation: {e}")


class Stem:
    def __init__(self):
        self.path = None
        self.name = None
        self.audio = None
        self.sample_rate = None
        self.frame_rate = None
        self.tensor = None
        self.event_pools = []

    def set_name(self, name):
        self.name = name
        Log.info(f"set stem name to {name}")
    
    def set_path(self, path):
        self.path = path
        Log.info(f"set stem path to {self.path}")

    def set_audio(self, audio):
        self.audio = audio
        Log.info(f"set stem object audio")
    
    def set_sample_rate(self, sr):
        self.sample_rate = sr
        Log.info(f"set sample rate")

    def set_frame_rate(self, rate):
        # Sets the frame_rate variable 
        self.frame_rate = rate
        Log.info(f"Frame rate set to {rate}")

    def set_tensor(self, tensor):
        self.tensor = tensor
        Log.info(f"set stem object tensor")

    def add_event_pool(self, event_pool_object):
        self.event_pools.append(event_pool_object)
        Log.info(f"added event pool object to stem object named '{self.name}'")

    def serialize(self):
        """
        Serializes the stem data into a JSON format.
        
        Args:
        stem (Stem): The stem object to serialize.
        
        Returns:
        dict: A dictionary containing serialized stem data.
        """
        try:
            stem_data = {
                'stem':{
                    "name": self.name,
                    "path": self.path,
                    "sample_rate" : self.sample_rate,
                    "frame_rate": self.frame_rate,
                    "audio": self.audio is not None,
                    "tensor": self.tensor is not None,
                    "event_pools": [pool.serialize() for pool in self.event_pools]
                }
            }
            # Log.info(f"{self.name} Completed stem serialization")
            return stem_data
        except Exception as e:
            Log.error(f"Error during stem serialization: {e}")
    
    def deserialize(self, data):
        try:
            stem_object_data = data['stem']
            self.set_name(stem_object_data.get('name', None))
            self.set_path(stem_object_data.get('path', None))
            self.set_sample_rate(stem_object_data.get('sample_rate', None))
            self.set_frame_rate(stem_object_data.get('frame_rate', None))
            t, _ = create_audio_tensor(self.path, self.sample_rate)
            self.set_tensor(t)
            a, _ = create_audio_data(self.path, self.sample_rate)
            self.set_audio(a)

            if 'event_pools' in stem_object_data:
                for pool_data in stem_object_data['event_pools']:
                    event_pool_object = event_pool()
                    event_pool_object.deserialize(pool_data)
                    self.add_event_pool(event_pool_object)
                    # Log.info(f"Deserialized event pool {event_pool_object.name}")
            # Log.info(f"Deserialized stem object named '{self.name}'")
        except Exception as e:
            Log.error(f"Error during stem deserialization: {e}")

class event_pool:
    def __init__(self):
        self.name = None
        self.objects = []

    def set_name(self, name):
        self.name = name
        Log.info(f"Event pool name set to {name}")

    def add_event(self, event):
        self.objects.append(event)
        Log.info(f"Event added: {event}")

    def set_event_list(self, event_list):
        self.objects = event_list
        Log.info(f"Event list set with {len(event_list)} events")

    def serialize(self):
        try:
            event_pool_data = {
                'event_pool': {
                    "name": self.name,
                    "events": [event.serialize() for event in self.objects if event is not None] if self.objects else False
                }
            }
            # Log.info(f"Serialized event_pool: {event_pool_data}")
            # Log.info(f"{self.name} Completed event_pool serialization")
            return event_pool_data
        except Exception as e:
            Log.error(f"Error during event_pool serialization: {e}")

    def deserialize(self, data):
        try: 
            if data:
                if 'event_pool' in data:
                    event_pool_data = data['event_pool']
                    # self.set_name(event_pool_data.get('name', None))
                    if event_pool_data['events']:
                        if event_pool_data['events'] is not None:
                            for event_data in event_pool_data['events']:
                                if event_data:
                                    e = event()
                                    e.deserialize(event_data)
                                    self.objects.append(e)
                    else:
                        self.objects = []
                        Log.warning("No events found")
                    # Log.info(f"Deserialized event pool for stem: '{self.name}'")
            else: 
                self.objects = []
                Log.warning("event pool deserialize didnt load, pool_data is null")
        except Exception as e:
            Log.error(f"Error during event_pool deserialization: {e}")

class event:
    def __init__(self):
        self.frame = None,
        self.name = None,
        self.category = None,
    
    def set_frame(self, frame):
        try:
            self.frame = int(frame)
            Log.info(f"Frame set to {self.frame}")
        except ValueError:
            Log.error(f"Invalid frame value: {frame}. Could not convert to int.")
    def set_name(self, name):
        try:
            if not isinstance(name, str):
                Log.error(f"Invalid name value: {name}. Name must be a string.")
                return
            self.name = name.strip()
            Log.info(f"Name set to {self.name}")
        except ValueError:
            Log.error(f"Invalid event name: '{name}' could not convert to string")

    def set_category(self, category):
        try:
            if not isinstance(category, str):
                Log.error(f"Invalid category value: {category}. Category must be a string.")
                return
            self.category = category.strip()
            Log.info(f"Category set to {self.category}")
        except ValueError:
            Log.error(f"Invalid event category: '{category}' could not convert to string")

    def serialize(self):
        try:
            info = {
                'event': {
                    "frame": self.frame,
                    "name": self.name,
                    "category": self.category,
                }
            }
            # Log.info(f"{self.name} Completed event serialization")

            # Log.info(f"Serialized event: {info}")
            return info
        except Exception as e:
            Log.error(f"Error during event serialization: {e}")

    def deserialize(self, data):
        try:
            if 'event' in data:
                event_data = data['event']
                self.frame = event_data.get('frame', None)
                self.name = event_data.get('name', None)
                self.category = event_data.get('category', None)
        except Exception as e:
            Log.error(f"Error during event deserialization: {e}")
