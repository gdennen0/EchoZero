from .audio import Audio
import os 
from message import Log

class AudioModel:
    """
    Model to store instances of audio.
    """
    def __init__(self):
        self.objects = [] # list of different audio objects
        self.selected_audio = None
        Log.info("Initialized Audio Model")

    def serialize(self):
        try:
            return [obj.serialize() for obj in self.objects]
        except (AttributeError, TypeError) as e:
            Log.error(f"Error during audio_model serialization: {e}")
            return []

    def deserialize(self, data):
        """
        Deserializes the audio model objects from data.
        """
        try:
            self.objects = [Audio().deserialize(obj_data) for obj_data in data]
            self.selected_audio = None
            Log.info("Audio model deserialized successfully")
        except (KeyError, TypeError) as e:
            Log.error(f"Error during audio_model deserialization: {e}")

    def reset(self):
        """
        Resets the audio model.
        """
        self.objects = []
        self.selected_audio = None
        Log.info("Reset Audio Model")

    def select(self, index):
        """
        Selects an audio object by index.
        """
        Log.info(f"Attempting to select audio object with index {index}")
        if index is not None:
            if 0 <= index < len(self.objects):
                self.selected_audio = self.objects[index]
                Log.info(f"Selected audio {self.selected_audio.name} at index: {index}")
            else:
                Log.error("Index out of range")
        else:
            Log.error(f"Passed index has a None value '{index}' to an audio object")

    def list(self):
        """
        Lists all audio objects.
        """
        Log.info('*' * 20 + "AUDIO OBJECTS" + '*' * 20)
        for index, obj in enumerate(self.objects):
            Log.info(f"Index: {index}, Audio Name: {obj.name}")
        Log.info('*' * 53)

    def generate_audio_object(self, dir, extension, data, tensor, sr, fps, type, name):
        """
        Generates an audio object and updates the necessary data.
        """
        Log.info("create_audio_object")
        a = Audio()
        a.set_dir(os.path.join(dir, "audio", name))
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
        """
        Adds an audio object to the model.
        """
        if isinstance(a, Audio):
            self.objects.append(a)
            Log.info("Added audio object to model objects")
        else:
            Log.error("Attempted to add a non-audio object.")

    def add_stems(self, stems):
        """
        Adds passed stems to the selected audio object's stems list.
        """
        if self.selected_audio:
            for stem in stems:
                self.selected_audio.stems.append(stem)
                Log.info(f"Added stem '{stem.name}' to {self.selected_audio.name}")
        else:
            Log.error("No audio object selected, please select an audio object first.")

    def delete(self, a_index):
        """
        Deletes an audio object by index.
        """
        try:
            name = self.objects[a_index].name
            del self.objects[a_index]
            Log.warning(f"Deleted audio object '{name}' at index: {a_index}")
        except IndexError:
            Log.error(f"Index {a_index} out of range. Cannot delete audio object.")

    def rename(self, a_index, new_name):
        """
        Renames an audio object by index.
        """
        try:
            old_name = self.objects[a_index].name
            self.objects[a_index].name = new_name
            Log.info(f"Renamed {old_name} to {new_name}")
        except IndexError:
            Log.error(f"Index {a_index} out of range. Cannot rename audio object.")

    def get_audio_file_path(self, index):
        """
        Gets the file path of an audio object by index.
        """
        try:
            return os.path.join(self.objects[index].directory, f"{self.objects[index].name}.mp3")
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get audio file path.")
            return ""

    def get_stems_file_path(self, index):
        """
        Gets the stems file path of an audio object by index.
        """
        try:
            return os.path.join(self.objects[index].directory, "Stems")
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get stems file path.")
            return ""

    def get_tensor(self, index):
        """
        Gets the tensor of an audio object by index.
        """
        try:
            return self.objects[index].tensor
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get tensor.")
            return None

    def get_sr(self, index):
        """
        Gets the sample rate of an audio object by index.
        """
        try:
            return self.objects[index].sample_rate
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get sample rate.")
            return 0

    def get_audio(self, index):
        """
        Gets the audio data of an audio object by index.
        """
        try:
            return self.objects[index].audio
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get audio data.")
            return None

    def get_object(self, index):
        """
        Gets an audio object by index.
        """
        try:
            return self.objects[index]
        except IndexError:
            Log.error(f"Index {index} out of range. Cannot get audio object.")
            return None