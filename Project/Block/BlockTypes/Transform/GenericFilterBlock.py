from Project.Data.Types.audio_data import AudioData
from Project.Block.block import Block
from Project.Block.Input.Types.audio_input import AudioInput
from Project.Block.Output.Types.audio_output import AudioOutput

from Utils.message import Log
import librosa
from Utils.tools import prompt_selection

class GenericFilterBlock(Block):
    name = "GenericFilter"
    def __init__(self):
        super().__init__()
        self.name = "GenericFilter"
        self.type = "GenericFilter"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")
        
        self.output.add_type(AudioOutput)
        self.output.add("AudioOutput")

        self.filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        self.filter_type = None
        self.cutoff = None
        self.cutoff_low = None
        self.cutoff_high = None

        self.command.add("list_filter_types", self.list_filter_types)
        self.command.add("set_filter_type", self.set_filter_type)
        self.command.add("start", self.start)

        Log.info(f"GenericFilter initialized")

    def list_filter_types(self):
        Log.list("Filter Types", self.filter_types)

    def set_filter_type(self, filter_type):
        if filter_type in self.filter_types:
            self.filter_type = filter_type
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
        
    def set_cutoff(self):
        if self.filter_type is None:
            Log.error("No filter type set for filtering.")
            return
        
        if self.filter_type == "bandpass" or self.filter_type == "bandstop":
            self.cutoff_low = prompt_selection("Please enter the cutoff frequencies (low): ", self.cutoff_low)
            self.cutoff_high = prompt_selection("Please enter the cutoff frequencies (high): ", self.cutoff_high)

            Log.info(f"Cutoff low set to {self.cutoff_low}")
            Log.info(f"Cutoff high set to {self.cutoff_high}")
        else:
            self.cutoff = prompt_selection("Please enter the cutoff frequencies: ")
            Log.info(f"Cutoff set to {self.cutoff}")

    def start(self, audio_data):
        for audio_object in audio_data:
            if not isinstance(audio_object, AudioData):
                Log.error("Input is not an instance of AudioData")
                return audio_object

            if self.filter_type is None:
                Log.error("No filter type set for filtering.")
                return audio_data

            Log.info(f"Applying {self.attribute.get('filter_type')} filter to the audio data.")

            y = audio_object.audio
            sr = audio_object.sample_rate

            if self.filter_type == "lowpass":
                y_filtered = librosa.effects.low_pass(y, sr=sr, cutoff=self.cutoff)
            elif self.filter_type == "highpass":
                y_filtered = librosa.effects.high_pass(y, sr=sr, cutoff=self.cutoff)
            elif self.filter_type == "bandpass":
                y_filtered = librosa.effects.band_pass(y, sr=sr, low=self.cutoff_low, high=self.cutoff_high)
            elif self.filter_type == "notch":
                y_filtered = librosa.effects.notch_filter(y, sr=sr, freq=self.cutoff)
            else:
                Log.error(f"Invalid filter type: {self.filter_type}, filter not applied.")
                y_filtered = y

            audio_object.set_audio(y_filtered)
            Log.info(f"{self.filter_type} filter applied successfully")
            return audio_object

    def process(self, audio_data):
        processed_data = []
        processed_data.append(self.start(audio_data))
        return processed_data
    

    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "filter_type": self.filter_type,
            "cutoff": self.cutoff,
            "cutoff_low": self.cutoff_low,
            "cutoff_high": self.cutoff_high
        }


    def load(self, data):
        self.filter_type = data.get("filter_type")
        self.cutoff = data.get("cutoff")
        self.cutoff_low = data.get("cutoff_low")
        self.cutoff_high = data.get("cutoff_high")

        self.reload()