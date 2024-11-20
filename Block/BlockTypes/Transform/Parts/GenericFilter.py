from message import Log
from Block.part import Part
import librosa
from tools import prompt_selection
from DataTypes.audio_data_type import AudioData

class GenericFilter(Part):
    def __init__(self):
        super().__init__()
        self.name = "GenericFilter"
        self.block_type = "Transform"
        self.filter_types = ["lowpass", "highpass", "bandpass", "bandstop"]
        self.filter_type = None
        self.cutoff = None
        self.cutoff_low = None
        self.cutoff_high = None
        self.command.add("list_filter_types", self.list_filter_types)
        self.command.add("set_filter_type", self.set_filter_type)
        self.command.add("start", self.start)

        self.add_input_type(AudioData())
        self.add_output_type(AudioData())

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

            Log.info(f"Applying {self.filter_type} filter to the audio data.")

            try:
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
            except Exception as e:
                Log.error(f"Error applying librosa filter: {e}")
                return audio_object