from Project.Block.block import Block
from Project.Data.Types.audio_data import AudioData
from Project.Block.Input.Types.audio_input import AudioInput
from Project.Block.Output.Types.event_output import EventOutput
from Project.Data.Types.event_data import EventData
from Project.Data.Types.event_item import EventItem
from Utils.tools import prompt_selection
import librosa
from Utils.message import Log
import os
from pathlib import Path

DEFAULT_ONSET_METHOD = "default"
DEFAULT_PRE_MAX = 3
DEFAULT_POST_MAX = 3
DEFAULT_PRE_AVG = 3
DEFAULT_POST_AVG = 3
DEFAULT_DELTA = 0.7
DEFAULT_WAIT = 30

class DetectOnsetsBlock(Block):
    name = "DetectOnsets"
    def __init__(self):
        super().__init__()
        self.name = "DetectOnsets" 
        self.type = "DetectOnsets"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")      


        self.onset_method = DEFAULT_ONSET_METHOD
        self.pre_max = DEFAULT_PRE_MAX
        self.post_max = DEFAULT_POST_MAX
        self.pre_avg = DEFAULT_PRE_AVG
        self.post_avg = DEFAULT_POST_AVG
        self.delta = DEFAULT_DELTA
        self.wait = DEFAULT_WAIT

        self.command.add("set_onset_method", self.set_onset_method)
   

    def process(self, input_data):
        event_data_list = []
        for audio_data in input_data:
            y = audio_data.data
            sr = audio_data.sample_rate
            
            # Calculate onset strength
            if self.onset_method == "energy":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.rms)
            elif self.onset_method == "spectral_flux":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            elif self.onset_method == "complex":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.complex_mel_spectrogram)
            elif self.onset_method == "melspectrogram":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.melspectrogram)
            else:  # default
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # Detect onset frames
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                pre_max=self.pre_max,
                post_max=self.post_max,
                pre_avg=self.pre_avg,
                post_avg=self.post_avg,
                delta=self.delta,
                wait=self.wait
            )

            onset_times = librosa.frames_to_time(onsets, sr=sr)

            event_data = EventData()
            event_data.name = "OnsetEvents"
            event_data.description = "Event data of Onsets"

            for i, onset in onset_times:
                event = EventItem()
                event.set_name("onset")
                event.set_description("onset timestamp")
                event.time = onset
                event.source = "DetectOnsets"
                event_data.add_item(event)

            event_data_list.append(event_data)

        return event_data_list
    
    def set_onset_method(self, method_type_name=None):
        method_types = {
            "default", 
            "energy", 
            "spectral_flux", 
            "complex", 
            "melspectrogram", 
            "rms",
        }
        if method_type_name:
            for name in method_types:
                if name == method_type_name:
                    self.onset_method = method_type_name
                    Log.info(f"Onset method set to {method_type_name}")
        else:
            _ , method_type_name = prompt_selection("Select method type to change to: ", method_types)
            self.onset_method = method_type_name
            Log.info(f"Onset method set to {method_type_name}")



    def save(self):
        return {
            "name": self.name,
            "type": self.type,
            "onset_method": self.onset_method,
            "pre_max": self.pre_max,
            "post_max": self.post_max,
            "pre_avg": self.pre_avg,
            "post_avg": self.post_avg,
            "delta": self.delta,
            "wait": self.wait,
            "data": self.data.save(),
            "input": self.input.save(),
            "output": self.output.save()
        }
    
    def load(self, data):
        self.name = data.get("name")
        self.type = data.get("type")
        self.onset_method = data.get("onset_method")
        self.pre_max = data.get("pre_max")
        self.post_max = data.get("post_max")
        self.pre_avg = data.get("pre_avg")
        self.post_avg = data.get("post_avg")
        self.delta = data.get("delta")
        self.wait = data.get("wait")
