from Project.Block.block import Block
from Project.Data.Types.audio_data import AudioData
from Project.Block.Input.Types.audio_input import AudioInput
from Project.Block.Output.Types.event_output import EventOutput
from Project.Data.Types.event_data import EventData
from Project.Data.Types.event_item import EventItem
from Utils.tools import prompt_selection, prompt
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
        self.command.add("set_pre_max", self.set_pre_max)
        self.command.add("set_post_max", self.set_post_max)
        self.command.add("set_pre_avg", self.set_pre_avg)
        self.command.add("set_post_avg", self.set_post_avg)
        self.command.add("set_delta", self.set_delta)
        self.command.add("set_wait", self.set_wait)
        self.command.add("list_settings", self.list_settings)

    def process(self, input_data):
        event_data_list = []
        for audio_data in input_data:
            y = audio_data.data
            sr = audio_data.sample_rate
            
            # Calculate onset strength
            if self.onset_method == "energy": # Suitable for percussive sounds where energy changes are prominent.
                onset_env = librosa.onset.onset_strength(y=y, feature=librosa.feature.rms)
            elif self.onset_method == "spectral_flux":  # Good for tracking changes in the spectral content, useful for complex sounds.
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            elif self.onset_method == "complex": # Captures both magnitude and phase information, beneficial for detailed onset detection.  d
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.complex_mel_spectrogram)
            elif self.onset_method == "melspectrogram": # Ideal for musical audio where perceptual aspects are important.
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=librosa.feature.melspectrogram)
            else:  # no transformation is applied
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

            for i, onset in enumerate(onset_times):
                event = EventItem()
                event.set_name(f"onset{i}")
                event.time = onset
                event.source = "DetectOnsets"
                event_data.add_item(event)
                event_data.set_source(audio_data)

            event_data_list.append(event_data)

        return event_data_list
    
    def set_onset_method(self, onset_method=None):
        method_types = {
            "default", 
            "energy", 
            "spectral_flux", 
            "complex", 
            "melspectrogram", 
            "rms",
        }
        if onset_method:
            for name in method_types:
                if name == onset_method:
                    self.onset_method = onset_method
                    Log.info(f"Onset method set to {onset_method}")
        else:
            onset_method = prompt_selection("Select method type to change to: ", method_types)
            self.onset_method = onset_method
            Log.info(f"Onset method set to {onset_method}")

    def set_pre_max(self, pre_max=None):
        if pre_max is None:
            pre_max = prompt(f"Enter new pre max value (current: {self.pre_max}): ")
        try:
            self.pre_max = int(pre_max)
            Log.info(f"Pre max set to {self.pre_max}")
        except ValueError:
            Log.error(f"Invalid value for pre_max: {pre_max}. It must be an integer.")

    def set_post_max(self, post_max=None):
        if post_max is None:
            post_max = prompt(f"Enter new post max value (current: {self.post_max}): ")
        try:
            self.post_max = int(post_max)
            Log.info(f"Post max set to {self.post_max}")
        except ValueError:
            Log.error(f"Invalid value for post_max: {post_max}. It must be an integer.")

    def set_pre_avg(self, pre_avg=None):
        if pre_avg is None:
            pre_avg = prompt(f"Enter new pre avg value (current: {self.pre_avg}): ")
        try:
            self.pre_avg = int(pre_avg)
            Log.info(f"Pre avg set to {self.pre_avg}")
        except ValueError:
            Log.error(f"Invalid value for pre_avg: {pre_avg}. It must be an integer.")

    def set_post_avg(self, post_avg=None):
        if post_avg is None:
            post_avg = prompt(f"Enter new post avg value (current: {self.post_avg}): ")
        try:
            self.post_avg = int(post_avg)
            Log.info(f"Post avg set to {self.post_avg}")
        except ValueError:
            Log.error(f"Invalid value for post_avg: {post_avg}. It must be an integer.")

    def set_delta(self, delta=None):
        if delta is None:
            delta = prompt(f"Enter new delta value (current: {self.delta}): ")
        try:
            self.delta = float(delta)
            Log.info(f"Delta set to {self.delta}")
        except ValueError:
            Log.error(f"Invalid value for delta: {delta}. It must be a float.")

    def set_wait(self, wait=None):
        if wait is None:
            wait = prompt(f"Enter new wait value (current: {self.wait}): ")
        try:
            self.wait = int(wait)
            Log.info(f"Wait set to {self.wait}")
        except ValueError:
            Log.error(f"Invalid value for wait: {wait}. It must be an integer.")

    def list_settings(self):
        Log.info(f"Block {self.name}s current settings:")
        Log.info(f"Onset method: {self.onset_method}")
        Log.info(f"Pre max: {self.pre_max}")
        Log.info(f"Post max: {self.post_max}")
        Log.info(f"Pre avg: {self.pre_avg}")
        Log.info(f"Post avg: {self.post_avg}")
        Log.info(f"Delta: {self.delta}")
        Log.info(f"Wait: {self.wait}")

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
        self.set_name(name=data.get("name"))
        self.set_type(type=data.get("type"))
        self.set_onset_method(onset_method=data.get("onset_method"))
        self.set_pre_max(pre_max=data.get("pre_max"))
        self.set_post_max(post_max=data.get("post_max"))
        self.set_pre_avg(pre_avg=data.get("pre_avg"))
        self.set_post_avg(post_avg=data.get("post_avg"))
        self.set_delta(delta=data.get("delta"))
        self.set_wait(wait=data.get("wait"))

        self.input.load(data.get("input")) # just need to reconnect the inputs
        self.reload()