from message import Log
import librosa
from tools import prompt_selection
from Data.Types.event_data import EventData
from Data.Types.event_item import EventItem
from Data.Types.audio_data import AudioData


DEFAULT_ONSET_METHOD = "default"
DEFAULT_PRE_MAX = 3
DEFAULT_POST_MAX = 3
DEFAULT_PRE_AVG = 3
DEFAULT_POST_AVG = 3
DEFAULT_DELTA = 0.7
DEFAULT_WAIT = 30

class OnsetDetection(Part): 
    name = "OnsetDetection"
    def __init__(self):
        super().__init__()
        self.set_name("OnsetDetection")  # maybe unnecessary now?
        self.type = "Analyze"

        self.detections = []
        self.onset_method = DEFAULT_ONSET_METHOD
        self.pre_max = DEFAULT_PRE_MAX
        self.post_max = DEFAULT_POST_MAX
        self.pre_avg = DEFAULT_PRE_AVG
        self.post_avg = DEFAULT_POST_AVG
        self.delta = DEFAULT_DELTA
        self.wait = DEFAULT_WAIT


    def start(self, audio_data):
        Log.info(f"Starting {self.name}")
        
        # Get audio data
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
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            pre_max=self.pre_max,
            post_max=self.post_max,
            pre_avg=self.pre_avg,
            post_avg=self.post_avg,
            delta=self.delta,
            wait=self.wait
        )

        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        event_data = EventData() # Initialize an EventData group

        for onset in onset_times:   # add an eventItem into the event_data group
            event = EventItem()
            event.set_name("onset")
            event.set_description("timestamp")
            event.set_data = onset

            event_data.add_item(event)

            Log.info(f"Onset Detected at time: {onset}")

        return event_data

    def set_onset_method(self, method_type_name):
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

