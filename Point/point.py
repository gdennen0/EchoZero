import librosa
from message import Log

# POINT CLASS
# The parent class for all point sub-objects, applicable broadly
class Point:
    def __init__(self):
        self.name = None
        self.type = None
        self.description = None

    def apply(self, data):
        Log.info(f"Point transform | Applying transformation: {self.name}")
        return data
    
class HighPassFilter(Point):
    def __init__(self):
        super().__init__()  # Calls the initializer of the parent 'Point' class
        self.name = "HighPass Filter"
        self.type = "Filter"
        self.description = "Applies a highpass filter to audio data."

    def apply(self, data, sr=22050, cutoff=1000):
        from scipy.signal import butter, sosfilt
        
        # Generate a highpass filter
        sos = butter(10, cutoff, btype='highpass', fs=sr, output='sos')
        filtered_data = sosfilt(sos, data)
        Log.info(f"HighPass Filter | Applied highpass filter with cutoff at {cutoff} Hz")
        return filtered_data

        
class Onset(Point):
    def __init__(self):
        super().__init__()  # Calls the initializer of the parent 'Point' class
        self.name = "Onset"
        self.type = "Onset"
        self.description = "Detects onsets in audio data using Librosa."

    def apply(self, data):
        onset_env = librosa.onset.onset_strength(y=data, sr=22050)
        events = librosa.onset.onset_detect(onset_envelope=onset_env, sr=22050)
        Log.info(f"Onset Detection | Detected {len(events)} events")
        return events