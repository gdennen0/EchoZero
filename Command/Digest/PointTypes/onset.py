import librosa
from message import Log

class Onset(Point):
    """
    returns: array of events
    """
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