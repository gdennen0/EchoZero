from message import Log
from scipy.signal import butter, sosfilt


class HighPassFilter(Point):
    def __init__(self):
        super().__init__()  # Calls the initializer of the parent 'Point' class
        self.name = "HighPass Filter"
        self.type = "Filter"
        self.description = "Applies a highpass filter to audio data."

    def apply(self, data, sr=22050, cutoff=1000):
        
        # Generate a highpass filter
        sos = butter(10, cutoff, btype='highpass', fs=sr, output='sos')
        filtered_data = sosfilt(sos, data)
        Log.info(f"HighPass Filter | Applied highpass filter with cutoff at {cutoff} Hz")
        return filtered_data