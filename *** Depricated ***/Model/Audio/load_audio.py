from Utils.message import Log
import librosa
import os

class AudioInstance:
    def __init__(self, selected_file=None, audio_dir=None):
        self.file_path = None
        self.sample_rate = None
        self.audioarray = None
        if selected_file:
            filename = selected_file[0] if isinstance(selected_file, tuple) else selected_file
            self.file_path = os.path.join(audio_dir, filename) if audio_dir else filename

    def loadaudio(self):
        if not self.file_path:
            Log.error("File path is not set.")
            return

        if not os.path.isfile(self.file_path):
            Log.error(f"File does not exist: {self.file_path}")
            return

        try:
            self.audioarray, self.sample_rate = librosa.load(self.file_path, sr=None)
            Log.info(f'Audio loaded from: {self.file_path}')
        except Exception as e:
            Log.error(f"Failed to load audio: {e}")

    