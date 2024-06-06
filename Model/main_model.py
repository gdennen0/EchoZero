from Model.audio import audio_model
from Model.tools import Log

class Main_Model:
    def __init__(self):
        self.audio_model = audio_model()
        Log.debug("Created Audio Model")