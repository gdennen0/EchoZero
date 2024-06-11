from Model.audio import audio_model
from Model.tools import Log

class Model:
    def __init__(self):
        Log.debug("")
        self.audio = audio_model()
        Log.debug("Created Audio Model")