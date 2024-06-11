from Model.tools import Log

class Command:
    def __init__(self, model):
        self.model = model
        pass

    def add_audio(self, a): # top level function to add audio to main program
        self.model.audio.add(a)
        Log.debug(f"Command: add_audio")
        