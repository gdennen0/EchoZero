from message import Log
class Command:
    def __init__(self, model):
        self.model = model
        Log.info("Initialized Command Module")
        pass

    def add_audio(self, a): # top level function to add audio to main program
        Log.command(f"Command initiated: 'add_audio'")
        self.model.audio.add(a)