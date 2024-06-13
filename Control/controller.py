from message import Log
from Control.load_audio import load_audio

# Main Controller Functions
"""
Responsible for routing the flow of information properly, this is the applications internal data exchange location

"""

class Control:
    def __init__(self, model):
        self.model = model          # i can see everything   # we have all the power now muahaa
        Log.info("Initialized Control Module")

    def load_audio(self, abs_path):
        Log.command(f"Command initiated: 'load_audio'")
        a = load_audio(abs_path)
        self.model.audio.add(a)

    