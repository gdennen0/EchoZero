from message import Log
from Control.load_audio import load_audio
from Control.audio_transformation import stem_separation

# Main Controller Functions
"""
Responsible for routing the flow of information properly, this is the applications internal data exchange location

# Control is the only place that data can be modified from

"""

# class Control:
#     def __init__(self, model):
#         self.model = model          # i can see everything   # we have all the power now muahaa
#         Log.info("Initialized Control Module")

#     def load_audio(self, abs_path):
#         a, sr = load_audio(abs_path)
#         self.model.audio.add(a)

#     def select_audio(self, index):
#         self.model.audio.select(index)

#     def generate_stems(self, abs_path, output_filepath):       
#         Log.command(f"Command initiated: 'generate_stems'")
#         stems = stem_separation(None, None, abs_path, output_filepath, "Demucs")
#         self.model.audio.add_stems(stems)

    