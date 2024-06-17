from message import Log
from audio_separator.separator import Separator
from message import Log

# ===================
# Audio Object Class
# ===================


# Separates audio file into stems
class stem_generation:
    def __init__(self, audio_tensor, sr, input_filepath, output_filepath, ai_model):
        self.at = audio_tensor # torch audio tensor
        self.sr = sr # sample rate
        self.input_filepath = input_filepath
        self.output_file = output_filepath
        self.model = ai_model
        self.separator = Separator() # Initializes an instance of the separator
        
    def separate_stems(self):
        self.separator.load_model(self.model) # loads a model based on user preferences
        self.output_file = self.separator.separate(self.output_file)
        Log.info(f'Separating Stems for file: {self.input_filepath} at {self.output_file}')
