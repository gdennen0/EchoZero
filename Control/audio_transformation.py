from Model.audio import audio
from message import Log
import torchaudio
from Model.transformations import stem_generation


# audio_transformation.py applies audio transformations to the audio tensor

# pre_transformation class performs pretransformations to the audio tensor (torchaudio)
class pre_transformation:
    def __init__(self, audio_tensor, sr, cutoff_freq, Q):
        self.t = audio_tensor
        self.sr = sr
        self.cut = cutoff_freq
        self.q = Q


    def highpass(self):
      torchaudio.functional.highpass_biquad(self.t, self.sr, self.cut, self.q)

    def lowpass(self):
        torchaudio.functional.lowwpass_biquad(self.t, self.sr, self.cut, self.q)


#stem_separation separates stems from audio data
class stem_separation:
    def __init__(self, audio_tensor, sr, input_file_path, output_filepath, ai_model):
        self.at = audio_tensor
        self.sr = sr
        self.input_file_path = input_file_path
        self.output_filepath = output_filepath
        self.ai_model = ai_model
        self.generate_stems = stem_generation(self.at, self.sr, self.input_file_path, self.output_filepath, self.ai_model)
