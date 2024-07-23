import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from message import Log

class PercussionFeatureExtractor:
    def __init__(self):
        self.name = "PercussionFeatureExtractor"
        self.type = "PercussionFeatureExtractor"
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=44100,
            n_fft=400,
            hop_length=160,
            n_mels=64
        )
        Log.info("Initialized PercussionFeatureExtractor")

    def extract_features(self, audio_tensor):
        t, sr = audio_tensor
        Log.info("Extracting percussion features")
        mel_spec = self.mel_spectrogram(t)
        mel_spec = mel_spec.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            try:
                features = self.model(mel_spec)
            except Exception as e:
                Log.error(f"Error extracting features: {str(e)}")
                features = None
            Log.info(f"Type of features: {type(features)}")

        return features

    def apply(self, audio_object):
        if audio_object.tensor is not None:
            Log.info(f"Tensor exists {audio_object.tensor}")
            features = self.extract_features(audio_object.tensor)
            audio_object.add_features(features)
            Log.info(f"Extracted percussion features for audio: {audio_object.name}")
        else:
            Log.error(f"No tensor data found for audio: {audio_object.name}")