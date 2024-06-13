import librosa
import torchaudio
import torchaudio.transforms as transforms
from Model.audio import audio
from message import Log

# ingest_audio class loads & standardizes the audio file's samplerate
# Args AUDIO_FILE should be the file location, and the TARGET_SAMPLERATE should be the constant for the target samplerate

TARGET_SAMPLERATE = 44100
FRAMERATE = 30

def load_audio(audio_path, target_sr=TARGET_SAMPLERATE):
    # Loads all necessary audio data
    audio_data, _ = create_audio_data(audio_path, target_sr)   # creates audio data array using librosa
    t, _ = create_audio_tensor(audio_path, target_sr)  # creates a tensor object with the audio file
    a = create_audio_object(audio_data, t, target_sr, FRAMERATE, None, "Default")  # creates an audio object and updates the necessary data
    return a
    
def create_audio_data(audio_path, target_sr):
    # creates audio data array using librosa
    # standardise sample rate??
    data, sr = librosa.load(audio_path, sr=target_sr)
    return data, sr

def create_audio_tensor(audio_path, target_sr):
    # creates a tensor object with the audio file
    # standardise sample rate??
    audio_tensor, sr = torchaudio.load(audio_path, target_sr)
    return audio_tensor, sr

def create_audio_object(data, tensor, sr, fps, type, name):
    # creates an audio object and updates the necessary data
    a = audio()
    a.set_original_data(data)
    a.set_tensor(tensor)
    a.set_sample_rate(sr)
    a.set_frame_rate(fps)
    a.set_type(type)
    a.set_name(name)
    return a

# def standardize_samplerate():
#     rate_transform = transforms.Resample(samplerate, target_samplerate)
#     resampled_waveform = rate_transform(waveform)
#     return resampled_waveform

# def standardize_samplerate_logic():
#     if self.samplerate > self.target_samplerate:
#         print(f"Target Sample Rate ({self.target_samplerate}) is less than current sample rate ({self.samplerate})")
#         self.waveform = self.standardize_samplerate(self.waveform)
#         return self.waveform
#     elif self.samplerate < self.target_samplerate:
#         print(f"Target Sample Rate ({self.target_samplerate}) is greater than current sample rate ({self.samplerate})")
#         self.waveform = self.standardize_samplerate(self.waveform)
#         return self.waveform
#     elif self.samplerate == self.target_samplerate:
#         print(f"Target Sample Rate ({self.target_samplerate}) equals current sample rate ({self.samplerate})")
#         pass

