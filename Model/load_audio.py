import torchaudio
import torchaudio.Transform as transforms

# ingest_audio class loads & standardizes the audio file's samplerate
# Args AUDIO_FILE should be the file location, and the TARGET_SAMPLERATE should be the constant for the target samplerate

class load_audio(AUDIO_FILE, TARGET_SAMPLERATE):
    def __init__(self):
        print("CLASS: inget_audio: __init__")
        self.audio_file = AUDIO_FILE
        self.target_samplerate = TARGET_SAMPLERATE
        self.samplerate = None
        self.waveform = None

    def load_audio(self):
        self.waveform, self.samplerate = torchaudio.load(AUDIO_FILE)
        print(f"Loaded Audio Waveform: {self.waveform}")
        print(f"Loaded Audio Samplerate: {self.samplerate}")

    def standardize_samplerate(self):
        rate_transform = transforms.Resample(self.samplerate, self.target_samplerate)
        resampled_waveform = rate_transform(self.waveform)
        return resampled_waveform

    def standardize_samplerate_logic(self):
        if self.samplerate > self.target_samplerate:
            print(f"Target Sample Rate ({self.target_samplerate}) is less than current sample rate ({self.samplerate})")
            self.waveform = self.standardize_samplerate(self.waveform)
            return self.waveform
        elif self.samplerate < self.target_samplerate:
            print(f"Target Sample Rate ({self.target_samplerate}) is greater than current sample rate ({self.samplerate})")
            self.waveform = self.standardize_samplerate(self.waveform)
            return self.waveform
        elif self.samplerate == self.target_samplerate:
            print(f"Target Sample Rate ({self.target_samplerate}) equals current sample rate ({self.samplerate})")
            pass