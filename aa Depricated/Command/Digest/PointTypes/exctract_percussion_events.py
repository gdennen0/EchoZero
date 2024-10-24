import librosa
import numpy as np
import soundfile as sf
import os
from .point import Point
from message import Log

class ExtractPercussionEvents(Point):
    def __init__(self):
        self.name = "ExtractPercussionEvents"
        self.type = "ExtractPercussionEvents"
        self.audio_path = None
        self.output_dir= None
        self.sr=22050
        self.energy_threshold=0.1 
        self.min_event_duration=0.5

#  creates a list of timestamps for percussion events. Iterates through the times, determines the start and end times for each sample, outputs the samples to project path

    def apply(self, audio_object):
        try:
            self.audio_path = audio_object.path 

            self.output_dir = os.path.join(os.path.dirname(audio_object.path), "perc")

            # Load the audio file
            Log.info(f"Ingesting path {self.audio_path} sr {self.sr}")
            y, sr = librosa.load(self.audio_path, sr=self.sr)

            
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            Log.info("Applied hpps")
            
            # Detect onsets in the percussive component
            onset_frames = librosa.onset.onset_detect(y=y_percussive, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            Log.info("Detected onsets")
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create audio clips for each detected onset
            for i, onset_time in enumerate(onset_times):
                # Define the start sample for the clip
                start_sample = int(onset_time * sr)
                
                # Determine the end sample based on energy threshold
                end_sample = start_sample
                while end_sample < len(y_percussive) and y_percussive[end_sample]**2 > self.energy_threshold:
                    end_sample += 1
                
                # Ensure minimum event duration
                if (end_sample - start_sample) < int(self.min_event_duration * sr):
                    end_sample = start_sample + int(self.min_event_duration * sr)
                
                # Extract the audio clip
                clip = y_percussive[start_sample:end_sample]
                
                # Save the audio clip
                time_start = librosa.samples_to_time(start_sample, sr=sr)
                time_end = librosa.samples_to_time(end_sample, sr=sr)
                output_path = f"{self.output_dir}/{i+1}_{time_start:.2f}_{time_end:.2f}.wav"
                sf.write(output_path, clip, sr)
                Log.info(f"Saved: {output_path}")
        except Exception as e:
            Log.error(f"An error occurred while extracting percussion events: {e}")

