from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Utils.tools import prompt_selection, prompt
import librosa
from src.Utils.message import Log
import os
from pathlib import Path
import numpy as np
from src.Project.Data.Types.audio_data import AudioData
import json

DEFAULT_ONSET_METHOD = "default"
DEFAULT_PRE_MAX = 3
DEFAULT_POST_MAX = 3
DEFAULT_PRE_AVG = 3
DEFAULT_POST_AVG = 3
DEFAULT_DELTA = 0.5
DEFAULT_WAIT = 10

class DetectOnsetsBlock(Block):
    """
    Detects onsets in audio using Librosa and extracts individual audio clips for each detected onset.
    Refines the end time of each onset by analyzing the short-time energy, helping avoid extra noise or trailing audio.
    """
    name = "DetectOnsets"
    type = "DetectOnsets"

    def __init__(self):
        super().__init__()
        self.name = "DetectOnsets"
        self.type = "DetectOnsets"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Default onset parameters
        self.onset_method = DEFAULT_ONSET_METHOD
        self.pre_max = DEFAULT_PRE_MAX
        self.post_max = DEFAULT_POST_MAX
        self.pre_avg = DEFAULT_PRE_AVG
        self.post_avg = DEFAULT_POST_AVG
        self.delta = DEFAULT_DELTA
        self.wait = DEFAULT_WAIT

        # Register commands for configuration
        self.command.add("set_onset_method", self.set_onset_method)
        self.command.add("set_pre_max", self.set_pre_max)
        self.command.add("set_post_max", self.set_post_max)
        self.command.add("set_pre_avg", self.set_pre_avg)
        self.command.add("set_post_avg", self.set_post_avg)
        self.command.add("set_delta", self.set_delta)
        self.command.add("set_wait", self.set_wait)
        self.command.add("list_settings", self.list_settings)

    def process(self, input_data):
        """
        Primary method for detecting onsets and extracting individual audio clips.
        """
        event_data_list = []

        for audio_data in input_data:
            audio_waveform = audio_data.data
            sample_rate = audio_data.sample_rate
            total_duration = librosa.get_duration(y=audio_waveform, sr=sample_rate)

            # Calculate onset envelope
            onset_envelope = self.calculate_onset_envelope(audio_waveform, sample_rate)

            # Detect onset frames
            onset_frames = librosa.onset.onset_detect(
                # onset_envelope: The array representing the onset strength curve of the audio
                # sr: The sample rate of the audio signal
                # pre_max: Number of frames before the current frame to use for local maximum filtering
                # post_max: Number of frames after the current frame to use for local maximum filtering
                # pre_avg: Number of frames before the current frame to use for local average filtering
                # post_avg: Number of frames after the current frame to use for local average filtering
                # delta: Onset detection threshold that determines how steep the increase in energy must be
                # wait: Minimum number of frames to wait between consecutive onsets
                onset_envelope=onset_envelope,
                sr=sample_rate,
                pre_max=self.pre_max,
                post_max=self.post_max,
                pre_avg=self.pre_avg,
                post_avg=self.post_avg,
                delta=self.delta,
                wait=self.wait,
            )

        

            onset_time_list = librosa.frames_to_time(onset_frames, sr=sample_rate)

            # Prepare a container for the events
            onset_events_data = EventData()
            onset_events_data.name = "OnsetEvents"
            onset_events_data.description = "Event data of single onsets from audio"
            onset_events_data.set_source(audio_data)

            for i, onset_time in enumerate(onset_time_list):
                # Default end time is the next onset or track end
                if i < len(onset_time_list) - 1:
                    default_end_time = onset_time_list[i + 1]
                else:
                    default_end_time = total_duration

                # Refine the end time using an energy-based approach
                refined_end_time = self.refine_onset_end(
                    audio_waveform, sample_rate, onset_time, default_end_time
                )

                start_sample = librosa.time_to_samples(onset_time, sr=sample_rate)
                end_sample = librosa.time_to_samples(refined_end_time, sr=sample_rate)

                # Slice the audio for this onset
                onset_clip = audio_waveform[start_sample:end_sample]

                # Create a new EventItem
                onset_event = EventItem()
                onset_event.set_name(f"onset_{i}")
                onset_event.time = onset_time
                onset_event.source = "DetectOnsets"

                # Embed audio data in event
                onset_audio_data = AudioData()
                onset_audio_data.set_name(f"onset_{i}")
                onset_audio_data.set_data(onset_clip)
                onset_audio_data.set_sample_rate(sample_rate)
                onset_audio_data.set_path(audio_data.path)

                onset_event.set_data(onset_audio_data)
                onset_events_data.add_item(onset_event)

            event_data_list.append(onset_events_data)

        return event_data_list

    def calculate_onset_envelope(self, audio_waveform, sample_rate):
        """
        Calculates the onset-strength envelope based on the selected onset method.
        """
        if self.onset_method == "energy":
            # We define a wrapper so that 'sr' is not passed as a keyword argument to rms().
            def custom_rms_wrapper(y, sr, n_fft, hop_length, center=True, **kwargs):
                # librosa.feature.rms does not accept 'sr', so we ignore it here
                # and pass the rest to the function. We also take the first row ([0])
                # since rms() returns a shape of (1, number_of_frames).
                return librosa.feature.rms(
                    y=y,
                    frame_length=n_fft,
                    hop_length=hop_length,
                    center=center,
                    **kwargs
                )[0]

            return librosa.onset.onset_strength(
                y=audio_waveform,
                sr=sample_rate,
                feature=custom_rms_wrapper
            )
        elif self.onset_method == "melspectrogram":
            return librosa.onset.onset_strength(
                y=audio_waveform,
                sr=sample_rate,
                feature=librosa.feature.melspectrogram
            )
        else:
            # "default" or any other mapping
            return librosa.onset.onset_strength(
                y=audio_waveform, 
                sr=sample_rate
            )

    def refine_onset_end(self, audio_waveform, sr, onset_time, next_onset_time, max_length_time=1.0):
        """
        Refine the end time of an onset by analyzing short-time RMS within the region
        from onset_time to next_onset_time.
        Also enforce a maximum length for the onset.

        1. Find the short-time RMS in the given segment.
        2. Determine the peak RMS in that segment.
        3. Once the RMS dips below a fraction of that peak for a short duration, 
            we consider it the end.
        4. If it never dips, fall back to next_onset_time.
        5. Ensure the onset length does not exceed max_length_time.
        """
        # Convert to samples for slicing
        start_sample = librosa.time_to_samples(onset_time, sr=sr)
        desired_end_time = onset_time + max_length_time
        end_time_cap = min(desired_end_time, next_onset_time)
        end_sample = librosa.time_to_samples(end_time_cap, sr=sr)
        if end_sample <= start_sample:
            return end_time_cap  # Fallback if there's an anomaly

        segment = audio_waveform[start_sample:end_sample]

        # Calculate short-time RMS for this segment
        rms_values = librosa.feature.rms(y=segment)[0]
        times_relative = librosa.frames_to_time(range(len(rms_values)), sr=sr)
        if len(rms_values) == 0:
            return end_time_cap

        peak_rms = np.max(rms_values)
        if peak_rms <= 0:
            return end_time_cap

        # Define a fraction of the peak RMS to consider "decay"
        decay_fraction = 0.2  
        decay_threshold = decay_fraction * peak_rms

        # Define how many consecutive frames must remain below threshold
        # before we decide it's the real "end"
        min_decay_frames = 4

        below_threshold_count = 0
        for i, rms_val in enumerate(rms_values):
            if rms_val < decay_threshold:
                below_threshold_count += 1
            else:
                below_threshold_count = 0

            if below_threshold_count >= min_decay_frames:
                # Compute actual time offset within the segment
                refined_time = onset_time + times_relative[i]
                # Make sure we don't exceed the end_time_cap
                refined_time = min(refined_time, end_time_cap)
                return refined_time

        return end_time_cap

    def set_onset_method(self, onset_method=None):
        """
        Set the onset detection method.
        """
        valid_methods = {
            "default", 
            "energy", 
            "mel_spectrogram",
        }
        if onset_method:
            if onset_method in valid_methods:
                self.onset_method = onset_method
                Log.info(f"Onset method set to {onset_method}")
            else:
                Log.error(f"Invalid method. Choose from: {valid_methods}")
        else:
            onset_method = prompt_selection("Select method type to change to: ", valid_methods)
            self.onset_method = onset_method
            Log.info(f"Onset method set to {onset_method}")

    def set_pre_max(self, pre_max=None):
        """
        Set the pre_max parameter for onset detection.
        """
        if pre_max is None:
            pre_max = prompt(f"Enter new pre_max value (current: {self.pre_max}): ")
        try:
            self.pre_max = int(pre_max)
            Log.info(f"Pre max set to {self.pre_max}")
        except ValueError:
            Log.error(f"Invalid value for pre_max: {pre_max}. Must be an integer.")

    def set_post_max(self, post_max=None):
        """
        Set the post_max parameter for onset detection.
        """
        if post_max is None:
            post_max = prompt(f"Enter new post_max value (current: {self.post_max}): ")
        try:
            self.post_max = int(post_max)
            Log.info(f"Post max set to {self.post_max}")
        except ValueError:
            Log.error(f"Invalid value for post_max: {post_max}. Must be an integer.")

    def set_pre_avg(self, pre_avg=None):
        """
        Set the pre_avg parameter for onset detection.
        """
        if pre_avg is None:
            pre_avg = prompt(f"Enter new pre_avg value (current: {self.pre_avg}): ")
        try:
            self.pre_avg = int(pre_avg)
            Log.info(f"Pre avg set to {self.pre_avg}")
        except ValueError:
            Log.error(f"Invalid value for pre_avg: {pre_avg}. Must be an integer.")

    def set_post_avg(self, post_avg=None):
        """
        Set the post_avg parameter for onset detection.
        """
        if post_avg is None:
            post_avg = prompt(f"Enter new post_avg value (current: {self.post_avg}): ")
        try:
            self.post_avg = int(post_avg)
            Log.info(f"Post avg set to {self.post_avg}")
        except ValueError:
            Log.error(f"Invalid value for post_avg: {post_avg}. Must be an integer.")

    def set_delta(self, delta=None):
        """
        Set the delta parameter for onset detection (threshold).
        """
        if delta is None:
            delta = prompt(f"Enter new delta value (current: {self.delta}): ")
        try:
            self.delta = float(delta)
            Log.info(f"Delta set to {self.delta}")
        except ValueError:
            Log.error(f"Invalid value for delta: {delta}. Must be a float.")

    def set_wait(self, wait=None):
        """
        Set the wait parameter for onset detection (minimum frames between onsets).
        """
        if wait is None:
            wait = prompt(f"Enter new wait value (current: {self.wait}): ")
        try:
            self.wait = int(wait)
            Log.info(f"Wait set to {self.wait}")
        except ValueError:
            Log.error(f"Invalid value for wait: {wait}. Must be an integer.")

    def list_settings(self):
        """
        Log all current settings for the user to review.
        """
        Log.info(f"Block {self.name}'s current settings:")
        Log.info(f"Onset method: {self.onset_method}")
        Log.info(f"Pre max: {self.pre_max}")
        Log.info(f"Post max: {self.post_max}")
        Log.info(f"Pre avg: {self.pre_avg}")
        Log.info(f"Post avg: {self.post_avg}")
        Log.info(f"Delta: {self.delta}")
        Log.info(f"Wait: {self.wait}")


    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "onset_method": self.onset_method,
            "pre_max": self.pre_max,
            "post_max": self.post_max,
            "pre_avg": self.pre_avg,
            "post_avg": self.post_avg,
            "delta": self.delta,
            "wait": self.wait,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)          

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        self.set_onset_method(onset_method=block_metadata.get("onset_method"))
        self.set_pre_max(pre_max=block_metadata.get("pre_max"))
        self.set_post_max(post_max=block_metadata.get("post_max"))
        self.set_pre_avg(pre_avg=block_metadata.get("pre_avg"))
        self.set_post_avg(post_avg=block_metadata.get("post_avg"))
        self.set_delta(delta=block_metadata.get("delta"))
        self.set_wait(wait=block_metadata.get("wait"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))  
        
        # push the results to the output ports       
        self.output.push_all(self.data.get_all())
                        