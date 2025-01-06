from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
import librosa
import numpy as np
from src.Utils.tools import prompt
from src.Utils.message import Log

DEFAULT_PRE_MAX = 3
DEFAULT_POST_MAX = 3
DEFAULT_PRE_AVG = 3
DEFAULT_POST_AVG = 3
DEFAULT_DELTA = 0.5
DEFAULT_WAIT = 10
DEFAULT_RISE_THRESHOLD = 0.1
DEFAULT_MIN_TIME = 0.5

class DetectPercussionBlock(Block):
    """
    A simplified block for detecting single-hit percussion onsets in audio using Librosa.
    This refactoring follows first principles: minimal complexity, clarity, and
    descriptive naming. Each detected onset now corresponds to exactly one clip.
    The clip ends upon detecting that amplitude has started to rise again after a fall,
    ensuring a single clean percussion sample.
    """
    name = "DetectPercussion"
    type = "DetectPercussion"
    
    def __init__(self):
        super().__init__()
        self.name = "DetectPercussion"
        self.type = "DetectPercussion"

        # Single audio input
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        # Single event output for detected percussion
        self.output.add_type(EventOutput)
        self.output.add("EventOutput")
        
        self.pre_max = DEFAULT_PRE_MAX
        self.post_max = DEFAULT_POST_MAX
        self.pre_avg = DEFAULT_PRE_AVG
        self.post_avg = DEFAULT_POST_AVG
        self.delta = DEFAULT_DELTA
        self.wait = DEFAULT_WAIT
        self.rise_threshold = DEFAULT_RISE_THRESHOLD
        self.min_time = DEFAULT_MIN_TIME

        self.command.add("set_pre_max", self.set_pre_max)
        self.command.add("set_post_max", self.set_post_max)
        self.command.add("set_pre_avg", self.set_pre_avg)
        self.command.add("set_post_avg", self.set_post_avg)
        self.command.add("set_delta", self.set_delta)
        self.command.add("set_wait", self.set_wait)
        self.command.add("set_rise_threshold", self.set_rise_threshold)
        self.command.add("set_min_time", self.set_min_time)

    def process(self, audio_input_data):
        """
        Detect percussion onsets, slice audio around each onset,
        and return EventData containing those slices.
        """
        percussion_events_list = []

        Log.info(f"***BEGIN PROCESSING BLOCK {self.name}***")
        for audio_data in audio_input_data:
            Log.info(f"-> Processing audio data: {audio_data.name}")
            waveform = audio_data.data
            sample_rate = audio_data.sample_rate

            # Use librosa's built-in onset detector with user-configurable parameters.
            onset_frames = librosa.onset.onset_detect(
                y=waveform,
                sr=sample_rate,
                units="samples",
                pre_max=self.pre_max,
                post_max=self.post_max,
                pre_avg=self.pre_avg,
                post_avg=self.post_avg,
                delta=self.delta,
                wait=self.wait,
                backtrack=True  # Enable backtracking for better percussion detection
            )

            # Prepare container for percussion events
            percussion_events_data = EventData()
            percussion_events_data.name = "PercussionEvents"
            percussion_events_data.description = "Detected percussion slices from audio"
            percussion_events_data.set_source(audio_data)

            # Slice audio around each detected onset
            for i, start_frame in enumerate(onset_frames):
                # Define a search limit up to the next onset, 0.5 seconds after start, or the end of the waveform
                if i < len(onset_frames) - 1:
                    next_onset = onset_frames[i + 1]
                    max_limit = start_frame + int(0.5 * sample_rate)
                    search_limit = min(next_onset, max_limit, len(waveform))
                else:
                    search_limit = min(len(waveform), start_frame + int(0.5 * sample_rate))

                end_frame = self._find_single_onset_clip_end(
                    waveform, start_frame, search_limit
                )

                # Extract the slice for this percussion event
                percussion_clip = waveform[start_frame:end_frame]

                # Build a new EventItem
                percussion_event = EventItem()
                percussion_event.set_name(f"percussion_{i}")
                percussion_event.time = librosa.samples_to_time(start_frame, sr=sample_rate)
                percussion_event.source = "DetectPercussion"

                # Create and embed AudioData
                percussion_audio_data = AudioData()
                percussion_audio_data.set_name(f"percussion_{i}")
                percussion_audio_data.set_data(percussion_clip)
                percussion_audio_data.set_sample_rate(sample_rate)
                percussion_audio_data.set_path(audio_data.path)

                percussion_event.set_data(percussion_audio_data)
                percussion_events_data.add_item(percussion_event)

            percussion_events_list.append(percussion_events_data)

        return percussion_events_list
    
    def _find_single_onset_clip_end(self, waveform, start_idx, search_limit):
        """
        From the onset start, find the end of a single percussion clip.
        1) Find the max amplitude (the peak).
        2) Once amplitude falls after the peak, if it starts to rise again (above a threshold),
        consider that the beginning of the next hit, so cut right before it.
        3) If it never rises again before 'search_limit', then the clip ends
        at 'search_limit'.

        :param waveform: Full audio waveform.
        :param start_idx: Starting index (onset index).
        :param search_limit: The maximum index to search to.
        :param rise_threshold: How much the amplitude must increase from one sample to
                            the next before considering it a true amplitude rise.
        """
        segment = waveform[start_idx:search_limit]
        abs_segment = np.abs(segment)

        # Find the peak amplitude in this segment
        peak_amplitude = 0.0
        peak_index = 0
        for i, amp in enumerate(abs_segment):
            if amp > peak_amplitude:
                peak_amplitude = amp
                peak_index = i

        # After hitting the peak, watch for amplitude falling
        # then rising again => cut
        has_started_falling = False

        # We'll scan from just after the peak to the end of the slice
        for i in range(peak_index + 1, len(abs_segment)):
            current_amp = abs_segment[i]
            previous_amp = abs_segment[i - 1]

            # Check if we're still going down or flat
            if current_amp < previous_amp:
                has_started_falling = True
            else:
                # If amplitude is slightly up but hasn't exceeded the threshold,
                # we ignore it. Only if it rises by more than "rise_threshold"
                # do we consider it a new rise.
                if has_started_falling and (current_amp - previous_amp) > self.rise_threshold:
                    # Cut just before amplitude increased significantly
                    return start_idx + i - 1

        # If we never detect a second rise, return the entire region
        return search_limit

    def list_settings(self):
        """
        Log all current settings for the user to review.
        """
        Log.info(f"Block {self.name}'s current settings:")
        Log.info(f"pre_max: {self.pre_max}")
        Log.info(f"post_max: {self.post_max}")
        Log.info(f"pre_avg: {self.pre_avg}")
        Log.info(f"post_avg: {self.post_avg}")
        Log.info(f"delta (threshold): {self.delta}")
        Log.info(f"wait (frames between onsets): {self.wait}")

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
            
    def set_rise_threshold(self, rise_threshold=None):
        """
        Set the rise_threshold parameter for onset detection.
        """
        if rise_threshold is None:
            rise_threshold = prompt(f"Enter new rise_threshold value (current: {self.rise_threshold}): ")
        try:
            self.rise_threshold = float(rise_threshold)
            Log.info(f"Rise threshold set to {self.rise_threshold}")
        except ValueError:
            Log.error(f"Invalid value for rise_threshold: {rise_threshold}. Must be a float.")

    def set_min_time(self, min_time=None):
        """
        Set the min_time parameter for onset detection.
        """
        if min_time is None:
            min_time = prompt(f"Enter new min_time value (current: {self.min_time}): ")
        try:
            self.min_time = float(min_time)
            Log.info(f"Min time set to {self.min_time}")
        except ValueError:
            Log.error(f"Invalid value for min_time: {min_time}. Must be a float.")

    def get_metadata(self):
        """
        Provide block metadata; you can adjust keys as needed.
        """
        return {
            "name": self.name,
            "type": self.type,
            "pre_max": self.pre_max,
            "post_max": self.post_max,
            "pre_avg": self.pre_avg,
            "post_avg": self.post_avg,
            "delta": self.delta,
            "wait": self.wait,
            "rise_threshold": self.rise_threshold,
            "min_time": self.min_time,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        """
        save block state.
        """
        self.data.save(save_dir)

    def load(self, block_dir):
        """
        load block state.
        """
        block_metadata = self.get_metadata_from_dir(block_dir)
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        # Restore config attributes
        self.set_pre_max(pre_max=block_metadata.get("pre_max"))
        self.set_post_max(post_max=block_metadata.get("post_max"))
        self.set_pre_avg(pre_avg=block_metadata.get("pre_avg"))
        self.set_post_avg(post_avg=block_metadata.get("post_avg"))
        self.set_delta(delta=block_metadata.get("delta"))
        self.set_wait(wait=block_metadata.get("wait"))
        self.set_rise_threshold(rise_threshold=block_metadata.get("rise_threshold"))
        self.set_min_time(min_time=block_metadata.get("min_time"))
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))
        # Push loaded data to outputs (if applicable)
        self.output.push_all(self.data.get_all())