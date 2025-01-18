from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
import librosa
import numpy as np
from src.Utils.tools import prompt, prompt_selection
from src.Utils.message import Log

DEFAULT_PRE_MAX = 3
DEFAULT_POST_MAX = 3
DEFAULT_PRE_AVG = 3
DEFAULT_POST_AVG = 3
DEFAULT_DELTA = 0.5
DEFAULT_WAIT = 10
DEFAULT_RISE_THRESHOLD = 0.1
DEFAULT_MIN_TIME = 0.5
DEFAULT_MAX_CLIP_LENGTH = 0.5
DEFAULT_TRAIL_SILENCE_THRESHOLD = 0.02
DEFAULT_TRAIL_SILENCE_FRAMES = 256

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
        self.backtrack = False

        # Additional parameters for slicing the audio
        self.max_clip_length = DEFAULT_MAX_CLIP_LENGTH    # seconds; the sample won't exceed this duration
        self.trail_silence_threshold = DEFAULT_TRAIL_SILENCE_THRESHOLD  # amplitude threshold for trailing silence
        self.trail_silence_frames = DEFAULT_TRAIL_SILENCE_FRAMES    # how many frames of silence before we cut

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
            onsets = self.detect_onsets(audio_data)            # Detect onsets
            event_data = self.slice_events(audio_data, onsets)            # Slice events
            percussion_events_list.append(event_data)            # Add the event data to the list
        return percussion_events_list
    
    def detect_onsets(self, audio_data):
        """
        Use librosa to detect onsets in the given audio data.
        Returns a list of onset indices (in samples).
        """

        # If you'd like to tweak additional params like hop_length, you can do so here
        onset_indices = librosa.onset.onset_detect(
            y=audio_data.get_data(),
            sr=audio_data.get_sample_rate(),
            units="samples",
            pre_max=self.pre_max,
            post_max=self.post_max,
            pre_avg=self.pre_avg,
            post_avg=self.post_avg,
            delta=self.delta,
            wait=self.wait,
            backtrack=False
        )

        Log.info(f"Detected {len(onset_indices)} onsets in {audio_data.name}.")
        return onset_indices
    
    def slice_events(self, audio_data, onset_indices):
        """
        Given the onset indices, slice the audio into individual hits.
        Returns an EventData containing all new EventItems.

        Enhancements:
        1. Detects secondary peaks within each sliced region and splits them if needed.
        2. Dynamically trims trailing silence.
        3. Trims excessive leading silence around each event.
        4. Enforces a min-clip duration and a max-clip duration.
        5. Handles boundary cases near start/end of the waveform.
        6. Optionally, can adapt thresholds based on global or local amplitude.
        """

        # Container for events
        event_data = EventData()
        event_data.name = "PercussionEvents"
        event_data.description = "Detected percussion slices from audio"
        event_data.set_source(audio_data)

        sr = audio_data.get_sample_rate()
        total_samples = len(audio_data.get_data())
        
        # Constants for advanced slicing
        min_clip_duration = 0.03  # 30 ms; ignore extremely short slices
        min_clip_length = int(min_clip_duration * sr)
        max_clip_length = int(self.max_clip_length * sr)  # from self
        pre_silence_offset = int(0.01 * sr)  # 10 ms backward look to remove leading silence
        # You could also make pre_silence_offset a user-tunable parameter

        # We iterate over official onsets, but we may add more if we detect internal peaks.
        # So let's store them in a list we can extend dynamically.
        working_onset_indices = list(onset_indices)

        # Sort and ensure uniqueness in case new indexes get inserted
        working_onset_indices.sort()
        secondary_onsets_to_add = []
        i = 0
        while i < len(working_onset_indices):
            start_idx = working_onset_indices[i]
            # Prevent out-of-range errors
            if start_idx >= total_samples:
                break

            if i < len(working_onset_indices) - 1:
                # Next onset or the maximum clip boundary
                next_onset_idx = working_onset_indices[i + 1]
                end_idx = min(start_idx + max_clip_length, next_onset_idx)
            else:
                # Last onset in the list
                end_idx = min(total_samples, start_idx + max_clip_length)

            # 1. Trim leading silence where appropriate (light approach).
            #    We'll only look back a small offset and see if there's a very low amplitude region.
            #    This helps remove big chunks of silence while still keeping the transient.
            start_idx = self.trim_leading_silence(audio_data.get_data(), start_idx, pre_silence_offset)

            # 2. Dynamic trailing silence detection (scan from end backward).
            end_idx = self.find_trailing_silence_dynamic(audio_data.get_data(), sr, start_idx, end_idx)

            # Enforce min clip length
            if end_idx - start_idx < min_clip_length:
                # If it's too short, skip it (or you could merge with next if you suspect the main onset is coming).
                i += 1
                continue

            # 3. Build the clip for further analysis
            clip = audio_data.get_data()[start_idx:end_idx]

            # 4. Check if there are additional peaks inside this clip that might warrant splitting.
            #    We use a simple amplitude-based approach here. 
            #    A more advanced approach could use another onset detection pass with more lenient parameters.
            additional_peaks = self.detect_secondary_peaks(clip, sr)
            # additional_peaks = []
            # If we found extra peaks, we insert them *relative* to the absolute waveform index
            # so that they become new onsets in the main list.
            if additional_peaks:
                for peak_sample in additional_peaks:
                    absolute_peak_idx = start_idx + peak_sample
                   
                    ms_offset = 10
                    if absolute_peak_idx > start_idx + ms_offset \
                        and absolute_peak_idx < end_idx - ms_offset \
                        and absolute_peak_idx not in working_onset_indices:
                            secondary_onsets_to_add.append(absolute_peak_idx)

                # We sort because we must keep them in ascending order for correct slicing
                working_onset_indices.sort()
                # Do not finalize an event yet. We'll handle it when the loop encounters it as a start_idx.
                # Move on to the next onset in the updated list
                i += 1
                continue

            # 5. Create the event for this slice
            # event_item = self.build_event_item(
            #     audio_data,
            #     clip,
            #     start_idx,
            #     i
            # )
            # event_data.add_item(event_item)

            i += 1  # Move to the next onset index

        if secondary_onsets_to_add:
            working_onset_indices.extend(secondary_onsets_to_add)
            working_onset_indices = list(sorted(set(working_onset_indices)))

        i = 0
        while i < len(working_onset_indices):
            start_idx = working_onset_indices[i]
            # Prevent out-of-range errors
            if start_idx >= total_samples:
                break

            if i < len(working_onset_indices) - 1:
                # Next onset or the maximum clip boundary
                next_onset_idx = working_onset_indices[i + 1]
                end_idx = min(start_idx + max_clip_length, next_onset_idx)
            else:
                # Last onset in the list
                end_idx = min(total_samples, start_idx + max_clip_length)

            # 1. Trim leading silence where appropriate (light approach).
            #    We'll only look back a small offset and see if there's a very low amplitude region.
            #    This helps remove big chunks of silence while still keeping the transient.
            start_idx = self.trim_leading_silence(audio_data.get_data(), start_idx, pre_silence_offset)

            # 2. Dynamic trailing silence detection (scan from end backward).
            end_idx = self.find_trailing_silence_dynamic(audio_data.get_data(), sr, start_idx, end_idx)

            # Enforce min clip length
            if end_idx - start_idx < min_clip_length:
                # If it's too short, skip it (or you could merge with next if you suspect the main onset is coming).
                i += 1
                continue

            # 3. Build the clip for further analysis
            clip = audio_data.get_data()[start_idx:end_idx]

            event_item = self.build_event_item(
                audio_data,
                clip,
                start_idx,
                i
            )
            event_data.add_item(event_item)

            i += 1  # Move to the next onset index

        return event_data


    def detect_secondary_peaks(self, clip, sr):
        """
        Quickly scans the clip for secondary peaks that exceed a fraction of the max amplitude.
        If found, returns a list of sample indices (relative to the clip) for those peaks.
        
        amplitude_factor: fraction of the clip's max amplitude to count as a peak
        min_peak_distance: minimum time (in seconds) between peaks to consider them distinct
        """
        amplitude_factor = 0.5  # Set the fraction of max amplitude to consider as a peak
        min_peak_distance = 0.03  # Set the minimum time between peaks to consider them distinct

        clip_abs = np.abs(clip)  # Get the absolute values of the clip to analyze amplitude
        max_amp = np.max(clip_abs)  # Find the maximum amplitude in the clip
        if max_amp == 0:
            return []  # Return an empty list if the maximum amplitude is zero

        threshold = amplitude_factor * max_amp  # Calculate the threshold for peak detection (so once it falls below this, it's not a peak)
        distance_samples = int(sr * min_peak_distance)  # Convert minimum peak distance to samples
        peaks = []  # Initialize a list to store detected peaks

        # Simple approach: gather all local maxima above threshold with a minimum separation.
        last_peak = None  # Initialize the last peak index
        for idx in range(1, len(clip_abs) - 1):
            # Check if the current sample is a local maximum above the threshold
            if clip_abs[idx] > threshold and clip_abs[idx] >= clip_abs[idx - 1] and clip_abs[idx] >= clip_abs[idx + 1]:
                # Check distance from previous peak
                if last_peak is None or (idx - last_peak) > distance_samples:
                    peaks.append(idx)  # Add the index to the peaks list
                    last_peak = idx  # Update the last peak index
                    Log.info(f"Detected {len(peaks)} secondary peaks.")  # Log the number of detected peaks
        # Return any discovered peaks
        return peaks  # Return the list of detected peaks


    def find_trailing_silence_dynamic(self, waveform, sr, start_idx, suggested_end_idx):
        """
        Scans backward from 'suggested_end_idx' to detect trailing silence.
        This version adjusts the threshold based on the slice's average amplitude.
        """
        end_idx = suggested_end_idx
        abs_wave = np.abs(waveform)

        if end_idx <= start_idx:
            return end_idx  # No valid segment

        slice_segment = abs_wave[start_idx:end_idx]
        if len(slice_segment) == 0:
            return end_idx

        # Compute mean amplitude in this slice
        slice_mean = np.mean(slice_segment)

        # Adjust trailing threshold to be either a small fraction of the slice_mean 
        # or a fallback absolute threshold
        dynamic_threshold = max(self.trail_silence_threshold, 0.1 * slice_mean)

        # We'll look for a consecutive region of 'trail_silence_frames' below dynamic_threshold
        frames_needed = self.trail_silence_frames
        counter = 0
        for idx in range(end_idx - 1, start_idx, -1):
            if abs_wave[idx] < dynamic_threshold:
                counter += 1
            else:
                counter = 0
            if counter >= frames_needed:
                # We've found a region of silence large enough; cut the clip here
                return idx - frames_needed + 1

        return end_idx


    def trim_leading_silence(self, waveform, start_idx, offset):
        """
        Attempts to trim some leading silence before the start_idx by scanning backward up to 'offset' samples.
        If within that range the amplitude is consistently below a fraction of its local max, move start_idx forward.
        """
        if start_idx <= offset:
            return start_idx  # Can't go negative

        segment_start = start_idx - offset
        segment_end = start_idx
        local_segment = np.abs(waveform[segment_start:segment_end])
        if len(local_segment) == 0:
            return start_idx

        local_mean = np.mean(local_segment)
        if local_mean < 0.1 * np.max(local_segment):
            # This means it's mostly quiet. We'll move the start_idx forward
            # until we find the point where amplitude starts rising significantly.
            # For simplicity, just find first index from segment_start that crosses 20% of local max.
            local_max = np.max(local_segment)
            cutoff = 0.2 * local_max
            for idx in range(len(local_segment)):
                if local_segment[idx] >= cutoff:
                    return segment_start + idx
            # If never crosses cutoff, just place onset at the original start - offset
            return segment_start
        return start_idx

    def build_event_item(self, original_audio_data, clip, start_idx, event_index):
        """
        Builds an EventItem (with embedded AudioData).
        """
        sr = original_audio_data.sample_rate

        # Create the container for the slice
        slice_audio_data = AudioData()
        slice_audio_data.set_name(f"percussion_{event_index}")
        slice_audio_data.set_data(clip)
        slice_audio_data.set_sample_rate(sr)
        slice_audio_data.set_path(original_audio_data.path)

        # Build the event
        event_item = EventItem()
        event_item.set_name(f"percussion_{event_index}")
        # Convert sample index to time in seconds
        event_item.time = librosa.samples_to_time(start_idx, sr=sr)
        event_item.source = "DetectPercussion"
        event_item.set_data(slice_audio_data)

        return event_item

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