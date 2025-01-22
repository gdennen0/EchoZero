from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Data.Types.event_data import EventData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log
import numpy as np
import librosa
from sklearn.metrics import silhouette_score
from src.Utils.tools import prompt_selection, prompt
import warnings

warnings.filterwarnings("ignore")

class DetectSongSegmentsBlock(Block):
    """
    A block to detect and segment major song sections by measuring
    rhythmic and harmonic changes across measures. This approach tries
    to follow typical music theory by:
      1) Estimating tempo and beat positions.
      2) Grouping beats into measures (assuming 4/4 time by default).
      3) Computing chroma to capture harmonic content per measure.
      4) Inserting boundaries where changes in the harmonic profile
         between consecutive measures exceed a threshold.
    """
    name = "DetectSongSegments"
    type = "DetectSongSegments"

    def __init__(self):
        super().__init__()
        self.name = "DetectSongSegments"
        self.type = "DetectSongSegments"

        # Define block inputs/outputs
        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

    def process(self, input_data):
        """
        Main processing function to detect segment boundaries for
        each AudioData object.

        Args:
            input_data (list of AudioData]): List of AudioData objects.

        Returns:
            list of EventData: Each item corresponds to segmented regions
                               labeled by cluster ID for each AudioData input.
        """
        if not input_data:
            Log.warning(f"{self.name}: No input data received.")
            return []

        results = []
        for audio_data in input_data:
            y = audio_data.data
            sr = audio_data.sample_rate

            Log.info(f"{self.name}: Processing audio '{audio_data.name}' at {sr} Hz")

            # Convert to mono if multi-channel
            if y.ndim > 1:
                y = librosa.to_mono(y)
                Log.debug(f"{self.name}: Converted audio to mono")

            # Ensure it's float32
            y = np.asanyarray(y, dtype=np.float32).flatten()

            # Detect segments via clustering
            segments = self._detect_segments(y, sr)

            # Convert segments to EventData
            event_data = EventData()
            event_data.name = f"{audio_data.name}_DetectedSegments"
            event_data.description = "Song segments based on beat-aligned harmonic changes"
            event_data.set_source(audio_data)

            for idx, (start_time, end_time) in enumerate(segments):
                event_item = EventItem()
                event_item.set_name(f"segment_{idx}")
                event_item.set_time(f"{start_time:.3f}-{end_time:.3f}")
                event_item.set_source(self.name)
                event_item.set_type("song_part")
                event_item.set_data(self.get_audio_clip(audio_data, start_time, end_time))
                event_item.set_classification("segment")
                event_data.add_item(event_item)

            results.append(event_data)

        return results

    def _detect_segments(self, y, sr):
        """
        Detect segments by:
          1) Estimating tempo and beats.
          2) Computing approximate measure boundaries (assuming 4/4).
          3) Computing average chroma per measure.
          4) Grouping multiple measures (8–16) into song parts,
             starting a new part if harmonic changes exceed a threshold.
        """
        # 1) Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) < 2:
            # Too few beats => just one segment
            return [(0.0, float(len(y)) / sr)]

        # 2) Convert beat frames to times
        beat_times = librosa.frames_to_time(beats, sr=sr)

        # Approximate measures (4 beats per measure in 4/4), ensuring the last beat
        # for measure i is precisely the (i+4-1)th beat if it exists
        measures = []
        measure_size = 4
        for i in range(0, len(beat_times), measure_size):
            start_beat = beat_times[max(0, i - 2)]
            end_idx = i + measure_size - 2

            if end_idx < len(beat_times):
                end_beat = beat_times[end_idx]
            else:
                # If there's not a full measure left, end on the last beat
                end_beat = beat_times[-1]

            measures.append((start_beat, end_beat))

        # 3) Compute average chroma per measure
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        measure_chroma = []
        for (m_start, m_end) in measures:
            start_frame = librosa.time_to_frames(m_start, sr=sr)
            end_frame = librosa.time_to_frames(m_end, sr=sr)
            start_frame = max(0, min(start_frame, chroma.shape[1]))
            end_frame = max(0, min(end_frame, chroma.shape[1]))
            if end_frame > start_frame:
                avg_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
            else:
                avg_chroma = np.mean(chroma, axis=1)
            measure_chroma.append(avg_chroma)

        # 4) Group measures into song parts
        # Typical parts are 8–16 measures. We place a boundary if:
        # - we've already reached 16 measures in the current group, OR
        # - we have at least 8 measures and the difference is above 'measure_diff_threshold'
        measure_diff_threshold = 0.25
        min_measures_per_part = 8
        max_measures_per_part = 16

        parts = []
        current_part_start_idx = 0

        for i in range(1, len(measure_chroma)):
            # Compare this measure's chroma to the previous measure's chroma
            distance = np.linalg.norm(measure_chroma[i] - measure_chroma[i - 1])
            measures_in_current_part = (i - current_part_start_idx)

            # Decide if we should start a new part here
            if measures_in_current_part >= max_measures_per_part:
                # Force boundary if we've hit 16 measures
                parts.append((current_part_start_idx, i))
                current_part_start_idx = i
            elif measures_in_current_part >= min_measures_per_part and distance > measure_diff_threshold:
                # If we have at least 8 measures and see a big harmonic jump, start a new part
                parts.append((current_part_start_idx, i))
                current_part_start_idx = i

        # Append final part
        if current_part_start_idx < len(measures):
            parts.append((current_part_start_idx, len(measures)))

        # Convert measure indices to actual time ranges
        segments = []
        for start_idx, end_idx in parts:
            # Start time is the beginning of the start_idx measure
            segment_start = measures[start_idx][0]
            # End time is the end of the end_idx-1 measure
            # (because end_idx is an exclusive boundary)
            segment_end = measures[end_idx - 1][1] if end_idx > 0 else measures[-1][1]
            segments.append((segment_start, segment_end))

        return segments

    def get_audio_clip(self, source_audio_data_item, start_time, end_time):
        """
        Extracts a time-based audio clip from the source AudioData.
        """
        sr = source_audio_data_item.sample_rate
        start_sample = max(int(start_time * sr), 0)
        end_sample = min(int(end_time * sr), len(source_audio_data_item.data))

        clip_data = source_audio_data_item.data[start_sample:end_sample]

        audio_data_item = AudioData()
        audio_data_item.set_data(clip_data)
        audio_data_item.set_sample_rate(sr)
        return audio_data_item

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))

        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input", {}))
        self.output.load(block_metadata.get("output", {}))

        # Push the results to the output ports
        self.output.push_all(self.data.get_all())