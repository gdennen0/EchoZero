from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
import librosa
import numpy as np

class DetectPercussionBlock(Block):
    """
    A simplified block for detecting percussion onsets in audio using Librosa.
    This refactoring follows first principles: we focus on clarity, descriptive naming,
    and minimal complexity.
    """
    name = "DetectPercussion"

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

    def process(self, audio_input_data):
        """
        Detect percussion onsets, slice audio around each onset,
        and return EventData containing those slices.
        """
        percussion_events_list = []

        for audio_data in audio_input_data:
            waveform = audio_data.data
            sample_rate = audio_data.sample_rate

            # Use librosa's built-in onset detector (frames in samples).
            onset_frames = librosa.onset.onset_detect(
                y=waveform,
                sr=sample_rate,
                units="samples"
            )

            # Prepare container for percussion events
            percussion_events_data = EventData()
            percussion_events_data.name = "PercussionEvents"
            percussion_events_data.description = "Detected percussion slices from audio"
            percussion_events_data.set_source(audio_data)

            # Slice audio around each detected onset
            for i, start_frame in enumerate(onset_frames):
                if i < len(onset_frames) - 1:
                    end_frame = onset_frames[i + 1]
                else:
                    end_frame = len(waveform)

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

    def get_metadata(self):
        """
        Provide block metadata; you can adjust keys as needed.
        """
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        """
        If needed, save block state.
        """
        self.data.save(save_dir)

    def load(self, block_dir):
        """
        If needed, load block state.
        """
        block_metadata = self.get_metadata_from_dir(block_dir)
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))
        # Push loaded data to outputs (if applicable)
        self.output.push_all(self.data.get_all())