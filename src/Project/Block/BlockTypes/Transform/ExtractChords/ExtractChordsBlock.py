from src.Project.Block.block import Block
from src.Project.Data.Types.audio_data import AudioData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Data.Types.event_data import EventData
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Utils.message import Log
import numpy as np
from madmom.features.chords import DeepChromaChordRecognitionProcessor, CNNChordFeatureProcessor, CRFChordRecognitionProcessor
import librosa
from madmom.audio.chroma import DeepChromaProcessor
from src.Utils.tools import prompt_selection, prompt

class ExtractChordsBlock(Block):
    """
    A block to extract chords from AudioData using Madmom's DeepChromaChordRecognitionProcessor.
    """
    name = "ExtractChords"
    type = "ExtractChords"

    def __init__(self):
        super().__init__()
        self.name = "ExtractChords"
        self.type = "ExtractChords"

        self.input.add_type(AudioInput)
        self.input.add("AudioInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        self.threshold = 0.1  # Confidence threshold for chord detection

        self.extract_types = ["deep_chroma_network", "cnn"]
        self.extract_type = "deep_chroma_network"

        self.command.add("extract_type", self.extract_type)
        self.command.add("set_threshold", self.set_threshold)

    def set_threshold(self, threshold=None):
        if threshold is None:
            threshold = prompt("Enter a threshold value", self.threshold)
        self.threshold = float(threshold)

    def set_extract_type(self, extract_type=None):
        if extract_type is None:
            extract_type = prompt_selection("Select an extract type", self.extract_type)
        self.extract_type = extract_type

    def process(self, input_data):
        """
        Process the input AudioData to extract chords.

        Args:
            input_data (list of AudioData): List containing AudioData objects.

        Returns:
            list of EventItem: List containing chord events.
        """
        
        if not input_data:
            Log.warning(f"{self.name}: No input data received.")
            return []
        
        chord_event_data_items = []

        for audio_data in input_data:
            y = audio_data.data
            sr = audio_data.sample_rate

            Log.info(f"{self.name}: Extracting chords from audio with sample rate {sr} Hz.")

            if y.ndim > 1:
                y = librosa.to_mono(y)
                Log.debug(f"{self.name}: Converted audio to mono. New shape: {y.shape}")
            # Verify that y is a 1D array
            if y.ndim != 1:
                Log.error(f"{self.name}: Audio data is not mono after conversion. Shape: {y.shape}")
                continue  # Skip processing this audio_data

            # Ensure y is a 1D NumPy array of type float32
            y = np.asarray(y, dtype=np.float32).flatten()
            Log.debug(f"{self.name}: Final audio data shape: {y.shape}")


            if self.extract_type == "deep_chroma_network":
                deep_chroma_processor = DeepChromaProcessor()
                chord_processor = DeepChromaChordRecognitionProcessor()

                chroma_vectors = deep_chroma_processor(y)
                chords = chord_processor(chroma_vectors)
                Log.info(f"extracted {len(chords)} chords")
            elif self.extract_type == "cnn":
                cnn_processor = CNNChordFeatureProcessor()
                crf_processor = CRFChordRecognitionProcessor()

                features = cnn_processor(y)
                chords = crf_processor(features)    
                Log.info(f"extracted {len(chords)} chords")

            event_data = EventData() # init blank event data object
            for start, end, chord in chords:
                Log.info(f"Chord: {chord}, start: {start:.2f}, end: {end:.2f}")
                
                event = EventItem()
                event.set_name(f"chord_{len(chord_event_data_items)}")
                event.set_time(f"{start:.3f}-{end:.3f}")
                event.set_source(self.name)
                event.set_type("chord")
                event.set_data(self.get_audio_clip(audio_data, start, end))
                event.set_classification(chord)
                event_data.add_item(event)
           
            Log.info(f"{self.name}: Extracted {len(chord_event_data_items)} chords.")
            chord_event_data_items.append(event_data)
        else:
            Log.warning(f" {self.name}: ndim > 1, audio data is not mono")

            
        return chord_event_data_items

    def get_audio_clip(self, source_audio_data_item, start_time, end_time):
        # Convert start and end times from seconds to sample indices
        start_sample = int(start_time * source_audio_data_item.sample_rate)
        end_sample = int(end_time * source_audio_data_item.sample_rate)

        # Ensure indices are within the bounds of the audio data
        start_sample = max(start_sample, 0)
        end_sample = min(end_sample, len(source_audio_data_item.data))

        # Extract the audio clip
        clip_data = source_audio_data_item.data[start_sample:end_sample]

        audio_data_item = AudioData()
        audio_data_item.set_data(clip_data)
        audio_data_item.set_sample_rate(source_audio_data_item.sample_rate)
        return audio_data_item

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "extract_type": self.extract_type,
            "threshold": self.threshold,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        self.data.save(save_dir)

    def load(self, block_dir):
        block_metadata = self.get_metadata_from_dir(block_dir)

        self.set_name(block_metadata.get("name", "ExtractChords"))
        self.set_type(block_metadata.get("type", "ExtractChords"))
        self.set_extract_type(block_metadata.get("extract_type", "deep_chroma_network"))
        self.set_threshold(block_metadata.get("threshold", 0.1))

        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input", {}))
        self.output.load(block_metadata.get("output", {}))

        # Push the results to the output ports
        self.output.push_all(self.data.get_all())