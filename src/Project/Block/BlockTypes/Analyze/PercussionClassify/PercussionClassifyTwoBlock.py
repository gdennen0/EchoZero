from src.Project.Block.block import Block
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
import librosa
import numpy as np
import torch
import os
from src.Utils.message import Log

class PercussionClassifyTwoBlock(Block):
    """
    A block that classifies percussion samples using a pre-trained CNN model.
    The model can classify drum types like kick, snare, closed hat, open hat, and clap.
    """
    name = "PercussionClassifyTwo"
    type = "PercussionClassifyTwo"

    def __init__(self):
        super().__init__()
        self.name = "PercussionClassifyTwo"
        self.type = "PercussionClassifyTwo"

        # Add input/output types
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Initialize model path
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "mel_cnn_model_high_v2.model")
        
        # Initialize classifier
        self.classifier = None
        self.initialize_classifier()

    def initialize_classifier(self):
        """Initialize the drum classifier model"""
        try:
            from src.Project.Block.BlockTypes.PyTorch.PercussionClassify.lib.drumclassifier_utils import DrumClassifier
            self.classifier = DrumClassifier(
                path_to_model=self.model_path,
                file_types=['wav', 'aif', 'flac', 'ogg'],
                hop_length=256
            )
            Log.info("Drum classifier initialized successfully")
        except Exception as e:
            Log.error(f"Failed to initialize drum classifier: {e}")
            self.classifier = None

    def process(self, event_data_list):
        """
        Process the input event data and classify each percussion sample.
        Returns a new EventData with classified samples.
        """
        if not self.classifier:
            Log.error("Drum classifier not initialized")
            return None

        Log.info(f"Processing {len(event_data_list)} event data items")
        
        # Create new event data for output
        output_event_data = EventData()
        output_event_data.name = "ClassifiedPercussion"
        output_event_data.description = "Percussion samples classified by type"

        for event_data in event_data_list:
            for item in event_data.items:
                # Get audio data from event item
                audio_data = item.data
                if not audio_data:
                    continue

                # Get audio samples and sample rate
                y = audio_data.get_data()
                sr = audio_data.get_sample_rate()

                # Transform audio to mel spectrogram
                M = self.classifier.transform(y, sr)
                if M is None:
                    continue

                # Classify the sample
                prediction = self.classifier.predict({item.name: M})
                if not prediction:
                    continue

                # Get the predicted class
                predicted_class = self.classifier.int2class[prediction[item.name]]
                
                # Create new event item with classification
                new_item = EventItem()
                new_item.set_name(f"{predicted_class}_{item.name}")
                new_item.time = item.time
                new_item.source = "PercussionClassify"
                new_item.set_data(audio_data)
                new_item.classification = predicted_class

                # Add to output event data
                output_event_data.add_item(new_item)

        return output_event_data

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata() if hasattr(self, 'data') else {}
        }

    def save(self, save_dir):
        if hasattr(self, 'data'):
            self.data.save(save_dir) 