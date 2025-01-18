import os
import glob
import numpy as np
import soundfile as sf

from src.Project.Block.block import Block
from src.Project.Block.Input.Types.event_input import EventInput
from src.Project.Block.Output.Types.event_output import EventOutput
from src.Project.Data.Types.event_data import EventData
from src.Project.Data.Types.event_item import EventItem
from src.Project.Data.Types.audio_data import AudioData
from src.Utils.message import Log
from src.Utils.tools import prompt
import librosa


class IngestDataSetBlock(Block):
    """
    IngestDataSetBlock scans a specified directory for folders that each
    represent a classification label. Inside each folder, it loads audio
    files as AudioData objects, creates EventItems labeled with the folder name,
    then stores them in an EventData object per classification.

    The idea is to create properly-classified audio events which can
    then be used by other blocks (e.g., PyTorchTrain) for training.
    """

    name = "IngestDataSet"
    type = "IngestDataSet"

    def __init__(self):
        super().__init__()
        self.name = "IngestDataSet"
        self.type = "IngestDataSet"

        # Input / Output definitions
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        self.output.add_type(EventOutput)
        self.output.add("EventOutput")

        # Directory to scan
        self.dataset_path = None

        # Register commands
        self.command.add("set_dataset_path", self.set_dataset_path)
        self.command.add("load_dataset", self.load_dataset)

    def set_dataset_path(self, path=None):
        """
        Sets the path to the dataset directory.
        The directory should have subfolders named by classification labels,
        each containing audio files to be ingested.
        """
        if path is None:
            path = prompt("Enter the path to the dataset directory: ")
        if not os.path.isdir(path):
            Log.error(f"{path} is not a valid directory.")
            return
        self.dataset_path = path
        Log.info(f"Dataset path set to: {self.dataset_path}")

    def load_dataset(self):
        """
        Loads audio files from the dataset path.
        Each subfolder is treated as a classification label.
        Audio files within each subfolder are loaded into AudioData,
        nested in an EventItem classified by the folder name, then
        grouped into an EventData object for that classification.
        """
        if not self.dataset_path:
            Log.error("No dataset path set. Use set_dataset_path first.")
            return

        # For each classification subfolder, create an EventData object
        for classification_folder in sorted(next(os.walk(self.dataset_path))[1]):
            classification_path = os.path.join(self.dataset_path, classification_folder)
            if not os.path.isdir(classification_path):
                Log.warning(f"Skipping non-directory {classification_path}")
                continue

            event_data = EventData()
            event_data.name = classification_folder
            event_data.description = f"Audio samples for classification: {classification_folder}"

            # Enumerate audio files and create EventItems
            file_counter = 0
            audio_files = glob.glob(os.path.join(classification_path, "*.wav")) \
                        + glob.glob(os.path.join(classification_path, "*.mp3")) \
                        + glob.glob(os.path.join(classification_path, "*.flac")) \
                        + glob.glob(os.path.join(classification_path, "*.aif")) \
                        + glob.glob(os.path.join(classification_path, "*.aiff"))

            for audio_file in sorted(audio_files):
                if not os.path.isfile(audio_file):
                    continue

                file_counter += 1
                item = EventItem()
                # Name pattern: e.g. "snare_001"
                item_name = f"{classification_folder}_{file_counter:03d}"
                item.set_name(item_name)
                item.set_classification(classification_folder)

                # Create an AudioData object and load the file
                audio_data = AudioData()
                audio_data.set_name(item_name)
                audio_data.set_path(audio_file)

        # Load from disk to get .data and sample_rate
                # Start of Selection
                data, samplerate = sf.read(audio_file)
                if samplerate != 44100:
                    data = librosa.resample(data.T, orig_sr=samplerate, target_sr=44100).T
                    samplerate = 44100
                    data = data.astype(np.float32)

                audio_data.set_data(data)
                audio_data.set_sample_rate(samplerate)

                # Attach AudioData to EventItem
                item.data = audio_data

                # Add item to EventData
                event_data.add_item(item)

            if file_counter > 0:
                # Store the full EventData object in this block
                self.data.add(event_data)
                Log.info(f"Ingested {file_counter} file(s) for classification '{classification_folder}'")
            else:
                Log.warning(f"No audio files found for classification folder '{classification_folder}'")

        # Optionally, push newly created EventData objects to outputs
        self.output.push_all(self.data.get_all())

    def process(self, input_data):
        """
        Bypass the input and return it as is, since ingestion
        is handled by load_dataset(), not typical data flow.
        """
        return input_data

    def get_metadata(self):
        return {
            "name": self.name,
            "type": self.type,
            "dataset_path": self.dataset_path,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }

    def save(self, save_dir):
        """
        Save the currently ingested data to disk.
        """
        self.data.save(save_dir)

    def load(self, block_dir):
        """
        Load the block's state (metadata) from disk.
        """
        block_metadata = self.get_metadata_from_dir(block_dir)

        # Load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.dataset_path = block_metadata.get("dataset_path")

        # Load sub-components
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # Push the loaded data to the output
        self.output.push_all(self.data.get_all()) 