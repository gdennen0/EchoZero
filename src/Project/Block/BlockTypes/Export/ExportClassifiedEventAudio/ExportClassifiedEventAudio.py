from src.Project.Block.block import Block
from src.Project.Block.Input.Types.audio_input import AudioInput
from src.Project.Block.Input.Types.event_input import EventInput
from src.Utils.message import Log
from src.Utils.tools import prompt_selection, prompt
import os
from pydub import AudioSegment
import soundfile as sf
import json

DEFAULT_EXPORT_AUDIO_PATH = os.path.join(os.getcwd(), "tmp", "audio_events")
DEFAULT_EXPORT_AUDIO_FILE_TYPE = "wav"
DEFAULT_EXPORT_AUDIO_BITRATE = "192k"
DEFAULT_EXPORT_AUDIO_CHANNELS = "stereo"
DEFAULT_EXPORT_AUDIO_SAMPLE_RATE = 44100

class ExportClassifiedEventAudio(Block):
    name = "ExportClassifiedEventAudio"
    type = "ExportClassifiedEventAudio"

    def __init__(self):
        super().__init__()
        self.name = "ExportClassifiedEventAudio"
        self.type = "ExportClassifiedEventAudio"

        # Initialize settings
        self.file_type = DEFAULT_EXPORT_AUDIO_FILE_TYPE
        self.audio_settings = {
            "bitrate": DEFAULT_EXPORT_AUDIO_BITRATE,
            "channels": DEFAULT_EXPORT_AUDIO_CHANNELS,
            "sample_rate": DEFAULT_EXPORT_AUDIO_SAMPLE_RATE,
        }
        self.destination_path = DEFAULT_EXPORT_AUDIO_PATH
        self.supported_file_types = ["wav", "mp3", "flac", "aac"]

        # Add port types and ports
        self.input.add_type(EventInput)
        self.input.add("EventInput")

        # Add commands
        self.command.add("select_file_type", self.select_file_type)
        self.command.add("set_audio_settings", self.set_audio_settings)
        self.command.add("set_destination_path", self.set_destination_path)
        self.command.add("export", self.export)
        self.command.add("reload", self.reload)
        Log.info(f"{self.name} initialized with supported file types: {self.supported_file_types}")

    def process(self, input_data):
        processed_data = input_data
        return processed_data

    def select_file_type(self, file_type=None):
        """Command to select the file type for export."""
        if file_type:
            if file_type in self.supported_file_types:
                self.file_type = file_type
                Log.info(f"Selected file type: {self.file_type}")
            else:
                Log.error(f"Unsupported file type passed as an argument: {file_type}")
        else:
            file_type = prompt_selection("Select the export file type:", self.supported_file_types)
            self.file_type = file_type
            Log.info(f"Selected file type: {self.file_type}")

    def set_audio_settings(self, bitrate=None, channels=None, sample_rate=None):
        """Command to set audio file settings."""
        if bitrate:
            self.audio_settings["bitrate"] = bitrate
        else:
            self.audio_settings["bitrate"] = prompt_selection("Select the export bitrate:", ["128k", "192k", "256k", "320k"])
        if channels:
            self.audio_settings["channels"] = channels
        else:
            self.audio_settings["channels"] = prompt_selection("Select the export channels:", ["mono", "stereo"])
        if sample_rate:
            self.audio_settings["sample_rate"] = sample_rate
        else:
            self.audio_settings["sample_rate"] = prompt_selection("Select the export sample rate:", ["44100", "48000", "96000"])

        Log.info(f"Set audio settings: {self.audio_settings}")

    def set_destination_path(self, path=None):
        """Command to set the destination path for the exported file."""
        if path:
            self.destination_path = path
            if not os.path.exists(self.destination_path):
                try:
                    os.makedirs(self.destination_path)
                    Log.info(f"Created destination directory: {self.destination_path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
        else:
            self.destination_path = prompt("Enter destination path: ")
            if not os.path.exists(self.destination_path):
                try:
                    os.makedirs(self.destination_path)
                    Log.info(f"Created destination directory: {self.destination_path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
        Log.info(f"Set destination path: {self.destination_path}")

    def export(self):
        """Command to export the audio data based on settings, organized by classification."""
        for event_data in self.data.get_all():
            Log.info(f"Exporting event_data {event_data.name}")
            for event in event_data.get_all():
                classification = event.classification  # Assuming each event has a 'classification' attribute
                audio_data_item = event.data
                audio = audio_data_item.data
                sr = audio_data_item.sample_rate

                if not audio_data_item:
                    Log.error("No audio data available to export.")
                    continue
                if not self.file_type:
                    Log.error("File type not selected.")
                    continue
                if not self.destination_path:
                    Log.error("Destination path not set.")
                    continue

                # Create classification folder
                classification_folder = os.path.join(self.destination_path, classification)
                if not os.path.exists(classification_folder):
                    try:
                        os.makedirs(classification_folder)
                        Log.info(f"Created classification directory: {classification_folder}")
                    except Exception as e:
                        Log.error(f"Failed to create classification directory: {e}")
                        continue

                # Construct the export file name
                export_file_name = f"{event.name}_export.{self.file_type}"
                export_file_path = os.path.join(classification_folder, export_file_name)

                # Export logic based on file type
                try:
                    if self.file_type == "wav":
                        self.export_wav(export_file_path, audio, sr)
                    elif self.file_type == "mp3":
                        self.export_mp3(export_file_path, audio, sr)
                    elif self.file_type == "flac":
                        self.export_flac(export_file_path, audio, sr)
                    elif self.file_type == "aac":
                        self.export_aac(export_file_path, audio, sr)
                    else:
                        Log.error(f"Unsupported file type: {self.file_type}")
                        continue
                except Exception as e:
                    Log.error(f"Failed to export {self.file_type} for event {event.name}: {e}")

    def export_wav(self, path, audio, sr):
        """Export audio data as WAV."""
        sf.write(path, audio, sr)
        Log.info(f"Exported WAV file to {path}")

    def export_mp3(self, path, audio, sr):
        """Export audio data as MP3."""
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sr,
            sample_width=audio.dtype.itemsize,
            channels=1 if self.audio_settings["channels"] == "mono" else 2
        )
        audio_segment.export(path, format="mp3", bitrate=self.audio_settings["bitrate"])
        Log.info(f"Exported MP3 file to {path}")

    def export_flac(self, path, audio, sr):
        """Export audio data as FLAC."""
        sf.write(path, audio, sr, format='FLAC')
        Log.info(f"Exported FLAC file to {path}")

    def export_aac(self, path, audio, sr):
        """Export audio data as AAC."""
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=self.audio_settings["sample_rate"],
            sample_width=audio.dtype.itemsize,
            channels=1 if self.audio_settings["channels"] == "mono" else 2
        )
        audio_segment.export(path, format="aac", bitrate=self.audio_settings["bitrate"])
        Log.info(f"Exported AAC file to {path}")

    def save(self, save_dir):
        # does not save any data, just metadata
        pass

    def get_metadata(self):
        metadata = {
            "name": self.name,
            "type": self.type,
            "file_type": self.file_type,
            "destination_path": self.destination_path,
            "audio_settings": self.audio_settings,
            "input": self.input.save(),
            "output": self.output.save(),
            "metadata": self.data.get_metadata()
        }
        return metadata

    def load(self, block_dir):
        # get block metadata
        block_metadata = self.get_metadata_from_dir(block_dir)

        # load attributes
        self.set_name(block_metadata.get("name"))
        self.set_type(block_metadata.get("type"))
        self.set_file_type(block_metadata.get("file_type"))
        self.set_destination_path(block_metadata.get("destination_path"))
        self.set_audio_settings(block_metadata.get("audio_settings"))

        # load sub components attributes
        self.data.load(block_metadata.get("metadata"), block_dir)
        self.input.load(block_metadata.get("input"))
        self.output.load(block_metadata.get("output"))

        # push the results to the output ports
        self.output.push_all(self.data.get_all())
