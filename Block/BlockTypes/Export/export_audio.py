from Block.block import Block
from Connections.port_types.audio_port import AudioPort
from message import Log
from tools import prompt_selection, prompt_selection_with_type, prompt
import os
from pydub import AudioSegment
import soundfile as sf

DEFAULT_EXPORT_AUDIO_PATH = os.path.join(os.getcwd(), "tmp")
DEFAULT_EXPORT_AUDIO_FILE_TYPE = "wav"
DEFAULT_EXPORT_AUDIO_BITRATE = "192k"
DEFAULT_EXPORT_AUDIO_CHANNELS = "stereo"
DEFAULT_EXPORT_AUDIO_SAMPLE_RATE = 44100

class ExportAudioBlock(Block):
    def __init__(self):
        super().__init__()
        self.name = "ExportAudio"
        self.type = "ExportAudio"

        # Initialize settings
        self.file_type = DEFAULT_EXPORT_AUDIO_FILE_TYPE
        self.audio_settings = {
            "bitrate": DEFAULT_EXPORT_AUDIO_BITRATE,
            "channels": DEFAULT_EXPORT_AUDIO_CHANNELS,
            "sample_rate": DEFAULT_EXPORT_AUDIO_SAMPLE_RATE
        }
        self.destination_path = DEFAULT_EXPORT_AUDIO_PATH

        # Add supported file types
        self.supported_file_types = ["wav", "mp3", "flac", "aac"]

        # Add commands
        self.command.add("select_file_type", self.select_file_type)
        self.command.add("set_audio_settings", self.set_audio_settings)
        self.command.add("set_destination_path", self.set_destination_path)
        self.command.add("export", self.export)
        self.command.add("reload", self.reload)

        # Add port types and ports
        self.add_port_type(AudioPort)
        self.add_input_port("AudioPort")
        # self.add_output_port("AudioPort")

        Log.info(f"{self.name} initialized with supported file types: {self.supported_file_types}")

    def select_file_type(self, file_type=None):
        """Command to select the file type for export."""
        if file_type:
            if file_type in self.supported_file_types:
                self.file_type = file_type
                Log.info(f"Selected file type: {self.file_type}")
            else:
                Log.error(f"Unsupported file type passed as an argument: {file_type}")
        else:
            file_type, _ = prompt_selection("Select the export file type:", self.supported_file_types)
            self.file_type = file_type
            Log.info(f"Selected file type: {self.file_type}")

    def set_audio_settings(self, bitrate=None, channels=None, sample_rate=None):
        """Command to set audio file settings."""
        # Example settings: bitrate, channels, sample rate
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
            if not os.path.exists(path):
                try:        
                    os.makedirs(path)
                    Log.info(f"Created destination directory: {path}")
                except Exception as e:
                    Log.error(f"Failed to create destination directory: {e}")
                return
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
        """Command to export the audio data based on settings."""
        if not self.data:
            Log.error("No audio data available to export.")
            return
        if not self.file_type:
            Log.error("File type not selected.")
            return
        if not self.destination_path:
            Log.error("Destination path not set.")
            return

        # Construct the export file name
        export_file_name = f"{self.name}_export.{self.file_type}"
        export_file_path = os.path.join(self.destination_path, export_file_name)

        try:
            # Export logic based on file type
            if self.file_type == "wav":
                self.export_wav(export_file_path)
            elif self.file_type == "mp3":
                self.export_mp3(export_file_path)
            elif self.file_type == "flac":
                self.export_flac(export_file_path)
            elif self.file_type == "aac":
                self.export_aac(export_file_path)
            else:
                Log.error(f"Unsupported file type: {self.file_type}")
                return

            Log.info(f"Exported audio to {export_file_path}")
        except Exception as e:
            Log.error(f"Failed to export audio: {e}")

    def export_wav(self, path):
        """Export audio data as WAV."""
        for item in self.data:
            sf.write(path, item.data, self.audio_settings.get("sample_rate", 44100))
            Log.info(f"Exported WAV file to {path}")

    def export_mp3(self, path):
        """Export audio data as MP3."""
        for item in self.data:
            audio_segment = AudioSegment(
                item.data.tobytes(),
            frame_rate=self.audio_settings.get("sample_rate", 44100),
            sample_width=self.data.dtype.itemsize,
            channels=1 if self.audio_settings.get("channels") == "mono" else 2
        )
            audio_segment.export(path, format="mp3", bitrate=self.audio_settings.get("bitrate", "192k"))
            Log.info(f"Exported MP3 file to {path}")

    def export_flac(self, path):
        """Export audio data as FLAC."""
        for item in self.data:
            sf.write(path, item.data, self.audio_settings.get("sample_rate", 44100), format='FLAC')
            Log.info(f"Exported FLAC file to {path}")

    def export_aac(self, path):
        """Export audio data as AAC."""
        for item in self.data:
            audio_segment = AudioSegment(
            self.data.tobytes(),
            frame_rate=self.audio_settings.get("sample_rate", 44100),
            sample_width=self.data.dtype.itemsize,
            channels=1 if self.audio_settings.get("channels") == "mono" else 2
        )
        audio_segment.export(path, format="aac", bitrate=self.audio_settings.get("bitrate", "192k"))
        Log.info(f"Exported AAC file to {path}")

    def reload(self):
        """Reload the block's data."""
        super().reload()
        Log.info(f"{self.name} reloaded successfully.")