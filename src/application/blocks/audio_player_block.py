"""
Audio Player Block Processor

Sink/endpoint block for inline audio preview.
Accepts audio input, provides no output -- the UI layer handles playback
via embedded controls in the node editor.
"""
from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.application.blocks import register_processor_class
from src.utils.message import Log


class AudioPlayerProcessor(BlockProcessor):
    """
    Processor for the AudioPlayer block.
    
    This is a sink block -- it accepts audio data on its input port
    but produces no outputs. The actual playback is handled by the
    AudioPlayerWidget embedded in the node editor UI.
    
    The processor validates that audio input is present and logs
    what audio items are available for playback.
    """

    def can_process(self, block) -> bool:
        return block.type == "AudioPlayer"

    def get_block_type(self) -> str:
        return "AudioPlayer"

    def get_status_levels(self, block, facade) -> list:
        return []

    def process(self, block, inputs, metadata=None):
        """
        Validate audio input exists. No outputs produced.
        
        The audio data items are accessed directly by the UI widget
        via the facade -- this processor just confirms valid input.
        """
        audio = inputs.get("audio")
        if audio is None:
            raise ProcessingError("No audio data connected to player")

        # Log available audio items for diagnostics
        if isinstance(audio, list):
            names = [getattr(item, 'name', '?') for item in audio]
            Log.info(f"AudioPlayer: {len(audio)} audio items available: {names}")
        else:
            name = getattr(audio, 'name', '?')
            Log.info(f"AudioPlayer: 1 audio item available: {name}")

        # Sink block -- no outputs
        return {}

    def cleanup(self, block) -> None:
        """
        Cleanup is handled by the AudioPlayerBlockItem UI layer,
        which owns the SimpleAudioPlayer instance.
        """
        pass


register_processor_class(AudioPlayerProcessor)
