"""
Processors: BlockExecutor implementations for each block type.
Exists because execution logic is block-type-specific and must be testable in isolation.
Each processor implements the BlockExecutor protocol and is registered with the ExecutionEngine.
"""

from echozero.processors.detect_onsets import DetectOnsetsProcessor
from echozero.processors.load_audio import LoadAudioProcessor
from echozero.processors.separate_audio import SeparateAudioProcessor

__all__ = ["LoadAudioProcessor", "DetectOnsetsProcessor", "SeparateAudioProcessor"]
