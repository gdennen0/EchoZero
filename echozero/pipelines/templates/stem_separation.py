"""
Stem separation pipeline template: LoadAudio → SeparateAudio.
Exists because source separation is the first step before per-instrument analysis.
Registers with the pipeline registry on import.
"""

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
from echozero.pipelines.registry import PromotedParam, pipeline_template


@pipeline_template(
    id="stem_separation",
    name="Stem Separation",
    description="Separate audio into individual stems (drums, bass, vocals, other) using Demucs.",
    promoted_params=(
        PromotedParam(
            key="model",
            name="Model",
            type=str,
            default="htdemucs",
            maps_to_block="separate",
            maps_to_setting="model",
        ),
        PromotedParam(
            key="device",
            name="Device",
            type=str,
            default="auto",
            maps_to_block="separate",
            maps_to_setting="device",
        ),
        PromotedParam(
            key="shifts",
            name="Quality Shifts",
            type=int,
            default=1,
            maps_to_block="separate",
            maps_to_setting="shifts",
        ),
        PromotedParam(
            key="two_stems",
            name="Two-Stem Mode",
            type=str,
            default=None,
            maps_to_block="separate",
            maps_to_setting="two_stems",
        ),
    ),
)
def build_stem_separation() -> Graph:
    """Build a LoadAudio → SeparateAudio graph."""
    load_block = Block(
        id="load",
        name="Load Audio",
        block_type="LoadAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(
            Port(name="audio_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={"file_path": ""}),
    )

    separate_block = Block(
        id="separate",
        name="Separate Audio",
        block_type="SeparateAudio",
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port(name="audio_in", port_type=PortType.AUDIO, direction=Direction.INPUT),
        ),
        output_ports=(
            Port(name="drums_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="bass_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="other_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
            Port(name="vocals_out", port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={
            "model": "htdemucs",
            "device": "auto",
            "shifts": 1,
            "output_format": "wav",
            "mp3_bitrate": 320,
        }),
    )

    connection = Connection(
        source_block_id="load",
        source_output_name="audio_out",
        target_block_id="separate",
        target_input_name="audio_in",
    )

    return Graph(
        blocks={"load": load_block, "separate": separate_block},
        connections=(connection,),
    )
