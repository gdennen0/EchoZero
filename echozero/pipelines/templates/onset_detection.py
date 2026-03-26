"""Onset Detection pipeline template."""

from echozero.domain.enums import BlockCategory, Direction, PortType
from echozero.domain.graph import Graph
from echozero.domain.types import Block, BlockSettings, Connection, Port
from echozero.pipelines.registry import PromotedParam, pipeline_template


@pipeline_template(
    id='onset_detection',
    name='Onset Detection',
    description='Detect note onsets in audio',
    promoted_params=[
        PromotedParam('audio_file', 'Audio File', str, required=True,
                      description='Path to audio file'),
        PromotedParam('threshold', 'Sensitivity', float, default=0.3,
                      description='Detection threshold (0.0-1.0)'),
        PromotedParam('method', 'Method', str, default='default',
                      description='Detection method'),
    ],
)
def build_onset_detection() -> Graph:
    """Build a LoadAudio -> DetectOnsets pipeline."""
    g = Graph()
    load = Block(
        id='load_audio',
        name='Load Audio',
        block_type='load_audio',
        category=BlockCategory.PROCESSOR,
        input_ports=(),
        output_ports=(
            Port(name='audio', port_type=PortType.AUDIO, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={'file_path': '', 'target_sample_rate': 44100}),
    )
    detect = Block(
        id='detect_onsets',
        name='Detect Onsets',
        block_type='detect_onsets',
        category=BlockCategory.PROCESSOR,
        input_ports=(
            Port(name='audio', port_type=PortType.AUDIO, direction=Direction.INPUT),
        ),
        output_ports=(
            Port(name='events', port_type=PortType.EVENT, direction=Direction.OUTPUT),
        ),
        settings=BlockSettings(entries={'threshold': 0.3, 'method': 'default'}),
    )
    g.add_block(load)
    g.add_block(detect)
    g.add_connection(Connection(
        source_block_id='load_audio',
        source_output_name='audio',
        target_block_id='detect_onsets',
        target_input_name='audio',
    ))
    return g
