"""Onset Detection pipeline template."""

from echozero.pipelines.block_specs import DetectOnsets, LoadAudio
from echozero.pipelines.params import KnobWidget, knob
from echozero.pipelines.pipeline import Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id='onset_detection',
    name='Onset Detection',
    description='Detect note onsets in audio',
    knobs={
        'audio_file': knob('', label='Audio File', widget=KnobWidget.FILE_PICKER,
                           file_types=('.wav', '.mp3', '.flac', '.aiff')),
        'threshold': knob(0.3, label='Sensitivity', min_value=0.0, max_value=1.0,
                          step=0.05, description='Detection threshold'),
        'min_gap': knob(0.05, label='Minimum Gap', min_value=0.0, max_value=1.0,
                        step=0.01, description='Minimum time between detected events'),
        'method': knob('default', label='Method',
                       widget=KnobWidget.DROPDOWN, options=('default', 'hfc', 'complex')),
        'backtrack': knob(True, label='Backtrack'),
        'timing_offset_ms': knob(
            0.0,
            label='Timing Offset (ms)',
            min_value=-100.0,
            max_value=100.0,
            step=1.0,
            description='Manual timing compensation applied after onset detection',
        ),
    },
)
def build_onset_detection(
    audio_file='',
    threshold=0.3,
    min_gap=0.05,
    method='default',
    backtrack=True,
    timing_offset_ms=0.0,
) -> Pipeline:
    """Build a LoadAudio -> DetectOnsets pipeline."""
    p = Pipeline('onset_detection', name='Onset Detection')
    load = p.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id='load_audio',
    )
    onsets = p.add(
        DetectOnsets(
            threshold=threshold,
            min_gap=min_gap,
            method=method,
            backtrack=backtrack,
            timing_offset_ms=timing_offset_ms,
        ),
        id='detect_onsets',
        audio_in=load.audio_out,
    )
    p.output('onsets', onsets.events_out)
    return p
