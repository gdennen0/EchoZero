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
        'method': knob('default', label='Method',
                       widget=KnobWidget.DROPDOWN, options=('default', 'hfc', 'complex')),
    },
)
def build_onset_detection(
    audio_file='',
    threshold=0.3,
    method='default',
) -> Pipeline:
    """Build a LoadAudio -> DetectOnsets pipeline."""
    p = Pipeline('onset_detection', name='Onset Detection')
    load = p.add(
        LoadAudio(file_path=audio_file, target_sample_rate=44100),
        id='load_audio',
    )
    onsets = p.add(
        DetectOnsets(threshold=threshold, method=method),
        id='detect_onsets',
        audio_in=load.audio_out,
    )
    p.output('onsets', onsets.events_out)
    return p
