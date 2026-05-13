"""
Stem separation pipeline template: LoadAudio → SeparateAudio.
Exists because source separation is the first step before per-instrument analysis.
Registers with the pipeline registry on import.
"""

from echozero.pipelines.block_specs import LoadAudio, Separator
from echozero.pipelines.params import knob, KnobWidget
from echozero.pipelines.pipeline import ArtifactPolicy, PersistenceMapping, Pipeline
from echozero.pipelines.registry import pipeline_template


@pipeline_template(
    id="stem_separation",
    name="Stem Separation",
    description="Separate audio into individual stems (drums, bass, vocals, other) using Demucs.",
    knobs={
        "audio_file": knob(
            "",
            label="Audio File",
            widget=KnobWidget.FILE_PICKER,
            file_types=(".wav", ".mp3", ".flac", ".aiff"),
        ),
        "model": knob(
            "latest_model",
            label="Model",
            widget=KnobWidget.DROPDOWN,
            options=("latest_model", "htdemucs", "htdemucs_ft", "mdx_extra", "mdx_extra_q"),
        ),
        "device": knob(
            "auto",
            label="Device",
            widget=KnobWidget.DROPDOWN,
            options=("auto", "cpu", "cuda", "mps"),
        ),
        "shifts": knob(
            1,
            label="Quality Shifts",
            min_value=0,
            max_value=10,
            step=1,
            description="More shifts = better quality, slower",
            advanced=True,
        ),
        "two_stems": knob(
            "none",
            label="Two-Stem Mode",
            widget=KnobWidget.DROPDOWN,
            options=("none", "vocals", "drums", "bass", "other"),
            description="Extract only one stem + remainder",
            advanced=True,
        ),
    },
)
def build_stem_separation(
    audio_file="",
    model="latest_model",
    device="auto",
    shifts=1,
    two_stems="none",
) -> Pipeline:
    """Build a LoadAudio → SeparateAudio pipeline."""
    p = Pipeline("stem_separation", name="Stem Separation")
    load = p.add(LoadAudio(file_path=audio_file), id="load")
    sep = p.add(
        Separator(
            model=model,
            device=device,
            shifts=shifts,
            output_format="wav",
            mp3_bitrate=320,
        ),
        id="separate",
        audio_in=load.audio_out,
    )
    for output_name, port_ref in (
        ("drums", sep.drums_out),
        ("bass", sep.bass_out),
        ("vocals", sep.vocals_out),
        ("other", sep.other_out),
    ):
        p.output(
            output_name,
            port_ref,
            data_type="audio",
            label=output_name.title(),
            persistence=PersistenceMapping(project_as_layer=True),
            artifact=ArtifactPolicy(
                artifact_kind="separated_audio",
                role=f"{output_name}_stem",
                source_input="audio_file",
            ),
        )
    return p
