"""
Pipeline infrastructure: Registry, Knobs, BlockSpecs, and template definitions.
Exists because pipelines are the executable unit — blocks + connections + knobs + outputs.
"""

from echozero.pipelines.params import Knob, KnobWidget, knob, extract_knobs, validate_bindings
from echozero.pipelines.registry import PipelineRegistry, PipelineTemplate, pipeline_template
from echozero.pipelines.pipeline import Pipeline, BlockHandle, PortRef, PipelineOutput
from echozero.pipelines.block_specs import (
    BlockSpec,
    PortSpec,
    LoadAudio,
    Separator,
    DetectOnsets,
    DetectSongSections,
    AudioFilter,
    Classify,
)
