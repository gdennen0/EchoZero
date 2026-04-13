"""Pipeline template definitions. Import all templates to register them."""

# Import templates to trigger registration via @pipeline_template decorator
from echozero.pipelines.templates import onset_detection  # noqa: F401
from echozero.pipelines.templates import stem_separation  # noqa: F401
from echozero.pipelines.templates import full_analysis  # noqa: F401
from echozero.pipelines.templates import drum_classification  # noqa: F401
