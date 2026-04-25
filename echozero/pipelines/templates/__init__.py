"""Pipeline template definitions. Import all templates to register them."""

# Import templates to trigger registration via @pipeline_template decorator
from . import drum_classification  # noqa: F401
from . import extract_classified_drums  # noqa: F401
from . import extract_song_drum_events  # noqa: F401
from . import full_analysis  # noqa: F401
from . import onset_detection  # noqa: F401
from . import stem_separation  # noqa: F401
