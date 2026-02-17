"""
MA3 Mapping Templates

Predefined mapping profiles for translating EchoZero layers/classifications
to MA3 tracks and track groups.

Templates:
- drums: Map drum classifications (kick, snare, hi-hat, etc.) to dedicated tracks
- stems: Map stems (vocals, bass, drums, other) to tracks
- single: All events -> single track
- custom: User-defined mappings
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class MappingMode(Enum):
    """How to handle multiple events/layers."""
    ONE_TO_ONE = "one_to_one"      # Each classification -> dedicated track
    MERGE_ALL = "merge_all"        # All classifications -> single track
    BY_LAYER = "by_layer"          # Each layer -> dedicated track
    CUSTOM = "custom"              # User-defined mapping


@dataclass
class TrackMapping:
    """Defines how a classification/layer maps to an MA3 track."""
    source_pattern: str           # Pattern to match (e.g., "kick", "snare", "*")
    target_track_name: str        # Name of the MA3 track
    event_type: str = "cmd"       # "cmd" or "fader"
    color: Optional[str] = None   # Optional track color
    properties: Dict[str, Any] = field(default_factory=dict)  # Extra properties


@dataclass
class MappingTemplate:
    """A complete mapping template."""
    name: str
    description: str
    mode: MappingMode
    track_group_name: str = "EchoZero Events"
    mappings: List[TrackMapping] = field(default_factory=list)
    default_mapping: Optional[TrackMapping] = None  # Fallback for unmatched
    
    def get_track_for_classification(self, classification: str) -> Optional[TrackMapping]:
        """Get the track mapping for a classification."""
        # Try exact match first
        for mapping in self.mappings:
            if mapping.source_pattern == classification:
                return mapping
        
        # Try pattern match (simple wildcard)
        for mapping in self.mappings:
            if self._pattern_matches(mapping.source_pattern, classification):
                return mapping
        
        # Use default if available
        return self.default_mapping
    
    def _pattern_matches(self, pattern: str, value: str) -> bool:
        """Simple pattern matching with * wildcard."""
        if pattern == "*":
            return True
        if pattern.startswith("*"):
            return value.endswith(pattern[1:])
        if pattern.endswith("*"):
            return value.startswith(pattern[:-1])
        return pattern == value


class MappingTemplateRegistry:
    """Registry of all available mapping templates."""
    
    _templates: Dict[str, MappingTemplate] = {}
    
    @classmethod
    def register(cls, template: MappingTemplate):
        """Register a mapping template."""
        cls._templates[template.name.lower()] = template
    
    @classmethod
    def get(cls, name: str) -> Optional[MappingTemplate]:
        """Get a template by name."""
        return cls._templates.get(name.lower())
    
    @classmethod
    def list_all(cls) -> List[MappingTemplate]:
        """List all registered templates."""
        return list(cls._templates.values())
    
    @classmethod
    def get_template_names(cls) -> List[str]:
        """Get list of template names."""
        return sorted(cls._templates.keys())


def _create_default_templates():
    """Create and register default mapping templates."""
    
    # Drums template - maps common drum classifications
    MappingTemplateRegistry.register(MappingTemplate(
        name="drums",
        description="Map drum classifications to dedicated tracks",
        mode=MappingMode.ONE_TO_ONE,
        track_group_name="Drums",
        mappings=[
            TrackMapping(source_pattern="kick", target_track_name="Kick", event_type="cmd"),
            TrackMapping(source_pattern="snare", target_track_name="Snare", event_type="cmd"),
            TrackMapping(source_pattern="hi-hat", target_track_name="Hi-Hat", event_type="cmd"),
            TrackMapping(source_pattern="hihat", target_track_name="Hi-Hat", event_type="cmd"),
            TrackMapping(source_pattern="tom", target_track_name="Toms", event_type="cmd"),
            TrackMapping(source_pattern="tom*", target_track_name="Toms", event_type="cmd"),
            TrackMapping(source_pattern="crash", target_track_name="Cymbals", event_type="cmd"),
            TrackMapping(source_pattern="ride", target_track_name="Cymbals", event_type="cmd"),
            TrackMapping(source_pattern="cymbal*", target_track_name="Cymbals", event_type="cmd"),
            TrackMapping(source_pattern="clap", target_track_name="Claps", event_type="cmd"),
            TrackMapping(source_pattern="rim", target_track_name="Snare", event_type="cmd"),
            TrackMapping(source_pattern="rimshot", target_track_name="Snare", event_type="cmd"),
        ],
        default_mapping=TrackMapping(
            source_pattern="*",
            target_track_name="Other Drums",
            event_type="cmd",
        ),
    ))
    
    # Stems template - maps stem types
    MappingTemplateRegistry.register(MappingTemplate(
        name="stems",
        description="Map stem types to dedicated tracks",
        mode=MappingMode.ONE_TO_ONE,
        track_group_name="Stems",
        mappings=[
            TrackMapping(source_pattern="vocals", target_track_name="Vocals", event_type="cmd"),
            TrackMapping(source_pattern="bass", target_track_name="Bass", event_type="cmd"),
            TrackMapping(source_pattern="drums", target_track_name="Drums", event_type="cmd"),
            TrackMapping(source_pattern="other", target_track_name="Other", event_type="cmd"),
            TrackMapping(source_pattern="piano", target_track_name="Keys", event_type="cmd"),
            TrackMapping(source_pattern="guitar", target_track_name="Guitar", event_type="cmd"),
            TrackMapping(source_pattern="keys", target_track_name="Keys", event_type="cmd"),
            TrackMapping(source_pattern="synth", target_track_name="Synth", event_type="cmd"),
            TrackMapping(source_pattern="strings", target_track_name="Strings", event_type="cmd"),
        ],
        default_mapping=TrackMapping(
            source_pattern="*",
            target_track_name="Other",
            event_type="cmd",
        ),
    ))
    
    # Single track template - all events to one track
    MappingTemplateRegistry.register(MappingTemplate(
        name="single",
        description="All events to a single track",
        mode=MappingMode.MERGE_ALL,
        track_group_name="Events",
        mappings=[
            TrackMapping(source_pattern="*", target_track_name="All Events", event_type="cmd"),
        ],
    ))
    
    # By Layer template - each layer becomes a track
    MappingTemplateRegistry.register(MappingTemplate(
        name="by_layer",
        description="Each layer becomes a separate track",
        mode=MappingMode.BY_LAYER,
        track_group_name="Layers",
        mappings=[],  # Dynamic - created from layer names
        default_mapping=TrackMapping(
            source_pattern="*",
            target_track_name="{layer_name}",  # Template variable
            event_type="cmd",
        ),
    ))
    
    # Custom template - empty, user-defined
    MappingTemplateRegistry.register(MappingTemplate(
        name="custom",
        description="User-defined custom mapping",
        mode=MappingMode.CUSTOM,
        track_group_name="Custom",
        mappings=[],
    ))
    
    # Notes template - for note/pitch data
    MappingTemplateRegistry.register(MappingTemplate(
        name="notes",
        description="Map notes by octave or pitch range",
        mode=MappingMode.ONE_TO_ONE,
        track_group_name="Notes",
        mappings=[
            TrackMapping(source_pattern="C*", target_track_name="C Notes", event_type="fader"),
            TrackMapping(source_pattern="D*", target_track_name="D Notes", event_type="fader"),
            TrackMapping(source_pattern="E*", target_track_name="E Notes", event_type="fader"),
            TrackMapping(source_pattern="F*", target_track_name="F Notes", event_type="fader"),
            TrackMapping(source_pattern="G*", target_track_name="G Notes", event_type="fader"),
            TrackMapping(source_pattern="A*", target_track_name="A Notes", event_type="fader"),
            TrackMapping(source_pattern="B*", target_track_name="B Notes", event_type="fader"),
        ],
        default_mapping=TrackMapping(
            source_pattern="*",
            target_track_name="Notes",
            event_type="fader",
        ),
    ))
    
    # Intensity-based template - for dynamics/amplitude
    MappingTemplateRegistry.register(MappingTemplate(
        name="intensity",
        description="Map events by intensity/amplitude level",
        mode=MappingMode.ONE_TO_ONE,
        track_group_name="Intensity",
        mappings=[
            TrackMapping(source_pattern="soft", target_track_name="Soft", event_type="fader", 
                        properties={"base_value": 0.3}),
            TrackMapping(source_pattern="medium", target_track_name="Medium", event_type="fader",
                        properties={"base_value": 0.6}),
            TrackMapping(source_pattern="loud", target_track_name="Loud", event_type="fader",
                        properties={"base_value": 0.9}),
            TrackMapping(source_pattern="accent", target_track_name="Accents", event_type="cmd"),
        ],
        default_mapping=TrackMapping(
            source_pattern="*",
            target_track_name="Default",
            event_type="cmd",
        ),
    ))


def create_custom_template(
    name: str,
    mappings: List[Dict[str, str]],
    track_group_name: str = "Custom",
    description: str = "",
) -> MappingTemplate:
    """
    Create a custom mapping template from a list of mappings.
    
    Args:
        name: Template name
        mappings: List of {"source": "pattern", "target": "track_name"} dicts
        track_group_name: Name for the MA3 track group
        description: Optional description
        
    Returns:
        New MappingTemplate
    """
    track_mappings = []
    for mapping in mappings:
        track_mappings.append(TrackMapping(
            source_pattern=mapping.get("source", "*"),
            target_track_name=mapping.get("target", "Default"),
            event_type=mapping.get("event_type", "cmd"),
        ))
    
    return MappingTemplate(
        name=name,
        description=description or f"Custom template: {name}",
        mode=MappingMode.CUSTOM,
        track_group_name=track_group_name,
        mappings=track_mappings,
    )


# Initialize default templates on module import
_create_default_templates()
