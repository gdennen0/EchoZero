"""
MA3 Routing Service

Handles routing configuration for MA3 to EchoZero event mapping.
Provides template-based and custom routing strategies.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json

from src.features.ma3.domain.ma3_event import MA3Event
from src.utils.message import Log


class RoutingStrategy(Enum):
    """Strategy for routing MA3 events to EchoZero."""
    TRACK_TO_TRACK = auto()  # MA3 track N → EZ layer N
    BY_CLASSIFICATION = auto()  # Route based on event properties
    TEMPLATE_BASED = auto()  # Use predefined template
    CUSTOM = auto()  # User-defined rules


@dataclass
class ClassificationRule:
    """
    Rule for mapping MA3 event properties to EZ classification.
    
    Example:
        - If MA3 event.name contains "kick" → EZ classification "kick"
        - If MA3 event.type == "fader" → EZ classification "fader"
    """
    
    # Condition
    ma3_property: str  # "name", "type", "cmd", "track", etc.
    condition: str  # "equals", "contains", "starts_with", "ends_with", "regex"
    value: Any  # Value to match
    
    # Result
    ez_classification: str
    ez_layer_id: Optional[str] = None
    
    def matches(self, ma3_event: MA3Event) -> bool:
        """Check if this rule matches the given MA3 event."""
        # Get property value from event
        if self.ma3_property == "name":
            prop_value = ma3_event.name
        elif self.ma3_property == "type":
            prop_value = ma3_event.event_type
        elif self.ma3_property == "cmd":
            prop_value = ma3_event.cmd or ""
        elif self.ma3_property == "track":
            prop_value = ma3_event.track
        elif self.ma3_property == "track_group":
            prop_value = ma3_event.track_group
        else:
            return False
        
        # Apply condition
        if self.condition == "equals":
            return prop_value == self.value
        elif self.condition == "contains":
            return self.value.lower() in str(prop_value).lower()
        elif self.condition == "starts_with":
            return str(prop_value).lower().startswith(self.value.lower())
        elif self.condition == "ends_with":
            return str(prop_value).lower().endswith(self.value.lower())
        elif self.condition == "regex":
            import re
            return bool(re.search(self.value, str(prop_value)))
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ma3_property': self.ma3_property,
            'condition': self.condition,
            'value': self.value,
            'ez_classification': self.ez_classification,
            'ez_layer_id': self.ez_layer_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassificationRule':
        """Create from dictionary."""
        return cls(
            ma3_property=data['ma3_property'],
            condition=data['condition'],
            value=data['value'],
            ez_classification=data['ez_classification'],
            ez_layer_id=data.get('ez_layer_id'),
        )


@dataclass
class RoutingConfig:
    """
    Configuration for routing MA3 events to EchoZero.
    
    Supports multiple routing strategies:
    - Template-based (predefined mappings)
    - Track-to-track (direct mapping)
    - Classification rules (property-based routing)
    """
    
    strategy: RoutingStrategy = RoutingStrategy.TEMPLATE_BASED
    template_name: str = "single_track"
    
    # Track-to-track mapping
    track_mapping: Dict[int, str] = field(default_factory=dict)  # MA3 track → EZ layer_id
    
    # Classification rules (evaluated in order)
    classification_rules: List[ClassificationRule] = field(default_factory=list)
    
    # Default fallback
    default_classification: str = "event"
    default_layer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy': self.strategy.name,
            'template_name': self.template_name,
            'track_mapping': self.track_mapping.copy(),
            'classification_rules': [r.to_dict() for r in self.classification_rules],
            'default_classification': self.default_classification,
            'default_layer_id': self.default_layer_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingConfig':
        """Create from dictionary."""
        return cls(
            strategy=RoutingStrategy[data.get('strategy', 'TEMPLATE_BASED')],
            template_name=data.get('template_name', 'single_track'),
            track_mapping=data.get('track_mapping', {}).copy(),
            classification_rules=[
                ClassificationRule.from_dict(r) 
                for r in data.get('classification_rules', [])
            ],
            default_classification=data.get('default_classification', 'event'),
            default_layer_id=data.get('default_layer_id'),
        )


class MA3RoutingService:
    """
    Service for routing MA3 events to EchoZero.
    
    Provides template-based and custom routing strategies.
    """
    
    # Predefined templates
    TEMPLATES = {
        'single_track': {
            'name': 'Single Track',
            'description': 'All MA3 events → single EZ track',
            'strategy': RoutingStrategy.TEMPLATE_BASED,
            'default_classification': 'ma3_event',
            'default_layer_id': 'ma3_layer',
        },
        'multi_track': {
            'name': 'Multi-Track',
            'description': 'MA3 track N → EZ layer N',
            'strategy': RoutingStrategy.TRACK_TO_TRACK,
            'default_classification': 'event',
        },
        'by_type': {
            'name': 'By Type',
            'description': 'Route by event type (cmd/fader)',
            'strategy': RoutingStrategy.BY_CLASSIFICATION,
            'rules': [
                {
                    'ma3_property': 'type',
                    'condition': 'equals',
                    'value': 'cmd',
                    'ez_classification': 'command',
                    'ez_layer_id': 'commands',
                },
                {
                    'ma3_property': 'type',
                    'condition': 'equals',
                    'value': 'fader',
                    'ez_classification': 'fader',
                    'ez_layer_id': 'faders',
                },
            ],
            'default_classification': 'event',
        },
        'by_name': {
            'name': 'By Name',
            'description': 'Route by event name patterns',
            'strategy': RoutingStrategy.BY_CLASSIFICATION,
            'rules': [
                {
                    'ma3_property': 'name',
                    'condition': 'contains',
                    'value': 'kick',
                    'ez_classification': 'kick',
                },
                {
                    'ma3_property': 'name',
                    'condition': 'contains',
                    'value': 'snare',
                    'ez_classification': 'snare',
                },
                {
                    'ma3_property': 'name',
                    'condition': 'contains',
                    'value': 'hat',
                    'ez_classification': 'hihat',
                },
            ],
            'default_classification': 'percussion',
        },
    }
    
    def __init__(self):
        """Initialize routing service."""
        self._current_config = RoutingConfig()
    
    @property
    def current_config(self) -> RoutingConfig:
        """Get current routing configuration."""
        return self._current_config
    
    def apply_template(self, template_name: str) -> bool:
        """
        Apply a predefined routing template.
        
        Args:
            template_name: Name of template to apply
            
        Returns:
            True if template was applied successfully
        """
        if template_name not in self.TEMPLATES:
            Log.error(f"Unknown routing template: {template_name}")
            return False
        
        template = self.TEMPLATES[template_name]
        Log.info(f"Applying routing template: {template['name']}")
        
        # Create new config from template
        config = RoutingConfig(
            strategy=template['strategy'],
            template_name=template_name,
            default_classification=template.get('default_classification', 'event'),
            default_layer_id=template.get('default_layer_id'),
        )
        
        # Add rules if present
        if 'rules' in template:
            for rule_data in template['rules']:
                rule = ClassificationRule.from_dict(rule_data)
                config.classification_rules.append(rule)
        
        self._current_config = config
        return True
    
    def set_config(self, config: RoutingConfig):
        """Set custom routing configuration."""
        self._current_config = config
        Log.info(f"Set custom routing config: {config.strategy.name}")
    
    def add_classification_rule(self, rule: ClassificationRule):
        """Add a classification rule to current config."""
        self._current_config.classification_rules.append(rule)
        Log.debug(f"Added classification rule: {rule.ma3_property} {rule.condition} {rule.value}")
    
    def set_track_mapping(self, ma3_track: int, ez_layer_id: str):
        """Set track-to-track mapping."""
        self._current_config.track_mapping[ma3_track] = ez_layer_id
        Log.debug(f"Set track mapping: MA3 track {ma3_track} → EZ layer {ez_layer_id}")
    
    def route_ma3_event(self, ma3_event: MA3Event) -> Tuple[str, Optional[str]]:
        """
        Route MA3 event to EchoZero classification and layer.
        
        Args:
            ma3_event: MA3Event to route
            
        Returns:
            Tuple of (classification, layer_id)
        """
        config = self._current_config
        
        # Strategy: Track-to-track
        if config.strategy == RoutingStrategy.TRACK_TO_TRACK:
            layer_id = config.track_mapping.get(ma3_event.track)
            if layer_id:
                classification = ma3_event.name or config.default_classification
                return (classification, layer_id)
        
        # Strategy: By classification (rules)
        if config.strategy in (RoutingStrategy.BY_CLASSIFICATION, RoutingStrategy.TEMPLATE_BASED):
            # Try each rule in order
            for rule in config.classification_rules:
                if rule.matches(ma3_event):
                    Log.debug(f"Matched rule: {rule.ez_classification}")
                    return (rule.ez_classification, rule.ez_layer_id)
        
        # Strategy: Template-based (already handled by rules above)
        
        # Fallback to defaults
        classification = ma3_event.name or config.default_classification
        layer_id = config.default_layer_id
        
        Log.debug(f"Using default routing: {classification}, layer={layer_id}")
        return (classification, layer_id)
    
    def get_available_templates(self) -> List[Dict[str, str]]:
        """Get list of available templates."""
        return [
            {
                'id': key,
                'name': template['name'],
                'description': template['description'],
            }
            for key, template in self.TEMPLATES.items()
        ]
    
    def save_config(self, filepath: str):
        """Save current config to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._current_config.to_dict(), f, indent=2)
            Log.info(f"Saved routing config to {filepath}")
        except Exception as e:
            Log.error(f"Failed to save routing config: {e}")
    
    def load_config(self, filepath: str) -> bool:
        """Load config from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self._current_config = RoutingConfig.from_dict(data)
            Log.info(f"Loaded routing config from {filepath}")
            return True
        except Exception as e:
            Log.error(f"Failed to load routing config: {e}")
            return False
