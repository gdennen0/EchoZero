"""
ShowManager Block Settings

Settings for the ShowManager block, which orchestrates MA3 communication.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

from src.features.show_manager.domain.layer_sync_types import SyncType

from .base_settings import BaseSettings


def normalize_ma3_coord(coord: str | None) -> str | None:
    """
    Normalize MA3 coordinate format.
    
    Converts dot-separated format (e.g., "1.1.1") to standard format (e.g., "tc1_tg1_tr1").
    Inlined here to avoid circular import with application/__init__.py.
    """
    if not coord:
        return coord
    if coord.startswith("tc"):
        return coord
    if "." in coord:
        parts = coord.split(".")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return f"tc{parts[0]}_tg{parts[1]}_tr{parts[2]}"
    return coord
from .block_settings import BlockSettingsManager


class SafetyLevel(Enum):
    """Safety level for sync operations."""
    CONFIRM_ALL = "confirm_all"           # Confirm every change
    CONFIRM_DESTRUCTIVE = "confirm_destructive"  # Confirm deletes only
    BATCH = "batch"                       # Batch changes, confirm batch
    NONE = "none"                         # No confirmation


class MappingTemplate(Enum):
    """Predefined mapping templates."""
    DRUMS = "drums"         # Kick->Kick track, Snare->Snare track, etc.
    STEMS = "stems"         # Vocals->Vocals track, Bass->Bass track, etc.
    SINGLE = "single"       # All events -> single track
    CUSTOM = "custom"       # User-defined mappings


@dataclass
class ShowManagerSettings(BaseSettings):
    """Settings for ShowManager block."""
    
    # Connection settings
    ma3_ip: str = "127.0.0.1"
    ma3_port: int = 9001  # Port to send OSC to MA3
    listen_port: int = 9000  # Port to receive OSC from MA3
    listen_address: str = "127.0.0.1"  # Address/interface to bind listener
    target_timecode: int = 1  # Target timecode number in MA3
    
    # Sync settings
    safety_level: str = SafetyLevel.CONFIRM_DESTRUCTIVE.value
    mapping_template: str = MappingTemplate.SINGLE.value
    auto_sync_enabled: bool = False
    sync_on_change: bool = True  # Real-time sync on changes
    sync_interval: int = 5  # Auto-sync interval in seconds (if enabled)
    apply_updates_enabled: bool = True  # Apply MA3 updates to Editor when hooked
    force_send_osc: bool = False  # Allow OSC sends without readiness checks (testing)
    
    # Synced layers configuration
    # List of synced layer entities (EditorLayerEntity or MA3TrackEntity dicts via entity.to_dict())
    synced_layers: List[Dict[str, Any]] = field(default_factory=list)
    naming_migrated: bool = False
    
    # Conflict resolution
    conflict_resolution_strategy: str = "prompt_user"  # "ma3_wins", "ez_wins", "prompt_user", "last_write_wins"
    
    # Layer mappings (MA3 coord -> EZ layer name)
    layer_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Custom mapping (legacy, used when mapping_template is "custom")
    custom_layer_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Connection state (not persisted, but tracked)
    last_connection_time: Optional[float] = None
    
    def get_safety_level(self) -> SafetyLevel:
        """Get safety level as enum."""
        try:
            return SafetyLevel(self.safety_level)
        except ValueError:
            return SafetyLevel.CONFIRM_DESTRUCTIVE
    
    def get_mapping_template(self) -> MappingTemplate:
        """Get mapping template as enum."""
        try:
            return MappingTemplate(self.mapping_template)
        except ValueError:
            return MappingTemplate.SINGLE

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary for storage.
        
        Overrides base implementation to avoid recursion issues with
        deeply nested synced_layers data when using asdict().
        """
        # Build dict manually to avoid deep copy recursion in asdict()
        return {
            "ma3_ip": self.ma3_ip,
            "ma3_port": self.ma3_port,
            "listen_port": self.listen_port,
            "listen_address": self.listen_address,
            "target_timecode": self.target_timecode,
            "safety_level": self.safety_level,
            "mapping_template": self.mapping_template,
            "auto_sync_enabled": self.auto_sync_enabled,
            "sync_on_change": self.sync_on_change,
            "sync_interval": self.sync_interval,
            "apply_updates_enabled": self.apply_updates_enabled,
            "force_send_osc": self.force_send_osc,
            # synced_layers is already a list of dicts - just copy the list
            "synced_layers": list(self.synced_layers) if self.synced_layers else [],
            "naming_migrated": self.naming_migrated,
            "conflict_resolution_strategy": self.conflict_resolution_strategy,
            # layer_mappings and custom_layer_mappings are simple dicts
            "layer_mappings": dict(self.layer_mappings) if self.layer_mappings else {},
            "custom_layer_mappings": dict(self.custom_layer_mappings) if self.custom_layer_mappings else {},
            "last_connection_time": self.last_connection_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShowManagerSettings":
        """Create settings from dictionary with backward compatibility."""
        if "layer_mappings" not in data and "custom_layer_mappings" in data:
            data = {**data, "layer_mappings": data.get("custom_layer_mappings") or {}}
        # Backfill new connection fields from legacy keys
        legacy_ip = data.get("ma3_ip", "127.0.0.1")
        legacy_port = data.get("ma3_port", 9001)
        if "listen_address" not in data:
            data = {**data, "listen_address": "127.0.0.1"}
        # Legacy live/session fallback (if present in older metadata)
        if "ma3_ip" not in data and "ma3_live_ip" in data:
            data = {**data, "ma3_ip": data.get("ma3_live_ip", legacy_ip)}
        if "ma3_port" not in data and "ma3_live_port" in data:
            data = {**data, "ma3_port": data.get("ma3_live_port", legacy_port)}
        if "ma3_ip" not in data and "ma3_session_ip" in data:
            data = {**data, "ma3_ip": data.get("ma3_session_ip", legacy_ip)}
        if "ma3_port" not in data and "ma3_session_port" in data:
            data = {**data, "ma3_port": data.get("ma3_session_port", legacy_port)}
        return super().from_dict(data)


class ShowManagerSettingsManager(BlockSettingsManager):
    """
    Settings manager for ShowManager block.
    
    Provides type-safe access to ShowManager configuration.
    """
    
    SETTINGS_CLASS = ShowManagerSettings
    
    @property
    def ma3_ip(self) -> str:
        return self._settings.ma3_ip
    
    @ma3_ip.setter
    def ma3_ip(self, value: str):
        if value != self._settings.ma3_ip:
            self._settings.ma3_ip = value
            self._save_setting("ma3_ip")
    
    @property
    def ma3_port(self) -> int:
        return self._settings.ma3_port
    
    @ma3_port.setter
    def ma3_port(self, value: int):
        if value != self._settings.ma3_port:
            self._settings.ma3_port = value
            self._save_setting("ma3_port")
    
    @property
    def listen_port(self) -> int:
        return self._settings.listen_port
    
    @listen_port.setter
    def listen_port(self, value: int):
        if value != self._settings.listen_port:
            self._settings.listen_port = value
            self._save_setting("listen_port")
    
    @property
    def listen_address(self) -> str:
        return self._settings.listen_address
    
    @listen_address.setter
    def listen_address(self, value: str):
        if value != self._settings.listen_address:
            self._settings.listen_address = value
            self._save_setting("listen_address")
    
    @property
    def target_timecode(self) -> int:
        return self._settings.target_timecode
    
    @target_timecode.setter
    def target_timecode(self, value: int):
        if value != self._settings.target_timecode:
            self._settings.target_timecode = value
            self._save_setting("target_timecode")
    
    @property
    def safety_level(self) -> SafetyLevel:
        return self._settings.get_safety_level()
    
    @safety_level.setter
    def safety_level(self, value: SafetyLevel):
        if value.value != self._settings.safety_level:
            self._settings.safety_level = value.value
            self._save_setting("safety_level")
    
    @property
    def mapping_template(self) -> MappingTemplate:
        return self._settings.get_mapping_template()
    
    @mapping_template.setter
    def mapping_template(self, value: MappingTemplate):
        if value.value != self._settings.mapping_template:
            self._settings.mapping_template = value.value
            self._save_setting("mapping_template")
    
    @property
    def auto_sync_enabled(self) -> bool:
        return self._settings.auto_sync_enabled
    
    @auto_sync_enabled.setter
    def auto_sync_enabled(self, value: bool):
        if value != self._settings.auto_sync_enabled:
            self._settings.auto_sync_enabled = value
            self._save_setting("auto_sync_enabled")

    @property
    def apply_updates_enabled(self) -> bool:
        return self._settings.apply_updates_enabled

    @apply_updates_enabled.setter
    def apply_updates_enabled(self, value: bool):
        if value != self._settings.apply_updates_enabled:
            self._settings.apply_updates_enabled = value
            self._save_setting("apply_updates_enabled")

    @property
    def force_send_osc(self) -> bool:
        return self._settings.force_send_osc

    @force_send_osc.setter
    def force_send_osc(self, value: bool):
        if value != self._settings.force_send_osc:
            self._settings.force_send_osc = value
            self._save_setting("force_send_osc")
    
    @property
    def custom_layer_mappings(self) -> Dict[str, str]:
        return self._settings.custom_layer_mappings
    
    @custom_layer_mappings.setter
    def custom_layer_mappings(self, value: Dict[str, str]):
        normalized = {
            (normalize_ma3_coord(coord) or coord): layer
            for coord, layer in (value or {}).items()
        }
        self._settings.custom_layer_mappings = normalized
        self._settings.layer_mappings = normalized or {}
        self._save_setting("custom_layer_mappings")
        self._save_setting("layer_mappings")

    @property
    def layer_mappings(self) -> Dict[str, str]:
        """Get MA3 track -> EZ layer mappings."""
        return self._settings.layer_mappings

    @layer_mappings.setter
    def layer_mappings(self, value: Dict[str, str]):
        normalized = {
            (normalize_ma3_coord(coord) or coord): layer
            for coord, layer in (value or {}).items()
        }
        self._settings.layer_mappings = normalized or {}
        self._settings.custom_layer_mappings = normalized or {}
        self._save_setting("layer_mappings")
        self._save_setting("custom_layer_mappings")

    def set_mapping(self, ma3_coord: str, layer_name: str) -> None:
        """Set a single MA3 -> EZ layer mapping."""
        ma3_coord = normalize_ma3_coord(ma3_coord) or ma3_coord
        mappings = self._settings.layer_mappings.copy()
        mappings[ma3_coord] = layer_name
        self.layer_mappings = mappings

    def set_layer_mapping(self, ma3_coord: str, layer_name: str) -> None:
        """Alias for set_mapping (compatibility)."""
        self.set_mapping(ma3_coord, layer_name)

    def get_mapping(self, ma3_coord: str) -> Optional[str]:
        """Get a single mapping by MA3 track coord."""
        ma3_coord = normalize_ma3_coord(ma3_coord) or ma3_coord
        return self._settings.layer_mappings.get(ma3_coord)

    def remove_mapping(self, ma3_coord: str) -> None:
        """Remove a mapping by MA3 track coord."""
        ma3_coord = normalize_ma3_coord(ma3_coord) or ma3_coord
        mappings = self._settings.layer_mappings.copy()
        mappings.pop(ma3_coord, None)
        self.layer_mappings = mappings

    def remove_layer_mapping(self, ma3_coord: str) -> None:
        """Alias for remove_mapping (compatibility)."""
        self.remove_mapping(ma3_coord)

    def clear_all_mappings(self) -> None:
        """Clear all mappings."""
        self.layer_mappings = {}

    def get_unmapped_ma3_tracks(self, track_coords: List[str]) -> List[str]:
        """Return MA3 tracks without a mapping."""
        mapped = set(normalize_ma3_coord(coord) or coord for coord in (self._settings.layer_mappings or {}).keys())
        return [coord for coord in track_coords if (normalize_ma3_coord(coord) or coord) not in mapped]

    def has_conflicts(self) -> bool:
        """Return True if multiple MA3 tracks map to same EZ layer."""
        targets = list(self._settings.layer_mappings.values())
        return len(targets) != len(set(targets))

    def get_conflicts(self) -> Dict[str, List[str]]:
        """Return conflicts as {layer_name: [ma3_coords]}."""
        conflicts: Dict[str, List[str]] = {}
        for ma3_coord, layer_name in self._settings.layer_mappings.items():
            conflicts.setdefault(layer_name, []).append(ma3_coord)
        return {name: coords for name, coords in conflicts.items() if len(coords) > 1}
    
    # =========================================================================
    # Synced Layers Management
    # =========================================================================
    
    @property
    def synced_layers(self) -> List[Dict[str, Any]]:
        """Get synced layers list (entity dicts from EditorLayerEntity.to_dict() or MA3TrackEntity.to_dict())."""
        synced = self._settings.synced_layers.copy()
        normalized = []
        for entity in synced:
            if not isinstance(entity, dict):
                continue
            settings = entity.get("settings") if isinstance(entity.get("settings"), dict) else {}
            if settings is None:
                settings = {}
            if 'sync_type' not in entity:
                entity = {**entity, 'sync_type': SyncType.SHOWMANAGER_LAYER.value}
            updated_settings = settings.copy()
            if "direction" not in updated_settings:
                updated_settings["direction"] = updated_settings.get("sync_direction", "BIDIRECTIONAL")
            if "conflict_strategy" not in updated_settings:
                updated_settings["conflict_strategy"] = updated_settings.get("conflict_resolution", "PROMPT_USER")
            if "auto_apply" not in updated_settings:
                updated_settings["auto_apply"] = False
            if "apply_updates_enabled" not in updated_settings:
                updated_settings["apply_updates_enabled"] = updated_settings.get("apply_updates_enabled", True)
            if updated_settings != settings:
                entity = {**entity, "settings": updated_settings}
            if entity.get("coord"):
                entity = {**entity, "coord": normalize_ma3_coord(entity.get("coord"))}
            if entity.get("mapped_ma3_track_id"):
                entity = {
                    **entity,
                    "mapped_ma3_track_id": normalize_ma3_coord(entity.get("mapped_ma3_track_id")),
                }
            normalized.append(entity)
        return normalized.copy()
    
    @synced_layers.setter
    def synced_layers(self, value: List[Dict[str, Any]]):
        """Set synced layers list."""
        if value != self._settings.synced_layers:
            self._settings.synced_layers = value.copy() if value else []
            self._save_setting("synced_layers")

    @property
    def naming_migrated(self) -> bool:
        return bool(self._settings.naming_migrated)

    @naming_migrated.setter
    def naming_migrated(self, value: bool):
        if bool(value) != bool(self._settings.naming_migrated):
            self._settings.naming_migrated = bool(value)
            self._save_setting("naming_migrated")
    
    @staticmethod
    def _get_editor_layer_id(entity_dict: Dict[str, Any]) -> Optional[str]:
        """Extract editor layer ID from entity dict (handles both legacy and current keys)."""
        return entity_dict.get('editor_layer_id') or entity_dict.get('layer_id')

    @staticmethod
    def _get_ma3_coord(entity_dict: Dict[str, Any]) -> Optional[str]:
        """Extract MA3 coord from entity dict (handles both legacy and current keys)."""
        return entity_dict.get('ma3_coord') or entity_dict.get('coord')

    def add_synced_layer(self, entity_dict: Dict[str, Any]) -> None:
        """
        Add a synced layer entity to the list.
        
        Args:
            entity_dict: Entity dictionary (from SyncLayerEntity.to_dict())
        """
        synced = self._settings.synced_layers.copy()
        # Normalize MA3 coord keys
        ma3_coord = self._get_ma3_coord(entity_dict)
        if ma3_coord:
            # Normalize whichever key is present
            normalized = normalize_ma3_coord(ma3_coord)
            if 'ma3_coord' in entity_dict:
                entity_dict = {**entity_dict, "ma3_coord": normalized}
            elif 'coord' in entity_dict:
                entity_dict = {**entity_dict, "coord": normalized}
        if entity_dict.get("mapped_ma3_track_id"):
            entity_dict = {
                **entity_dict,
                "mapped_ma3_track_id": normalize_ma3_coord(entity_dict.get("mapped_ma3_track_id")),
            }
        entity_id = self._get_editor_layer_id(entity_dict) or self._get_ma3_coord(entity_dict)
        if entity_id:
            # Remove existing entry if present
            synced = [
                e for e in synced
                if (self._get_editor_layer_id(e) or self._get_ma3_coord(e)) != entity_id
            ]
        synced.append(entity_dict)
        self._settings.synced_layers = synced
        self._save_setting("synced_layers")

    def rename_editor_layer(self, old_name: str, new_name: str) -> None:
        """Rename an Editor layer in synced_layers and mappings."""
        if not old_name or not new_name or old_name == new_name:
            return
        synced = self._settings.synced_layers.copy()
        updated = False
        for idx, entity in enumerate(synced):
            if not isinstance(entity, dict):
                continue
            layer_id = self._get_editor_layer_id(entity)
            if layer_id == old_name:
                # Update whichever key is present
                updates = {"name": new_name}
                if 'editor_layer_id' in entity:
                    updates["editor_layer_id"] = new_name
                if 'layer_id' in entity:
                    updates["layer_id"] = new_name
                synced[idx] = {**entity, **updates}
                updated = True
            if entity.get("mapped_editor_layer_id") == old_name:
                synced[idx] = {**entity, "mapped_editor_layer_id": new_name}
                updated = True
        if updated:
            self._settings.synced_layers = synced
            self._save_setting("synced_layers")
        mappings = self._settings.layer_mappings.copy()
        remapped = False
        for coord, layer_name in list(mappings.items()):
            if layer_name == old_name:
                mappings[coord] = new_name
                remapped = True
        if remapped:
            self.layer_mappings = mappings

    def move_ma3_mapping(self, old_coord: str, new_coord: str) -> None:
        """Move mapping from one MA3 coord to another."""
        if not old_coord or not new_coord or old_coord == new_coord:
            return
        old_coord = normalize_ma3_coord(old_coord) or old_coord
        new_coord = normalize_ma3_coord(new_coord) or new_coord
        mappings = self._settings.layer_mappings.copy()
        mapped_layer = mappings.pop(old_coord, None)
        if mapped_layer:
            mappings[new_coord] = mapped_layer
        self.layer_mappings = mappings
    
    def remove_synced_layer(self, entity_type: str, entity_id: str) -> bool:
        """
        Remove a synced layer entity from the list.
        
        Args:
            entity_type: "editor" or "ma3"
            entity_id: layer_id for editor, coord for ma3
            
        Returns:
            True if removed, False if not found
        """
        synced = self._settings.synced_layers.copy()
        original_count = len(synced)
        
        if entity_type == "ma3":
            entity_id = normalize_ma3_coord(entity_id) or entity_id
        if entity_type == "editor":
            synced = [e for e in synced if self._get_editor_layer_id(e) != entity_id]
        elif entity_type == "ma3":
            synced = [e for e in synced if self._get_ma3_coord(e) != entity_id]
        else:
            return False
        
        if len(synced) < original_count:
            self._settings.synced_layers = synced
            self._save_setting("synced_layers")
            return True
        return False
    
    def get_synced_layer(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a synced layer entity by type and ID.
        
        Args:
            entity_type: "editor" or "ma3"
            entity_id: layer_id for editor, coord for ma3
            
        Returns:
            Entity dict if found, None otherwise
        """
        if entity_type == "ma3":
            entity_id = normalize_ma3_coord(entity_id) or entity_id
        for entity_dict in self._settings.synced_layers:
            if entity_type == "editor" and self._get_editor_layer_id(entity_dict) == entity_id:
                return entity_dict
            elif entity_type == "ma3" and self._get_ma3_coord(entity_dict) == entity_id:
                return entity_dict
        return None
    
    def update_synced_layer(self, entity_type: str, entity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a synced layer entity.
        
        Args:
            entity_type: "editor" or "ma3"
            entity_id: layer_id for editor, coord for ma3
            updates: Dict of fields to update
            
        Returns:
            True if updated, False if not found
        """
        synced = self._settings.synced_layers.copy()
        updated = False
        
        if entity_type == "ma3":
            entity_id = normalize_ma3_coord(entity_id) or entity_id
            if "coord" in updates and updates.get("coord"):
                updates = {**updates, "coord": normalize_ma3_coord(updates.get("coord"))}
            if "ma3_coord" in updates and updates.get("ma3_coord"):
                updates = {**updates, "ma3_coord": normalize_ma3_coord(updates.get("ma3_coord"))}
        for i, entity_dict in enumerate(synced):
            match = False
            if entity_type == "editor" and self._get_editor_layer_id(entity_dict) == entity_id:
                match = True
            elif entity_type == "ma3" and self._get_ma3_coord(entity_dict) == entity_id:
                match = True
            
            if match:
                synced[i] = {**entity_dict, **updates}
                updated = True
                break
        
        if updated:
            self._settings.synced_layers = synced
            self._save_setting("synced_layers")
        
        return updated
    
    def get_synced_editor_layers(self) -> List[Dict[str, Any]]:
        """Get all synced Editor layer entities."""
        return [e for e in self._settings.synced_layers if self._get_editor_layer_id(e)]
    
    def get_synced_ma3_tracks(self) -> List[Dict[str, Any]]:
        """Get all synced MA3 track entities."""
        return [e for e in self._settings.synced_layers if self._get_ma3_coord(e)]
    
    def set_custom_layer_mapping(self, layer_name: str, track_name: str):
        """Set a single custom layer->track mapping (legacy method)."""
        mappings = self._settings.custom_layer_mappings.copy()
        mappings[layer_name] = track_name
        self.custom_layer_mappings = mappings
    
    def remove_custom_layer_mapping(self, layer_name: str):
        """Remove a custom layer mapping (legacy method)."""
        mappings = self._settings.custom_layer_mappings.copy()
        if layer_name in mappings:
            del mappings[layer_name]
            self.custom_layer_mappings = mappings
    
    @property
    def sync_on_change(self) -> bool:
        return self._settings.sync_on_change
    
    @sync_on_change.setter
    def sync_on_change(self, value: bool):
        if value != self._settings.sync_on_change:
            self._settings.sync_on_change = value
            self._save_setting("sync_on_change")
    
    @property
    def sync_interval(self) -> int:
        return self._settings.sync_interval
    
    @sync_interval.setter
    def sync_interval(self, value: int):
        if value != self._settings.sync_interval:
            self._settings.sync_interval = value
            self._save_setting("sync_interval")
    
    @property
    def conflict_resolution_strategy(self) -> str:
        return self._settings.conflict_resolution_strategy
    
    @conflict_resolution_strategy.setter
    def conflict_resolution_strategy(self, value: str):
        if value != self._settings.conflict_resolution_strategy:
            self._settings.conflict_resolution_strategy = value
            self._save_setting("conflict_resolution_strategy")
    