"""
Layer Manager
=============

Manages timeline layers independently of event classifications.

Layers are explicit user-controlled constructs. Any event can be
placed on any layer - classification does NOT determine layer.

This is the single source of truth for:
- Layer ordering and positioning
- Layer heights
- Layer visibility and lock state
- Y-coordinate to layer mapping
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from PyQt6.QtGui import QColor
from PyQt6.QtCore import QObject, pyqtSignal

from ..types import TimelineLayer
from ..constants import TRACK_HEIGHT, TRACK_SPACING
from ..core.style import TimelineStyle


class LayerManager(QObject):
    """
    Manages timeline layers.
    
    Single source of truth for layer state. All layer operations
    go through this manager.
    
    Signals:
        layers_changed: Emitted when layers are added, removed, or reordered
        layer_updated: Emitted when a layer's properties change (id)
    """
    
    layers_changed = pyqtSignal()
    layer_updated = pyqtSignal(str)  # layer_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Layer storage: id -> TimelineLayer
        self._layers: Dict[str, TimelineLayer] = {}
        
        # Layer order: list of layer IDs in display order (top to bottom)
        self._order: List[str] = []
        
        # Auto-increment counter for generating layer IDs
        self._next_id = 0
        
        # Default layer height for new layers (can be updated from settings)
        self._default_layer_height = TRACK_HEIGHT
    
    def set_default_layer_height(self, height: int):
        """Set the default height for new layers."""
        self._default_layer_height = max(20, min(height, 200))
    
    def get_default_layer_height(self) -> int:
        """Get the default height for new layers."""
        return self._default_layer_height
    
    # =========================================================================
    # Layer CRUD Operations
    # =========================================================================
    
    def create_layer(
        self,
        name: str,
        layer_id: Optional[str] = None,
        index: int = -1,
        height: float = None,
        color: Optional[str] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        group_index: Optional[int] = None,
        is_synced: bool = False,
        show_manager_block_id: Optional[str] = None,
        ma3_track_coord: Optional[str] = None,
        derived_from_ma3: bool = False
    ) -> TimelineLayer:
        """
        Create a new layer.
        
        Args:
            name: Display name for the layer
            layer_id: Optional ID (auto-generated if not provided)
            index: Position in layer order (-1 = append at end)
            height: Layer height in pixels
            color: Optional hex color (auto-assigned if not provided)
            
        Returns:
            The created TimelineLayer (or existing layer if name already exists)
            If existing layer is returned and index is specified, layer is moved to that index
        """
        from src.utils.message import Log
        import traceback
        
        # DEBUG: Log where layer creation is being called from
        stack = traceback.extract_stack()
        caller_info = f"{stack[-2].filename}:{stack[-2].lineno}" if len(stack) >= 2 else "unknown"
        Log.debug(f"[LAYER_CREATE] LayerManager.create_layer() called for layer '{name}' from {caller_info}")
        Log.debug(f"[LAYER_CREATE]   group_name='{group_name}', group_id='{group_id}', index={index}")
        
        # Check if layer with this name already exists (prevent duplicates)
        if group_id is not None:
            existing = self.get_layer_by_name_and_group(name, group_id)
        else:
            existing = self.get_layer_by_name(name)
        if existing:
            Log.debug(f"[LAYER_CREATE]   Layer '{name}' already exists (ID: {existing.id}), returning existing")
            # If index is specified and layer exists, move it to the correct position
            # This ensures layers are in the correct order when recreating from stored order
            if index >= 0:
                current_index = self.get_layer_index(existing.id)
                if current_index != index:
                    self.reorder_layer(existing.id, index)
            return existing
        
        # Generate ID if not provided
        if layer_id is None:
            layer_id = self._generate_id()
        
        # Ensure unique ID
        if layer_id in self._layers:
            raise ValueError(f"Layer with ID '{layer_id}' already exists")
        
        # Determine index
        if index < 0 or index > len(self._order):
            index = len(self._order)
        
        # Use default height if not provided
        if height is None:
            height = self._default_layer_height
        
        # Auto-assign color if not provided
        if color is None:
            color = TimelineStyle.get_layer_color(len(self._layers)).name()
        
        # Create layer
        layer = TimelineLayer(
            id=layer_id,
            name=name,
            index=index,
            height=height,
            color=color,
            group_id=group_id,
            group_name=group_name,
            group_index=group_index,
            is_synced=is_synced,
            show_manager_block_id=show_manager_block_id,
            ma3_track_coord=ma3_track_coord,
            derived_from_ma3=derived_from_ma3
        )
        
        # Store
        self._layers[layer_id] = layer
        self._order.insert(index, layer_id)
        
        # Update indices for all layers
        self._reindex_layers()
        
        from src.utils.message import Log
        Log.info(f"[LAYER_CREATE] ✓ Created new layer '{name}' (ID: {layer_id}) with group_name='{group_name}'")
        
        self.layers_changed.emit()
        return layer
    
    def delete_layer(self, layer_id: str) -> bool:
        """
        Delete a layer.
        
        Args:
            layer_id: ID of layer to delete
            
        Returns:
            True if layer was deleted
            
        Note:
            Events on this layer should be moved elsewhere before deletion.
        """
        if layer_id not in self._layers:
            return False
        
        del self._layers[layer_id]
        self._order.remove(layer_id)
        
        # Update indices
        self._reindex_layers()
        
        self.layers_changed.emit()
        return True
    
    def update_layer(
        self,
        layer_id: str,
        name: Optional[str] = None,
        height: Optional[float] = None,
        color: Optional[str] = None,
        locked: Optional[bool] = None,
        visible: Optional[bool] = None,
        collapsed: Optional[bool] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        group_index: Optional[int] = None,
        check_synced_restrictions: bool = True
    ) -> bool:
        """
        Update layer properties.
        
        Args:
            layer_id: ID of layer to update
            **kwargs: Properties to update
            
        Returns:
            True if layer was found and updated
        """
        if layer_id not in self._layers:
            return False
        
        layer = self._layers[layer_id]
        
        # Check synced layer restrictions
        if check_synced_restrictions and layer.is_synced:
            # Synced layers cannot have name changed (derived from MA3)
            if name is not None and name != layer.name:
                from src.utils.message import Log
                Log.warning(f"Cannot rename synced layer '{layer.name}' - name is derived from MA3 source")
                return False
        
        if name is not None:
            layer.name = name
        if height is not None:
            layer.height = max(20.0, min(200.0, height))
        if color is not None:
            layer.color = color
        if locked is not None:
            layer.locked = locked
        if visible is not None:
            layer.visible = visible
        if collapsed is not None:
            layer.collapsed = collapsed
        if group_id is not None:
            layer.group_id = group_id
        if group_name is not None:
            layer.group_name = group_name
        if group_index is not None:
            layer.group_index = group_index
        
        self.layer_updated.emit(layer_id)
        return True
    
    def reorder_layer(self, layer_id: str, new_index: int) -> bool:
        """
        Move a layer to a new position.
        
        Args:
            layer_id: ID of layer to move
            new_index: New position (0 = top)
            
        Returns:
            True if layer was moved
        """
        if layer_id not in self._layers:
            return False
        
        current_index = self._order.index(layer_id)
        if current_index == new_index:
            return True
        
        # Remove and reinsert
        self._order.remove(layer_id)
        
        # Adjust index if we removed from before the target
        if current_index < new_index:
            new_index -= 1
        
        new_index = max(0, min(len(self._order), new_index))
        self._order.insert(new_index, layer_id)
        
        # Update indices
        self._reindex_layers()
        
        self.layers_changed.emit()
        return True
    
    def clear(self):
        """Remove all layers."""
        self._layers.clear()
        self._order.clear()
        self._next_id = 0
        self.layers_changed.emit()
    
    # =========================================================================
    # Layer Queries
    # =========================================================================
    
    def get_layer(self, layer_id: str) -> Optional[TimelineLayer]:
        """Get layer by ID."""
        return self._layers.get(layer_id)
    
    def get_layer_at_index(self, index: int) -> Optional[TimelineLayer]:
        """Get layer by visual index."""
        if 0 <= index < len(self._order):
            return self._layers.get(self._order[index])
        return None
    
    def get_layer_by_name(self, name: str) -> Optional[TimelineLayer]:
        """Get first layer with matching name."""
        for layer in self._layers.values():
            if layer.name == name:
                return layer
        return None

    def get_layer_by_name_and_group(self, name: str, group_id: str) -> Optional[TimelineLayer]:
        """Get first layer with matching name and group_id."""
        for layer in self._layers.values():
            if layer.name == name and layer.group_id == group_id:
                return layer
        return None
    
    def get_all_layers(self) -> List[TimelineLayer]:
        """Get all layers in display order."""
        return [self._layers[layer_id] for layer_id in self._order]
    
    def get_layer_ids(self) -> List[str]:
        """Get all layer IDs in display order."""
        return self._order.copy()
    
    def get_layer_names(self) -> List[str]:
        """Get all layer names in display order."""
        return [self._layers[layer_id].name for layer_id in self._order]
    
    def get_layer_count(self) -> int:
        """Get number of layers."""
        return len(self._layers)
    
    def get_synced_layers(self) -> List[TimelineLayer]:
        """Get all layers that are marked as synced with ShowManager/MA3."""
        return [layer for layer in self.get_all_layers() if layer.is_synced]
    
    def get_synced_layer_names(self) -> List[str]:
        """Get names of all synced layers."""
        return [layer.name for layer in self.get_synced_layers()]
    
    def has_layer(self, layer_id: str) -> bool:
        """Check if layer exists."""
        return layer_id in self._layers
    
    def get_first_layer_id(self) -> Optional[str]:
        """Get ID of first layer (for default placement)."""
        return self._order[0] if self._order else None
    
    def get_layer_index(self, layer_id: str) -> int:
        """Get visual index of a layer (-1 if not found)."""
        try:
            return self._order.index(layer_id)
        except ValueError:
            return -1
    
    # =========================================================================
    # Coordinate Mapping (Single Source of Truth)
    # =========================================================================
    
    def get_layer_y_position(self, layer_id: str) -> float:
        """
        Get the Y coordinate (top edge) of a layer in scene coordinates.
        
        Accounts for group header heights (headers act as "folders" that take up space).
        
        Args:
            layer_id: ID of the layer
            
        Returns:
            Y coordinate in pixels (from top of scene)
        """
        if layer_id not in self._layers:
            return 0.0
        
        GROUP_HEADER_HEIGHT = 18  # Height of group header dividers
        
        y = 0.0
        processed_groups: set = set()
        
        for lid in self._order:
            if lid == layer_id:
                # Check if this is the first layer in its group (needs header above)
                layer = self._layers[lid]
                if layer.visible and layer.group_id and layer.group_id not in processed_groups:
                    # Add header height before first layer in group
                    y += GROUP_HEADER_HEIGHT
                return y
            
            layer = self._layers[lid]
            if layer.visible:
                # Check if this is the first layer in its group (add header height)
                if layer.group_id and layer.group_id not in processed_groups:
                    processed_groups.add(layer.group_id)
                    y += GROUP_HEADER_HEIGHT  # Add header space before first layer in group
                
                y += layer.height + TRACK_SPACING
        
        return y
    
    def get_layer_y_position_by_index(self, index: int) -> float:
        """
        Get Y coordinate by layer index.
        
        Args:
            index: Visual index of the layer
            
        Returns:
            Y coordinate in pixels
        """
        if 0 <= index < len(self._order):
            return self.get_layer_y_position(self._order[index])
        return 0.0
    
    def get_layer_height(self, layer_id: str) -> float:
        """Get height of a layer."""
        layer = self._layers.get(layer_id)
        return layer.height if layer else TRACK_HEIGHT
    
    def get_layer_height_by_index(self, index: int) -> float:
        """Get height of a layer by index."""
        if 0 <= index < len(self._order):
            return self.get_layer_height(self._order[index])
        return TRACK_HEIGHT
    
    def get_layer_from_y(self, y: float) -> Optional[TimelineLayer]:
        """
        Get the layer at a given Y coordinate.
        
        Args:
            y: Y coordinate in scene space
            
        Returns:
            The layer at that position, or None if outside all layers
        """
        layer_id = self.get_layer_id_from_y(y)
        return self._layers.get(layer_id) if layer_id else None
    
    def get_layer_id_from_y(self, y: float) -> Optional[str]:
        """
        Get layer ID at a given Y coordinate.
        
        This is the SINGLE SOURCE OF TRUTH for Y -> layer mapping.
        All other code should use this method.
        
        Accounts for group header heights (headers take up space above first layer in group).
        If y falls within a header area, maps to the first layer in that group.
        
        Args:
            y: Y coordinate in scene space
            
        Returns:
            Layer ID, or None if outside all layers
        """
        if not self._order:
            return None
        
        GROUP_HEADER_HEIGHT = 18  # Height of group header dividers
        
        # Clamp to non-negative
        y = max(0.0, y)
        
        current_y = 0.0
        processed_groups: set = set()
        
        for layer_id in self._order:
            layer = self._layers[layer_id]
            if not layer.visible:
                continue
            
            # Check if this is the first layer in its group (has header above)
            has_header = layer.group_id and layer.group_id not in processed_groups
            if has_header:
                processed_groups.add(layer.group_id)
                header_y = current_y
                header_bottom = header_y + GROUP_HEADER_HEIGHT
                
                # If y falls within header area, return this layer (first in group)
                if y < header_bottom:
                    return layer_id
                
                # Move past header
                current_y = header_bottom
            
            # Check if y falls within this layer
            layer_bottom = current_y + layer.height
            if y < layer_bottom:
                return layer_id
            
            current_y = layer_bottom + TRACK_SPACING
        
        # Y is beyond all layers - return last visible layer
        for layer_id in reversed(self._order):
            if self._layers[layer_id].visible:
                return layer_id
        
        return self._order[-1] if self._order else None
    
    def get_layer_index_from_y(self, y: float) -> int:
        """
        Get layer index from Y coordinate.
        
        Convenience wrapper around get_layer_id_from_y.
        
        Args:
            y: Y coordinate in scene space
            
        Returns:
            Layer index (0-based), or 0 if no layers
        """
        layer_id = self.get_layer_id_from_y(y)
        if layer_id:
            return self.get_layer_index(layer_id)
        return 0
    
    def get_total_height(self) -> float:
        """
        Get total height of all visible layers including group headers.
        
        Accounts for group header heights (headers act as "folders" that take up space).
        
        Returns:
            Total height in pixels
        """
        if not self._order:
            return TRACK_HEIGHT
        
        GROUP_HEADER_HEIGHT = 18  # Height of group header dividers
        
        total = 0.0
        visible_count = 0
        processed_groups: set = set()
        
        for layer_id in self._order:
            layer = self._layers[layer_id]
            if layer.visible:
                # Check if this is the first layer in its group (has header above)
                if layer.group_id and layer.group_id not in processed_groups:
                    processed_groups.add(layer.group_id)
                    total += GROUP_HEADER_HEIGHT  # Add header height
                
                total += layer.height
                visible_count += 1
        
        # Add spacing between layers
        if visible_count > 1:
            total += TRACK_SPACING * (visible_count - 1)
        
        return total
    
    # =========================================================================
    # Bulk Operations
    # =========================================================================
    
    def set_layers(self, layers: List[TimelineLayer]) -> None:
        """
        Set all layers at once (replaces existing).
        
        Args:
            layers: List of TimelineLayer objects
        """
        self._layers.clear()
        self._order.clear()
        
        # Sort by index
        sorted_layers = sorted(layers, key=lambda l: l.index)
        
        for layer in sorted_layers:
            self._layers[layer.id] = layer
            self._order.append(layer.id)
        
        self._reindex_layers()
        self.layers_changed.emit()
    
    def create_layers_from_classifications(
        self,
        classifications: List[str],
        name_prefix: str = "",
        preserved_order: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Create layers from a list of classifications.
        
        Convenience method for backwards compatibility. Creates one
        layer per unique classification. Checks for existing layers
        by name to prevent duplicates.
        
        Args:
            classifications: List of classification names
            name_prefix: Optional prefix for layer names
            preserved_order: Optional list of layer names in desired order.
                            Layers in this list will be created first in this order.
                            Remaining layers will be appended in their natural order.
            
        Returns:
            Mapping of classification -> layer_id
        """
        mapping = {}
        unique = list(dict.fromkeys(classifications))  # Preserve order, remove dups
        
        # If preserved_order is provided, use it to order layers
        if preserved_order:
            # Create ordered list: preserved layers first (in order), then remaining layers
            ordered_names = []
            preserved_set = set(preserved_order)
            unique_set = set(unique)
            
            # Add preserved layers in order (if they exist in classifications)
            # Match by exact name (preserved order should already be normalized)
            for preserved_name in preserved_order:
                # Check if preserved name matches any classification (exact match)
                if preserved_name in unique_set:
                    name = f"{name_prefix}{preserved_name}" if name_prefix else preserved_name
                    if name not in ordered_names:  # Prevent duplicates
                        ordered_names.append(name)
                else:
                    # Also check with prefix if name_prefix is used
                    if name_prefix:
                        full_preserved = f"{name_prefix}{preserved_name}"
                        if full_preserved in unique_set and full_preserved not in ordered_names:
                            ordered_names.append(full_preserved)
            
            # Add remaining layers not in preserved order (maintain their relative order from unique)
            for classification in unique:
                name = f"{name_prefix}{classification}" if name_prefix else classification
                # Check both with and without prefix to avoid duplicates
                name_without_prefix = classification
                if name not in ordered_names and name_without_prefix not in preserved_set:
                    ordered_names.append(name)
            
            # Create layers in the ordered sequence
            from src.utils.message import Log
            Log.debug(f"[LAYER_CREATE] create_layers_from_classifications() creating {len(ordered_names)} layers from preserved order")
            for i, name in enumerate(ordered_names):
                layer = self.create_layer(name=name, index=i)
                # Map back to original classification (without prefix)
                classification = name[len(name_prefix):] if name_prefix and name.startswith(name_prefix) else name
                mapping[classification] = layer.id
        else:
            # Default behavior: create in natural order
            from src.utils.message import Log
            Log.debug(f"[LAYER_CREATE] create_layers_from_classifications() creating {len(unique)} layers from classifications")
            for i, classification in enumerate(unique):
                name = f"{name_prefix}{classification}" if name_prefix else classification
                Log.debug(f"[LAYER_CREATE]   Creating layer from classification: '{classification}' -> layer name: '{name}'")
                # create_layer now checks for existing layers by name, preventing duplicates
                layer = self.create_layer(name=name, index=i)
                mapping[classification] = layer.id
        
        return mapping
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    def _generate_id(self) -> str:
        """Generate a unique layer ID."""
        while True:
            layer_id = f"layer_{self._next_id}"
            self._next_id += 1
            if layer_id not in self._layers:
                return layer_id
    
    def _reindex_layers(self):
        """Update index property for all layers based on order."""
        for i, layer_id in enumerate(self._order):
            self._layers[layer_id].index = i
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'layers': [self._layers[lid].to_dict() for lid in self._order],
            'next_id': self._next_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerManager':
        """Deserialize from dictionary."""
        manager = cls()
        
        if 'layers' in data:
            layers = [TimelineLayer.from_dict(d) for d in data['layers']]
            manager.set_layers(layers)
        
        if 'next_id' in data:
            manager._next_id = data['next_id']
        
        return manager



