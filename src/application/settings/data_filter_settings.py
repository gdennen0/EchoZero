"""
Input Filter Settings

Settings manager for block input filter selections.
Filter selections are stored per-input-port in block.metadata["filter_selections"].
Only input ports are filtered; output ports display expected outputs only.

Format: {port_name: {output_name: bool, ...}}
- True = item enabled (passes filter)
- False = item disabled (filtered out)
"""
from typing import Dict, Optional
from dataclasses import dataclass, field

from .block_settings import BlockSettingsManager
from .base_settings import BaseSettings
from src.utils.message import Log


@dataclass
class DataFilterSettings(BaseSettings):
    """
    Settings schema for input filter selections.
    
    Stores per-input-port filter selections: {input_port_name: {output_name: bool, ...}}
    Simple dict with true/false for each output name.
    """
    filter_selections: Dict[str, Dict[str, bool]] = field(default_factory=dict)


class BlockDataFilterSettingsManager(BlockSettingsManager):
    """
    Settings manager for block input filter selections.
    
    Provides type-safe access to filter selections stored in block.metadata.
    Filter selections are per-input-port: {input_port_name: {output_name: bool, ...}}
    """
    SETTINGS_CLASS = DataFilterSettings
    
    def get_port_selection(self, port_name: str) -> Optional[Dict[str, bool]]:
        """
        Get filter state for an input port.
        
        Args:
            port_name: Input port name
            
        Returns:
            Dict mapping output names to bool (True=enabled, False=disabled), or None if no selection
        """
        selections = self._settings.filter_selections
        return selections.get(port_name)
    
    def set_port_selection(self, port_name: str, filter_state: Dict[str, bool]) -> None:
        """
        Set filter state for an input port.
        
        Args:
            port_name: Input port name
            filter_state: Dict mapping output names to bool (True=enabled, False=disabled)
        """
        if not self._settings.filter_selections:
            self._settings.filter_selections = {}
        
        self._settings.filter_selections[port_name] = filter_state
        self._save_setting('filter_selections')
        self._do_save(immediate=True)
        
        enabled_count = sum(1 for v in filter_state.values() if v)
        Log.debug(
            f"BlockDataFilterSettingsManager: Set filter for {port_name}: "
            f"{enabled_count}/{len(filter_state)} enabled"
        )
        
        # Recalculate expected outputs immediately
        self._recalculate_expected_outputs()
    
    def set_item_state(self, port_name: str, output_name: str, enabled: bool) -> None:
        """
        Set state for a single filter item.
        
        Args:
            port_name: Input port name
            output_name: Output name to set state for
            enabled: True to enable, False to disable
        """
        if not self._settings.filter_selections:
            self._settings.filter_selections = {}
        
        if port_name not in self._settings.filter_selections:
            self._settings.filter_selections[port_name] = {}
        
        self._settings.filter_selections[port_name][output_name] = enabled
        self._save_setting('filter_selections')
        self._do_save(immediate=True)
    
    def clear_port_selection(self, port_name: str) -> None:
        """
        Clear filter selection for a port.
        
        Args:
            port_name: Port name
        """
        if self._settings.filter_selections and port_name in self._settings.filter_selections:
            del self._settings.filter_selections[port_name]
            self._save_setting('filter_selections')
            self._do_save(immediate=True)
            
            # Recalculate expected outputs immediately
            self._recalculate_expected_outputs()
    
    def get_all_selections(self) -> Dict[str, Dict[str, bool]]:
        """
        Get all filter selections.
        
        Returns:
            Dictionary mapping port names to filter state dicts
        """
        return dict(self._settings.filter_selections)
    
    def _recalculate_expected_outputs(self) -> None:
        """
        Recalculate expected outputs for this block and downstream blocks.
        
        Called when filter selections change, as filters affect what outputs
        connection-based blocks will produce.
        """
        if not self._facade or not self._block_id:
            return
        
        try:
            # Recalculate for this block
            if hasattr(self._facade, '_recalculate_expected_outputs'):
                self._facade._recalculate_expected_outputs(self._block_id)
            
            # Recalculate for downstream blocks
            if hasattr(self._facade, 'connection_repo'):
                connections = self._facade.connection_repo.list_by_block(self._block_id)
                downstream_ids = {
                    conn.target_block_id for conn in connections
                    if conn.source_block_id == self._block_id
                }
                
                for downstream_id in downstream_ids:
                    if hasattr(self._facade, '_recalculate_expected_outputs'):
                        self._facade._recalculate_expected_outputs(downstream_id)
        except Exception as e:
            Log.debug(f"BlockDataFilterSettingsManager: Failed to recalculate expected outputs: {e}")
