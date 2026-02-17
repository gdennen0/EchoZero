"""
ShowManager Block Processor

Orchestrates bidirectional communication between EchoZero and grandMA3.
Uses a manipulator port to connect to Editor blocks for command exchange.

The ShowManager:
- Sends commands to Editor blocks (local) or MA3 (via OSC)
- Receives change notifications from both sides
- Manages synchronization state and conflict resolution
"""

from typing import Dict, Optional, Any, List, TYPE_CHECKING
import json
import time

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.value_objects.port_type import MANIPULATOR_TYPE
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class ShowManagerBlockProcessor(BlockProcessor):
    """
    Processor for ShowManager block type.
    
    The ShowManager block is an orchestrator that:
    - Connects to Editor blocks via manipulator port
    - Sends commands to both Editor and MA3
    - Manages synchronization between EchoZero and MA3
    - Handles mapping templates and conflict resolution
    
    Unlike data processing blocks, ShowManager doesn't transform data.
    It issues commands and coordinates state between systems.
    """
    
    def can_process(self, block: Block) -> bool:
        return block.type == "ShowManager"
    
    def get_block_type(self) -> str:
        return "ShowManager"
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for ShowManager block.
        
        Status levels:
        - Error (0): Not connected to Editor block OR MA3 OSC not configured/listening
        - Ready (1): Connected to Editor block AND MA3 OSC configured
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        
        def _get_connection_state(blk: Block, f: "ApplicationFacade") -> tuple[bool, bool]:
            """
            Get connection state for ShowManager block.
            
            Returns:
                Tuple of (has_editor_connection, ma3_ready)
            """
            # Reload block from repository to ensure we have latest metadata
            try:
                block_result = f.describe_block(blk.id)
                if block_result.success and block_result.data:
                    blk = block_result.data
            except Exception as e:
                Log.debug(f"ShowManager status check: Failed to reload block, using provided block: {e}")
            
            # Check Editor connection
            has_editor_connection = False
            if hasattr(f, 'connection_service'):
                connections = f.connection_service.list_connections_by_block(blk.id)
                # Check for manipulator port connections (bidirectional)
                for conn in connections:
                    # Check if this ShowManager is connected to an Editor via manipulator port
                    if (conn.source_block_id == blk.id and conn.source_output_name == "manipulator") or \
                       (conn.target_block_id == blk.id and conn.target_input_name == "manipulator"):
                        # Verify the other end is an Editor block
                        other_block_id = conn.target_block_id if conn.source_block_id == blk.id else conn.source_block_id
                        other_result = f.describe_block(other_block_id)
                        if other_result.success and other_result.data and other_result.data.type == "Editor":
                            has_editor_connection = True
                            break
            
            # Check MA3 OSC configuration and listener status
            metadata = blk.metadata or {}
            ma3_ip = metadata.get("ma3_ip", "").strip()
            ma3_port = metadata.get("ma3_port", 0)
            ma3_configured = bool(ma3_ip and ma3_port > 0)
            
            # Check listener status from service (source of truth - works even when panel is closed)
            listening = False
            if hasattr(f, 'show_manager_listener_service') and f.show_manager_listener_service:
                listening = f.show_manager_listener_service.is_listening(blk.id)
            
            # MA3 is ready if configured AND listening
            ma3_ready = ma3_configured and listening
            

            return has_editor_connection, ma3_ready
        
        def check_neither_connected(blk: Block, f: "ApplicationFacade") -> bool:
            """
            Check if neither Editor nor MA3 is connected.
            Returns False (activates error level) when both are not connected.
            Returns True (passes to next level) when at least one is connected.
            """
            has_editor, ma3_ready = _get_connection_state(blk, f)
            neither_connected = not has_editor and not ma3_ready
            
            # Return False (activate error level) if neither is connected
            # Return True (pass to next level) if at least one is connected
            result = not neither_connected
            Log.debug(f"ShowManager status check: check_neither_connected returns {result} (editor={has_editor}, ma3={ma3_ready}, neither={neither_connected})")
            return result
        
        def check_only_one_connected(blk: Block, f: "ApplicationFacade") -> bool:
            """
            Check if exactly one of Editor or MA3 is connected (but not both).
            Returns False (activates warning level) when exactly one is connected.
            Returns True (passes to next level) when both are connected or neither is connected.
            """
            has_editor, ma3_ready = _get_connection_state(blk, f)
            only_one = (has_editor and not ma3_ready) or (not has_editor and ma3_ready)
            
            # Return False (activate warning level) if exactly one is connected
            # Return True (pass to next level) if both are connected or neither is connected
            result = not only_one
            Log.debug(f"ShowManager status check: check_only_one_connected returns {result} (editor={has_editor}, ma3={ma3_ready}, only_one={only_one})")
            return result
        
        return [
            BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Not Connected",
                color="#ff6b6b",  # Red
                conditions=[check_neither_connected]
            ),
            BlockStatusLevel(
                priority=1,
                name="warning",
                display_name="Partially Connected",
                color="#ffa94d",  # Orange
                conditions=[check_only_one_connected]
            ),
            BlockStatusLevel(
                priority=2,
                name="ready",
                display_name="Connected",
                color="#51cf66",  # Green
                conditions=[]
            )
        ]
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        ShowManager has no data outputs.
        
        It communicates via the manipulator port, which is bidirectional
        and doesn't produce output data items.
        """
        return {}
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process ShowManager block.
        
        The ShowManager doesn't process data in the traditional sense.
        It orchestrates communication between systems. Processing here
        means checking connection status and updating internal state.
        
        Args:
            block: Block entity
            inputs: Dict with 'manipulator' key (optional, bidirectional)
            metadata: Optional processing metadata
            
        Returns:
            Empty dict (no data outputs)
        """
        Log.info(f"ShowManagerBlockProcessor: Processing block '{block.name}'")
        
        # ShowManager processing involves:
        # 1. Checking connection status to MA3
        # 2. Validating manipulator connection to Editor
        # 3. Updating sync state
        
        # Get settings from block metadata (settings are at top level, not nested)
        # ShowManagerSettingsManager saves settings at metadata top level
        metadata = block.metadata or {}
        
        # Log configuration
        ma3_ip = metadata.get("ma3_ip", "127.0.0.1")
        ma3_port = metadata.get("ma3_port", 9001)
        target_timecode = metadata.get("target_timecode", 1)
        
        Log.debug(f"ShowManager config: MA3={ma3_ip}:{ma3_port}, TC={target_timecode}")
        
        # Check for manipulator input (bidirectional connection to Editor)
        manipulator_input = inputs.get("manipulator")
        if manipulator_input:
            Log.debug("ShowManager: Manipulator port connected")
        else:
            Log.debug("ShowManager: No manipulator connection")
        
        # ShowManager doesn't produce output data items
        return {}
    
    def get_default_inputs(self) -> Dict[str, Any]:
        """Get default input ports for ShowManager block."""
        return {}
    
    def get_default_outputs(self) -> Dict[str, Any]:
        """Get default output ports for ShowManager block."""
        return {}
    
    def get_default_bidirectional(self) -> Dict[str, Any]:
        """Get default bidirectional ports for ShowManager block."""
        return {
            "manipulator": MANIPULATOR_TYPE,
        }


def create_show_manager_block(
    facade,
    name: str = "ShowManager",
    position_x: float = 0.0,
    position_y: float = 0.0,
) -> Optional[Block]:
    """
    Factory function to create a ShowManager block.
    
    Args:
        facade: ApplicationFacade instance
        name: Block name
        position_x: X position
        position_y: Y position
        
    Returns:
        Created Block entity, or None if creation failed
    """
    from src.features.blocks.domain import Block
    from src.shared.domain.value_objects.port_type import MANIPULATOR_TYPE
    
    # Create block with manipulator input port
    result = facade.add_block(
        block_type="ShowManager",
        name=name,
        x=position_x,
        y=position_y,
    )
    
    if not result.success:
        Log.error(f"Failed to create ShowManager block: {result.message}")
        return None
    
    block = result.data
    
    # Configure ports
    from src.features.blocks.domain import PortDirection
    block.add_port("manipulator", MANIPULATOR_TYPE, PortDirection.BIDIRECTIONAL)
    
    # Set default metadata directly (not nested under "settings")
    # ShowManagerSettingsManager saves/loads settings at metadata top level
    from src.application.settings.show_manager_settings import ShowManagerSettings
    default_settings = ShowManagerSettings()
    block.metadata.update(default_settings.to_dict())
    
    # Update block in repository
    facade.project_repo.update_block(block)
    
    Log.info(f"Created ShowManager block: {block.name}")
    return block


# Register the processor
register_processor_class(ShowManagerBlockProcessor)
