"""
Connection Helper

Shared logic for connection UI operations.
All connection methods (drag, dialog, toolbar) use this helper
to ensure consistent behavior and validation.
"""
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from src.application.api.application_facade import ApplicationFacade
from src.application.api.result_types import CommandResult
from src.application.commands import CreateConnectionCommand, DeleteConnectionCommand
from src.utils.message import Log

if TYPE_CHECKING:
    from PyQt6.QtGui import QUndoStack


@dataclass
class PortInfo:
    """Information about a port for UI display"""
    block_id: str
    block_name: str
    port_name: str
    port_type: str
    is_output: bool
    is_connected: bool = False  # For inputs, indicates if already connected


class ConnectionHelper:
    """
    Helper class for connection UI operations.
    
    Centralizes:
    - Port enumeration and filtering
    - Compatibility checking
    - Connection creation through facade
    """
    
    def __init__(self, facade: ApplicationFacade, undo_stack: Optional["QUndoStack"] = None):
        self.facade = facade
        self.undo_stack = undo_stack
    
    def get_all_blocks(self) -> List[dict]:
        """
        Get all blocks in current project.
        
        Returns:
            List of block summaries with id and name
        """
        result = self.facade.list_blocks()
        if result.success and result.data:
            return [{"id": b.id, "name": b.name, "type": b.type} for b in result.data]
        return []
    
    def get_block_outputs(self, block_id: str) -> List[PortInfo]:
        """
        Get output ports for a block.
        
        Args:
            block_id: Block to get outputs for
            
        Returns:
            List of PortInfo for each output port
        """
        result = self.facade.describe_block(block_id)
        if not result.success or not result.data:
            return []
        
        block = result.data
        ports = []
        
        output_ports = block.get_outputs()
        for port_name, port in output_ports.items():
            ports.append(PortInfo(
                block_id=block.id,
                block_name=block.name,
                port_name=port_name,
                port_type=port.port_type.name,
                is_output=True
            ))
        
        return ports
    
    def get_block_inputs(self, block_id: str) -> List[PortInfo]:
        """
        Get input ports for a block, with connection status.
        
        Args:
            block_id: Block to get inputs for
            
        Returns:
            List of PortInfo for each input port
        """
        result = self.facade.describe_block(block_id)
        if not result.success or not result.data:
            return []
        
        block = result.data
        
        # Get existing connections to check which inputs are connected
        # Note: Multiple connections per input are now allowed (e.g., Event ports)
        connected_inputs = set()
        conn_result = self.facade.list_connections()
        if conn_result.success and conn_result.data:
            for conn in conn_result.data:
                if conn.target_block_id == block_id:
                    connected_inputs.add(conn.target_input_name)
        
        ports = []
        input_ports = block.get_inputs()
        for port_name, port in input_ports.items():
            ports.append(PortInfo(
                block_id=block.id,
                block_name=block.name,
                port_name=port_name,
                port_type=port.port_type.name,
                is_output=False,
                is_connected=port_name in connected_inputs
            ))
        
        return ports
    
    def get_block_bidirectional_ports(self, block_id: str) -> List[PortInfo]:
        """
        Get bidirectional ports for a block, with connection status.
        
        Args:
            block_id: Block to get bidirectional ports for
            
        Returns:
            List of PortInfo for each bidirectional port
        """
        result = self.facade.describe_block(block_id)
        if not result.success or not result.data:
            return []
        
        block = result.data
        
        # Check for existing bidirectional connections
        # Bidirectional ports should only have one connection (1:1 relationship)
        connected_bidirectional = set()
        conn_result = self.facade.list_connections()
        if conn_result.success and conn_result.data:
            for conn in conn_result.data:
                # Check if either end is this block's bidirectional port
                if conn.source_block_id == block_id:
                    source_block_result = self.facade.describe_block(conn.source_block_id)
                    if source_block_result.success and source_block_result.data:
                        source_block = source_block_result.data
                        bidirectional_ports = source_block.get_bidirectional()
                        if conn.source_output_name in bidirectional_ports:
                            connected_bidirectional.add(conn.source_output_name)
                
                if conn.target_block_id == block_id:
                    target_block_result = self.facade.describe_block(conn.target_block_id)
                    if target_block_result.success and target_block_result.data:
                        target_block = target_block_result.data
                        bidirectional_ports = target_block.get_bidirectional()
                        if conn.target_input_name in bidirectional_ports:
                            connected_bidirectional.add(conn.target_input_name)
        
        ports = []
        bidirectional_ports = block.get_bidirectional()
        for port_name, port in bidirectional_ports.items():
            ports.append(PortInfo(
                block_id=block.id,
                block_name=block.name,
                port_name=port_name,
                port_type=port.port_type.name,
                is_output=False,  # Bidirectional, but we use False for consistency
                is_connected=port_name in connected_bidirectional
            ))
        
        return ports
    
    def get_compatible_targets(
        self, 
        source_block_id: str, 
        source_port_name: str
    ) -> List[Tuple[dict, List[PortInfo]]]:
        """
        Get blocks with compatible input ports for a given output.
        
        Args:
            source_block_id: Source block ID
            source_port_name: Source output port name
            
        Returns:
            List of (block_info, compatible_ports) tuples
        """
        # Get source port type
        source_result = self.facade.describe_block(source_block_id)
        if not source_result.success or not source_result.data:
            return []
        
        source_block = source_result.data
        output_ports = source_block.get_outputs()
        source_port = output_ports.get(source_port_name)
        if not source_port:
            return []
        
        source_port_type = source_port.port_type
        
        # Find compatible targets
        compatible = []
        blocks = self.get_all_blocks()
        
        for block_info in blocks:
            # Skip self-connections
            if block_info["id"] == source_block_id:
                continue
            
            # Get inputs for this block
            inputs = self.get_block_inputs(block_info["id"])
            
            # Filter to compatible inputs
            # Note: Event ports can now have multiple connections, so we don't filter by is_connected
            # For other port types, you may still want to check is_connected if needed
            compatible_inputs = [
                p for p in inputs 
                if source_port_type.name == p.port_type  # Simple name comparison
            ]
            
            if compatible_inputs:
                compatible.append((block_info, compatible_inputs))
        
        return compatible
    
    def get_compatible_sources(
        self, 
        target_block_id: str, 
        target_port_name: str
    ) -> List[Tuple[dict, List[PortInfo]]]:
        """
        Get blocks with compatible output ports for a given input.
        
        Args:
            target_block_id: Target block ID
            target_port_name: Target input port name
            
        Returns:
            List of (block_info, compatible_ports) tuples
        """
        # Get target port type
        target_result = self.facade.describe_block(target_block_id)
        if not target_result.success or not target_result.data:
            return []
        
        target_block = target_result.data
        input_ports = target_block.get_inputs()
        target_port = input_ports.get(target_port_name)
        if not target_port:
            return []
        
        target_port_type = target_port.port_type
        
        # Find compatible sources
        compatible = []
        blocks = self.get_all_blocks()
        
        for block_info in blocks:
            # Skip self-connections
            if block_info["id"] == target_block_id:
                continue
            
            # Get outputs for this block
            outputs = self.get_block_outputs(block_info["id"])
            
            # Filter to compatible outputs
            compatible_outputs = [
                p for p in outputs 
                if target_port_type.name == p.port_type  # Simple name comparison
            ]
            
            if compatible_outputs:
                compatible.append((block_info, compatible_outputs))
        
        return compatible
    
    def check_compatibility(
        self,
        source_block_id: str,
        source_port_name: str,
        target_block_id: str,
        target_port_name: str
    ) -> Tuple[bool, str]:
        """
        Check if two ports can be connected.
        
        Handles both regular (input/output) and bidirectional connections.
        
        Returns:
            (is_compatible, reason) tuple
        """
        # Get source block
        source_result = self.facade.describe_block(source_block_id)
        if not source_result.success or not source_result.data:
            return False, "Source block not found"
        
        source_block = source_result.data
        
        # Get target block
        target_result = self.facade.describe_block(target_block_id)
        if not target_result.success or not target_result.data:
            return False, "Target block not found"
        
        target_block = target_result.data
        
        # Check if either port is bidirectional
        source_bidirectional = source_block.get_bidirectional()
        target_bidirectional = target_block.get_bidirectional()
        source_is_bidirectional = source_port_name in source_bidirectional
        target_is_bidirectional = target_port_name in target_bidirectional
        
        # Bidirectional to bidirectional connection
        if source_is_bidirectional and target_is_bidirectional:
            source_port_obj = source_bidirectional[source_port_name]
            target_port_obj = target_bidirectional[target_port_name]
            source_type = source_port_obj.port_type
            target_type = target_port_obj.port_type
            
            # Check type compatibility
            if not source_type.is_compatible_with(target_type):
                return False, f"Type mismatch: {source_type.name} <-> {target_type.name}"
            
            # Check if either port is already connected (1:1 relationship for bidirectional)
            bidirectional_ports_source = self.get_block_bidirectional_ports(source_block_id)
            bidirectional_ports_target = self.get_block_bidirectional_ports(target_block_id)
            
            for port in bidirectional_ports_source:
                if port.port_name == source_port_name and port.is_connected:
                    return False, f"Bidirectional port '{source_port_name}' is already connected"
            
            for port in bidirectional_ports_target:
                if port.port_name == target_port_name and port.is_connected:
                    return False, f"Bidirectional port '{target_port_name}' is already connected"
            
            return True, "Compatible (bidirectional)"
        
        # Mixed connection (bidirectional to regular) - not allowed
        if source_is_bidirectional or target_is_bidirectional:
            return False, "Cannot connect bidirectional ports to regular input/output ports"
        
        # Regular input/output connection
        output_ports = source_block.get_outputs()
        source_port = output_ports.get(source_port_name)
        if not source_port:
            return False, f"Output port '{source_port_name}' not found"
        
        input_ports = target_block.get_inputs()
        target_port = input_ports.get(target_port_name)
        if not target_port:
            return False, f"Input port '{target_port_name}' not found"
        
        source_type = source_port.port_type
        target_type = target_port.port_type
        
        # Check type compatibility
        if not source_type.is_compatible_with(target_type):
            return False, f"Type mismatch: {source_type.name} -> {target_type.name}"
        
        # Note: Multiple connections to the same input port are now allowed
        # This enables connecting multiple EventDataItems to Event input ports
        
        return True, "Compatible"
    
    def create_connection(
        self,
        source_block_id: str,
        source_port_name: str,
        target_block_id: str,
        target_port_name: str
    ) -> CommandResult:
        """
        Create a connection between two ports.
        
        Args:
            source_block_id: Source block ID
            source_port_name: Source output port name
            target_block_id: Target block ID  
            target_port_name: Target input port name
            
        Returns:
            CommandResult with connection data or error
        """
        Log.info(
            f"ConnectionHelper: Creating connection "
            f"{source_block_id}.{source_port_name} -> "
            f"{target_block_id}.{target_port_name}"
        )
        
        # Use CommandBus for undoable command
        cmd = CreateConnectionCommand(
            self.facade,
            source_block_id, source_port_name,
            target_block_id, target_port_name
        )
        self.facade.command_bus.execute(cmd)
        return CommandResult.success_result(message="Connection created")
    
    def get_existing_connections(self) -> List[dict]:
        """
        Get all existing connections in the project.
        
        Returns:
            List of connection dictionaries with id, source_block_id, source_block_name,
            source_port_name, target_block_id, target_block_name, target_port_name
        """
        result = self.facade.list_connections()
        if not result.success or not result.data:
            return []
        
        # Build a lookup map from block_id to block_name
        blocks = self.get_all_blocks()
        block_name_map = {block["id"]: block["name"] for block in blocks}
        
        # Build connection dictionaries with block names looked up from IDs
        connections = []
        for conn in result.data:
            connections.append({
                "id": conn.id,
                "source_block_id": conn.source_block_id,
                "source_block_name": block_name_map.get(conn.source_block_id, conn.source_block_id),
                "source_port_name": conn.source_output_name,
                "target_block_id": conn.target_block_id,
                "target_block_name": block_name_map.get(conn.target_block_id, conn.target_block_id),
                "target_port_name": conn.target_input_name,
            })
        
        return connections
    
    def get_connections_for_block(self, block_id: str) -> List[dict]:
        """
        Get all connections involving a specific block.
        
        Args:
            block_id: Block ID to get connections for
            
        Returns:
            List of connection dictionaries
        """
        all_connections = self.get_existing_connections()
        return [
            conn for conn in all_connections
            if conn["source_block_id"] == block_id or conn["target_block_id"] == block_id
        ]
    
    def disconnect_connection(self, connection_id: str) -> CommandResult:
        """
        Disconnect a connection by ID.
        
        Args:
            connection_id: Connection ID to disconnect
            
        Returns:
            CommandResult
        """
        Log.info(f"ConnectionHelper: Disconnecting connection {connection_id}")
        
        # Use CommandBus for undoable command
        cmd = DeleteConnectionCommand(self.facade, connection_id)
        self.facade.command_bus.execute(cmd)
        return CommandResult.success_result(message="Connection deleted")
    
    def disconnect_by_port(self, block_id: str, port_name: str) -> CommandResult:
        """
        Disconnect all connections to a specific input port on a block.
        
        Args:
            block_id: Block ID
            port_name: Input port name to disconnect (disconnects all connections to this port)
            
        Returns:
            CommandResult
        """
        Log.info(f"ConnectionHelper: Disconnecting all connections to {block_id}.{port_name}")
        
        # Find all connections to this port, then disconnect them
        connections = self.get_existing_connections()
        port_connections = [
            conn for conn in connections
            if conn["target_block_id"] == block_id and conn["target_port_name"] == port_name
        ]
        
        if not port_connections:
            return CommandResult.error_result(message=f"No connections found for {block_id}.{port_name}")
        
        # Disconnect all connections to this port
        for conn in port_connections:
            result = self.disconnect_connection(conn["id"])
            if not result.success:
                return result
        
        return CommandResult.success_result(
            message=f"Disconnected {len(port_connections)} connection(s) from {block_id}.{port_name}"
        )

