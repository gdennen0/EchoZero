"""
Block entity

Represents a processing node in the EchoZero project graph.
Blocks process data and can be connected to other blocks.

UNIFIED: Ports are now unified Port entities stored in a single dictionary.
Port names are unique per direction - same name can exist in different directions.
Uses composite keys: "{direction}:{port_name}" for internal storage.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import uuid

from src.features.blocks.domain.port import Port
from src.shared.domain.value_objects.port_type import PortType
from src.features.blocks.domain.port_direction import PortDirection


@dataclass
class Block:
    """
    Block entity - represents a processing node.
    
    A block has:
    - Identity (id, name - unique within project)
    - Type (determines processing behavior)
    - Parent project reference
    - Unified ports dictionary (composite keys: "{direction}:{port_name}" -> Port)
    - Metadata for additional information
    
    Port structure:
    - Single unified dictionary: ports: Dict[str, Port]
    - Composite keys: "{direction}:{port_name}" (e.g., "input:audio", "output:audio")
    - Port names are unique per direction - same name can exist in different directions
    - Example: {"input:audio": Port(...), "output:audio": Port(...)}
    
    Bidirectional ports are for two-way command/data exchange (e.g., ShowManager <-> Editor).
    They connect to other bidirectional ports, not to inputs or outputs.
    
    Connections reference blocks directly by block_id + port_name + direction.
    """
    id: str
    project_id: str
    name: str
    type: str
    ports: Dict[str, Port] = field(default_factory=dict)  # Composite key -> Port
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize block with defaults"""
        if not self.id:
            self.id = str(uuid.uuid4())
        
        # Validate required fields
        if not self.name or not self.name.strip():
            raise ValueError("Block name cannot be empty")
        if not self.type or not self.type.strip():
            raise ValueError("Block type cannot be empty")
        if not self.project_id:
            raise ValueError("Project ID cannot be empty")
    
    def rename(self, new_name: str):
        """Rename block"""
        if not new_name or not new_name.strip():
            raise ValueError("Block name cannot be empty")
        self.name = new_name.strip()
    
    def _make_port_key(self, name: str, direction: PortDirection) -> str:
        """Create composite key for port lookup"""
        return f"{direction.value}:{name}"
    
    def add_port(self, name: str, port_type: PortType, direction: PortDirection, 
                 metadata: Optional[dict] = None) -> None:
        """
        Add a port to the block.
        
        Port names are unique per direction - same name can exist in different directions.
        
        Args:
            name: Port name
            port_type: PortType value object
            direction: PortDirection enum (INPUT, OUTPUT, or BIDIRECTIONAL)
            metadata: Optional port-specific configuration
        
        Raises:
            ValueError: If port name is empty or port already exists for this direction
        """
        if not name or not name.strip():
            raise ValueError("Port name cannot be empty")
        
        key = self._make_port_key(name, direction)
        if key in self.ports:
            raise ValueError(f"{direction.value.capitalize()} port '{name}' already exists on block {self.name}")
        
        self.ports[key] = Port(name=name, port_type=port_type, 
                               direction=direction, metadata=metadata or {})
    
    def get_port(self, name: str, direction: PortDirection) -> Optional[Port]:
        """
        Get port by name and direction.
        
        Args:
            name: Port name
            direction: PortDirection to search
        
        Returns:
            Port if found, None otherwise
        """
        key = self._make_port_key(name, direction)
        return self.ports.get(key)
    
    def get_port_type(self, name: str, direction: PortDirection) -> Optional[PortType]:
        """
        Get port type by name for a specific direction.
        
        Args:
            name: Port name
            direction: PortDirection to search
        
        Returns:
            PortType if found, None otherwise
        """
        port = self.get_port(name, direction)
        return port.port_type if port else None
    
    def has_port(self, name: str, direction: PortDirection) -> bool:
        """
        Check if port exists with given name and direction.
        
        Args:
            name: Port name
            direction: PortDirection to check
        
        Returns:
            True if port exists, False otherwise
        """
        key = self._make_port_key(name, direction)
        return key in self.ports
    
    def get_ports_by_direction(self, direction: PortDirection) -> Dict[str, Port]:
        """
        Get all ports of a specific direction, keyed by port name.
        
        Args:
            direction: PortDirection to filter by
        
        Returns:
            Dictionary mapping port name -> Port
        """
        result = {}
        prefix = f"{direction.value}:"
        for key, port in self.ports.items():
            if key.startswith(prefix):
                port_name = key[len(prefix):]
                result[port_name] = port
        return result
    
    def get_inputs(self) -> Dict[str, Port]:
        """Convenience: Get all input ports, keyed by port name"""
        return self.get_ports_by_direction(PortDirection.INPUT)
    
    def get_outputs(self) -> Dict[str, Port]:
        """Convenience: Get all output ports, keyed by port name"""
        return self.get_ports_by_direction(PortDirection.OUTPUT)
    
    def get_bidirectional(self) -> Dict[str, Port]:
        """Convenience: Get all bidirectional ports, keyed by port name"""
        return self.get_ports_by_direction(PortDirection.BIDIRECTIONAL)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Uses composite keys for ports: "{direction}:{port_name}"
        """
        from src.shared.domain.value_objects.port_type import get_port_type
        
        # Serialize ports with composite keys
        ports_dict = {}
        for key, port in self.ports.items():
            ports_dict[key] = {
                "port_type": port.port_type.name,
                "direction": port.direction.value,
                "metadata": port.metadata
            }
        
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "type": self.type,
            "ports": ports_dict,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Block':
        """
        Create from dictionary.
        
        Uses unified ports format with composite keys: "{direction}:{port_name}"
        """
        from src.shared.domain.value_objects.port_type import get_port_type
        
        ports = {}
        
        # Parse unified ports format with composite keys
        if "ports" in data and isinstance(data["ports"], dict):
            for key, port_data in data["ports"].items():
                if isinstance(port_data, dict):
                    port_type_name = port_data.get("port_type", port_data.get("type", ""))
                    direction_str = port_data.get("direction", "")
                    metadata = port_data.get("metadata", {})
                    
                    try:
                        direction = PortDirection.from_string(direction_str)
                        port_type = get_port_type(port_type_name)
                        ports[key] = Port(
                            name=key.split(":", 1)[1] if ":" in key else key,
                            port_type=port_type,
                            direction=direction,
                            metadata=metadata
                        )
                    except (ValueError, KeyError) as e:
                        # Skip invalid ports
                        continue
        
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            name=data["name"],
            type=data["type"],
            ports=ports,
            metadata=data.get("metadata", {})
        )

