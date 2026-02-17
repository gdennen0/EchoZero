"""
Topological Sort for Block Execution Order

Determines the correct execution order for blocks based on their connections.
Blocks without dependencies execute first, followed by blocks that depend on them.
"""
from typing import List, Dict, Set, Optional
from collections import deque

from src.features.blocks.domain import Block
from src.features.connections.domain import Connection
from src.features.blocks.domain import PortDirection
from src.utils.message import Log


class CyclicDependencyError(Exception):
    """Exception raised when blocks have circular dependencies"""
    
    def __init__(self, cycle: List[str]):
        """
        Initialize cyclic dependency error.
        
        Args:
            cycle: List of block IDs forming the cycle
        """
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)} -> {cycle[0]}")


def topological_sort_blocks(
    blocks: List[Block],
    connections: List[Connection]
) -> List[Block]:
    """
    Topologically sort blocks for execution order.
    
    Uses Kahn's algorithm to determine execution order:
    1. Blocks with no incoming connections execute first
    2. After a block executes, blocks that depend on it become available
    3. Process continues until all blocks are executed
    
    Args:
        blocks: List of Block entities to sort
        connections: List of Connection entities defining dependencies
        
    Returns:
        List of blocks in execution order
        
    Raises:
        CyclicDependencyError: If blocks have circular dependencies
        ValueError: If connections reference non-existent blocks
    """
    if not blocks:
        return []
    
    # Build block lookup by ID
    block_map: Dict[str, Block] = {block.id: block for block in blocks}
    
    # Validate all connections reference existing blocks
    for conn in connections:
        if conn.source_block_id not in block_map:
            raise ValueError(f"Connection references non-existent source block: {conn.source_block_id}")
        if conn.target_block_id not in block_map:
            raise ValueError(f"Connection references non-existent target block: {conn.target_block_id}")
    
    # Build dependency graph
    # incoming_count[block_id] = number of blocks that must execute before this block
    incoming_count: Dict[str, int] = {block.id: 0 for block in blocks}
    
    # outgoing_connections[block_id] = list of blocks that depend on this block
    outgoing_connections: Dict[str, List[str]] = {block.id: [] for block in blocks}
    
    # Build graph from connections
    for conn in connections:
        # Target block depends on source block
        incoming_count[conn.target_block_id] += 1
        outgoing_connections[conn.source_block_id].append(conn.target_block_id)
    
    # Find blocks with no dependencies (can execute first)
    queue = deque()
    for block_id, count in incoming_count.items():
        if count == 0:
            queue.append(block_id)
    
    # Topological sort
    result: List[Block] = []
    processed_count = 0
    
    while queue:
        # Get next block with no dependencies
        block_id = queue.popleft()
        block = block_map[block_id]
        result.append(block)
        processed_count += 1
        
        # Reduce dependency count for blocks that depend on this one
        for dependent_id in outgoing_connections[block_id]:
            incoming_count[dependent_id] -= 1
            # If dependent block now has no dependencies, add to queue
            if incoming_count[dependent_id] == 0:
                queue.append(dependent_id)
    
    # Check for cycles (if not all blocks processed, there's a cycle)
    if processed_count != len(blocks):
        # Find the cycle
        remaining = [bid for bid, count in incoming_count.items() if count > 0]
        if remaining:
            cycle = _find_cycle(block_map, outgoing_connections, remaining[0])
            raise CyclicDependencyError(cycle)
        else:
            raise ValueError("Topological sort failed: not all blocks processed")
    
    Log.info(f"Topological sort: Ordered {len(result)} blocks for execution")
    return result


def _find_cycle(
    block_map: Dict[str, Block],
    outgoing_connections: Dict[str, List[str]],
    start_block_id: str
) -> List[str]:
    """
    Find a cycle in the dependency graph.
    
    Uses DFS to find a cycle starting from the given block.
    
    Args:
        block_map: Map of block ID to Block entity
        outgoing_connections: Map of block ID to list of dependent block IDs
        start_block_id: Block ID to start cycle detection from
        
    Returns:
        List of block IDs forming the cycle
    """
    visited: Set[str] = set()
    path: List[str] = []
    
    def dfs(block_id: str) -> Optional[List[str]]:
        """DFS to find cycle"""
        if block_id in visited:
            # Found a cycle
            if block_id in path:
                # Cycle starts here
                cycle_start = path.index(block_id)
                return path[cycle_start:] + [block_id]
            return None
        
        visited.add(block_id)
        path.append(block_id)
        
        for dependent_id in outgoing_connections.get(block_id, []):
            cycle = dfs(dependent_id)
            if cycle:
                return cycle
        
        path.pop()
        return None
    
    cycle = dfs(start_block_id)
    if cycle:
        return cycle
    
    # Fallback: return a simple cycle with the problematic block
    return [start_block_id]


def validate_block_graph(
    blocks: List[Block],
    connections: List[Connection]
) -> tuple[bool, Optional[str]]:
    """
    Validate that the block graph is valid and executable.
    
    Checks:
    - All connections reference existing blocks
    - All connections reference existing ports
    - Port types are compatible
    - No circular dependencies
    
    Args:
        blocks: List of Block entities
        connections: List of Connection entities
        
    Returns:
        Tuple of (is_valid, error_message)
        If valid, returns (True, None)
        If invalid, returns (False, error_message)
    """
    # Build block lookup
    block_map: Dict[str, Block] = {block.id: block for block in blocks}
    
    # Check all connections reference existing blocks
    for conn in connections:
        if conn.source_block_id not in block_map:
            return False, f"Connection references non-existent source block: {conn.source_block_id}"
        
        if conn.target_block_id not in block_map:
            return False, f"Connection references non-existent target block: {conn.target_block_id}"
        
        source_block = block_map[conn.source_block_id]
        target_block = block_map[conn.target_block_id]
        
        # Check ports exist
        if not source_block.has_port(conn.source_output_name, PortDirection.OUTPUT):
            return False, f"Source block '{source_block.name}' has no output port '{conn.source_output_name}'"
        
        if not target_block.has_port(conn.target_input_name, PortDirection.INPUT):
            return False, f"Target block '{target_block.name}' has no input port '{conn.target_input_name}'"
        
        # Check port types are compatible
        source_port_type = source_block.get_port_type(conn.source_output_name, PortDirection.OUTPUT)
        target_port_type = target_block.get_port_type(conn.target_input_name, PortDirection.INPUT)
        
        if source_port_type and target_port_type:
            if not source_port_type.is_compatible_with(target_port_type):
                return False, (
                    f"Port type mismatch: {source_block.name}.{conn.source_output_name} "
                    f"({source_port_type.name}) -> {target_block.name}.{conn.target_input_name} "
                    f"({target_port_type.name})"
                )
    
    # Check for cycles
    try:
        topological_sort_blocks(blocks, connections)
    except CyclicDependencyError as e:
        return False, str(e)
    except ValueError as e:
        return False, str(e)
    
    return True, None

