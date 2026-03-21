"""
Graph: Directed acyclic graph aggregate root with full invariant enforcement.
Exists because the pipeline-as-data model (FP1) requires a validated, traversable DAG structure.
Owns blocks and connections — all validation happens here, not on individual entities.
"""

from __future__ import annotations

from collections import deque

from echozero.domain.enums import Direction, PortType
from echozero.domain.types import Block, Connection, Port
from echozero.errors import ValidationError


class Graph:
    """The directed acyclic graph of blocks and connections — enforces all pipeline invariants."""

    def __init__(self) -> None:
        self.blocks: dict[str, Block] = {}
        self.connections: list[Connection] = []

    # -- Mutators -----------------------------------------------------------

    def add_block(self, block: Block) -> None:
        """Insert a block into the graph, rejecting duplicate IDs."""
        if block.id in self.blocks:
            raise ValidationError(f"Duplicate block ID: {block.id}")
        self.blocks[block.id] = block

    def remove_block(self, block_id: str) -> None:
        """Remove a block and all connections that reference it."""
        if block_id not in self.blocks:
            raise ValidationError(f"Block not found: {block_id}")
        del self.blocks[block_id]
        self.connections = [
            c
            for c in self.connections
            if c.source_block_id != block_id and c.target_block_id != block_id
        ]

    def add_connection(self, connection: Connection) -> None:
        """Add a connection after validating all invariants."""
        self._validate_connection(connection)
        self.connections.append(connection)
        if self.has_cycle():
            self.connections.pop()
            raise ValidationError(
                f"Connection {connection.source_block_id} -> "
                f"{connection.target_block_id} would create a cycle"
            )

    def remove_connection(self, connection: Connection) -> None:
        """Remove a connection by value equality."""
        try:
            self.connections.remove(connection)
        except ValueError:
            raise ValidationError("Connection not found") from None

    # -- Queries ------------------------------------------------------------

    def validate(self) -> None:
        """Check all graph invariants: no cycles, valid ports, valid connections."""
        if self.has_cycle():
            raise ValidationError("Graph contains a cycle")
        for connection in self.connections:
            self._validate_connection(connection)

    def topological_sort(self) -> list[str]:
        """Return block IDs in execution order using Kahn's algorithm."""
        in_degree: dict[str, int] = {bid: 0 for bid in self.blocks}
        adjacency: dict[str, list[str]] = {bid: [] for bid in self.blocks}

        for conn in self.connections:
            adjacency[conn.source_block_id].append(conn.target_block_id)
            in_degree[conn.target_block_id] += 1

        queue: deque[str] = deque(
            bid for bid, degree in in_degree.items() if degree == 0
        )
        result: list[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.blocks):
            raise ValidationError("Graph contains a cycle")

        return result

    def downstream_of(self, block_id: str) -> set[str]:
        """Return all transitive descendants of a block."""
        if block_id not in self.blocks:
            raise ValidationError(f"Block not found: {block_id}")

        adjacency: dict[str, list[str]] = {bid: [] for bid in self.blocks}
        for conn in self.connections:
            adjacency[conn.source_block_id].append(conn.target_block_id)

        visited: set[str] = set()
        queue: deque[str] = deque(adjacency[block_id])

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                queue.extend(adjacency[current])

        return visited

    def has_cycle(self) -> bool:
        """Detect cycles using DFS with coloring."""
        white = set(self.blocks.keys())
        gray: set[str] = set()

        adjacency: dict[str, list[str]] = {bid: [] for bid in self.blocks}
        for conn in self.connections:
            adjacency[conn.source_block_id].append(conn.target_block_id)

        def _dfs(node: str) -> bool:
            white.discard(node)
            gray.add(node)
            for neighbor in adjacency[node]:
                if neighbor in gray:
                    return True
                if neighbor in white and _dfs(neighbor):
                    return True
            gray.discard(node)
            return False

        while white:
            node = next(iter(white))
            if _dfs(node):
                return True

        return False

    # -- Internal -----------------------------------------------------------

    def _validate_connection(self, connection: Connection) -> None:
        """Validate a single connection against all invariants."""
        src_id = connection.source_block_id
        tgt_id = connection.target_block_id

        if src_id == tgt_id:
            raise ValidationError("Self-connections are not allowed")

        if src_id not in self.blocks:
            raise ValidationError(f"Source block not found: {src_id}")
        if tgt_id not in self.blocks:
            raise ValidationError(f"Target block not found: {tgt_id}")

        source_block = self.blocks[src_id]
        target_block = self.blocks[tgt_id]

        source_port = self._find_port(
            source_block, connection.source_output_name, Direction.OUTPUT
        )
        if source_port is None:
            raise ValidationError(
                f"Output port '{connection.source_output_name}' "
                f"not found on block '{src_id}'"
            )

        target_port = self._find_port(
            target_block, connection.target_input_name, Direction.INPUT
        )
        if target_port is None:
            raise ValidationError(
                f"Input port '{connection.target_input_name}' "
                f"not found on block '{tgt_id}'"
            )

        if source_port.port_type != target_port.port_type:
            raise ValidationError(
                f"Port type mismatch: "
                f"{source_port.port_type.name} -> {target_port.port_type.name}"
            )

        if target_port.port_type == PortType.AUDIO:
            existing = sum(
                1
                for c in self.connections
                if c.target_block_id == tgt_id
                and c.target_input_name == connection.target_input_name
            )
            if existing >= 1:
                raise ValidationError(
                    f"Audio input '{connection.target_input_name}' on block "
                    f"'{tgt_id}' already has a connection (fan-in not allowed for Audio)"
                )

    @staticmethod
    def _find_port(
        block: Block, port_name: str, direction: Direction
    ) -> Port | None:
        """Look up a port by name and direction on a block."""
        if direction == Direction.OUTPUT:
            ports = block.output_ports
        elif direction == Direction.INPUT:
            ports = block.input_ports
        else:
            ports = block.control_ports

        for port in ports:
            if port.name == port_name:
                return port
        return None
