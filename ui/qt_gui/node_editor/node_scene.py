"""
Node Scene

QGraphicsScene managing blocks and connections.
"""
from typing import Dict, Optional, Tuple, List
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsItem, QGraphicsPathItem
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QTimer
from PyQt6.QtGui import QPen, QColor, QBrush, QPainterPath, QUndoStack

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import CreateConnectionCommand, DeleteBlockCommand
from src.utils.message import Log

from ui.qt_gui.node_editor.block_item import BlockItem
from ui.qt_gui.node_editor.audio_player_block_item import AudioPlayerBlockItem
from ui.qt_gui.node_editor.audio_filter_block_item import AudioFilterBlockItem
from ui.qt_gui.node_editor.eq_bands_block_item import EQBandsBlockItem
from ui.qt_gui.node_editor.audio_negate_block_item import AudioNegateBlockItem
from ui.qt_gui.node_editor.connection_item import ConnectionItem
from ui.qt_gui.connection.connection_helper import ConnectionHelper
from ui.qt_gui.design_system import Colors, Sizes


class NodeScene(QGraphicsScene):
    """
    Graphics scene for node editor.
    
    Manages:
    - Block items (visual nodes)
    - Connection items (visual edges)
    - Connection drawing interaction
    - Block positioning (with debounced saves to database)
    """
    
    block_selected = pyqtSignal(str)  # Emits block ID
    block_double_clicked = pyqtSignal(str)  # Emits block ID for panel opening
    
    def __init__(self, facade: ApplicationFacade, undo_stack: Optional[QUndoStack] = None):
        super().__init__()
        self.facade = facade
        self.undo_stack = undo_stack
        self.connection_helper = ConnectionHelper(facade, undo_stack=undo_stack)
        
        # Scene size
        self.setSceneRect(-5000, -5000, 10000, 10000)
        
        # Item tracking
        self.block_items: Dict[str, BlockItem] = {}  # block_id -> BlockItem
        self.connection_items: Dict[str, ConnectionItem] = {}  # connection_id -> ConnectionItem
        
        # Connection dragging state
        self._drag_connection: Optional[QGraphicsPathItem] = None
        self._drag_source_block_id: Optional[str] = None
        self._drag_source_port: Optional[str] = None
        self._drag_is_output: bool = False
        self._drag_is_bidirectional: bool = False  # Track if source port is bidirectional
        self._drag_start_pos: Optional[QPointF] = None
        
        # Grid settings
        self.grid_size = 20
        self.draw_grid = True
        
        # Selection
        self.selectionChanged.connect(self._on_selection_changed)
        
        # Position save debouncing (avoid excessive DB writes during drag)
        self._position_save_timers: Dict[str, QTimer] = {}  # block_id -> QTimer
        self._pending_positions: Dict[str, QPointF] = {}  # block_id -> position
        
        # Connection retry mechanism for connections that fail to render initially
        self._pending_connections: List = []  # Connections that failed to render, will retry
        
        # Refresh guard to prevent duplicate refreshes
        self._is_refreshing = False
    
    def drawBackground(self, painter, rect):
        """Draw grid background"""
        super().drawBackground(painter, rect)
        
        if not self.draw_grid:
            return
        
        # Draw grid
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)
        
        # Grid lines
        lines = []
        
        # Vertical lines
        x = left
        while x < rect.right():
            lines.append((x, rect.top(), x, rect.bottom()))
            x += self.grid_size
        
        # Horizontal lines
        y = top
        while y < rect.bottom():
            lines.append((rect.left(), y, rect.right(), y))
            y += self.grid_size
        
        # Draw lines
        pen = QPen(QColor(60, 60, 60))
        pen.setWidth(0)
        painter.setPen(pen)
        
        for line in lines:
            x1, y1, x2, y2 = line
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
    
    def refresh(self):
        """
        Refresh scene with current project state.
        
        STANDARDIZED LOAD SEQUENCE:
        1. Load ALL block summaries (fast)
        2. Load ALL full block entities (with skip_port_check for performance) - BATCHED
        3. Create ALL block items (synchronously)
        4. Position ALL blocks (synchronously)
        5. Load ALL connections (synchronously)
        6. Render ALL connections (synchronously)
        
        No async operations, no timers, no delays - everything loads in order.
        """
        # Guard against duplicate refreshes
        if self._is_refreshing:
            Log.debug("NodeScene: Refresh already in progress, skipping duplicate call")
            return
        
        self._is_refreshing = True
        try:
            # Clear existing items
            self.clear()
            self.block_items.clear()
            self.connection_items.clear()
            self._pending_connections.clear()
            
            # Check if we have a current project
            current_project_id = self.facade.get_current_project_id()
            if not current_project_id:
                return
            
            # STEP 1: Load ALL block summaries (fast, no DB writes)
            result = self.facade.list_blocks()
            if not result.success or not result.data:
                return
            
            block_summaries = result.data
            
            # STEP 2: Load ALL full block entities (BATCHED for efficiency)
            # Use batch loading instead of individual calls
            block_ids = [bs.id for bs in block_summaries]
            if block_ids:
                batch_result = self.facade.get_blocks_batch(block_ids)
                if batch_result.success and batch_result.data:
                    full_blocks = list(batch_result.data.values())
                else:
                    # Fallback to individual loading if batch fails
                    Log.debug("NodeScene: Batch load failed, falling back to individual loads")
                    full_blocks = []
                    for block_summary in block_summaries:
                        full_result = self.facade.describe_block(block_summary.id, skip_port_check=True)
                        if full_result.success and full_result.data:
                            full_blocks.append(full_result.data)
                        else:
                            Log.warning(f"Could not load full block for {block_summary.id}")
            else:
                full_blocks = []
            
            # STEP 3: Create ALL block items (synchronously, no delays)
            for block in full_blocks:
                try:
                    self._add_block_item_from_entity(block)
                except Exception as e:
                    Log.error(f"Failed to create block item for {block.name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            # STEP 4: Position ALL blocks (synchronously)
            # Auto-layout only blocks that don't have saved positions
            if self.block_items:
                blocks_without_positions = self._get_blocks_without_positions()
                if blocks_without_positions:
                    self.auto_layout("hierarchical", blocks_to_layout=blocks_without_positions)
            
            # STEP 5: Load ALL connections (synchronously, after all blocks are ready)
            result = self.facade.list_connections()
            if result.success and result.data:
                for conn in result.data:
                    self._add_connection_item(conn)
            
            # STEP 6: Force scene update (synchronously)
            self.update()
            
            # Update block data states (synchronously, no timers)
            self._update_all_block_states()
        finally:
            self._is_refreshing = False
    
    def _add_block_item_from_entity(self, block):
        """Add a block item from a full Block entity (optimized, no describe_block call)"""
        try:
            # Use specialized block items for blocks with embedded controls
            if block.type == "AudioPlayer":
                block_item = AudioPlayerBlockItem(block, self.facade, undo_stack=self.undo_stack)
            elif block.type == "AudioFilter":
                block_item = AudioFilterBlockItem(block, self.facade, undo_stack=self.undo_stack)
            elif block.type == "EQBands":
                block_item = EQBandsBlockItem(block, self.facade, undo_stack=self.undo_stack)
            elif block.type == "AudioNegate":
                block_item = AudioNegateBlockItem(block, self.facade, undo_stack=self.undo_stack)
            else:
                block_item = BlockItem(block, self.facade, undo_stack=self.undo_stack)
            
            # Connect signals
            block_item.signals.position_changed.connect(self._on_block_position_changed)
            block_item.signals.selected.connect(self._on_block_selected)
            block_item.signals.double_clicked.connect(self._on_block_double_clicked)
            block_item.signals.port_clicked.connect(self._on_port_clicked)
            
            # Add item to scene FIRST (Qt requirement)
            self.addItem(block_item)
            self.block_items[block.id] = block_item
            
            # Restore position from ui_state
            position_result = self.facade.get_ui_state("block_position", block.id)
            
            # Temporarily disconnect position_changed to prevent save during initial positioning
            block_item.signals.position_changed.disconnect(self._on_block_position_changed)
            
            has_position = False
            if position_result.success and position_result.data:
                pos = position_result.data
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    try:
                        x = float(pos["x"])
                        y = float(pos["y"])
                        block_item.setPos(x, y)
                        has_position = True
                    except (ValueError, TypeError, KeyError):
                        pass
            
            if not has_position:
                # Find a non-overlapping position
                new_pos = self._find_non_overlapping_position(exclude_block_id=block.id)
                block_item.setPos(new_pos)
                self._pending_positions[block.id] = new_pos
                self._save_block_position(block.id)
            
            # Reconnect position_changed
            block_item.signals.position_changed.connect(self._on_block_position_changed)
            
            # Verify block is in scene
            if block_item.scene() != self:
                Log.error(f"Block '{block.name}' was not properly added to scene!")
        
        except Exception as e:
            Log.error(f"Error adding block item for {block.name if hasattr(block, 'name') else block.id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_all_block_states(self):
        """Update data state for all blocks in the scene"""
        for block_item in self.block_items.values():
            if hasattr(block_item, '_update_data_state'):
                block_item._update_data_state()
    
    def refresh_from_database(self):
        """
        Refresh block positions from database after undo/redo.
        
        This is a lightweight refresh that only updates positions,
        without recreating all block/connection items.
        """
        for block_id, block_item in self.block_items.items():
            result = self.facade.get_ui_state("block_position", block_id)
            if result.success and result.data:
                pos = result.data
                new_x = pos.get("x", 0)
                new_y = pos.get("y", 0)
                
                # Update position without triggering save
                try:
                    block_item.signals.position_changed.disconnect(self._on_block_position_changed)
                except TypeError:
                    pass
                
                block_item.setPos(new_x, new_y)
                
                # Update connections
                for conn in block_item.connections:
                    conn.update_position()
                
                try:
                    block_item.signals.position_changed.connect(self._on_block_position_changed)
                except TypeError:
                    pass
        
        # Force visual refresh
        self.update()
    
    def _blocks_have_positions(self) -> bool:
        """
        Check if blocks have saved positions (Phase B: using ui_state).
        
        Returns True if ANY block has a saved position (indicates user has
        positioned blocks before, so we should respect those positions).
        """
        # Check if any blocks have positions in ui_state
        for block_id in self.block_items.keys():
            result = self.facade.get_ui_state("block_position", block_id)
            if result.success and result.data:
                pos = result.data
                # Verify position data is valid (has x and y)
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    return True
        return False
    
    def _get_blocks_without_positions(self) -> List[str]:
        """
        Get list of block IDs that don't have saved positions.
        
        Returns:
            List of block IDs that need auto-layout
        """
        blocks_without_positions = []
        
        for block_id in self.block_items.keys():
            result = self.facade.get_ui_state("block_position", block_id)
            if not result.success or not result.data:
                blocks_without_positions.append(block_id)
            else:
                pos = result.data
                # Verify position data is valid
                if not isinstance(pos, dict) or 'x' not in pos or 'y' not in pos:
                    blocks_without_positions.append(block_id)
        
        return blocks_without_positions
    
    def _find_non_overlapping_position(self, exclude_block_id: str = None) -> QPointF:
        """
        Find a position for a new block that doesn't overlap existing blocks.
        
        Algorithm:
        1. If no existing blocks, place at origin
        2. Otherwise, try placing to the right of the rightmost block
        3. If row is too wide (>5 blocks), start a new row below
        
        Args:
            exclude_block_id: Block ID to exclude from overlap check (the new block)
            
        Returns:
            QPointF with non-overlapping position
        """
        from ui.qt_gui.design_system import Sizes
        
        BLOCK_WIDTH = Sizes.BLOCK_WIDTH
        BLOCK_HEIGHT = Sizes.BLOCK_HEIGHT
        X_SPACING = 50  # Horizontal gap between blocks
        Y_SPACING = 40  # Vertical gap between rows
        MAX_ROW_WIDTH = 1500  # Max width before wrapping to new row
        
        # Get all existing block positions (excluding the new block if specified)
        existing_positions = []
        for block_id, block_item in self.block_items.items():
            if block_id != exclude_block_id:
                pos = block_item.pos()
                existing_positions.append({
                    'x': pos.x(),
                    'y': pos.y(),
                    'right': pos.x() + BLOCK_WIDTH,
                    'bottom': pos.y() + BLOCK_HEIGHT
                })
        
        # If no existing blocks, place at origin
        if not existing_positions:
            return QPointF(0, 0)
        
        # Find the bounding box of all existing blocks
        min_x = min(p['x'] for p in existing_positions)
        max_x = max(p['right'] for p in existing_positions)
        min_y = min(p['y'] for p in existing_positions)
        max_y = max(p['bottom'] for p in existing_positions)
        
        # Try placing to the right of the rightmost block
        new_x = max_x + X_SPACING
        
        # Find the y position of blocks near the right edge (same row)
        rightmost_blocks = [p for p in existing_positions if p['right'] >= max_x - BLOCK_WIDTH]
        if rightmost_blocks:
            # Align with the topmost block in the rightmost column
            new_y = min(p['y'] for p in rightmost_blocks)
        else:
            new_y = min_y
        
        # If new position would make row too wide, start a new row
        if new_x + BLOCK_WIDTH > MAX_ROW_WIDTH:
            new_x = min_x  # Start at left edge
            new_y = max_y + Y_SPACING  # Below all existing blocks
        
        # Verify position doesn't overlap (shouldn't happen but safety check)
        def overlaps(x, y):
            for p in existing_positions:
                if (x < p['right'] and x + BLOCK_WIDTH > p['x'] and
                    y < p['bottom'] and y + BLOCK_HEIGHT > p['y']):
                    return True
            return False
        
        # If somehow overlapping, find a clear spot
        attempts = 0
        while overlaps(new_x, new_y) and attempts < 20:
            new_x += BLOCK_WIDTH + X_SPACING
            if new_x + BLOCK_WIDTH > MAX_ROW_WIDTH:
                new_x = min_x
                new_y += BLOCK_HEIGHT + Y_SPACING
            attempts += 1
        
        return QPointF(new_x, new_y)
    
    def _add_block_item(self, block_summary):
        """
        Add a block item to the scene (DEPRECATED - use _add_block_item_from_entity instead).
        
        This method is kept for backwards compatibility but should not be used during refresh().
        """
        try:
            # Get full block entity (list_blocks returns BlockSummary, we need full Block)
            # Skip port check during scene refresh to avoid blocking database writes
            full_block_result = self.facade.describe_block(block_summary.id, skip_port_check=True)
            if not full_block_result.success or not full_block_result.data:
                Log.warning(f"Could not get full block for {block_summary.id}: {full_block_result.message if hasattr(full_block_result, 'message') else 'Unknown error'}")
                return
            
            block = full_block_result.data
            
            try:
                block_item = BlockItem(block, self.facade, undo_stack=self.undo_stack)
            except Exception as e:
                Log.error(f"Failed to create BlockItem for {block.name} ({block.id}): {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Connect signals (we'll disconnect position_changed temporarily during positioning)
            block_item.signals.position_changed.connect(self._on_block_position_changed)
            block_item.signals.selected.connect(self._on_block_selected)
            block_item.signals.double_clicked.connect(self._on_block_double_clicked)
            block_item.signals.port_clicked.connect(self._on_port_clicked)
            
            # Add item to scene FIRST (Qt requirement for proper positioning)
            self.addItem(block_item)
            self.block_items[block.id] = block_item
            
            # Restore position from ui_state (Phase B: database-centric)
            position_result = self.facade.get_ui_state("block_position", block.id)
            
            # Temporarily disconnect position_changed to prevent save during initial positioning
            block_item.signals.position_changed.disconnect(self._on_block_position_changed)
            
            # Check if we have valid position data
            has_position = False
            
            # Get position from ui_state
            if position_result.success and position_result.data:
                pos = position_result.data
                Log.debug(f"Position from ui_state for '{block.name}': {pos}")
                
                # Apply position if valid
                if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                    try:
                        x = float(pos["x"])
                        y = float(pos["y"])
                        
                        # Set position AFTER item is in scene (critical for Qt)
                        block_item.setPos(x, y)
                        # Force scene update to ensure position is applied
                        self.update()
                        has_position = True
                        
                        # Verify position was set correctly
                        actual_pos = block_item.pos()
                        Log.debug(f"Restored position for '{block.name}': Source=({x}, {y}), Actual=({actual_pos.x()}, {actual_pos.y()})")
                        
                        if abs(actual_pos.x() - x) > 0.1 or abs(actual_pos.y() - y) > 0.1:
                            Log.warning(f"Position mismatch for '{block.name}': expected ({x}, {y}), got ({actual_pos.x()}, {actual_pos.y()}), retrying...")
                            # Try setting again with explicit scene coordinates
                            block_item.setPos(QPointF(x, y))
                            self.update()
                            actual_pos = block_item.pos()
                            Log.debug(f"After retry: ({actual_pos.x()}, {actual_pos.y()})")
                    except (ValueError, TypeError, KeyError) as e:
                        Log.warning(f"Invalid position data for '{block.name}': {e}, data: {pos}")
            else:
                Log.debug(f"   No position data found for '{block.name}' in ui_state")
            
            if not has_position:
                # Find a non-overlapping position for this block
                new_pos = self._find_non_overlapping_position(exclude_block_id=block.id)
                block_item.setPos(new_pos)
                Log.info(f"Placed new block '{block.name}' at non-overlapping position: ({new_pos.x()}, {new_pos.y()})")
                
                # Save this position immediately so it persists
                self._pending_positions[block.id] = new_pos
                self._save_block_position(block.id)
            
            # Reconnect position_changed AFTER initial positioning
            block_item.signals.position_changed.connect(self._on_block_position_changed)
            
            # Update data state after block is fully added to scene
            # This ensures the LED shows correct state on load
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, block_item._update_data_state)
            
            # Verify block is actually in scene
            if block_item.scene() != self:
                Log.error(f"Block '{block.name}' was not properly added to scene!")
            else:
                Log.debug(f"Block '{block.name}' successfully added to scene at {block_item.pos()}")
        
        except Exception as e:
            Log.error(f"Error adding block item for {block_summary.name if hasattr(block_summary, 'name') else block_summary.id}: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_connection_item(self, connection):
        """Add a connection item to the scene"""
        # Find source and target blocks
        src_block_item = self.block_items.get(connection.source_block_id)
        tgt_block_item = self.block_items.get(connection.target_block_id)
        
        if not src_block_item or not tgt_block_item:
            # Block items not ready yet - queue for retry
            # This fixes the delay issue where ShowManager-Editor connections appear late
            if connection not in self._pending_connections:
                self._pending_connections.append(connection)
                Log.debug(f"Connection {connection.id} queued for retry (blocks not ready): "
                         f"source={connection.source_block_id}, target={connection.target_block_id}")
            else:
                Log.warning(f"Cannot draw connection {connection.id}: block not found (already queued)")
            return
        
        # ConnectionSummary has source_output_name and target_input_name, not source_port/target_port
        try:
            conn_item = ConnectionItem(
                src_block_item,
                connection.source_output_name,
                tgt_block_item,
                connection.target_input_name,
                connection
            )
            
            self.addItem(conn_item)
            self.connection_items[connection.id] = conn_item
            
            # Register with block items
            src_block_item.add_connection(conn_item)
            tgt_block_item.add_connection(conn_item)
            
            # Remove from pending if it was there
            if connection in self._pending_connections:
                self._pending_connections.remove(connection)
                Log.debug(f"Successfully rendered connection {connection.id} on retry")
        except Exception as e:
            Log.error(f"Error creating connection item for {connection.id}: {e}")
            # Queue for retry if not already queued
            if connection not in self._pending_connections:
                self._pending_connections.append(connection)
    
    def _retry_pending_connections(self):
        """Retry connections that failed to render initially"""
        if not self._pending_connections:
            return
        
        retried = []
        still_pending = []
        
        for conn in self._pending_connections:
            # Try again
            src_block_item = self.block_items.get(conn.source_block_id)
            tgt_block_item = self.block_items.get(conn.target_block_id)
            
            if src_block_item and tgt_block_item:
                # Blocks are now ready - try to add connection
                try:
                    conn_item = ConnectionItem(
                        src_block_item,
                        conn.source_output_name,
                        tgt_block_item,
                        conn.target_input_name,
                        conn
                    )
                    
                    self.addItem(conn_item)
                    self.connection_items[conn.id] = conn_item
                    
                    src_block_item.add_connection(conn_item)
                    tgt_block_item.add_connection(conn_item)
                    
                    retried.append(conn)
                    Log.info(f"Successfully rendered connection {conn.id} on retry")
                except Exception as e:
                    Log.warning(f"Error retrying connection {conn.id}: {e}")
                    still_pending.append(conn)
            else:
                still_pending.append(conn)
        
        # Update pending list
        self._pending_connections = still_pending
        
        if retried:
            Log.info(f"Retried {len(retried)} connections successfully")
            self.update()  # Force scene update
        
        # If there are still pending connections, retry once more after a short delay
        if still_pending:
            QTimer.singleShot(100, self._retry_pending_connections)
    
    def _on_block_position_changed(self, block_id: str, pos: QPointF):
        """Handle block position change - save to ui_state with debouncing (Phase B: database-centric)"""
        # Store pending position
        self._pending_positions[block_id] = pos
        
        # Cancel existing timer if any
        if block_id in self._position_save_timers:
            self._position_save_timers[block_id].stop()
            self._position_save_timers[block_id].deleteLater()
        
        # Create new timer for debounced save (500ms delay)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._save_block_position(block_id))
        timer.start(500)  # Wait 500ms before saving
        
        self._position_save_timers[block_id] = timer
    
    def _save_block_position(self, block_id: str):
        """Actually save the block position to database (called after debounce)"""
        # Get pending position
        if block_id not in self._pending_positions:
            return
        
        pos = self._pending_positions[block_id]
        
        # Get block name for better debugging
        block_item = self.block_items.get(block_id)
        block_name = block_item.block.name if block_item else "Unknown"
        
        Log.debug(f"Saving position for block '{block_name}' ({block_id}): ({pos.x()}, {pos.y()})")
        
        # Save to ui_state table (not block metadata)
        # Include block name for easier debugging and database inspection
        position_data = {
            "x": pos.x(),
            "y": pos.y(),
            "block_name": block_name  # Added for debugging
        }
        
        result = self.facade.set_ui_state("block_position", block_id, position_data)
        if not result.success:
            Log.warning(f"Failed to save block position for '{block_name}': {result.message}")
        else:
            Log.debug(f"Successfully saved position for block '{block_name}' ({block_id})")
        
        # Clean up
        del self._pending_positions[block_id]
        if block_id in self._position_save_timers:
            del self._position_save_timers[block_id]
    
    def _on_block_selected(self, block_id: str):
        """Handle block selection"""
        # Emit to any listeners (e.g., properties panel)
        self.block_selected.emit(block_id)
    
    def _on_block_double_clicked(self, block_id: str):
        """Handle block double-click (open panel)"""
        Log.debug(f"Block double-clicked: {block_id}")
        self.block_double_clicked.emit(block_id)
    
    def _on_selection_changed(self):
        """Handle selection change"""
        selected_items = self.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            if isinstance(item, BlockItem):
                self.block_selected.emit(item.block.id)
    
    def get_selected_block_id(self) -> Optional[str]:
        """Get currently selected block ID"""
        selected_items = self.selectedItems()
        
        if selected_items:
            item = selected_items[0]
            if isinstance(item, BlockItem):
                return item.block.id
        
        return None
    
    def selected_blocks(self) -> List[str]:
        """
        Get list of selected block IDs (for session state).
        
        Returns:
            List of block IDs
        """
        block_ids = []
        for item in self.selectedItems():
            if isinstance(item, BlockItem):
                block_ids.append(item.block.id)
        return block_ids
    
    def select_block(self, block_id: str):
        """
        Select a block by ID (for session restoration).
        
        Args:
            block_id: Block ID to select
        """
        # Clear current selection
        self.clearSelection()
        
        # Find and select the block
        if block_id in self.block_items:
            block_item = self.block_items[block_id]
            block_item.setSelected(True)
            Log.debug(f"Selected block: {block_id}")
    
    def select_all_blocks(self):
        """Select all blocks in the scene."""
        for block_item in self.block_items.values():
            block_item.setSelected(True)
        Log.debug(f"Selected all blocks ({len(self.block_items)})")
    
    def flush_pending_position_saves(self):
        """
        Force-save all pending block positions immediately.
        
        Should be called when:
        - Window is closing
        - Project is being saved
        - Before project load/switch
        """
        if not self._pending_positions:
            return
        
        Log.info(f"Flushing {len(self._pending_positions)} pending position saves")
        
        # Stop all timers
        for timer in self._position_save_timers.values():
            timer.stop()
            timer.deleteLater()
        self._position_save_timers.clear()
        
        # Save all pending positions immediately
        for block_id in list(self._pending_positions.keys()):
            self._save_block_position(block_id)
    
    def delete_selected_items(self):
        """Delete currently selected items (blocks/connections)"""
        selected = self.selectedItems()
        if not selected:
            return
        
        blocks_to_delete = []
        connections_to_delete = []
        
        for item in selected:
            if isinstance(item, BlockItem):
                blocks_to_delete.append(item.block)
            elif isinstance(item, ConnectionItem):
                if item.connection_data and hasattr(item.connection_data, 'id'):
                    connections_to_delete.append(item.connection_data.id)
        
        # Delete connections first (they depend on blocks)
        if connections_to_delete:
            from src.application.commands import DeleteConnectionCommand
            
            if len(connections_to_delete) > 1:
                self.facade.command_bus.begin_macro(f"Delete {len(connections_to_delete)} Connections")
            
            for conn_id in connections_to_delete:
                cmd = DeleteConnectionCommand(self.facade, conn_id)
                self.facade.command_bus.execute(cmd)
                Log.info(f"Deleted connection: {conn_id}")
            
            if len(connections_to_delete) > 1:
                self.facade.command_bus.end_macro()
        
        # Delete blocks
        if blocks_to_delete:
            if len(blocks_to_delete) > 1:
                self.facade.command_bus.begin_macro(f"Delete {len(blocks_to_delete)} Blocks")
            
            for block in blocks_to_delete:
                cmd = DeleteBlockCommand(self.facade, block.id)
                self.facade.command_bus.execute(cmd)
                Log.info(f"Deleted block: {block.name}")
            
            if len(blocks_to_delete) > 1:
                self.facade.command_bus.end_macro()
        
        # Refresh scene
        self.refresh()
    
    def auto_layout(self, layout_type: str = "grid", blocks_to_layout: List[str] = None):
        """
        Auto-arrange blocks with specified layout algorithm.
        
        Args:
            layout_type: Layout algorithm to use (grid, horizontal, vertical, etc.)
            blocks_to_layout: Optional list of block IDs to layout. If None, layouts all blocks.
                             If provided, only layouts blocks in this list (preserves others).
        """
        if not self.block_items:
            return
        
        # If no specific blocks provided, layout all blocks (backward compatibility)
        if blocks_to_layout is None:
            blocks_to_layout = list(self.block_items.keys())
        
        if not blocks_to_layout:
            Log.info("No blocks to auto-layout")
            return
        
        Log.info(f"Auto-layouting {len(blocks_to_layout)} blocks: {layout_type}")
        
        if layout_type == "grid":
            self._layout_grid(blocks_to_layout)
        elif layout_type == "horizontal":
            self._layout_horizontal(blocks_to_layout)
        elif layout_type == "vertical":
            self._layout_vertical(blocks_to_layout)
        elif layout_type == "circular":
            self._layout_circular(blocks_to_layout)
        elif layout_type == "hierarchical":
            self._layout_hierarchical(blocks_to_layout)
        else:
            self._layout_grid(blocks_to_layout)
    
    def _layout_grid(self, blocks_to_layout: List[str]):
        """Grid layout - blocks in rows and columns"""
        x_spacing = 250
        y_spacing = 150
        columns = 3
        
        # Temporarily disconnect position_changed to prevent saves during layout
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.disconnect(self._on_block_position_changed)
        
        for i, block_id in enumerate(blocks_to_layout):
            if block_id not in self.block_items:
                continue
            block_item = self.block_items[block_id]
            col = i % columns
            row = i // columns
            x = col * x_spacing
            y = row * y_spacing
            block_item.setPos(x, y)
        
        # Reconnect and save positions
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.connect(self._on_block_position_changed)
                # Save position
                pos = block_item.pos()
                self._on_block_position_changed(block_id, pos)
    
    def _layout_horizontal(self, blocks_to_layout: List[str]):
        """Horizontal flow - blocks in a single row"""
        x_spacing = 250
        x = 0
        
        # Temporarily disconnect position_changed
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                self.block_items[block_id].signals.position_changed.disconnect(self._on_block_position_changed)
        
        for block_id in blocks_to_layout:
            if block_id not in self.block_items:
                continue
            block_item = self.block_items[block_id]
            block_item.setPos(x, 0)
            x += x_spacing
        
        # Reconnect and save
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.connect(self._on_block_position_changed)
                pos = block_item.pos()
                self._on_block_position_changed(block_id, pos)
    
    def _layout_vertical(self, blocks_to_layout: List[str]):
        """Vertical flow - blocks in a single column"""
        y_spacing = 150
        y = 0
        
        # Temporarily disconnect position_changed
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                self.block_items[block_id].signals.position_changed.disconnect(self._on_block_position_changed)
        
        for block_id in blocks_to_layout:
            if block_id not in self.block_items:
                continue
            block_item = self.block_items[block_id]
            block_item.setPos(0, y)
            y += y_spacing
        
        # Reconnect and save
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.connect(self._on_block_position_changed)
                pos = block_item.pos()
                self._on_block_position_changed(block_id, pos)
    
    def _layout_circular(self, blocks_to_layout: List[str]):
        """Circular layout - blocks arranged in a circle"""
        import math
        
        count = len(blocks_to_layout)
        if count == 0:
            return
        
        radius = max(200, count * 50)  # Adjust radius based on count
        center_x = 0
        center_y = 0
        
        # Temporarily disconnect position_changed
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                self.block_items[block_id].signals.position_changed.disconnect(self._on_block_position_changed)
        
        for i, block_id in enumerate(blocks_to_layout):
            if block_id not in self.block_items:
                continue
            block_item = self.block_items[block_id]
            angle = (2 * math.pi * i) / count
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            block_item.setPos(x, y)
        
        # Reconnect and save
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.connect(self._on_block_position_changed)
                pos = block_item.pos()
                self._on_block_position_changed(block_id, pos)
    
    def _layout_hierarchical(self, blocks_to_layout: List[str]):
        """Hierarchical layout - try to organize by connections (simple version)"""
        # Build a simple dependency graph (only for blocks to layout)
        block_levels = {}
        blocks_to_layout_set = set(blocks_to_layout)
        
        # Find blocks with no inputs (sources) - only from blocks_to_layout
        for block_id in blocks_to_layout:
            if block_id not in self.block_items:
                continue
            block_item = self.block_items[block_id]
            has_input_connections = any(
                conn.connection_data.target_block_id == block_id 
                and conn.connection_data.source_block_id in blocks_to_layout_set
                for conn in self.connection_items.values()
            )
            
            if not has_input_connections:
                block_levels[block_id] = 0
        
        # Assign levels based on connections (only within blocks_to_layout)
        max_iterations = 10
        for _ in range(max_iterations):
            changed = False
            for conn_id, conn in self.connection_items.items():
                src_id = conn.connection_data.source_block_id
                tgt_id = conn.connection_data.target_block_id
                
                # Only consider connections between blocks we're laying out
                if src_id not in blocks_to_layout_set or tgt_id not in blocks_to_layout_set:
                    continue
                
                src_level = block_levels.get(src_id, -1)
                tgt_level = block_levels.get(tgt_id, -1)
                
                if src_level >= 0:
                    new_level = src_level + 1
                    if tgt_level < 0 or tgt_level < new_level:
                        block_levels[tgt_id] = new_level
                        changed = True
            
            if not changed:
                break
        
        # Assign default level to unconnected blocks (only from blocks_to_layout)
        for block_id in blocks_to_layout:
            if block_id not in block_levels:
                block_levels[block_id] = 0
        
        # Position blocks by level
        level_blocks = {}
        for block_id, level in block_levels.items():
            if level not in level_blocks:
                level_blocks[level] = []
            level_blocks[level].append(block_id)
        
        x_spacing = 300
        y_spacing = 150
        
        # Temporarily disconnect position_changed
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                self.block_items[block_id].signals.position_changed.disconnect(self._on_block_position_changed)
        
        for level, block_ids in sorted(level_blocks.items()):
            for i, block_id in enumerate(block_ids):
                x = level * x_spacing
                y = i * y_spacing - (len(block_ids) - 1) * y_spacing / 2  # Center vertically
                block_item = self.block_items[block_id]
                block_item.setPos(x, y)
        
        # Reconnect and save
        for block_id in blocks_to_layout:
            if block_id in self.block_items:
                block_item = self.block_items[block_id]
                block_item.signals.position_changed.connect(self._on_block_position_changed)
                pos = block_item.pos()
                self._on_block_position_changed(block_id, pos)

    # ==================== Drag-to-Connect ====================
    
    def _on_port_clicked(self, block_id: str, port_name: str, is_output: bool, is_bidirectional: bool, scene_pos: QPointF):
        """Handle port click to start connection drag"""
        Log.info(f"Port clicked: {block_id}.{port_name} (output={is_output}, bidirectional={is_bidirectional})")
        
        self._drag_source_block_id = block_id
        self._drag_source_port = port_name
        self._drag_is_output = is_output
        self._drag_is_bidirectional = is_bidirectional
        self._drag_start_pos = scene_pos
        
        # Create temporary connection line
        self._drag_connection = QGraphicsPathItem()
        self._drag_connection.setPen(QPen(Colors.ACCENT_YELLOW, 2, Qt.PenStyle.DashLine))
        self._drag_connection.setZValue(100)  # Above everything
        self.addItem(self._drag_connection)
        
        # Update line to current mouse position
        self._update_drag_connection(scene_pos)
    
    def _update_drag_connection(self, end_pos: QPointF):
        """Update the temporary connection line during drag"""
        if not self._drag_connection or not self._drag_start_pos:
            return
        
        # Check if item still exists and is in scene (may have been deleted by scene refresh)
        try:
            if self._drag_connection.scene() is None:
                # Item was removed from scene, cancel the drag
                self._cancel_connection_drag()
                return
        except RuntimeError:
            # Item has been deleted, cancel the drag
            self._cancel_connection_drag()
            return
        
        start_pos = self._drag_start_pos
        
        # Create bezier curve like real connections
        path = QPainterPath(start_pos)
        
        ctrl_offset = abs(end_pos.x() - start_pos.x()) * 0.5
        if ctrl_offset < 50:
            ctrl_offset = 50
        
        if self._drag_is_output:
            # Dragging from output (right side) - curve goes right
            ctrl1 = QPointF(start_pos.x() + ctrl_offset, start_pos.y())
            ctrl2 = QPointF(end_pos.x() - ctrl_offset, end_pos.y())
        else:
            # Dragging from input (left side) - curve goes left
            ctrl1 = QPointF(start_pos.x() - ctrl_offset, start_pos.y())
            ctrl2 = QPointF(end_pos.x() + ctrl_offset, end_pos.y())
        
        path.cubicTo(ctrl1, ctrl2, end_pos)
        try:
            self._drag_connection.setPath(path)
        except RuntimeError:
            # Item was deleted during path creation, cancel the drag
            self._cancel_connection_drag()
    
    def _cancel_connection_drag(self):
        """Cancel the current connection drag"""
        if self._drag_connection:
            try:
                # Check if item is still in scene before removing
                if self._drag_connection.scene() is not None:
                    self.removeItem(self._drag_connection)
            except RuntimeError:
                # Item has already been deleted (e.g., scene was refreshed)
                pass
            self._drag_connection = None
        
        self._drag_source_block_id = None
        self._drag_source_port = None
        self._drag_is_output = False
        self._drag_is_bidirectional = False
        self._drag_start_pos = None
    
    def _complete_connection_drag(self, target_block_id: str, target_port: str, target_is_output: bool, target_is_bidirectional: bool):
        """Complete the connection drag by creating the connection"""
        if not self._drag_source_block_id or not self._drag_source_port:
            return
        
        # Reject mixed bidirectional/regular connections (defensive check)
        if self._drag_is_bidirectional != target_is_bidirectional:
            Log.warning("Cannot connect bidirectional port to regular input/output port")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "Cannot Connect", "Cannot connect bidirectional ports to regular input/output ports")
            self._cancel_connection_drag()
            return
        
        # Handle bidirectional connections differently
        if self._drag_is_bidirectional and target_is_bidirectional:
            # Both ports are bidirectional - use block order (first clicked = source)
            source_block_id = self._drag_source_block_id
            source_port = self._drag_source_port
            dest_block_id = target_block_id
            dest_port = target_port
        elif self._drag_is_output:
            # Dragging from output to input (regular connection)
            source_block_id = self._drag_source_block_id
            source_port = self._drag_source_port
            dest_block_id = target_block_id
            dest_port = target_port
        else:
            # Dragging from input to output (regular connection)
            source_block_id = target_block_id
            source_port = target_port
            dest_block_id = self._drag_source_block_id
            dest_port = self._drag_source_port
        
        # Check compatibility
        is_compatible, reason = self.connection_helper.check_compatibility(
            source_block_id, source_port,
            dest_block_id, dest_port
        )
        
        if not is_compatible:
            Log.warning(f"Cannot connect: {reason}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "Cannot Connect", reason)
            self._cancel_connection_drag()
            return
        
        # Remove drag visual BEFORE creating connection (scene will refresh after connection)
        self._cancel_connection_drag()
        
        # Create connection (this will trigger scene refresh via event bus)
        result = self.connection_helper.create_connection(
            source_block_id, source_port,
            dest_block_id, dest_port
        )
        
        if result.success:
            Log.info(f"Connection created via drag")
            # Scene will refresh via event bus
        else:
            Log.error(f"Failed to create connection: {result.message}")
    
    def _find_port_at_scene_pos(self, scene_pos: QPointF) -> Optional[Tuple[str, str, bool, bool]]:
        """
        Find if there's a port at the given scene position.
        
        Returns:
            (block_id, port_name, is_output, is_bidirectional) or None
        """
        for block_id, block_item in self.block_items.items():
            # Convert scene pos to item local coords
            local_pos = block_item.mapFromScene(scene_pos)
            port_info = block_item.port_at_position(local_pos)
            
            if port_info:
                port_name, is_output, is_bidirectional = port_info
                return (block_id, port_name, is_output, is_bidirectional)
        
        return None
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for connection dragging"""
        if self._drag_connection:
            # Verify item still exists before updating
            try:
                if self._drag_connection.scene() is None:
                    # Item was removed, cancel drag
                    self._cancel_connection_drag()
                    super().mouseMoveEvent(event)
                    return
            except RuntimeError:
                # Item has been deleted, cancel drag
                self._cancel_connection_drag()
                super().mouseMoveEvent(event)
                return
            
            self._update_drag_connection(event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release for connection completion"""
        if self._drag_connection and event.button() == Qt.MouseButton.LeftButton:
            # Check if we're over a valid port
            port_info = self._find_port_at_scene_pos(event.scenePos())
            
            if port_info:
                target_block_id, target_port, target_is_output, target_is_bidirectional = port_info
                
                # Can't connect to same block
                if target_block_id == self._drag_source_block_id:
                    Log.debug("Cannot connect block to itself")
                    self._cancel_connection_drag()
                # Bidirectional ports can ONLY connect to bidirectional ports
                elif self._drag_is_bidirectional and not target_is_bidirectional:
                    Log.debug("Cannot connect bidirectional port to regular port")
                    self._cancel_connection_drag()
                elif not self._drag_is_bidirectional and target_is_bidirectional:
                    Log.debug("Cannot connect regular port to bidirectional port")
                    self._cancel_connection_drag()
                # Bidirectional ports can connect to bidirectional ports
                elif self._drag_is_bidirectional and target_is_bidirectional:
                    # Both are bidirectional - allow connection
                    self._complete_connection_drag(target_block_id, target_port, target_is_output, target_is_bidirectional)
                # Can't connect output to output or input to input (unless both are bidirectional)
                elif target_is_output == self._drag_is_output:
                    Log.debug("Cannot connect same port types")
                    self._cancel_connection_drag()
                else:
                    # Regular input/output connection
                    self._complete_connection_drag(target_block_id, target_port, target_is_output, target_is_bidirectional)
            else:
                self._cancel_connection_drag()
            
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle escape to cancel connection drag"""
        if event.key() == Qt.Key.Key_Escape and self._drag_connection:
            self._cancel_connection_drag()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    # ==================== Connection Dialog ====================
    
    def _on_connection_dialog_requested(self, block_id: str):
        """Open connection dialog for a block"""
        from ui.qt_gui.connection.connection_dialog import ConnectionDialog
        
        # Show dialog with this block as source
        if ConnectionDialog.show_dialog(self.facade, block_id, parent=None):
            # Connection created - scene will refresh via event bus
            Log.info("Connection created via dialog")

