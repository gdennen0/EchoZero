"""
Node Graphics View

Custom QGraphicsView with zoom, pan, and interaction handling.
"""
from PyQt6.QtWidgets import QGraphicsView, QMenu, QInputDialog
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtProperty
from PyQt6.QtGui import QPainter, QWheelEvent

from src.utils.message import Log


class NodeGraphicsView(QGraphicsView):
    """
    Custom graphics view for node editor.
    
    Features:
    - Mouse wheel zoom
    - Middle mouse drag to pan
    - Anti-aliasing
    - Grid background
    """
    
    def __init__(self, scene):
        super().__init__(scene)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        # Enable drag mode
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        
        # Interaction settings
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Zoom settings
        self._zoom_level = 1.0
        self.min_zoom = 0.3
        self.max_zoom = 3.0
        self._zoom_base_step = 1.15
        
        # Pan mode
        self.is_panning = False
        self.pan_start_pos = QPointF()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom using native Qt view scaling."""
        zoom_modifier = (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        )
        if event.modifiers() & zoom_modifier:
            event.accept()

            # Prefer wheel angle delta; fall back to pixel delta for high-resolution devices.
            angle_y = event.angleDelta().y()
            if angle_y:
                steps = angle_y / 120.0
            else:
                pixel_y = event.pixelDelta().y()
                if pixel_y == 0:
                    return
                steps = pixel_y / 240.0

            zoom_factor = self._zoom_base_step ** steps
            current_zoom = self.transform().m11()
            target_zoom = current_zoom * zoom_factor
            clamped_zoom = max(self.min_zoom, min(target_zoom, self.max_zoom))

            if clamped_zoom == current_zoom:
                return

            self.scale(clamped_zoom / current_zoom, clamped_zoom / current_zoom)
            self._zoom_level = self.transform().m11()
        else:
            # Default scroll behavior
            super().wheelEvent(event)
    
    def mousePressEvent(self, event):
        """Handle mouse press for panning"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for panning"""
        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Delete/Backspace/Cmd+Delete - delete selected items
        is_delete = event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        is_cmd_delete = (event.key() == Qt.Key.Key_Delete and 
                        (event.modifiers() & Qt.KeyboardModifier.MetaModifier or 
                         event.modifiers() & Qt.KeyboardModifier.ControlModifier))
        
        if is_delete or is_cmd_delete:
            if self.scene():
                self.scene().delete_selected_items()
            event.accept()
        
        # F2 - rename selected block
        elif event.key() == Qt.Key.Key_F2:
            self._rename_selected_block()
            event.accept()
        
        # Enter - open panel for selected block
        elif event.key() == Qt.Key.Key_Return and not event.modifiers():
            self._open_selected_panel()
            event.accept()
        
        # Ctrl+A - select all
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.scene():
                for item in self.scene().items():
                    item.setSelected(True)
            event.accept()
        
        # Ctrl+D or Cmd+D - duplicate selected block
        elif event.key() == Qt.Key.Key_D and (event.modifiers() & Qt.KeyboardModifier.ControlModifier or 
                                               event.modifiers() & Qt.KeyboardModifier.MetaModifier):
            self._duplicate_selected_block()
            event.accept()
        
        # Escape - clear selection
        elif event.key() == Qt.Key.Key_Escape:
            if self.scene():
                self.scene().clearSelection()
            event.accept()
        
        # Home - fit all in view
        elif event.key() == Qt.Key.Key_Home:
            self._fit_blocks_in_view()
            event.accept()
        
        else:
            super().keyPressEvent(event)
    
    def _rename_selected_block(self):
        """Rename the currently selected block"""
        if not self.scene():
            return
        
        selected = self.scene().selectedItems()
        if not selected:
            return
        
        from ui.qt_gui.node_editor.block_item import BlockItem
        block_items = [item for item in selected if isinstance(item, BlockItem)]
        
        if block_items:
            block_item = block_items[0]
            block_item._rename_block()
    
    def _open_selected_panel(self):
        """Open panel for the selected block"""
        if not self.scene():
            return
        
        selected = self.scene().selectedItems()
        if not selected:
            return
        
        from ui.qt_gui.node_editor.block_item import BlockItem
        block_items = [item for item in selected if isinstance(item, BlockItem)]
        
        if block_items:
            block_item = block_items[0]
            block_item.signals.double_clicked.emit(block_item.block.id)
    
    def _duplicate_selected_block(self):
        """Duplicate the currently selected block"""
        scene = self.scene()
        if not scene or not hasattr(scene, 'facade'):
            return
        
        selected = scene.selectedItems()
        if not selected:
            return
        
        from ui.qt_gui.node_editor.block_item import BlockItem
        block_items = [item for item in selected if isinstance(item, BlockItem)]
        
        if block_items:
            block_item = block_items[0]
            from src.application.commands import DuplicateBlockCommand
            cmd = DuplicateBlockCommand(scene.facade, block_item.block.id)
            scene.facade.command_bus.execute(cmd)
            # Scene will refresh via event bus
    
    def _fit_blocks_in_view(self):
        """Fit all blocks in view"""
        if not self.scene() or not hasattr(self.scene(), 'block_items'):
            return
        
        if not self.scene().block_items:
            self.reset_view()
            return
        
        # Calculate bounding rect of all blocks
        bounding_rect = QRectF()
        
        for block_item in self.scene().block_items.values():
            item_rect = block_item.sceneBoundingRect()
            bounding_rect = bounding_rect.united(item_rect)
        
        # Add some padding
        padding = 50
        bounding_rect.adjust(-padding, -padding, padding, padding)
        
        # Fit in view
        self.fitInView(bounding_rect, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Update zoom level
        self._zoom_level = self.transform().m11()
    
    def reset_view(self):
        """Reset view to default zoom and center"""
        self.resetTransform()
        self._zoom_level = 1.0
        self.centerOn(0, 0)
    
    # ==================== Session Management (Phase C) ====================
    
    def set_zoom_level(self, zoom: float):
        """
        Set zoom level programmatically (for session restoration).
        
        Args:
            zoom: Zoom level (1.0 = 100%)
        """
        if self.min_zoom <= zoom <= self.max_zoom:
            # Reset transform and apply new zoom
            self.resetTransform()
            self.scale(zoom, zoom)
            self._zoom_level = zoom
    
    @pyqtProperty(float)
    def zoom_level(self):
        """Get current zoom level"""
        return self._zoom_level
    
    @zoom_level.setter
    def zoom_level(self, value: float):
        """Set zoom level directly."""
        if value == self._zoom_level:
            return

        clamped = max(self.min_zoom, min(value, self.max_zoom))
        self.resetTransform()
        self.scale(clamped, clamped)
        self._zoom_level = self.transform().m11()
    
    def get_viewport_center(self) -> dict:
        """
        Get the center point of the current viewport in scene coordinates.
        
        Returns:
            Dict with 'x' and 'y' keys
        """
        center_in_view = self.viewport().rect().center()
        center_in_scene = self.mapToScene(center_in_view)
        return {"x": center_in_scene.x(), "y": center_in_scene.y()}
    
    def center_on_point(self, x: float, y: float):
        """
        Center the viewport on a specific point in scene coordinates.
        
        Args:
            x: X coordinate in scene
            y: Y coordinate in scene
        """
        self.centerOn(x, y)
    
    # ==================== Context Menu ====================
    
    def contextMenuEvent(self, event):
        """
        Show context menu on right-click.
        
        When clicking on empty background, shows an "Add Block" submenu
        with all available block types. The new block is positioned at the
        scene location of the right-click.
        """
        scene = self.scene()
        if not scene or not hasattr(scene, 'facade'):
            super().contextMenuEvent(event)
            return
        
        # Check if click is on empty space (no item under cursor)
        scene_pos = self.mapToScene(event.pos())
        item_at_pos = scene.itemAt(scene_pos, self.transform())
        
        if item_at_pos is not None:
            # Clicked on an existing item - let default handling take over
            super().contextMenuEvent(event)
            return
        
        # Build context menu for empty background
        menu = QMenu(self)
        
        # "Add Block" submenu
        add_block_menu = menu.addMenu("Add Block")
        
        # Load block types from registry
        result = scene.facade.list_block_types()
        if result.success and result.data:
            block_types = sorted(result.data, key=lambda bt: bt["name"])
            
            for bt in block_types:
                action = add_block_menu.addAction(bt["name"])
                action.triggered.connect(
                    lambda checked, type_id=bt["type_id"], pos=scene_pos: 
                        self._add_block_at_position(type_id, pos)
                )
        else:
            no_types = add_block_menu.addAction("No block types available")
            no_types.setEnabled(False)
        
        menu.exec(event.globalPos())
    
    def _add_block_at_position(self, block_type: str, scene_pos: QPointF):
        """
        Create a new block and position it at the given scene coordinates.
        
        Args:
            block_type: Block type ID (e.g., "LoadAudio", "Separator")
            scene_pos: Scene coordinates where the block should be placed
        """
        scene = self.scene()
        if not scene or not hasattr(scene, 'facade'):
            return
        
        facade = scene.facade
        
        # Generate a unique block name
        default_name = self._generate_unique_block_name(facade, block_type)
        
        # Prompt user for block name
        name, ok = QInputDialog.getText(
            self,
            "Add Block",
            f"Name for {block_type} block:",
            text=default_name
        )
        
        if not ok or not name:
            return
        
        # Validate name uniqueness
        name = name.strip()
        if not name:
            return
        
        is_valid, error_msg = self._validate_block_name(facade, name)
        if not is_valid:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Block Name", error_msg)
            # Retry
            return self._add_block_at_position(block_type, scene_pos)
        
        # Create block via undoable command
        from src.application.commands import AddBlockCommand
        cmd = AddBlockCommand(facade, block_type, name)
        facade.command_bus.execute(cmd)
        
        Log.info(f"Added block '{name}' ({block_type}) at ({scene_pos.x():.0f}, {scene_pos.y():.0f})")
        
        # After scene refresh, position the new block at the right-click location
        # The block gets created and scene refreshes via event bus.
        # We save the desired position so it appears where the user clicked.
        self._position_new_block(facade, name, scene_pos)
    
    def _position_new_block(self, facade, block_name: str, scene_pos: QPointF):
        """
        Position a newly created block at the specified scene coordinates.
        
        Finds the block by name after creation and sets its position
        in the ui_state, then refreshes the scene.
        
        Args:
            facade: ApplicationFacade instance
            block_name: Name of the newly created block
            scene_pos: Desired scene coordinates
        """
        scene = self.scene()
        if not scene:
            return
        
        # Find the newly created block by name
        result = facade.list_blocks()
        if result.success and result.data:
            for block_summary in result.data:
                if block_summary.name == block_name:
                    # Save position to ui_state so it persists
                    position_data = {
                        "x": scene_pos.x(),
                        "y": scene_pos.y(),
                        "block_name": block_name
                    }
                    facade.set_ui_state("block_position", block_summary.id, position_data)
                    Log.debug(f"Set position for new block '{block_name}' at ({scene_pos.x():.0f}, {scene_pos.y():.0f})")
                    break
        
        # Refresh scene to show the new block at the correct position
        scene.refresh()
    
    def _generate_unique_block_name(self, facade, block_type: str) -> str:
        """
        Generate a unique block name for the given block type.
        
        Args:
            facade: ApplicationFacade instance
            block_type: Block type ID
            
        Returns:
            Unique block name string
        """
        result = facade.list_blocks()
        existing_names = set()
        if result.success and result.data:
            existing_names = {block.name for block in result.data}
        
        base_name = f"{block_type}1"
        if base_name not in existing_names:
            return base_name
        
        counter = 2
        while True:
            name = f"{block_type}{counter}"
            if name not in existing_names:
                return name
            counter += 1
    
    def _validate_block_name(self, facade, name: str):
        """
        Validate that a block name is unique.
        
        Args:
            facade: ApplicationFacade instance
            name: Block name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name or not name.strip():
            return False, "Block name cannot be empty"
        
        result = facade.list_blocks()
        if result.success and result.data:
            existing_names = {block.name for block in result.data}
            if name in existing_names:
                return False, f"A block named '{name}' already exists. Please choose a different name."
        
        return True, ""

