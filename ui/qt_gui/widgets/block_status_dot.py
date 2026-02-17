"""
Block Status Dot Widget

Lightweight reusable status indicator widget that automatically updates
when block status changes. Can be placed anywhere in the UI.

Uses BlockStatusService as the single source of truth for all blocks.
Subscribes to StatusChanged events for immediate updates.
"""
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from src.application.events.events import BlockChanged, StatusChanged
from ui.qt_gui.design_system import Colors
from src.utils.message import Log


class BlockStatusDot(QWidget):
    """
    Lightweight status dot widget that auto-updates from centralized status source.
    
    Automatically subscribes to StatusChanged events for immediate updates.
    Uses BlockStatusService as the single source of truth for all blocks.
    Includes periodic refresh as fallback during migration.
    
    Usage:
        status_dot = BlockStatusDot(block_id, facade, parent=header)
        layout.addWidget(status_dot)
    """
    
    def __init__(self, block_id: str, facade, parent=None):
        """
        Initialize block status dot widget.
        
        Args:
            block_id: Block identifier
            facade: ApplicationFacade instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade
        self._current_status = None
        
        self._setup_ui()
        self._subscribe_to_events()
        self._update_status()
    
    def _setup_ui(self):
        """Setup the widget UI"""
        # Status indicator (dot only)
        self.status_dot = QLabel("●")
        status_font = QFont()
        status_font.setPointSize(10)
        self.status_dot.setFont(status_font)
        self.status_dot.setStyleSheet(f"color: {Colors.ACCENT_BLUE.name()};")
        self.status_dot.setToolTip("Ready")
        
        # Simple layout - just the dot
        from PyQt6.QtWidgets import QHBoxLayout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.status_dot)
    
    def _subscribe_to_events(self):
        """Subscribe to status change events"""
        if not self.facade or not hasattr(self.facade, 'event_bus') or not self.facade.event_bus:
            return
        
        # Primary: Subscribe to StatusChanged events for immediate updates
        self.facade.event_bus.subscribe("StatusChanged", self._on_status_changed)
        # Fallback: Also subscribe to BlockChanged during migration
        self.facade.event_bus.subscribe("BlockChanged", self._on_block_changed)
        self.facade.event_bus.subscribe("BlockUpdated", self._on_block_updated)
    
    def _on_status_changed(self, event: StatusChanged):
        """Handle StatusChanged event - immediate update from single source of truth"""
        if not event.data:
            Log.debug(f"BlockStatusDot: StatusChanged event has no data for block {self.block_id}")
            return
        
        # Check if this event is for our block
        event_block_id = event.data.get('block_id')
        if event_block_id != self.block_id:
            # Not for this block, ignore
            return
        
        Log.debug(f"BlockStatusDot: Received StatusChanged event for block {self.block_id}")
        
        # Status object is in event data
        status_dict = event.data.get('status')
        if status_dict:
            # Reconstruct status from dict
            from src.features.blocks.domain import BlockStatus, BlockStatusLevel
            level_dict = status_dict.get('level', {})
            level = BlockStatusLevel(
                priority=level_dict.get('priority', 0),
                name=level_dict.get('name', 'ready'),
                display_name=level_dict.get('display_name', 'Ready'),
                color=level_dict.get('color', Colors.STATUS_SUCCESS.name()),
                conditions=[]
            )
            status = BlockStatus(
                level=level,
                message=status_dict.get('message')
            )
            Log.debug(f"BlockStatusDot: Applying status {status.display_name} (color: {status.color}) to block {self.block_id}")
            self._apply_status(status)
        else:
            # Fallback: recalculate status
            Log.debug(f"BlockStatusDot: StatusChanged event missing status dict, recalculating for block {self.block_id}")
            self._update_status()
    
    def _on_block_changed(self, event: BlockChanged):
        """Handle block changed event - fallback during migration"""
        if not event.data:
            return
        
        # Check if this event is for our block
        event_block_id = event.data.get('block_id') or event.data.get('id')
        if event_block_id == self.block_id:
            # Small delay to ensure data is persisted, then recalculate
            QTimer.singleShot(100, self._update_status)
    
    def _on_block_updated(self, event):
        """Handle block updated event - trigger status refresh"""
        if not event.data:
            return
        
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            self._update_status()
    
    def _update_status(self):
        """Update the displayed status from BlockStatusService"""
        if not self.facade or not self.block_id:
            self._set_status_default()
            return
        
        # Always use BlockStatusService (single source of truth)
        if not hasattr(self.facade, 'block_status_service') or not self.facade.block_status_service:
            self._set_status_default()
            return
        
        try:
            # Force recalculation to ensure we get the latest status
            # This bypasses cache and ensures StatusChanged events are published if status changed
            status = self.facade.block_status_service.get_block_status(self.block_id, self.facade, force_recalculate=True)
            
            if status:
                self._apply_status(status)
            else:
                self._set_status_default()
        except Exception as e:
            Log.warning(f"BlockStatusDot: Failed to update block status for {self.block_id}: {e}")
            self._set_status_default()
    
    def _apply_status(self, status):
        """Apply status to UI"""
        self._current_status = status
        
        # Update dot color
        color = status.color
        self.status_dot.setStyleSheet(f"color: {color};")
        
        # Update tooltip with diagnostic information
        tooltip = self._build_diagnostic_tooltip(status)
        self.status_dot.setToolTip(tooltip)
        
        Log.debug(f"BlockStatusDot: Updated UI for block {self.block_id} - color: {color}, status: {status.display_name}")
    
    def _build_diagnostic_tooltip(self, status) -> str:
        """Build detailed diagnostic tooltip showing condition evaluation"""
        if not self.facade or not hasattr(self.facade, 'block_status_service') or not self.facade.block_status_service:
            # Fallback to simple tooltip
            tooltip = status.display_name
            if status.message:
                tooltip = f"{status.display_name}: {status.message}"
            return tooltip
        
        try:
            diagnostics = self.facade.block_status_service.get_block_status_diagnostics(
                self.block_id,
                self.facade
            )
            
            if diagnostics.get("error"):
                return f"{status.display_name}\n\nError: {diagnostics['error']}"
            
            # Build detailed tooltip
            lines = [f"Status: {status.display_name}"]
            
            if status.message:
                lines.append(f"Message: {status.message}")
            
            lines.append("")  # Blank line
            
            # Show evaluation for each level
            for level_info in diagnostics.get("levels", []):
                level = level_info["level"]
                is_active = level_info["is_active"]
                reason = level_info.get("reason", "Unknown")
                
                # Mark active level
                marker = "→ " if is_active else "  "
                lines.append(f"{marker}{level['display_name']} (Priority {level['priority']})")
                
                if level_info["condition_count"] > 0:
                    lines.append(f"    {reason}")
                else:
                    lines.append(f"    {reason}")
                
                # Show condition details if there are failures
                if not level_info["all_conditions_pass"] and level_info.get("failed_conditions"):
                    for failed in level_info["failed_conditions"][:3]:  # Limit to 3
                        lines.append(f"      ✗ {failed}")
                
                # Show actionable guidance for active level with failures
                if is_active and not level_info["all_conditions_pass"] and level_info.get("actionable_guidance"):
                    lines.append("")
                    lines.append("How to fix:")
                    for step in level_info["actionable_guidance"]:
                        lines.append(f"  {step}")
                
                lines.append("")  # Blank line between levels
            
            return "\n".join(lines)
            
        except Exception as e:
            Log.debug(f"BlockStatusDot: Error building diagnostic tooltip: {e}")
            # Fallback to simple tooltip
            tooltip = status.display_name
            if status.message:
                tooltip = f"{status.display_name}: {status.message}"
            return tooltip
    
    def _set_status_default(self):
        """Set default status display"""
        self.status_dot.setStyleSheet(f"color: {Colors.ACCENT_BLUE.name()};")
        self.status_dot.setToolTip("Ready")
    
    def closeEvent(self, event):
        """Clean up event subscriptions"""
        # Unsubscribe from events
        if self.facade and hasattr(self.facade, 'event_bus') and self.facade.event_bus:
            try:
                self.facade.event_bus.unsubscribe("StatusChanged", self._on_status_changed)
                self.facade.event_bus.unsubscribe("BlockChanged", self._on_block_changed)
                self.facade.event_bus.unsubscribe("BlockUpdated", self._on_block_updated)
            except:
                pass
        super().closeEvent(event)
