"""
Base class for all block panels.

Provides common structure: header, content area, footer.
Subclasses override create_content_widget() to provide block-specific UI.
Implements IStatefulWindow for saving/restoring internal state.
"""
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QMessageBox, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.application.api.application_facade import ApplicationFacade
from src.application.commands import (
    UpdateBlockMetadataCommand,
    ConfigureBlockCommand,
    BatchUpdateMetadataCommand,
)
from src.features.blocks.domain import Block
from src.utils.message import Log
from ui.qt_gui.design_system import Colors, Spacing, Typography, Sizes, get_stylesheet, on_theme_changed, disconnect_theme_changed
from ui.qt_gui.core.window_state_types import IStatefulWindow
from ui.qt_gui.widgets.block_status_dot import BlockStatusDot


class BlockPanelBase(QDockWidget, IStatefulWindow):
    """
    Base class for all block panels.
    
    Provides standardized structure:
    - Header: block name, type, status
    - Content: block-specific widget (override create_content_widget())
    - Footer: common actions (Save, Close)
    
    Subclasses should override:
    - create_content_widget(): Return block-specific UI widget
    - refresh(): Update UI with current block data
    """
    
    # Signals
    panel_closed = pyqtSignal(str)  # Emits block_id when panel is closed
    
    def __init__(self, block_id: str, facade: ApplicationFacade, parent=None):
        """
        Initialize block panel.
        
        Args:
            block_id: ID of the block this panel represents
            facade: Application facade for operations
            parent: Parent widget (must be MainWindow for docking to work)
        """
        # Initialize with a temporary title - will be updated after loading block data
        # The title parameter is important for Qt's dock system to work correctly
        super().__init__("Loading...", parent)
        
        self.block_id = block_id
        self.facade = facade
        self.block: Block = None
        self._is_saving = False  # Guard flag to prevent refresh during save
        
        # Configure dock widget - same settings as main docks (Node Editor, Batch Runner)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        
        # Setup UI
        self._setup_ui()
        
        # Load block data (this will update the title)
        self._load_block_data()
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Connect to dock state changes for safe reparenting
        self.topLevelChanged.connect(self._on_dock_level_changed)
        
        # Auto-refresh local styling when theme changes (Tier 3 propagation).
        # The global QApplication stylesheet covers default widget styles;
        # this hook lets subclasses re-apply variant/context-specific overrides.
        on_theme_changed(self._on_theme_changed)
    
    def _setup_ui(self):
        """Setup the panel UI structure"""
        self._separators = []  # Track separator lines for theme refresh
        
        # Main container widget - explicit background to prevent flash during refresh
        main_widget = QWidget()
        main_widget.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        self.header_widget = self._create_header()
        main_layout.addWidget(self.header_widget)
        
        # Separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
        separator1.setFixedHeight(1)
        self._separators.append(separator1)
        main_layout.addWidget(separator1)
        
        # Content area (block-specific) - wrapped in scroll area
        self._content_container = QWidget()
        self._content_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._content_container.setStyleSheet(f"background-color: {Colors.BG_DARK.name()};")
        content_layout = QVBoxLayout(self._content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        self.content_widget = self.create_content_widget()
        self.content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # Wrap content in scroll area to handle panels that are too tall
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self.content_widget)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {Colors.BG_DARK.name()};
                border: none;
            }}
        """)
        content_layout.addWidget(self._scroll_area, 1)
        
        main_layout.addWidget(self._content_container, 1)
        
        # Store main_widget for theme refresh
        self._main_widget = main_widget
        
        # Separator line
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet(f"background-color: {Colors.BORDER.name()};")
        separator2.setFixedHeight(1)
        self._separators.append(separator2)
        main_layout.addWidget(separator2)
        
        # Footer
        self.footer_widget = self._create_footer()
        main_layout.addWidget(self.footer_widget)
        
        # Set as dock widget content
        self.setWidget(main_widget)
        
        # Apply styling
        self.setStyleSheet(f"""
            QDockWidget {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
            }}
            QDockWidget::title {{
                background-color: {Colors.BG_MEDIUM.name()};
                padding: {Spacing.SM}px;
            }}
        """)
    
    def _create_header(self) -> QWidget:
        """Create common header with block info"""
        header = QWidget()
        header.setFixedHeight(40)  # Fixed height for header
        header.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(
            Spacing.MD,
            Spacing.SM,
            Spacing.MD,
            Spacing.SM
        )
        
        # Block type icon/label
        self.type_label = QLabel("◼")  # Will be updated with block type
        type_font = QFont()
        type_font.setPointSize(16)
        self.type_label.setFont(type_font)
        layout.addWidget(self.type_label)
        
        # Block name
        self.name_label = QLabel("Loading...")
        name_font = Typography.heading_font()
        self.name_label.setFont(name_font)
        layout.addWidget(self.name_label)
        
        layout.addStretch()
        
        # Status indicator - unified widget that auto-updates
        self.status_dot = BlockStatusDot(self.block_id, self.facade, parent=header)
        layout.addWidget(self.status_dot)
        
        return header
    
    def _create_footer(self) -> QWidget:
        """Create common footer with actions"""
        footer = QWidget()
        # Height must fit button content: 6px padding * 2 + text line ~16px + borders; 44px leaves room after margins
        footer.setFixedHeight(44)
        footer.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(
            Spacing.MD,
            Spacing.SM,
            Spacing.MD,
            Spacing.SM
        )
        
        # Status message
        self.status_message = QLabel("Ready")
        status_font = Typography.default_font()
        status_font.setPointSize(11)
        self.status_message.setFont(status_font)
        self.status_message.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(self.status_message)
        
        layout.addStretch()

        # Pull Data (Overwrite) button - applies to all blocks
        pull_btn = QPushButton("Pull Data")
        pull_btn.setToolTip("Overwrite this block's local inputs and re-pull from its incoming connections")
        pull_btn.clicked.connect(self._on_pull_data_clicked)
        layout.addWidget(pull_btn)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        return footer

    def _on_pull_data_clicked(self):
        """Overwrite local inputs by pulling from connections (with confirmation)."""
        if not self.block_id or not self.facade:
            return

        reply = QMessageBox.question(
            self,
            "Pull Data (Overwrite)",
            "Overwrite this block's local inputs and re-pull from its connections?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        result = self.facade.pull_block_inputs_overwrite(self.block_id)
        if hasattr(result, "success") and result.success:
            self.set_status_message("Pulled local inputs", error=False)
            # Refresh panel UI; subclasses may read local inputs via facade
            self._load_block_data()
            return

        # Handle missing upstream data (MVP)
        missing_txt = ""
        try:
            if hasattr(result, "errors") and result.errors:
                # We encode missing details into errors as: "missing=[...]"
                for e in result.errors:
                    if isinstance(e, str) and e.startswith("missing="):
                        missing_txt = e[len("missing="):]
                        break
        except Exception:
            missing_txt = ""

        if missing_txt:
            msg_lines = ["Upstream has no data for one or more connections:", ""]
            msg_lines.append(missing_txt)
            msg_lines.append("")
            msg_lines.append("Execute upstream blocks first, then pull again.")
            QMessageBox.information(self, "Pull Data", "\n".join(msg_lines))
            self.set_status_message("Pull failed: upstream missing data", error=True)
        else:
            QMessageBox.warning(self, "Pull Data", getattr(result, "message", "Pull failed"))
            self.set_status_message("Pull failed", error=True)
    
    def create_content_widget(self) -> QWidget:
        """
        Create block-specific content widget.
        
        Override this method in subclasses to provide custom UI.
        
        Returns:
            QWidget with block-specific controls
        """
        # Default implementation (no specific panel)
        label = QLabel("No panel implementation for this block type")
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label
    
    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
    
    def _load_block_data(self):
        """Load block data from facade"""
        result = self.facade.describe_block(self.block_id)
        if result.success:
            self.block = result.data
            self._update_header()
            self.refresh()
            # Status indicator auto-updates via BlockStatusDot widget
            # Check for filter warnings after loading data
            self._check_and_display_filter_warnings()
        else:
            Log.error(f"Failed to load block data for {self.block_id}: {result.message}")
            self.status_message.setText("Error loading block")
            self.status_message.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
    
    def _update_header(self):
        """Update header with block information"""
        if not self.block:
            return
        
        # Get block type icon
        type_icons = {
            "LoadAudio": "",
            "Separator": "️",
            "TranscribeNote": "",
            "PlotEvents": "",
            "ExportAudio": "",
            "CommandSequencer": "️",
            "LoadMultiple": "",
            "Editor": "",
            "EditorV2": "",
        }
        icon = type_icons.get(self.block.type, "◼")
        self.type_label.setText(icon)
        
        # Set block name and type
        self.name_label.setText(f"{self.block.type}: \"{self.block.name}\"")
        
        # Update window title
        self.setWindowTitle(f"{self.block.type} - {self.block.name}")
    
    def refresh_for_undo(self):
        """
        Refresh panel state after undo/redo operation.
        
        Called by MainWindow when the undo stack changes.
        Override in subclasses that need special undo refresh behavior.
        By default, just calls refresh().
        """
        self.refresh()
    
    def refresh(self):
        """
        Refresh panel with current block data.
        
        Override this method in subclasses to update block-specific UI.
        Called after block data is loaded or updated.
        """
        pass
    
    # ==================== IStatefulWindow Implementation ====================
    
    def get_internal_state(self) -> Dict[str, Any]:
        """
        Get internal state for saving (IStatefulWindow interface).
        
        Override in subclasses to save panel-specific state.
        By default, saves nothing.
        
        Returns:
            Dictionary of internal state (must be JSON-serializable)
        """
        return {}
    
    def restore_internal_state(self, state: Dict[str, Any]) -> None:
        """
        Restore internal state after loading (IStatefulWindow interface).
        
        Override in subclasses to restore panel-specific state.
        
        Args:
            state: Dictionary of internal state (from get_internal_state())
        """
        pass
    
    def _subscribe_to_events(self):
        """Subscribe to block update events"""
        self.facade.event_bus.subscribe("BlockUpdated", self._on_block_updated)
        # Status indicator updates are handled by BlockStatusDot widget
    
    def _on_block_updated(self, event):
        """Handle block update event"""
        # Skip refresh if this panel triggered the update (prevents value reset during save)
        if self._is_saving:
            Log.debug(f"BlockPanel: Skipping refresh during save for {self.block_id}")
            return
        
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            Log.info(f"BlockPanel: Block {self.block_id} updated, refreshing panel")
            self._load_block_data()
    
    def set_status_message(self, message: str, error: bool = False):
        """
        Set status message in footer.
        
        Args:
            message: Status message text
            error: If True, display as error (red text)
        """
        self.status_message.setText(message)
        if error:
            self.status_message.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        else:
            self.status_message.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
    
    def _check_and_display_filter_warnings(self):
        """Check for filter issues and display warnings in status"""
        if not self.facade or not self.block_id:
            return
        
        # Reload block to get latest metadata
        result = self.facade.describe_block(self.block_id)
        if not result.success or not result.data:
            return
        
        block = result.data
        
        # Check if block has filter_selections that might cause issues
        filter_selections = block.metadata.get("filter_selections", {})
        if not filter_selections:
            return
        
        # For each port with filter_selections, verify output names exist
        warnings = []
        for port_name, selected_output_names in filter_selections.items():
            if not selected_output_names:
                continue
            
            # Get expected outputs from source blocks (what filters should be based on)
            if hasattr(self.facade, 'data_filter_manager') and self.facade.data_filter_manager:
                try:
                    # Get source block info to check expected outputs
                    source_info = self.facade.data_filter_manager.get_source_block_info(
                        block, port_name, facade=self.facade
                    )
                    
                    if not source_info:
                        # No source blocks - filters might be stale
                        continue
                    
                    # Collect all expected output names from all source blocks
                    expected_output_names = set()
                    source_blocks = source_info.get('source_blocks', [])
                    
                    # Backward compatibility: single source format
                    if not source_blocks:
                        expected_outputs = source_info.get('expected_outputs', [])
                        if expected_outputs:
                            expected_output_names.update(expected_outputs)
                    else:
                        # Multiple sources format
                        for source_block_info in source_blocks:
                            expected_outputs = source_block_info.get('expected_outputs', [])
                            if expected_outputs:
                                expected_output_names.update(expected_outputs)
                    
                    selected_set = set(selected_output_names)
                    
                    # Check for invalid output names (not in expected outputs from sources)
                    invalid_names = selected_set - expected_output_names
                    if invalid_names:
                        warnings.append(
                            f"Filter for '{port_name}' contains invalid output names: "
                            f"{invalid_names}"
                        )
                except Exception as e:
                    Log.debug(f"BlockPanelBase: Could not check filter warnings for {port_name}: {e}")
        
        if warnings:
            warning_text = "Filter warnings: " + "; ".join(warnings)
            self.set_status_message(warning_text, error=True)
    
    # ==================== Metadata Management (Phase B) ====================
    
    def set_block_metadata_key(self, key: str, value: any, success_message: str = None):
        """
        Set a single metadata key-value pair (UNDOABLE via CommandBus).
        
        Replaces direct mutations like: self.block.metadata["key"] = value
        
        Args:
            key: Metadata key
            value: Value to set
            success_message: Optional success message for status bar
            
        Returns:
            True if command was executed successfully
            
        Example:
            # Instead of: self.block.metadata["min_note"] = 36
            self.set_block_metadata_key("min_note", 36, "Note range updated")
        """
        # Set guard flag to prevent refresh during save (prevents value reset)
        self._is_saving = True
        try:
            # Use undoable command via CommandBus
            cmd = UpdateBlockMetadataCommand(
                self.facade,
                self.block_id,
                key,
                value,
                description=f"Set {key}"
            )
            result = self.facade.command_bus.execute(cmd)
            
            if result:
                Log.debug(f"Set metadata {key}={value} for block {self.block_id}")
                if success_message:
                    self.set_status_message(success_message)
            else:
                Log.error(f"Failed to set metadata {key}")
                self.set_status_message(f"Failed to update {key}", error=True)
            
            return result
        finally:
            self._is_saving = False
    
    # ==================== Block Configuration (Undoable) ====================
    
    def execute_block_setting(
        self,
        command_name: str,
        new_value: any,
        old_value: any = None,
        success_message: str = "Updated",
        description: str = None
    ):
        """
        Execute a block configuration command (UNDOABLE via CommandBus).
        
        This is the preferred way to change block settings like model, device, etc.
        All changes go through CommandBus for undo/redo support.
        
        Args:
            command_name: Command to execute (e.g., "set_model", "set_device")
            new_value: New value to set
            old_value: Optional previous value (captured if not provided)
            success_message: Status message on success
            description: Optional custom description for command history
            
        Returns:
            True if command was executed successfully
            
        Example:
            # Instead of: facade.execute_block_command(...)
            self.execute_block_setting(
                "set_model",
                "htdemucs_ft",
                old_value="htdemucs",
                success_message="Model updated"
            )
        """
        # Set guard flag to prevent refresh during save (prevents value reset)
        self._is_saving = True
        try:
            # Use undoable command via CommandBus
            cmd = ConfigureBlockCommand(
                self.facade,
                self.block_id,
                command_name,
                new_value,
                old_value=old_value,
                description=description
            )
            result = self.facade.command_bus.execute(cmd)
            
            if result:
                self.set_status_message(success_message, error=False)
                Log.debug(f"Block setting '{command_name}' updated to {new_value}")
            else:
                self.set_status_message(f"Failed to update {command_name}", error=True)
                Log.error(f"Block setting '{command_name}' failed")
            
            return result
        finally:
            self._is_saving = False
    
    def set_multiple_metadata(
        self,
        updates: dict,
        success_message: str = "Settings updated",
        description: str = None
    ):
        """
        Update multiple metadata keys at once (UNDOABLE via CommandBus).
        
        More efficient than multiple set_block_metadata_key calls.
        
        Args:
            updates: Dict of key -> value to update
            success_message: Status message on success
            description: Optional custom description
            
        Returns:
            True if command was executed successfully
            
        Example:
            self.set_multiple_metadata({
                "min_note": 36,
                "max_note": 84
            }, success_message="Note range updated")
        """
        # Set guard flag to prevent refresh during save (prevents value reset)
        self._is_saving = True
        try:
            cmd = BatchUpdateMetadataCommand(
                self.facade,
                self.block_id,
                updates,
                description=description
            )
            result = self.facade.command_bus.execute(cmd)
            
            if result:
                self.set_status_message(success_message, error=False)
                Log.debug(f"Updated {len(updates)} metadata keys for block {self.block_id}")
            else:
                self.set_status_message(f"Failed to update settings", error=True)
            
            return result
        finally:
            self._is_saving = False
    
    # =====================================================================
    # Theme Change Propagation (Tier 3)
    # =====================================================================
    
    def _on_theme_changed(self):
        """Called automatically when the application theme changes.
        
        Three-step process for instant, complete visual refresh:
        
        1. **Clear** all child local stylesheets.  This removes the stale
           baked-in color values and lets the global ``QApplication``
           stylesheet take over for default-styled widgets.
        2. **Re-apply** the base panel frame styles (dock border, header
           background, separators) which genuinely differ from the global.
        3. **Re-apply** subclass variant styles via ``_apply_local_styles()``
           (primary buttons, warning banners, etc.).
        """
        self._clear_child_stylesheets()
        self._apply_base_styles()
        self._apply_local_styles()
    
    def _clear_child_stylesheets(self):
        """Clear all local stylesheets on child widgets.
        
        Local ``setStyleSheet()`` calls have higher CSS specificity than
        the global ``QApplication`` stylesheet.  When their baked-in color
        values become stale after a theme change, clearing them allows the
        freshly-regenerated global stylesheet to take effect immediately.
        
        Variant-specific overrides are restored by ``_apply_local_styles()``
        right after this method runs.
        """
        from PyQt6.QtWidgets import QWidget
        for child in self.findChildren(QWidget):
            if child.styleSheet():
                child.setStyleSheet("")
    
    def _apply_base_styles(self):
        """Re-apply base BlockPanelBase frame styles with current theme tokens.
        
        These styles are genuinely different from the global defaults
        (e.g. QDockWidget border, header background, separator color).
        """
        # Dock widget frame
        self.setStyleSheet(f"""
            QDockWidget {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
            }}
            QDockWidget::title {{
                background-color: {Colors.BG_MEDIUM.name()};
                padding: {Spacing.SM}px;
            }}
        """)
        
        # Header background
        if hasattr(self, 'header_widget') and self.header_widget:
            self.header_widget.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        
        # Separator lines
        border_color = f"background-color: {Colors.BORDER.name()};"
        for sep in getattr(self, '_separators', []):
            sep.setStyleSheet(border_color)
    
    def _apply_local_styles(self):
        """Override in subclasses to re-apply variant/context-specific styles.
        
        Called automatically on theme change. Subclasses that use local
        ``setStyleSheet()`` with variant colors (primary buttons, warning
        banners, colored indicators) should override this to re-evaluate
        their f-strings against the new ``Colors`` values.
        
        The global ``QApplication`` stylesheet covers default widget styling,
        so only genuinely non-default overrides need to go here.
        """
        pass
    
    def _on_dock_level_changed(self, floating: bool):
        """
        Handle dock level changes (floating/docked).
        
        Subclasses can override to add custom behavior during dock state changes.
        This is called when the panel transitions between floating and docked states,
        including when tabifying with other docks.
        """
        Log.debug(f"BlockPanel {self.block_id}: Dock level changed, floating={floating}")
        
        # When floating, set window flags to behave like a regular window
        # This prevents the panel from disappearing when the app loses focus on macOS
        if floating:
            # Set window flags to match main window behavior (regular Window type)
            # By default, Qt uses Tool window type for floating docks, which causes them
            # to hide on macOS when the app loses focus. Setting to Window type fixes this.
            # We preserve existing hint flags but change the window type to Window
            current_flags = self.windowFlags()
            # Keep only hint flags (not window type flags)
            # Window type flags are: Widget, Window, Dialog, Tool, Sheet, Drawer, Popup
            # We'll explicitly set to Window type
            hint_mask = (
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.WindowStaysOnBottomHint |
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.CustomizeWindowHint |
                Qt.WindowType.WindowTitleHint |
                Qt.WindowType.WindowSystemMenuHint |
                Qt.WindowType.WindowMinimizeButtonHint |
                Qt.WindowType.WindowMaximizeButtonHint |
                Qt.WindowType.WindowCloseButtonHint |
                Qt.WindowType.WindowContextHelpButtonHint |
                Qt.WindowType.WindowShadeButtonHint
            )
            hint_flags = current_flags & hint_mask
            # Set to regular Window type (same as main window) with preserved hints
            new_flags = Qt.WindowType.Window | hint_flags
            self.setWindowFlags(new_flags)
            # Must show the window again after changing flags
            self.show()
    
    def create_filter_widget(
        self,
        port_name: str
    ):
        """
        Create an input filter widget for an input port.
        
        Helper method to easily add filter widgets to block panels.
        
        Args:
            port_name: Input port name
            
        Returns:
            InputFilterWidget instance, or None if services not available
        """
        if not self.block or not self.facade:
            return None
        
        # Check if services are available
        if not hasattr(self.facade, 'data_filter_manager') or not self.facade.data_filter_manager:
            Log.warning("BlockPanelBase: Data filter manager not available")
            return None
        
        if not hasattr(self.facade, 'data_state_service') or not self.facade.data_state_service:
            Log.warning("BlockPanelBase: Data state service not available")
            # Continue without state service (filter widget will work without status indicators)
        
        try:
            from ui.qt_gui.block_panels.components.data_filter_widget import InputFilterWidget
            
            widget = InputFilterWidget(
                block=self.block,
                port_name=port_name,
                facade=self.facade,
                data_filter_manager=self.facade.data_filter_manager,
                data_state_service=getattr(self.facade, 'data_state_service', None),
                parent=self
            )
            
            # Connect selection changed signal to refresh if needed
            widget.selection_changed.connect(lambda: self._on_filter_selection_changed(port_name))
            
            return widget
        except Exception as e:
            Log.error(f"BlockPanelBase: Failed to create filter widget: {e}")
            return None
    
    def create_expected_outputs_display(self):
        """
        Create an expected outputs display widget.
        
        Shows what this block expects to output based on processor.get_expected_outputs().
        
        Returns:
            ExpectedOutputsDisplay instance, or None if services not available
        """
        if not self.block or not self.facade:
            return None
        
        try:
            from ui.qt_gui.block_panels.components.expected_outputs_display import ExpectedOutputsDisplay
            
            widget = ExpectedOutputsDisplay(
                block=self.block,
                facade=self.facade,
                parent=self
            )
            
            return widget
        except Exception as e:
            Log.error(f"BlockPanelBase: Failed to create expected outputs display: {e}")
            return None
    
    def add_port_filter_sections(self, layout: QVBoxLayout) -> None:
        """
        Add filter sections for input ports and expected outputs display.
        
        Automatically creates filter widgets for all input ports and displays
        expected outputs for output ports.
        
        Args:
            layout: QVBoxLayout to add filter sections to
        """
        if not self.block:
            return
        
        from PyQt6.QtWidgets import QGroupBox
        
        # Add input port filters
        input_ports = self.block.get_inputs()
        if input_ports:
            input_group = QGroupBox("Input Data Filtering")
            input_layout = QVBoxLayout(input_group)
            input_layout.setSpacing(Spacing.SM)
            
            has_input_filters = False
            for port_name in input_ports.keys():
                filter_widget = self.create_filter_widget(port_name)
                if filter_widget:
                    input_layout.addWidget(filter_widget)
                    has_input_filters = True
            
            if has_input_filters:
                layout.addWidget(input_group)
        
        # Add expected outputs display
        output_ports = self.block.get_outputs()
        if output_ports:
            expected_outputs_widget = self.create_expected_outputs_display()
            if expected_outputs_widget:
                layout.addWidget(expected_outputs_widget)
    
    def _on_filter_selection_changed(self, port_name: str):
        """
        Handle filter selection change.
        
        Override in subclasses if needed to react to filter changes.
        
        Args:
            port_name: Port name that changed
        """
        pass
    
    def closeEvent(self, event):
        """Handle panel close event"""
        # Unsubscribe from events
        try:
            self.facade.event_bus.unsubscribe("BlockUpdated", self._on_block_updated)
            # BlockStatusDot widget handles its own event cleanup
        except:
            pass
        
        # Disconnect theme signal to prevent callbacks on deleted widgets
        try:
            disconnect_theme_changed(self._on_theme_changed)
        except (TypeError, RuntimeError):
            pass
        
        # Emit closed signal
        self.panel_closed.emit(self.block_id)
        
        # Accept close event
        event.accept()

