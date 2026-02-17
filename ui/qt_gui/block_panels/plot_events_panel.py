"""
PlotEvents block panel.

Provides UI for configuring event visualization settings.
Simple checkbox-based interface for enabling/disabling plot types.
"""

from typing import Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QGroupBox, QSpinBox, QCheckBox,
    QPushButton, QScrollArea, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
import io

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.application.settings.plot_events_settings import PlotEventsSettingsManager
from src.utils.message import Log
from src.shared.domain.entities import EventDataItem


@register_block_panel("PlotEvents")
class PlotEventsPanel(BlockPanelBase):
    """
    Panel for PlotEvents block configuration.
    
    Simple checkbox-based interface for enabling/disabling plot types.
    Multiple plot types can be enabled simultaneously.
    """
    
    def __init__(self, block_id: str, facade, parent=None):
        # Call parent init first (sets up UI structure)
        # Note: parent.__init__ calls refresh() before _settings_manager exists,
        # so refresh() must be defensive
        super().__init__(block_id, facade, parent)
        
        # Initialize settings manager AFTER parent init
        self._settings_manager = PlotEventsSettingsManager(facade, block_id, parent=self)
        
        # Connect to settings changes for UI updates
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        
        # Auto-refresh timer for debouncing checkbox changes
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(300)  # 300ms debounce
        self._refresh_timer.timeout.connect(self._on_refresh_timer)
        
        # Track enabled plot types (for multiple selection)
        self._enabled_plot_types: set = set()
        
        # Refresh UI now that settings manager is ready
        # (parent's refresh() was called before manager existed)
        if self.block:
            self.refresh()
    
    def create_content_widget(self) -> QWidget:
        """Create PlotEvents-specific UI"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(Spacing.MD)
        
        # Plot types group with checkboxes
        plot_types_group = QGroupBox("Plot Types")
        plot_types_layout = QVBoxLayout(plot_types_group)
        plot_types_layout.setSpacing(Spacing.SM)
        
        # Bars checkbox
        self.bars_check = QCheckBox("Bars")
        self.bars_check.setChecked(True)  # Default enabled
        self.bars_check.stateChanged.connect(self._on_plot_type_changed)
        plot_types_layout.addWidget(self.bars_check)
        
        # Markers checkbox
        self.markers_check = QCheckBox("Markers")
        self.markers_check.setChecked(False)
        self.markers_check.stateChanged.connect(self._on_plot_type_changed)
        plot_types_layout.addWidget(self.markers_check)
        
        # Piano Roll checkbox
        self.piano_roll_check = QCheckBox("Piano Roll")
        self.piano_roll_check.setChecked(False)
        self.piano_roll_check.stateChanged.connect(self._on_plot_type_changed)
        plot_types_layout.addWidget(self.piano_roll_check)
        
        layout.addWidget(plot_types_group)
        
        # Figure settings group
        figure_group = QGroupBox("Figure Settings")
        figure_layout = QFormLayout(figure_group)
        figure_layout.setSpacing(Spacing.SM)
        
        # Figure width
        self.width_spin = QSpinBox()
        self.width_spin.setRange(4, 32)
        self.width_spin.setValue(12)
        self.width_spin.setSuffix(" inches")
        self.width_spin.valueChanged.connect(self._on_figure_size_changed)
        figure_layout.addRow("Width:", self.width_spin)
        
        # Figure height
        self.height_spin = QSpinBox()
        self.height_spin.setRange(3, 24)
        self.height_spin.setValue(8)
        self.height_spin.setSuffix(" inches")
        self.height_spin.valueChanged.connect(self._on_figure_size_changed)
        figure_layout.addRow("Height:", self.height_spin)
        
        # DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setSingleStep(24)
        self.dpi_spin.setValue(100)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.valueChanged.connect(self._on_dpi_changed)
        self.dpi_spin.setToolTip("Higher DPI = higher resolution image")
        figure_layout.addRow("DPI:", self.dpi_spin)
        
        layout.addWidget(figure_group)
        
        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)
        display_layout.setSpacing(Spacing.SM)
        
        # Show labels
        self.show_labels_check = QCheckBox()
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self._on_show_labels_changed)
        display_layout.addRow("Show Labels:", self.show_labels_check)
        
        # Show grid
        self.show_grid_check = QCheckBox()
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(self._on_show_grid_changed)
        display_layout.addRow("Show Grid:", self.show_grid_check)
        
        # Show energy/amplitude subplots
        self.show_energy_amplitude_check = QCheckBox()
        self.show_energy_amplitude_check.setChecked(False)
        self.show_energy_amplitude_check.stateChanged.connect(self._on_show_energy_amplitude_changed)
        self.show_energy_amplitude_check.setToolTip(
            "Show energy and amplitude subplots (3-panel layout):\n"
            "• ON: Creates 3-panel visualization with main plot, RMS energy levels, and waveform\n"
            "• Requires audio data to be available via event metadata (audio_id)\n"
            "• Energy plot shows per-clip thresholds from DetectOnsets metadata\n"
            "• Waveform plot shows onsets and clip boundaries\n\n"
            "Useful for: Debugging onset detection, verifying clip boundaries, analyzing energy decay"
        )
        display_layout.addRow("Show Energy/Amplitude:", self.show_energy_amplitude_check)
        
        layout.addWidget(display_group)
        
        # Refresh button
        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh Plots")
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        self.refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 8px 20px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
        """)
        refresh_layout.addWidget(self.refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(Spacing.SM)
        
        # Preview image area
        self.preview_label = QLabel("Click 'Refresh Plots' to generate preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(300)
        self.preview_label.setStyleSheet(f"""
            background-color: {Colors.BG_DARK.name()};
            border: 1px solid {Colors.BORDER.name()};
            border-radius: {border_radius(4)};
            color: {Colors.TEXT_SECONDARY.name()};
        """)
        self.preview_label.setWordWrap(True)
        
        # Scroll area for preview
        preview_scroll = QScrollArea()
        preview_scroll.setWidget(self.preview_label)
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setMinimumHeight(300)
        preview_layout.addWidget(preview_scroll, 1)
        
        layout.addWidget(preview_group, 1)  # Give preview stretch
        
        # Info note
        info_label = QLabel(
            "Enable plot types using checkboxes above. Plots refresh automatically when checkboxes change, "
            "or click 'Refresh Plots' to manually refresh. Generated plots are saved to the output directory when the block executes."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        return widget
    
    def refresh(self):
        """Update UI with current settings from settings manager"""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        if not self.block or not self._settings_manager.is_loaded():
            return
        
        # Load settings from settings manager (single source of truth)
        try:
            plot_style = self._settings_manager.plot_style
            figsize_width = self._settings_manager.figsize_width
            figsize_height = self._settings_manager.figsize_height
            dpi = self._settings_manager.dpi
            show_labels = self._settings_manager.show_labels
            show_grid = self._settings_manager.show_grid
            show_energy_amplitude = self._settings_manager.show_energy_amplitude
        except Exception as e:
            Log.error(f"PlotEventsPanel: Failed to load settings: {e}")
            return
        
        # Block signals while updating
        self.bars_check.blockSignals(True)
        self.markers_check.blockSignals(True)
        self.piano_roll_check.blockSignals(True)
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.dpi_spin.blockSignals(True)
        self.show_labels_check.blockSignals(True)
        self.show_grid_check.blockSignals(True)
        self.show_energy_amplitude_check.blockSignals(True)
        
        # Set plot type checkboxes based on current plot_style
        # Default to bars if plot_style matches, otherwise set based on style
        self.bars_check.setChecked(plot_style == "bars")
        self.markers_check.setChecked(plot_style == "markers")
        self.piano_roll_check.setChecked(plot_style == "piano_roll")
        
        # Update enabled plot types set
        self._enabled_plot_types.clear()
        if self.bars_check.isChecked():
            self._enabled_plot_types.add("bars")
        if self.markers_check.isChecked():
            self._enabled_plot_types.add("markers")
        if self.piano_roll_check.isChecked():
            self._enabled_plot_types.add("piano_roll")
        
        # If no plot types enabled, default to bars
        if not self._enabled_plot_types:
            self._enabled_plot_types.add("bars")
            self.bars_check.setChecked(True)
        
        # Set figure size
        self.width_spin.setValue(int(figsize_width))
        self.height_spin.setValue(int(figsize_height))
        Log.debug(f"PlotEventsPanel: Set figure size to {figsize_width}x{figsize_height}")
        
        # Set DPI
        self.dpi_spin.setValue(int(dpi))
        Log.debug(f"PlotEventsPanel: Set DPI to {dpi}")
        
        # Set display options
        self.show_labels_check.setChecked(show_labels)
        self.show_grid_check.setChecked(show_grid)
        self.show_energy_amplitude_check.setChecked(show_energy_amplitude)
        Log.debug(f"PlotEventsPanel: Set show_labels={show_labels}, show_grid={show_grid}, show_energy_amplitude={show_energy_amplitude}")
        
        # Unblock signals
        self.bars_check.blockSignals(False)
        self.markers_check.blockSignals(False)
        self.piano_roll_check.blockSignals(False)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self.dpi_spin.blockSignals(False)
        self.show_labels_check.blockSignals(False)
        self.show_grid_check.blockSignals(False)
        self.show_energy_amplitude_check.blockSignals(False)
        
        # Update status
        self.set_status_message("Settings loaded")
    
    def _on_plot_type_changed(self, state: int):
        """Handle plot type checkbox change - triggers auto-refresh"""
        # Update enabled plot types set
        self._enabled_plot_types.clear()
        if self.bars_check.isChecked():
            self._enabled_plot_types.add("bars")
        if self.markers_check.isChecked():
            self._enabled_plot_types.add("markers")
        if self.piano_roll_check.isChecked():
            self._enabled_plot_types.add("piano_roll")
        
        # If no plot types enabled, prevent unchecking the last one
        if not self._enabled_plot_types:
            # Re-enable the checkbox that was just unchecked
            sender = self.sender()
            if sender == self.bars_check:
                self.bars_check.setChecked(True)
                self._enabled_plot_types.add("bars")
            elif sender == self.markers_check:
                self.markers_check.setChecked(True)
                self._enabled_plot_types.add("markers")
            elif sender == self.piano_roll_check:
                self.piano_roll_check.setChecked(True)
                self._enabled_plot_types.add("piano_roll")
            return
        
        # Update settings manager with first enabled plot type (for backward compatibility)
        # The actual plot generation will use all enabled types
        first_style = next(iter(self._enabled_plot_types))
        try:
            self._settings_manager.plot_style = first_style
        except ValueError as e:
            Log.warning(f"PlotEventsPanel: Failed to update plot_style: {e}")
        
        # Trigger auto-refresh (debounced)
        self._refresh_timer.stop()
        self._refresh_timer.start()
    
    def _on_refresh_timer(self):
        """Handle debounced refresh timer - generates preview"""
        self._generate_and_show_preview()
    
    def _on_refresh_clicked(self):
        """Handle manual refresh button click"""
        self._generate_and_show_preview()
    
    def _on_figure_size_changed(self):
        """Handle figure size change (undoable via settings manager)"""
        width = self.width_spin.value()
        height = self.height_spin.value()
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.figsize_width = float(width)
            self._settings_manager.figsize_height = float(height)
            self.set_status_message(f"Size set to {width}x{height} inches", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_dpi_changed(self, value: int):
        """Handle DPI change (undoable via settings manager)"""
        # Update via settings manager (single pathway, auto-saves, undoable)
        # Settings manager handles change detection internally
        try:
            self._settings_manager.dpi = value
            self.set_status_message(f"DPI set to {value}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_show_labels_changed(self, state: int):
        """Handle show labels checkbox change (undoable via settings manager)"""
        show = state == Qt.CheckState.Checked.value
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.show_labels = show
            self.set_status_message(f"Labels: {'on' if show else 'off'}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_show_grid_changed(self, state: int):
        """Handle show grid checkbox change (undoable via settings manager)"""
        show = state == Qt.CheckState.Checked.value
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.show_grid = show
            self.set_status_message(f"Grid: {'on' if show else 'off'}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def _on_show_energy_amplitude_changed(self, state: int):
        """Handle show energy/amplitude checkbox change (undoable via settings manager)"""
        show = state == Qt.CheckState.Checked.value
        
        # Update via settings manager (single pathway, auto-saves, undoable)
        try:
            self._settings_manager.show_energy_amplitude = show
            self.set_status_message(f"Energy/Amplitude subplots: {'on' if show else 'off'}", error=False)
        except ValueError as e:
            self.set_status_message(str(e), error=True)
            self.refresh()
    
    def refresh_for_undo(self):
        """
        Refresh panel after undo/redo operation.
        
        Reloads settings from database to ensure UI reflects current state.
        Single source of truth: block.metadata in database.
        """
        # Reload settings manager from database (undo may have changed metadata)
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.reload_from_storage()
        
        # Refresh UI with current settings
        self.refresh()
    
    def _on_block_updated(self, event):
        """
        Handle block update event - reload settings and refresh UI.
        
        This ensures panel stays in sync when settings change via quick actions
        or other sources. Single source of truth: block.metadata in database.
        """
        updated_block_id = event.data.get('id')
        if updated_block_id == self.block_id:
            # Skip if we triggered this update (prevents refresh loop)
            if self._is_saving:
                Log.debug(f"PlotEventsPanel: Skipping refresh during save for {self.block_id}")
                return
            
            Log.debug(f"PlotEventsPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
            else:
                Log.warning(f"PlotEventsPanel: Failed to reload block {self.block_id}")
                return
            
            # Reload settings from database (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                self._settings_manager.reload_from_storage()
                Log.debug(f"PlotEventsPanel: Settings manager reloaded from database")
            else:
                Log.warning(f"PlotEventsPanel: Settings manager not available")
                return
            
            # Refresh UI to reflect changes (now that both block and settings are reloaded)
            # Use QTimer.singleShot to ensure refresh happens after event processing
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)
    
    def _on_setting_changed(self, setting_name: str):
        """
        React to settings changes from this panel's settings manager.
        
        Note: Changes from other sources (quick actions) are handled via
        _on_block_updated() which reloads from database.
        """
        if setting_name in ['plot_style', 'figsize_width', 'figsize_height', 'dpi', 'show_labels', 'show_grid', 'show_energy_amplitude']:
            # Refresh UI to reflect change
            self.refresh()
    
    # ==================== Preview Methods ====================
    
    def _generate_and_show_preview(self):
        """Generate and display preview plot(s) for enabled plot types"""
        if not self.block:
            return
        
        try:
            # Get event data from connected source blocks
            event_data_items = self._get_event_data_for_preview()
            
            if not event_data_items:
                self.preview_label.setText(
                    "No event data found for preview.\n\n"
                    "Connect a block that outputs events (e.g., TranscribeNote, DetectOnsets) "
                    "to this PlotEvents block and execute it first."
                )
                self.set_status_message("No event data available", error=True)
                return
            
            # Check if any plot types are enabled
            if not self._enabled_plot_types:
                self.preview_label.setText("Please enable at least one plot type")
                self.set_status_message("No plot types enabled", error=True)
                return
            
            # Generate preview plot (show first enabled type)
            pixmap = self._generate_preview_plot(event_data_items)
            
            if pixmap:
                self.preview_label.setPixmap(pixmap)
                # Don't use setScaledContents - preserve the calculated aspect ratio
                # The scroll area will handle scrolling if needed
                self.preview_label.setScaledContents(False)
                # Set minimum size to ensure pixmap is visible
                pixmap_size = pixmap.size()
                self.preview_label.setMinimumSize(pixmap_size.width(), pixmap_size.height())
                enabled_types_str = ", ".join(sorted(self._enabled_plot_types))
                self.set_status_message(f"Preview generated ({enabled_types_str})")
            else:
                self.preview_label.setText("Failed to generate preview")
                self.set_status_message("Preview failed", error=True)
                
        except Exception as e:
            Log.error(f"PlotEventsPanel: Preview error: {e}")
            import traceback
            traceback.print_exc()
            self.preview_label.setText(f"Error generating preview:\n{str(e)}")
            self.set_status_message("Preview error", error=True)
    
    def _get_event_data_for_preview(self) -> list:
        """Get EventDataItem objects from connected source blocks"""
        event_items = []
        
        try:
            # Access facade's internal data_item_repo
            if not hasattr(self.facade, 'data_item_repo') or not self.facade.data_item_repo:
                Log.warning("PlotEventsPanel: data_item_repo not available")
                return event_items
            
            # Get connections to this block
            connections_result = self.facade.list_connections()
            if not connections_result.success:
                return event_items
            
            # Find connections to this block's "events" input
            for conn in connections_result.data:
                if conn.target_block_id == self.block_id and conn.target_input_name == "events":
                    # Load data items from source block (same pattern as execution engine)
                    source_data_items = self.facade.data_item_repo.list_by_block(conn.source_block_id)
                    
                    # Filter for EventDataItem outputs matching the source output port
                    matching_items = [
                        item for item in source_data_items
                        if item.metadata.get('output_port') == conn.source_output_name
                        and isinstance(item, EventDataItem)
                    ]
                    
                    # Check if events are loaded in memory
                    for item in matching_items:
                        events = item.get_events()
                        if events:  # Events are in memory
                            event_items.append(item)
                        elif item.file_path and item.event_count > 0:  # Try to load from file if events exist
                            try:
                                # Load events from file if available
                                import json
                                from pathlib import Path
                                
                                file_path = Path(item.file_path)
                                if file_path.exists() and file_path.suffix == '.json':
                                    with open(file_path, 'r') as f:
                                        data = json.load(f)
                                        # Handle both full EventDataItem dict and events-only array
                                        if isinstance(data, dict):
                                            events_data = data.get("events", [])
                                        elif isinstance(data, list):
                                            events_data = data
                                        else:
                                            events_data = []
                                        
                                        # Load events into item
                                        for event_data in events_data:
                                            from src.domain.entities.event_data_item import Event
                                            event = Event.from_dict(event_data)
                                            item.add_event(event.time, event.classification, 
                                                         event.duration, event.metadata)
                                        
                                        if item.get_events():
                                            event_items.append(item)
                            except Exception as e:
                                Log.debug(f"PlotEventsPanel: Could not load events from file: {e}")
        
        except Exception as e:
            Log.error(f"PlotEventsPanel: Error getting event data: {e}")
            import traceback
            traceback.print_exc()
        
        return event_items
    
    def _generate_preview_plot(self, event_data_items: list) -> QPixmap:
        """Generate preview plot for the first enabled plot type and return as QPixmap"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpl_patches
            
            if not event_data_items or not self._enabled_plot_types:
                return None
            
            # Get current block settings from settings manager (single source of truth)
            if hasattr(self, '_settings_manager') and self._settings_manager:
                figsize_width = self._settings_manager.figsize_width
                figsize_height = self._settings_manager.figsize_height
                show_labels = self._settings_manager.show_labels
                show_grid = self._settings_manager.show_grid
            else:
                # Fallback to metadata if settings manager not available
                figsize_width = self.block.metadata.get("figsize_width", 12.0)
                figsize_height = self.block.metadata.get("figsize_height", 8.0)
                show_labels = self.block.metadata.get("show_labels", True)
                show_grid = self.block.metadata.get("show_grid", True)
            
            dpi = 100  # Lower DPI for preview
            color_by_classification = self.block.metadata.get("color_by_classification", True)
            
            # Combine events from all event items for preview
            all_events = []
            for event_item in event_data_items:
                events = event_item.get_events()
                if events:
                    all_events.extend(events)
            
            if not all_events:
                return None
            
            # Sort events by time
            sorted_events = sorted(all_events, key=lambda e: e.time)
            
            # Calculate time range for width scaling
            times = [e.time for e in sorted_events]
            durations = [e.duration for e in sorted_events]
            min_time = min(times) if times else 0.0
            max_time = max(t + d for t, d in zip(times, durations)) if times else 1.0
            time_range = max_time - min_time
            
            # Ensure minimum time range (handle edge case where all events are at same time)
            if time_range < 0.01:
                # If all events are at essentially the same time, use a small default range
                time_range = 1.0
                max_time = min_time + time_range
            
            # Use first enabled plot type for preview
            plot_style = next(iter(sorted(self._enabled_plot_types)))
            
            # Scale width based on time range (each plot type independently)
            # Use logarithmic scaling for better handling of wide time ranges
            # Base width for 10 seconds of data
            base_width_for_10s = 10.0
            reference_time = 10.0
            
            if time_range > 0:
                if time_range <= 1.0:
                    # Very short durations: use fixed width
                    preview_width = 8.0
                elif time_range <= 60.0:
                    # Short to medium durations: linear scaling
                    scaled_width = base_width_for_10s * (time_range / reference_time)
                    preview_width = max(6.0, min(scaled_width, 16.0))
                else:
                    # Long durations: logarithmic scaling to prevent excessive width
                    import math
                    log_factor = math.log10(time_range / reference_time)
                    scaled_width = base_width_for_10s * (1 + log_factor * 0.5)  # Slower growth
                    preview_width = max(10.0, min(scaled_width, 20.0))
            else:
                # Fallback for zero or very small time range
                preview_width = 8.0
            
            # Log the calculated values for debugging
            Log.debug(f"PlotEventsPanel: Time range: {time_range:.3f}s, min: {min_time:.3f}s, max: {max_time:.3f}s, width: {preview_width:.1f}in")
            
            # Estimate height based on number of classifications (for better scaling)
            num_classifications = len(set(e.classification if e.classification else "Event" for e in all_events))
            base_height = 6
            height_per_class = 0.8
            estimated_height = base_height + (num_classifications - 1) * height_per_class
            preview_height = min(estimated_height, figsize_height, 10)  # Cap at reasonable max
            
            # Create figure with data-scaled width
            # IMPORTANT: Use calculated preview_width, not figsize_width from settings
            fig, ax = plt.subplots(figsize=(preview_width, preview_height), dpi=dpi)
            Log.debug(f"PlotEventsPanel: Created figure with size: {preview_width:.1f}x{preview_height:.1f} inches")
            
            # Get unique classifications for color mapping
            # Handle empty classifications
            classifications = []
            for event in sorted_events:
                cls = event.classification if event.classification else "Event"
                if cls not in classifications:
                    classifications.append(cls)
            color_map = self._create_color_map(classifications) if color_by_classification else {}
            
            # Plot based on style - each plot type handles its own x-axis limits
            if plot_style == "piano_roll":
                self._plot_piano_roll_preview(ax, sorted_events, color_map, show_labels, min_time, max_time, time_range)
            elif plot_style == "markers":
                self._plot_markers_preview(ax, sorted_events, color_map, show_labels, min_time, max_time, time_range)
            else:  # bars (default)
                self._plot_timeline_preview(ax, sorted_events, color_map, show_labels, min_time, max_time, time_range)
            
            # Configure plot appearance
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_ylabel('Events', fontsize=10)
            
            # Add title with event source and plot type info
            if len(event_data_items) == 1:
                title = f"{event_data_items[0].name}\n{len(sorted_events)} events"
            else:
                total_events = sum(len(item.get_events()) for item in event_data_items)
                source_names = ", ".join([item.name for item in event_data_items[:2]])
                if len(event_data_items) > 2:
                    source_names += f" (+{len(event_data_items) - 2} more)"
                title = f"{source_names}\n{total_events} events total"
            
            if len(self._enabled_plot_types) > 1:
                enabled_str = ", ".join(sorted(self._enabled_plot_types))
                title += f" (Preview: {plot_style}, Enabled: {enabled_str})"
            else:
                title += f" ({plot_style})"
            ax.set_title(title, fontsize=11)
            
            if show_grid:
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Add legend if color-coded
            if color_by_classification and classifications:
                handles = [mpl_patches.Patch(color=color_map.get(c, 'blue'), label=c) 
                          for c in sorted(classifications)]
                ax.legend(handles=handles, loc='upper right', fontsize=8, 
                         framealpha=0.9, ncol=min(len(classifications), 4))
            
            plt.tight_layout()
            
            # Convert to QPixmap
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            buf.close()
            plt.close(fig)
            
            return pixmap
            
        except ImportError:
            Log.error("PlotEventsPanel: matplotlib not available for preview")
            return None
        except Exception as e:
            Log.error(f"PlotEventsPanel: Error generating preview: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_color_map(self, classifications: list) -> dict:
        """Create a color map for classifications"""
        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            if not classifications:
                return {}
            
            cmap = cm.get_cmap('tab20' if len(classifications) <= 20 else 'hsv')
            colors = [cmap(i / len(classifications)) for i in range(len(classifications))]
            return {cls: mcolors.to_hex(color) for cls, color in zip(classifications, colors)}
        except:
            return {}
    
    def _plot_timeline_preview(self, ax, events, color_map, show_labels, min_time=None, max_time=None, time_range=None):
        """Plot events as horizontal bars grouped by classification (timeline view)"""
        if not events:
            return
        
        # Calculate time range if not provided
        if min_time is None or max_time is None or time_range is None:
            times = [e.time for e in events]
            durations = [e.duration for e in events]
            min_time = min(times)
            max_time = max(t + d for t, d in zip(times, durations))
            time_range = max_time - min_time
        
        # Group events by classification for better readability
        from collections import defaultdict
        events_by_class = defaultdict(list)
        for event in events:
            classification = event.classification if event.classification else "Event"
            events_by_class[classification].append(event)
        
        # Sort classifications for consistent ordering
        classifications = sorted(events_by_class.keys())
        
        # Assign row numbers to each classification
        class_to_row = {cls: i for i, cls in enumerate(classifications)}
        
        # Plot events, grouping by classification
        # Limit total rows to prevent plots from becoming too tall
        max_total_rows = 50
        row_offset = 0
        y_ticks = []
        y_labels = []
        max_events_per_class = min(30, max_total_rows // max(len(classifications), 1))
        
        for classification in classifications:
            class_events = sorted(events_by_class[classification], key=lambda e: e.time)
            color = color_map.get(classification, 'steelblue') if color_map else 'steelblue'
            
            # Limit events per classification for preview
            original_count = len(class_events)
            if len(class_events) > max_events_per_class:
                class_events = class_events[:max_events_per_class]
            
            # Check if we've exceeded total row limit
            if row_offset + len(class_events) > max_total_rows:
                remaining_rows = max_total_rows - row_offset
                if remaining_rows > 0:
                    class_events = class_events[:remaining_rows]
                else:
                    break
            
            # Plot all events of this classification in a contiguous block
            for i, event in enumerate(class_events):
                row = row_offset + i
                
                if event.duration > 0:
                    # Draw bar for events with duration
                    ax.barh(row, event.duration, left=event.time, height=0.7, 
                           color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                else:
                    # Draw marker for instantaneous events (onsets)
                    ax.plot(event.time, row, 'o', color=color, markersize=7, 
                           markeredgecolor='black', markeredgewidth=0.5, zorder=3)
                
                # Add label if enabled (only for first few events to avoid clutter)
                if show_labels and i < 5:
                    label = classification
                    # Add additional info from metadata
                    if 'frequency_hz' in event.metadata:
                        freq = event.metadata['frequency_hz']
                        label += f"\n{freq:.0f}Hz"
                    elif 'midi_note' in event.metadata:
                        midi = event.metadata['midi_note']
                        label += f"\nM{midi}"
                    
                    # Position label to avoid overlap
                    label_x = event.time
                    if event.duration > 0:
                        label_x = event.time + event.duration / 2
                    
                    ax.text(label_x, row, label, fontsize=7, va='center', 
                           ha='center', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    alpha=0.8, edgecolor='none'))
            
            # Add classification label on y-axis (middle of this classification's block)
            if class_events:
                mid_row = row_offset + len(class_events) / 2 - 0.5
                y_ticks.append(mid_row)
                display_count = len(class_events)
                if original_count > display_count:
                    y_labels.append(f"{classification}\n({display_count}/{original_count})")
                else:
                    y_labels.append(f"{classification}\n({display_count})")
            
            # Move to next classification block
            row_offset += len(class_events) + 1  # +1 for spacing between classifications
            
            # Stop if we've hit the limit
            if row_offset >= max_total_rows:
                break
        
        # Set y-axis with proper scaling
        if y_ticks and row_offset > 0:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=8)
            # Set limits with padding
            ax.set_ylim(-0.5, row_offset - 0.5)
        else:
            ax.set_yticks([])
            ax.set_ylim(-0.5, 0.5)
        
        ax.invert_yaxis()
        
        # Set x-axis limits with proper padding (scaled to data)
        if time_range > 0:
            padding = max(0.1, time_range * 0.05)  # 5% padding or 0.1s minimum
        else:
            padding = 0.1
        
        x_min = max(0, min_time - padding)
        x_max = max_time + padding
        
        # Ensure we have a valid range
        if x_max <= x_min:
            x_max = x_min + 1.0  # Default 1 second range if invalid
        
        ax.set_xlim(x_min, x_max)
        
        # Log for debugging
        Log.debug(f"PlotEventsPanel: Timeline x-axis limits: [{x_min:.3f}, {x_max:.3f}], range: {x_max - x_min:.3f}s")
        
        # Add note if events were truncated
        total_events = len(events)
        truncated = any(len(events_by_class[cls]) > max_events_per_class for cls in classifications)
        if truncated:
            ax.text(0.5, 0.02, f"Showing up to {max_events_per_class} events per classification ({total_events} total)", 
                   transform=ax.transAxes, ha='center', fontsize=7, 
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
    
    def _plot_markers_preview(self, ax, events, color_map, show_labels, min_time=None, max_time=None, time_range=None):
        """Plot events as markers, grouped by classification"""
        if not events:
            return
        
        # Calculate time range if not provided
        if min_time is None or max_time is None or time_range is None:
            times = [e.time for e in events]
            durations = [e.duration for e in events]
            min_time = min(times)
            max_time = max(t + d for t, d in zip(times, durations))
            time_range = max_time - min_time
        
        # Group events by classification
        from collections import defaultdict
        events_by_class = defaultdict(list)
        for event in events:
            classification = event.classification if event.classification else "Event"
            events_by_class[classification].append(event)
        
        # Sort classifications for consistent ordering
        classifications = sorted(events_by_class.keys())
        
        # Limit events per classification for preview
        max_events_per_class = 50
        y_positions = {}
        y_offset = 0
        
        for classification in classifications:
            class_events = sorted(events_by_class[classification], key=lambda e: e.time)
            if len(class_events) > max_events_per_class:
                class_events = class_events[:max_events_per_class]
            
            color = color_map.get(classification, 'steelblue') if color_map else 'steelblue'
            y_pos = 1.0 + y_offset * 0.3  # Space out different classifications
            
            times = [e.time for e in class_events]
            
            # Plot stems
            ax.stem(times, [y_pos] * len(times), linefmt='gray', markerfmt='', 
                   basefmt=' ', bottom=0)
            
            # Plot markers
            for event in class_events:
                ax.plot(event.time, y_pos, 'o', color=color, markersize=9, 
                       markeredgecolor='black', markeredgewidth=0.5, zorder=3)
                
                # Add duration indicator if present
                if event.duration > 0:
                    ax.plot([event.time, event.time + event.duration], 
                           [y_pos - 0.05, y_pos - 0.05], 
                           color=color, linewidth=2, alpha=0.6, zorder=2)
                
                # Add label if enabled
                if show_labels and len(class_events) <= 20:  # Only label if not too many
                    label = classification
                    if 'frequency_hz' in event.metadata:
                        freq = event.metadata['frequency_hz']
                        label += f"\n{freq:.0f}Hz"
                    ax.text(event.time, y_pos + 0.08, label, fontsize=7, 
                           rotation=45, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    alpha=0.7, edgecolor='none'))
            
            y_positions[classification] = y_pos
            y_offset += 1
        
        # Set y-axis with classification labels
        if y_positions:
            max_y = max(y_positions.values())
            ax.set_ylim(-0.2, max_y + 0.3)
            y_ticks = list(y_positions.values())
            y_labels = [f"{cls}\n({len(events_by_class[cls])})" 
                       for cls in classifications]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=8)
        else:
            ax.set_ylim(-0.1, 1.5)
            ax.set_yticks([])
        
        # Set x-axis limits with proper padding (scaled to data)
        if time_range > 0:
            padding = max(0.1, time_range * 0.05)  # 5% padding or 0.1s minimum
        else:
            padding = 0.1
        
        x_min = max(0, min_time - padding)
        x_max = max_time + padding
        
        # Ensure we have a valid range
        if x_max <= x_min:
            x_max = x_min + 1.0  # Default 1 second range if invalid
        
        ax.set_xlim(x_min, x_max)
        
        # Log for debugging
        Log.debug(f"PlotEventsPanel: Markers x-axis limits: [{x_min:.3f}, {x_max:.3f}], range: {x_max - x_min:.3f}s")
    
    def _plot_piano_roll_preview(self, ax, events, color_map, show_labels, min_time=None, max_time=None, time_range=None):
        """Plot events as piano roll (pitch vs time)"""
        import matplotlib.patches as mpl_patches
        import numpy as np
        
        if not events:
            return
        
        # Calculate time range if not provided
        if min_time is None or max_time is None or time_range is None:
            times = [e.time for e in events]
            durations = [e.duration for e in events]
            min_time = min(times)
            max_time = max(t + d for t, d in zip(times, durations))
            time_range = max_time - min_time
        
        # Extract MIDI notes or assign based on classification
        midi_notes = []
        valid_events = []
        
        for event in events:
            midi_note = event.metadata.get('midi_note')
            if midi_note is None:
                # Try to parse from classification (e.g., "C4" -> 60)
                try:
                    midi_note = self._note_name_to_midi(event.classification)
                except:
                    # If no MIDI note and can't parse, assign sequential numbers based on classification
                    classification = event.classification if event.classification else "Event"
                    # Use hash of classification to get consistent note assignment
                    midi_note = 60 + (hash(classification) % 24) - 12  # Range: 48-72 (C3-C5)
            
            midi_notes.append(midi_note)
            valid_events.append((event, midi_note))
        
        if not valid_events:
            return
        
        # Limit for preview
        max_events = 100
        if len(valid_events) > max_events:
            valid_events = valid_events[:max_events]
            midi_notes = midi_notes[:max_events]
        
        # Plot events
        for event, midi_note in valid_events:
            color = color_map.get(event.classification, 'steelblue') if color_map else 'steelblue'
            
            if event.duration > 0:
                # Draw note as rectangle
                rect = mpl_patches.Rectangle((event.time, midi_note - 0.4), 
                                        event.duration, 0.8,
                                        facecolor=color, alpha=0.7, 
                                        edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
            else:
                # Draw marker for instantaneous events
                ax.plot(event.time, midi_note, 'o', color=color, markersize=7,
                       markeredgecolor='black', markeredgewidth=0.5, zorder=3)
            
            # Add label if enabled and not too many events
            if show_labels and len(valid_events) <= 30:
                label = event.classification if event.classification else f"M{midi_note}"
                if 'frequency_hz' in event.metadata:
                    freq = event.metadata['frequency_hz']
                    label += f"\n{freq:.0f}Hz"
                ax.text(event.time, midi_note + 0.6, label, fontsize=6, 
                       rotation=45, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.7, edgecolor='none'))
        
        # Set y-axis to MIDI note range
        if midi_notes:
            min_note = max(0, min(midi_notes) - 2)
            max_note = min(127, max(midi_notes) + 2)
            ax.set_ylim(min_note, max_note)
            
            # Add note name labels on y-axis (limit to reasonable number)
            note_range = max_note - min_note
            if note_range <= 24:  # 2 octaves
                step = 1
            elif note_range <= 48:  # 4 octaves
                step = 2
            else:
                step = 4
            
            note_ticks = range(int(min_note), int(max_note) + 1, step)
            note_labels = [self._midi_to_note_name(n) for n in note_ticks]
            ax.set_yticks(note_ticks)
            ax.set_yticklabels(note_labels, fontsize=8)
        else:
            ax.set_ylim(48, 72)  # Default to one octave around middle C
            ax.set_yticks([])
        
        ax.set_ylabel('Pitch (MIDI Note)', fontsize=9)
        
        # Set x-axis limits with proper padding (scaled to data)
        if time_range > 0:
            padding = max(0.1, time_range * 0.05)  # 5% padding or 0.1s minimum
        else:
            padding = 0.1
        
        x_min = max(0, min_time - padding)
        x_max = max_time + padding
        
        # Ensure we have a valid range
        if x_max <= x_min:
            x_max = x_min + 1.0  # Default 1 second range if invalid
        
        ax.set_xlim(x_min, x_max)
        
        # Log for debugging
        Log.debug(f"PlotEventsPanel: Piano roll x-axis limits: [{x_min:.3f}, {x_max:.3f}], range: {x_max - x_min:.3f}s")
    
    def _note_name_to_midi(self, note_name: str) -> int:
        """Convert note name to MIDI number (e.g., 'C4' -> 60)"""
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                   'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        # Parse note name (e.g., "C#4" or "C4")
        if len(note_name) >= 2:
            if note_name[1] == '#':
                note = note_name[:2]
                octave = int(note_name[2:])
            else:
                note = note_name[0]
                octave = int(note_name[1:])
            
            return (octave + 1) * 12 + note_map.get(note, 0)
        
        return 60  # Default to middle C
    
    def _midi_to_note_name(self, midi_number: int) -> str:
        """Convert MIDI number to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_number // 12) - 1
        note = note_names[midi_number % 12]
        return f"{note}{octave}"
    
    def cleanup(self):
        """Clean up resources - called when panel is closed"""
        # Stop refresh timer
        if hasattr(self, '_refresh_timer'):
            self._refresh_timer.stop()
        
        super().cleanup() if hasattr(super(), 'cleanup') else None
    

