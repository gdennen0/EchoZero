"""
Input Filter Widget

Simple widget for filtering block input data items by expected outputs.
Uses processor.get_expected_outputs() as the single source of truth.
"""
from typing import List, Optional, Dict
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QShowEvent

from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem
from src.shared.domain.data_state import DataState
from src.shared.application.services.data_filter_manager import DataFilterManager
from src.application.settings.data_filter_settings import BlockDataFilterSettingsManager
from src.application.api.application_facade import ApplicationFacade
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from src.utils.message import Log


class InputFilterWidget(QWidget):
    """
    Simple widget for filtering input data items by expected outputs.
    
    Single source of truth: processor.get_expected_outputs(source_block)
    Displays checkboxes for each expected output name from the source block.
    """
    
    selection_changed = pyqtSignal(list)  # Emits list of selected output names
    
    def __init__(
        self,
        block: Block,
        port_name: str,
        facade: ApplicationFacade,
        data_filter_manager: DataFilterManager,
        data_state_service=None,
        parent=None
    ):
        super().__init__(parent)
        
        self.block = block
        self.port_name = port_name
        self.facade = facade
        self._filter_manager = data_filter_manager
        self._data_state_service = data_state_service
        self._settings_manager = BlockDataFilterSettingsManager(facade, block.id, parent=self)
        
        self._item_checkboxes: dict[str, QCheckBox] = {}
        self._items_by_output_name: dict[str, DataItem] = {}
        self._source_sections: Dict[str, QWidget] = {}
        self._previewed_once = False
        
        self._setup_ui()
        QTimer.singleShot(0, self._load_preview)
    
    def _setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Info label
        self.info_label = QLabel("Loading...")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 10pt;")
        layout.addWidget(self.info_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Refresh")
        self.preview_button.clicked.connect(self._load_preview)
        button_layout.addWidget(self.preview_button)
        
        if self._data_state_service:
            self.status_label = QLabel("")
            self.status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;")
            button_layout.addWidget(self.status_label)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Selection label
        selection_label = QLabel("Select inputs to process:")
        selection_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
        layout.addWidget(selection_label)
        
        # Scrollable container for source sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setStyleSheet(
            f"background-color: {Colors.BG_MEDIUM.name()}; "
            f"border: 1px solid {Colors.BORDER.name()}; "
            f"border-radius: {border_radius(4)};"
        )
        
        self.checkbox_container = QWidget()
        # Explicit background to prevent flash during refresh
        self.checkbox_container.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()};")
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setSpacing(Spacing.SM)
        self.checkbox_layout.setContentsMargins(8, 8, 8, 8)
        
        scroll_area.setWidget(self.checkbox_container)
        layout.addWidget(scroll_area)
        
        # Store source sections for easy access
        self._source_sections: Dict[str, QWidget] = {}
        
        # Select All / Deselect All
        select_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._on_select_all)
        select_all_btn.setMaximumWidth(100)
        select_buttons.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._on_deselect_all)
        deselect_all_btn.setMaximumWidth(100)
        select_buttons.addWidget(deselect_all_btn)
        select_buttons.addStretch()
        layout.addLayout(select_buttons)
    
    def _load_preview(self):
        """Load preview - simplified and fast"""
        try:
            # Get source block info (handles multiple connections)
            source_info = self._filter_manager.get_source_block_info(
                self.block, self.port_name, facade=self.facade
            )
            
            if not source_info:
                self._show_no_outputs_message()
                return
            
            # Get source blocks info
            source_blocks = source_info.get('source_blocks', [])
            if not source_blocks:
                # Backward compatibility: single source format
                expected_outputs = source_info.get('expected_outputs', [])
                if not expected_outputs:
                    self._show_no_outputs_message()
                    return
                self._load_single_source_preview_simple(source_info, expected_outputs)
                return
            
            # Collect all expected outputs
            all_expected_names = []
            for source_block_info in source_blocks:
                expected_outputs = source_block_info.get('expected_outputs', [])
                if expected_outputs and expected_outputs != []:
                    all_expected_names.extend(expected_outputs)
            
            if not all_expected_names:
                self._show_no_outputs_message()
                return
            
            # Get saved filter state (dict: {name: bool})
            saved_filter = self._settings_manager.get_port_selection(self.port_name)
            
            # Merge saved filter with all expected outputs (missing outputs default to True)
            # This ensures that if new expected outputs appear, they're enabled by default
            merged_filter = {}
            for name in all_expected_names:
                # Use saved value if exists, otherwise default to True (enabled)
                merged_filter[name] = saved_filter.get(name, True) if saved_filter else True
            
            # If we added new outputs (not in saved filter), save the merged filter
            if saved_filter is None or len(merged_filter) > len(saved_filter):
                self._settings_manager.set_port_selection(self.port_name, merged_filter)
                saved_filter = merged_filter
            elif saved_filter:
                # Use saved filter as-is (it already has all expected outputs)
                saved_filter = saved_filter
            
            # Update info label
            self._update_info_label_from_sources(source_blocks)
            
            # Clear existing sections
            self._clear_checkboxes()
            
            # Create separate section for each source block
            Log.info(f"InputFilterWidget: Creating sections for {len(source_blocks)} source block(s)")
            for idx, source_block_info in enumerate(source_blocks):
                source_block_id = source_block_info.get('source_block_id')
                source_block_name = source_block_info.get('source_block_name', 'Unknown')
                source_output_name = source_block_info.get('source_output_name', '')
                expected_outputs = source_block_info.get('expected_outputs', [])
                
                Log.info(
                    f"InputFilterWidget: Processing source[{idx}]: '{source_block_name}' "
                    f"(ID: {source_block_id}, output: {source_output_name}), "
                    f"expected_outputs: {expected_outputs} (type: {type(expected_outputs).__name__}, len: {len(expected_outputs) if expected_outputs else 0})"
                )
                
                if expected_outputs == []:
                    # Upstream block has all filters disabled
                    Log.info(f"InputFilterWidget: Source[{idx}] '{source_block_name}' has empty expected_outputs (all disabled) - creating empty section")
                    section_key = f"{source_block_id}:{source_output_name}"
                    section_widget = self._create_empty_filter_section(
                        source_block_name,
                        source_output_name
                    )
                    self.checkbox_layout.addWidget(section_widget)
                    self._source_sections[section_key] = section_widget
                    Log.info(f"InputFilterWidget: Added empty section for source[{idx}] '{source_block_name}'")
                    continue
                elif not expected_outputs:
                    Log.info(f"InputFilterWidget: Source[{idx}] '{source_block_name}' has no expected_outputs (None/empty) - skipping")
                    continue
                
                # Create section for this source
                Log.info(f"InputFilterWidget: Creating section for source[{idx}] '{source_block_name}' with {len(expected_outputs)} expected output(s)")
                section_key = f"{source_block_id}:{source_output_name}"
                section_widget = self._create_source_section_simple(
                    source_block_name,
                    source_output_name,
                    expected_outputs,
                    saved_filter
                )
                
                self.checkbox_layout.addWidget(section_widget)
                self._source_sections[section_key] = section_widget
                Log.info(f"InputFilterWidget: Added section for source[{idx}] '{source_block_name}' (section_key: {section_key})")
            
            Log.info(f"InputFilterWidget: Created {len(self._source_sections)} section(s) total, {len(self._item_checkboxes)} checkbox(es)")
            
            self.checkbox_layout.addStretch()
            self._previewed_once = True
            
            # Update status indicator
            if self._data_state_service:
                state = self._data_state_service.get_port_data_state(
                    self.block.id,
                    self.port_name,
                    is_input=True
                )
                self._update_status_indicator(state)
            
        except Exception as e:
            Log.error(f"InputFilterWidget: Error loading preview: {e}")
            import traceback
            traceback.print_exc()
            self._show_error(str(e))
    
    def _load_single_source_preview_simple(self, source_info: Dict, expected_outputs: List[str]):
        """Load preview for single source - simplified"""
        source_output_name = source_info.get('source_output_name', '')
        source_name = source_info.get('source_block_name', 'Unknown')
        
        # Get saved filter state
        saved_filter = self._settings_manager.get_port_selection(self.port_name)
        
        # If no saved filter, default all to True
        if saved_filter is None:
            saved_filter = {name: True for name in expected_outputs}
            self._settings_manager.set_port_selection(self.port_name, saved_filter)
        
        # Update info label
        self.info_label.setText(
            f"Filtering from '{source_name}' ({source_output_name} port).\n"
            f"Unchecked items will be skipped."
        )
        
        # Create checkboxes with saved state
        for output_name in sorted(expected_outputs):
            enabled = saved_filter.get(output_name, True)  # Default True if not in filter
            self._create_checkbox_simple(output_name, enabled)
    
    def _create_source_section_simple(
        self,
        source_block_name: str,
        source_output_name: str,
        expected_outputs: List[str],
        saved_filter: Dict[str, bool]
    ) -> QWidget:
        """Create a separate section for a source block's outputs - simplified"""
        # Group box for this source
        group = QGroupBox(f"{source_block_name} ({source_output_name} port)")
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {Colors.TEXT_SECONDARY.name()};
                font-weight: 600;
            }}
        """)
        
        section_layout = QVBoxLayout(group)
        section_layout.setSpacing(Spacing.XS)
        section_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create checkboxes for this source's outputs
        for output_name in sorted(expected_outputs):
            enabled = saved_filter.get(output_name, True)  # Default True if not in filter
            checkbox = self._create_checkbox_for_section_simple(
                output_name,
                enabled,
                section_layout
            )
        
        return group
    
    def _create_empty_filter_section(
        self,
        source_block_name: str,
        source_output_name: str
    ) -> QWidget:
        """Create a section indicating that upstream block has all filters disabled"""
        # Group box for this source
        group = QGroupBox(f"{source_block_name} ({source_output_name} port)")
        group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {Colors.ACCENT_ORANGE.name()};
                border-radius: {border_radius(4)};
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {Colors.ACCENT_ORANGE.name()};
                font-weight: 600;
            }}
        """)
        
        section_layout = QVBoxLayout(group)
        section_layout.setSpacing(Spacing.XS)
        section_layout.setContentsMargins(8, 8, 8, 8)
        
        # Message indicating all filters are disabled upstream
        message_label = QLabel(
            f"No outputs available.\n"
            f"Upstream block '{source_block_name}' has all filters disabled."
        )
        message_label.setWordWrap(True)
        message_label.setStyleSheet(
            f"color: {Colors.ACCENT_ORANGE.name()}; "
            f"font-style: italic; "
            f"padding: 8px;"
        )
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        section_layout.addWidget(message_label)
        
        return group
    
    def _create_checkbox_for_section_simple(
        self,
        output_name: str,
        enabled: bool,
        layout: QVBoxLayout
    ) -> QCheckBox:
        """Create checkbox and add to layout - simplified"""
        # Extract item name from output_name (e.g., "bass" from "audio:bass")
        if ':' in output_name:
            _, item_name = output_name.split(':', 1)
            label = item_name.replace('_', ' ').title()
        else:
            label = output_name
        
        # Note: If multiple sources have the same output_name, we create separate checkboxes
        # but they share the same key in _item_checkboxes (last one wins for state tracking)
        # This is OK because the filter state is per output_name, not per source
        if output_name in self._item_checkboxes:
            Log.debug(f"InputFilterWidget: output_name '{output_name}' already has a checkbox (multiple sources) - creating new one for this section")
        
        # Create checkbox (even if duplicate output_name - each source section gets its own)
        checkbox = QCheckBox(label)
        # Block signals while setting initial state
        checkbox.blockSignals(True)
        checkbox.setChecked(enabled)
        checkbox.blockSignals(False)
        checkbox.stateChanged.connect(self._on_checkbox_changed)
        
        # Tooltip
        checkbox.setToolTip(f"Output: {output_name}")
        
        # Store checkbox (will overwrite if duplicate, but that's OK - we track by output_name)
        self._item_checkboxes[output_name] = checkbox
        layout.addWidget(checkbox)
        
        return checkbox
    
    def _create_checkbox_simple(self, output_name: str, enabled: bool):
        """Create checkbox for single source - simplified"""
        self._create_checkbox_for_section_simple(output_name, enabled, self.checkbox_layout)
    
    
    def _update_info_label_from_sources(self, source_blocks: List[Dict]):
        """Update info label with source block information"""
        if not source_blocks:
            self.info_label.setText(f"No connection to '{self.port_name}' port.")
            return
        
        if len(source_blocks) == 1:
            source_block_info = source_blocks[0]
            source_name = source_block_info.get('source_block_name', 'Unknown')
            port = source_block_info.get('source_output_name', '')
            self.info_label.setText(
                f"Filtering from '{source_name}' ({port} port).\n"
                f"Unchecked items will be skipped."
            )
        else:
            # Multiple sources
            source_names = [sb.get('source_block_name', 'Unknown') for sb in source_blocks]
            self.info_label.setText(
                f"Filtering from {len(source_blocks)} source(s): {', '.join(source_names)}\n"
                f"Each source shown in separate section below.\n"
                f"Unchecked items will be skipped."
            )
    
    def _create_checkbox(self, output_name: str, item: Optional[DataItem], checked: bool):
        """Create checkbox for an expected output (used by single source preview)"""
        self._create_checkbox_for_section(output_name, item, checked, self.checkbox_layout)
    
    def _clear_checkboxes(self):
        """Clear all checkboxes and source sections"""
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._item_checkboxes.clear()
        self._items_by_output_name.clear()
        self._source_sections.clear()
    
    def _show_no_outputs_message(self):
        """Show message when no expected outputs"""
        label = QLabel(
            f"No expected outputs for this port.\n"
            f"Connect a source block to see outputs."
        )
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding: 8px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.checkbox_layout.addWidget(label)
        self.info_label.setText("No expected outputs defined.")
    
    def _show_error(self, error_msg: str):
        """Show error message"""
        label = QLabel(f"Error: {error_msg}")
        label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()};")
        self.checkbox_layout.addWidget(label)
    
    def _update_status_indicator(self, state: DataState):
        """Update status indicator with data state"""
        if not hasattr(self, 'status_label'):
            return
        
        status_text = f"Status: {state.display_name}"
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"color: {state.color}; font-size: 9pt;")
        self.status_label.setToolTip(f"Data state: {state.display_name}")
    
    def _on_select_all(self):
        """Select all checkboxes"""
        # Block signals to batch the changes
        for checkbox in self._item_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        # Only trigger one save/recalculation for all changes
        self._on_checkbox_changed()
    
    def _on_deselect_all(self):
        """Deselect all checkboxes"""
        # Block signals to batch the changes
        for checkbox in self._item_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        # Only trigger one save/recalculation for all changes
        self._on_checkbox_changed()
    
    def _on_checkbox_changed(self):
        """Handle checkbox change - save filter state"""
        # Build dict with current checkbox states
        # Only save states for checkboxes that exist (user has interacted with)
        filter_state = {
            name: checkbox.isChecked()
            for name, checkbox in self._item_checkboxes.items()
        }
        
        # Get current saved filter to preserve any outputs not in current checkboxes
        # (e.g., if expected outputs changed but user hasn't seen new checkboxes yet)
        current_saved = self._settings_manager.get_port_selection(self.port_name)
        if current_saved:
            # Merge: update with current checkbox states, preserve others
            merged_state = dict(current_saved)
            merged_state.update(filter_state)
            filter_state = merged_state
        
        self._settings_manager.set_port_selection(self.port_name, filter_state)
        
        # Emit list of enabled names for backward compatibility
        enabled_names = [name for name, enabled in filter_state.items() if enabled]
        self.selection_changed.emit(enabled_names)
    
    def get_selected_output_names(self) -> List[str]:
        """Get selected output names (for backward compatibility)"""
        return [
            name for name, checkbox in self._item_checkboxes.items()
            if checkbox.isChecked()
        ]
    
    def get_filter_state(self) -> Dict[str, bool]:
        """Get filter state dict"""
        return {
            name: checkbox.isChecked()
            for name, checkbox in self._item_checkboxes.items()
        }
    
    def showEvent(self, event: QShowEvent):
        """Refresh when shown"""
        super().showEvent(event)
        if not self._previewed_once:
            self._load_preview()


# Backward compatibility alias
DataFilterWidget = InputFilterWidget
