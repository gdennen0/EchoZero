"""
Event Filter Dialog

Dialog for configuring event filters in the Editor block.
Part of the "processes" improvement area.

Allows filtering events by:
- Classification (include/exclude specific types)
- Time range (min/max time)
- Duration range (min/max duration)
- Metadata key-value pairs
"""
from typing import Optional, Set, Dict, Any, List, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QLineEdit, QDoubleSpinBox, QFormLayout,
    QScrollArea, QWidget, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.application.api.application_facade import ApplicationFacade
from src.shared.application.services.event_filter_manager import EventFilterManager, EventFilter
from src.application.settings.editor_event_filter_settings import EditorEventFilterSettingsManager
from src.features.blocks.domain import Block
from src.shared.domain.entities import EventDataItem
from ui.qt_gui.design_system import Colors, Spacing, Typography, ThemeAwareMixin, border_radius
from src.utils.message import Log


class EventFilterDialog(ThemeAwareMixin, QDialog):
    """
    Dialog for configuring event filters in the Editor block.
    """
    
    def __init__(self, block_id: str, facade: ApplicationFacade, parent=None):
        super().__init__(parent)
        self.block_id = block_id
        self.facade = facade
        
        # Initialize settings manager (follows settings abstraction pattern)
        self._settings_manager = EditorEventFilterSettingsManager(facade, block_id, parent=self)
        
        # Initialize filter manager with block getter function (for reading filters)
        self._filter_manager = EventFilterManager()
        def get_block(block_id: str) -> Optional[Block]:
            result = self.facade.describe_block(block_id)
            return result.data if result.success else None
        self._filter_manager.set_block_getter(get_block)
        
        # Get block info
        block_result = self.facade.describe_block(block_id)
        if not block_result.success or not block_result.data:
            Log.error(f"EventFilterDialog: Block {block_id} not found")
            self.reject()
            return
        
        self.block: Block = block_result.data
        
        # Load available classifications from current events
        self._available_classifications: Set[str] = set()
        self._load_available_classifications()
        
        # Load available metadata keys from current events
        self._available_metadata_keys: Dict[str, Set[Any]] = {}  # key -> set of example values
        self._load_available_metadata_keys()
        
        # UI components
        self._classification_checkboxes: Dict[str, QCheckBox] = {}
        self._excluded_classification_checkboxes: Dict[str, QCheckBox] = {}
        self._min_time_spinbox: Optional[QDoubleSpinBox] = None
        self._max_time_spinbox: Optional[QDoubleSpinBox] = None
        self._min_duration_spinbox: Optional[QDoubleSpinBox] = None
        self._max_duration_spinbox: Optional[QDoubleSpinBox] = None
        self._filter_enabled_checkbox: Optional[QCheckBox] = None
        self._metadata_filter_widgets: Dict[str, Dict[str, Any]] = {}  # key -> {checkbox, operator, value_widget}
        
        self._setup_ui()
        self._load_filter()
        self._init_theme_aware()
    
    def _load_available_classifications(self):
        """Load available event classifications from current event data"""
        try:
            # Get events from local state or connected inputs
            local_inputs = self.facade.block_local_state_repo.get_inputs(self.block_id) if self.facade.block_local_state_repo else {}
            event_ids = local_inputs.get("events", [])
            
            if not event_ids:
                # Try to get from connections
                connections_result = self.facade.list_connections()
                if connections_result.success:
                    for conn in connections_result.data:
                        if conn.target_block_id == self.block_id and conn.target_input_name == "events":
                            # Get source block's event outputs
                            source_items = self.facade.data_item_repo.list_by_block(conn.source_block_id) if self.facade.data_item_repo else []
                            for item in source_items:
                                if isinstance(item, EventDataItem):
                                    for event in item.get_events():
                                        if event.classification:
                                            self._available_classifications.add(event.classification)
            
            # Also check from local state event items
            if isinstance(event_ids, str):
                event_ids = [event_ids]
            
            for event_id in event_ids:
                if self.facade.data_item_repo:
                    item = self.facade.data_item_repo.get(event_id)
                    if isinstance(item, EventDataItem):
                        for event in item.get_events():
                            if event.classification:
                                self._available_classifications.add(event.classification)
        except Exception as e:
            Log.warning(f"EventFilterDialog: Failed to load classifications: {e}")
    
    def _load_available_metadata_keys(self):
        """Load available metadata keys from current event data"""
        try:
            # Get events from local state or connected inputs
            local_inputs = self.facade.block_local_state_repo.get_inputs(self.block_id) if self.facade.block_local_state_repo else {}
            event_ids = local_inputs.get("events", [])
            
            if not event_ids:
                # Try to get from connections
                connections_result = self.facade.list_connections()
                if connections_result.success:
                    for conn in connections_result.data:
                        if conn.target_block_id == self.block_id and conn.target_input_name == "events":
                            # Get source block's event outputs
                            source_items = self.facade.data_item_repo.list_by_block(conn.source_block_id) if self.facade.data_item_repo else []
                            for item in source_items:
                                if isinstance(item, EventDataItem):
                                    for event in item.get_events():
                                        if event.metadata:
                                            for key, value in event.metadata.items():
                                                if key not in self._available_metadata_keys:
                                                    self._available_metadata_keys[key] = set()
                                                # Store example value (limit to 10 examples per key)
                                                if len(self._available_metadata_keys[key]) < 10:
                                                    self._available_metadata_keys[key].add(value)
            
            # Also check from local state event items
            if isinstance(event_ids, str):
                event_ids = [event_ids]
            
            for event_id in event_ids:
                if self.facade.data_item_repo:
                    item = self.facade.data_item_repo.get(event_id)
                    if isinstance(item, EventDataItem):
                        for event in item.get_events():
                            if event.metadata:
                                for key, value in event.metadata.items():
                                    if key not in self._available_metadata_keys:
                                        self._available_metadata_keys[key] = set()
                                    # Store example value (limit to 10 examples per key)
                                    if len(self._available_metadata_keys[key]) < 10:
                                        self._available_metadata_keys[key].add(value)
        except Exception as e:
            Log.warning(f"EventFilterDialog: Failed to load metadata keys: {e}")
    
    def _setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle(f"Event Filtering: {self.block.name}")
        self.setMinimumSize(400, 500)
        self.resize(560, 640)
        
        # Apply dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(Spacing.SM)
        main_layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        # Header
        header_label = QLabel(f"Filter Events for: {self.block.name}")
        header_label.setFont(Typography.heading_font())
        header_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; padding-bottom: {Spacing.SM}px;")
        main_layout.addWidget(header_label)
        
        # Description
        desc_label = QLabel(
            "Configure filters to show/hide events in the timeline and block outputs. "
            "Filtered events are excluded from visualization and processing."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; padding-bottom: {Spacing.MD}px;")
        main_layout.addWidget(desc_label)
        
        # Scroll area for filter options
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"background-color: transparent; border: none;")
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(Spacing.SM)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Filter enabled checkbox
        self._filter_enabled_checkbox = QCheckBox("Enable Event Filtering")
        self._filter_enabled_checkbox.setChecked(True)
        self._filter_enabled_checkbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600; font-size: 12pt;")
        self._filter_enabled_checkbox.toggled.connect(self._on_filter_enabled_toggled)
        scroll_layout.addWidget(self._filter_enabled_checkbox)
        
        # Classification filter group
        classification_group = self._create_classification_group()
        scroll_layout.addWidget(classification_group)
        
        # Time range filter group
        time_group = self._create_time_range_group()
        scroll_layout.addWidget(time_group)
        
        # Duration filter group
        duration_group = self._create_duration_group()
        scroll_layout.addWidget(duration_group)
        
        # Metadata filter group
        metadata_group = self._create_metadata_group()
        scroll_layout.addWidget(metadata_group)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        _btn_style = f"""
            QPushButton {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                padding: 6px 16px;
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.name()};
            }}
        """
        _primary_btn_style = f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                padding: 6px 16px;
                border: none;
                border-radius: {border_radius(4)};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
        """

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._on_clear_filter)
        clear_button.setStyleSheet(_btn_style)
        button_layout.addWidget(clear_button)
        
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply)
        apply_button.setStyleSheet(_primary_btn_style)
        button_layout.addWidget(apply_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setStyleSheet(_btn_style)
        button_layout.addWidget(close_button)
        
        main_layout.addLayout(button_layout)
    
    def _group_box_style(self) -> str:
        """Shared compact group box stylesheet."""
        return f"""
            QGroupBox {{
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                margin-top: 6px;
                padding: 8px 6px 6px 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 6px;
                padding: 0 3px;
            }}
        """

    def _create_classification_group(self) -> QGroupBox:
        """Create classification filter group"""
        group = QGroupBox("Classification Filter")
        group.setStyleSheet(self._group_box_style())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(Spacing.XS)
        
        # Include classifications
        include_label = QLabel("Include (empty = all):")
        include_label.setWordWrap(True)
        include_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-weight: 600;")
        layout.addWidget(include_label)
        
        include_scroll = QScrollArea()
        include_scroll.setWidgetResizable(True)
        include_scroll.setMaximumHeight(120)
        include_scroll.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()}; border: 1px solid {Colors.BORDER.name()}; border-radius: {border_radius(4)};")
        
        include_container = QWidget()
        include_layout = QVBoxLayout(include_container)
        include_layout.setSpacing(2)
        include_layout.setContentsMargins(4, 4, 4, 4)
        
        if self._available_classifications:
            for classification in sorted(self._available_classifications):
                checkbox = QCheckBox(classification)
                checkbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
                include_layout.addWidget(checkbox)
                self._classification_checkboxes[classification] = checkbox
        else:
            no_class_label = QLabel("No classifications found in current events")
            no_class_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-style: italic;")
            include_layout.addWidget(no_class_label)
        
        include_scroll.setWidget(include_container)
        layout.addWidget(include_scroll)
        
        # Exclude classifications
        exclude_label = QLabel("Exclude:")
        exclude_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-weight: 600;")
        layout.addWidget(exclude_label)
        
        exclude_scroll = QScrollArea()
        exclude_scroll.setWidgetResizable(True)
        exclude_scroll.setMaximumHeight(120)
        exclude_scroll.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()}; border: 1px solid {Colors.BORDER.name()}; border-radius: {border_radius(4)};")
        
        exclude_container = QWidget()
        exclude_layout = QVBoxLayout(exclude_container)
        exclude_layout.setSpacing(2)
        exclude_layout.setContentsMargins(4, 4, 4, 4)
        
        if self._available_classifications:
            for classification in sorted(self._available_classifications):
                checkbox = QCheckBox(classification)
                checkbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
                exclude_layout.addWidget(checkbox)
                self._excluded_classification_checkboxes[classification] = checkbox
        else:
            no_class_label = QLabel("No classifications found in current events")
            no_class_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-style: italic;")
            exclude_layout.addWidget(no_class_label)
        
        exclude_scroll.setWidget(exclude_container)
        layout.addWidget(exclude_scroll)
        
        return group
    
    def _create_time_range_group(self) -> QGroupBox:
        """Create time range filter group"""
        group = QGroupBox("Time Range")
        group.setStyleSheet(self._group_box_style())
        
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(Spacing.XS)
        
        self._min_time_spinbox = QDoubleSpinBox()
        self._min_time_spinbox.setMinimumWidth(60)
        self._min_time_spinbox.setMinimum(0.0)
        self._min_time_spinbox.setMaximum(999999.0)
        self._min_time_spinbox.setDecimals(3)
        self._min_time_spinbox.setSuffix(" s")
        self._min_time_spinbox.setSpecialValueText("No limit")
        self._min_time_spinbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        layout.addRow("Min:", self._min_time_spinbox)
        
        self._max_time_spinbox = QDoubleSpinBox()
        self._max_time_spinbox.setMinimumWidth(60)
        self._max_time_spinbox.setMinimum(0.0)
        self._max_time_spinbox.setMaximum(999999.0)
        self._max_time_spinbox.setDecimals(3)
        self._max_time_spinbox.setSuffix(" s")
        self._max_time_spinbox.setSpecialValueText("No limit")
        self._max_time_spinbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        layout.addRow("Max:", self._max_time_spinbox)
        
        return group
    
    def _create_duration_group(self) -> QGroupBox:
        """Create duration filter group"""
        group = QGroupBox("Duration")
        group.setStyleSheet(self._group_box_style())
        
        layout = QFormLayout(group)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(Spacing.XS)
        
        self._min_duration_spinbox = QDoubleSpinBox()
        self._min_duration_spinbox.setMinimumWidth(60)
        self._min_duration_spinbox.setMinimum(0.0)
        self._min_duration_spinbox.setMaximum(999999.0)
        self._min_duration_spinbox.setDecimals(3)
        self._min_duration_spinbox.setSuffix(" s")
        self._min_duration_spinbox.setSpecialValueText("No limit")
        self._min_duration_spinbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        layout.addRow("Min:", self._min_duration_spinbox)
        
        self._max_duration_spinbox = QDoubleSpinBox()
        self._max_duration_spinbox.setMinimumWidth(60)
        self._max_duration_spinbox.setMinimum(0.0)
        self._max_duration_spinbox.setMaximum(999999.0)
        self._max_duration_spinbox.setDecimals(3)
        self._max_duration_spinbox.setSuffix(" s")
        self._max_duration_spinbox.setSpecialValueText("No limit")
        self._max_duration_spinbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        layout.addRow("Max:", self._max_duration_spinbox)
        
        return group
    
    def _create_metadata_group(self) -> QGroupBox:
        """Create metadata filter group"""
        group = QGroupBox("Metadata")
        group.setStyleSheet(self._group_box_style())
        
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(Spacing.XS)
        
        # Description
        desc_label = QLabel("Filter by metadata key-value pairs.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;")
        layout.addWidget(desc_label)
        
        # Scroll area for metadata filters
        metadata_scroll = QScrollArea()
        metadata_scroll.setWidgetResizable(True)
        metadata_scroll.setMaximumHeight(240)
        metadata_scroll.setStyleSheet(f"background-color: {Colors.BG_MEDIUM.name()}; border: 1px solid {Colors.BORDER.name()}; border-radius: {border_radius(4)};")
        
        metadata_container = QWidget()
        metadata_layout = QVBoxLayout(metadata_container)
        metadata_layout.setSpacing(Spacing.XS)
        metadata_layout.setContentsMargins(4, 4, 4, 4)
        
        if self._available_metadata_keys:
            for key in sorted(self._available_metadata_keys.keys()):
                # Get example values to determine value type
                example_values = list(self._available_metadata_keys[key])
                value_type = self._infer_value_type(example_values)
                
                # Create row widget for this metadata key
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(Spacing.XS)
                
                # Checkbox to enable filtering for this key
                checkbox = QCheckBox(key)
                checkbox.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: 600;")
                checkbox.toggled.connect(lambda checked, k=key: self._on_metadata_filter_toggled(k, checked))
                row_layout.addWidget(checkbox)
                
                # Operator dropdown
                operator_combo = QComboBox()
                operator_combo.addItems(["==", "!=", ">", "<", ">=", "<=", "contains", "not contains", "in", "not in"])
                operator_combo.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
                operator_combo.setEnabled(False)
                row_layout.addWidget(operator_combo)
                
                # Value input widget (type depends on value type)
                value_widget = self._create_metadata_value_widget(value_type)
                value_widget.setEnabled(False)
                row_layout.addWidget(value_widget, 1)
                
                # Type hint label
                type_hint = QLabel(f"({value_type})")
                type_hint.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 9pt;")
                row_layout.addWidget(type_hint)
                
                # Store widgets
                self._metadata_filter_widgets[key] = {
                    "checkbox": checkbox,
                    "operator": operator_combo,
                    "value_widget": value_widget,
                    "value_type": value_type
                }
                
                metadata_layout.addWidget(row_widget)
        else:
            no_metadata_label = QLabel("No metadata keys found in current events")
            no_metadata_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-style: italic; padding: 20px;")
            no_metadata_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metadata_layout.addWidget(no_metadata_label)
        
        metadata_scroll.setWidget(metadata_container)
        layout.addWidget(metadata_scroll)
        
        return group
    
    def _infer_value_type(self, example_values: List[Any]) -> str:
        """Infer the type of metadata values from examples"""
        if not example_values:
            return "string"
        
        # Check if all are numeric
        all_numeric = all(isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).replace('-', '', 1).isdigit()) for v in example_values)
        if all_numeric:
            return "number"
        
        # Check if all are booleans
        all_bool = all(isinstance(v, bool) for v in example_values)
        if all_bool:
            return "boolean"
        
        # Default to string
        return "string"
    
    def _create_metadata_value_widget(self, value_type: str) -> QWidget:
        """Create appropriate value input widget based on type"""
        if value_type == "number":
            widget = QDoubleSpinBox()
            widget.setMinimumWidth(50)
            widget.setMinimum(-999999.0)
            widget.setMaximum(999999.0)
            widget.setDecimals(3)
            widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        elif value_type == "boolean":
            widget = QComboBox()
            widget.setMinimumWidth(50)
            widget.addItems(["true", "false"])
            widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()};")
        else:  # string
            widget = QLineEdit()
            widget.setMinimumWidth(50)
            widget.setPlaceholderText("Value...")
            widget.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; background-color: {Colors.BG_LIGHT.name()}; padding: 3px;")
        
        return widget
    
    def _on_metadata_filter_toggled(self, key: str, enabled: bool):
        """Handle metadata filter checkbox toggle"""
        if key in self._metadata_filter_widgets:
            widgets = self._metadata_filter_widgets[key]
            widgets["operator"].setEnabled(enabled)
            widgets["value_widget"].setEnabled(enabled)
    
    def _on_filter_enabled_toggled(self, enabled: bool):
        """Handle filter enabled toggle"""
        # Enable/disable all filter controls
        for checkbox in self._classification_checkboxes.values():
            checkbox.setEnabled(enabled)
        for checkbox in self._excluded_classification_checkboxes.values():
            checkbox.setEnabled(enabled)
        if self._min_time_spinbox:
            self._min_time_spinbox.setEnabled(enabled)
        if self._max_time_spinbox:
            self._max_time_spinbox.setEnabled(enabled)
        if self._min_duration_spinbox:
            self._min_duration_spinbox.setEnabled(enabled)
        if self._max_duration_spinbox:
            self._max_duration_spinbox.setEnabled(enabled)
        
        # Enable/disable metadata filters
        for widgets in self._metadata_filter_widgets.values():
            widgets["checkbox"].setEnabled(enabled)
            if widgets["checkbox"].isChecked():
                widgets["operator"].setEnabled(enabled)
                widgets["value_widget"].setEnabled(enabled)
    
    def _load_filter(self):
        """Load existing filter from settings manager"""
        # Load from settings manager (single source of truth)
        filter_dict = self._settings_manager.get_event_filter_dict()
        if not filter_dict or not filter_dict.get("enabled", True):
            return
        
        try:
            filter = EventFilter.from_dict(filter_dict)
        except Exception as e:
            Log.warning(f"EventFilterDialog: Failed to load filter: {e}")
            return
        
        # Load enabled classifications
        if filter.enabled_classifications:
            for classification in filter.enabled_classifications:
                if classification in self._classification_checkboxes:
                    self._classification_checkboxes[classification].setChecked(True)
        
        # Load excluded classifications
        if filter.excluded_classifications:
            for classification in filter.excluded_classifications:
                if classification in self._excluded_classification_checkboxes:
                    self._excluded_classification_checkboxes[classification].setChecked(True)
        
        # Load time range
        if filter.min_time is not None and self._min_time_spinbox:
            self._min_time_spinbox.setValue(filter.min_time)
        if filter.max_time is not None and self._max_time_spinbox:
            self._max_time_spinbox.setValue(filter.max_time)
        
        # Load duration range
        if filter.min_duration is not None and self._min_duration_spinbox:
            self._min_duration_spinbox.setValue(filter.min_duration)
        if filter.max_duration is not None and self._max_duration_spinbox:
            self._max_duration_spinbox.setValue(filter.max_duration)
        
        # Load metadata filters
        if filter.metadata_filters:
            for key, filter_spec in filter.metadata_filters.items():
                if key in self._metadata_filter_widgets:
                    widgets = self._metadata_filter_widgets[key]
                    
                    # Handle new format with operator
                    if isinstance(filter_spec, dict) and "operator" in filter_spec:
                        operator = filter_spec.get("operator", "==")
                        value = filter_spec.get("value")
                        
                        widgets["checkbox"].setChecked(True)
                        widgets["operator"].setCurrentText(operator)
                        self._set_metadata_value_widget(widgets["value_widget"], value, widgets["value_type"])
                        widgets["operator"].setEnabled(True)
                        widgets["value_widget"].setEnabled(True)
                    else:
                        # Legacy format: exact match
                        widgets["checkbox"].setChecked(True)
                        widgets["operator"].setCurrentText("==")
                        self._set_metadata_value_widget(widgets["value_widget"], filter_spec, widgets["value_type"])
                        widgets["operator"].setEnabled(True)
                        widgets["value_widget"].setEnabled(True)
        
        # Load enabled state
        if self._filter_enabled_checkbox:
            self._filter_enabled_checkbox.setChecked(filter.enabled)
            self._on_filter_enabled_toggled(filter.enabled)
    
    def _set_metadata_value_widget(self, widget: QWidget, value: Any, value_type: str):
        """Set value on metadata value widget"""
        if isinstance(widget, QDoubleSpinBox):
            try:
                widget.setValue(float(value))
            except (ValueError, TypeError):
                widget.setValue(0.0)
        elif isinstance(widget, QComboBox):
            if value_type == "boolean":
                widget.setCurrentText("true" if value else "false")
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value) if value is not None else "")
    
    def _get_metadata_value_widget(self, widget: QWidget, value_type: str) -> Any:
        """Get value from metadata value widget"""
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QComboBox):
            if value_type == "boolean":
                return widget.currentText() == "true"
            return widget.currentText()
        elif isinstance(widget, QLineEdit):
            return widget.text()
        return None
    
    def _on_clear_filter(self):
        """Clear all filter settings"""
        # Clear checkboxes
        for checkbox in self._classification_checkboxes.values():
            checkbox.setChecked(False)
        for checkbox in self._excluded_classification_checkboxes.values():
            checkbox.setChecked(False)
        
        # Clear spinboxes
        if self._min_time_spinbox:
            self._min_time_spinbox.setValue(0.0)
        if self._max_time_spinbox:
            self._max_time_spinbox.setValue(0.0)
        if self._min_duration_spinbox:
            self._min_duration_spinbox.setValue(0.0)
        if self._max_duration_spinbox:
            self._max_duration_spinbox.setValue(0.0)
        
        # Clear metadata filters
        for widgets in self._metadata_filter_widgets.values():
            widgets["checkbox"].setChecked(False)
            widgets["operator"].setCurrentIndex(0)
            widgets["operator"].setEnabled(False)
            if isinstance(widgets["value_widget"], QDoubleSpinBox):
                widgets["value_widget"].setValue(0.0)
            elif isinstance(widgets["value_widget"], QComboBox):
                widgets["value_widget"].setCurrentIndex(0)
            elif isinstance(widgets["value_widget"], QLineEdit):
                widgets["value_widget"].clear()
            widgets["value_widget"].setEnabled(False)
        
        # Disable filter
        if self._filter_enabled_checkbox:
            self._filter_enabled_checkbox.setChecked(False)
    
    def _on_apply(self):
        """Apply filter settings via settings manager"""
        if not self._filter_enabled_checkbox or not self._filter_enabled_checkbox.isChecked():
            # Filter disabled - reset to defaults via settings manager
            self._settings_manager.reset_to_defaults()
            self.accept()
            return
        
        # Build filter from UI
        enabled_classifications_set = {
            classification for classification, checkbox in self._classification_checkboxes.items()
            if checkbox.isChecked()
        } if self._classification_checkboxes else set()
        # If no classifications selected, pass None to mean "include all"
        enabled_classifications = enabled_classifications_set if enabled_classifications_set else None
        
        excluded_classifications_set = {
            classification for classification, checkbox in self._excluded_classification_checkboxes.items()
            if checkbox.isChecked()
        } if self._excluded_classification_checkboxes else set()
        # Empty set is fine for excluded (means exclude none)
        excluded_classifications = excluded_classifications_set if excluded_classifications_set else None
        
        min_time = self._min_time_spinbox.value() if self._min_time_spinbox and self._min_time_spinbox.value() > 0 else None
        max_time = self._max_time_spinbox.value() if self._max_time_spinbox and self._max_time_spinbox.value() > 0 else None
        min_duration = self._min_duration_spinbox.value() if self._min_duration_spinbox and self._min_duration_spinbox.value() > 0 else None
        max_duration = self._max_duration_spinbox.value() if self._max_duration_spinbox and self._max_duration_spinbox.value() > 0 else None
        
        # Build metadata filters with operators
        metadata_filters = {}
        for key, widgets in self._metadata_filter_widgets.items():
            if widgets["checkbox"].isChecked():
                operator = widgets["operator"].currentText()
                value = self._get_metadata_value_widget(widgets["value_widget"], widgets["value_type"])
                
                # Convert "not contains" to "not_contains" for internal format
                if operator == "not contains":
                    operator = "not_contains"
                elif operator == "not in":
                    operator = "not_in"
                
                # Store in new format with operator
                metadata_filters[key] = {
                    "operator": operator,
                    "value": value
                }
        
        # Validate time range
        if min_time is not None and max_time is not None and min_time > max_time:
            QMessageBox.warning(self, "Invalid Filter", "Minimum time cannot be greater than maximum time.")
            return
        
        # Validate duration range
        if min_duration is not None and max_duration is not None and min_duration > max_duration:
            QMessageBox.warning(self, "Invalid Filter", "Minimum duration cannot be greater than maximum duration.")
            return
        
        # Create filter dict
        filter_dict = {
            "enabled_classifications": enabled_classifications if enabled_classifications else None,
            "excluded_classifications": excluded_classifications if excluded_classifications else None,
            "min_time": min_time,
            "max_time": max_time,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "metadata_filters": metadata_filters if metadata_filters else {},
            "enabled": True
        }
        
        # Save via settings manager (follows settings abstraction pattern)
        # This automatically handles persistence, undo support, and events
        self._settings_manager.set_from_event_filter_dict(filter_dict)
        
        # Publish event to refresh UI (settings manager may not emit BlockUpdated)
        if self.facade.event_bus:
            from src.application.events import BlockUpdated
            self.facade.event_bus.publish(BlockUpdated(
                project_id=self.facade.current_project_id,
                data={"id": self.block.id, "name": self.block.name, "type": self.block.type}
            ))
        
        self.accept()

