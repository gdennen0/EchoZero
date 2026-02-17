"""
ShowManager block panel.

Provides UI for MA3 connection management, sync controls, and testing.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QGroupBox, QLineEdit, QSpinBox, QComboBox,
    QCheckBox, QFrame, QTextEdit, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QTabWidget, QDoubleSpinBox, QInputDialog, QDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QBrush
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from ui.qt_gui.block_panels.block_panel_base import BlockPanelBase
from ui.qt_gui.block_panels.panel_registry import register_block_panel
from ui.qt_gui.design_system import Colors, Spacing, border_radius
from ui.qt_gui.widgets.block_status_dot import BlockStatusDot
from src.application.settings.show_manager_settings import (
    ShowManagerSettingsManager, SafetyLevel, MappingTemplate
)
from src.utils.message import Log

from src.application.events.events import MA3OscOutbound, MA3OscInbound
from src.features.ma3.domain.ma3_event import MA3Event
from src.features.ma3.domain.osc_message import OSCMessage, MessageType, ChangeType
from src.features.ma3.infrastructure.osc_parser import get_osc_parser
from src.features.ma3.application.ma3_layer_mapping_service import (
    MA3LayerMappingService, MA3TrackInfo,
)
from src.features.show_manager.application.sync_system_manager import SyncSystemManager
from src.features.show_manager.domain import (
    SyncLayerEntity, SyncSource, SyncStatus,
)
import json
import time

class RefreshableComboBox(QComboBox):
    """QComboBox that calls a refresh callback before showing the dropdown."""
    
    def __init__(self, refresh_callback=None, parent=None):
        super().__init__(parent)
        self._refresh_callback = refresh_callback
    
    def showPopup(self):
        """Refresh items before showing the dropdown."""
        if self._refresh_callback:
            self._refresh_callback()
        super().showPopup()

def _compute_events_checksum(events: List[Dict[str, Any]]) -> int:
    total = 0
    for i, evt in enumerate(events or [], start=1):
        try:
            time_val = float(evt.get("time", 0.0))
        except (ValueError, TypeError):
            time_val = 0.0
        idx_val = evt.get("idx", i)
        try:
            idx_int = int(idx_val)
        except (ValueError, TypeError):
            idx_int = i
        total += int(round(time_val * 1000.0)) + idx_int
    return total

def _merge_event_chunks(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for _, chunk in sorted(entry.get("chunks", {}).items(), key=lambda item: item[1]["offset"]):
        merged.extend(chunk.get("events") or [])
    return merged

@register_block_panel("ShowManager")
class ShowManagerPanel(BlockPanelBase):
    """Panel for ShowManager block - MA3 integration control center"""
    
    def __init__(self, block_id: str, facade, parent=None):
        self._reconcile_cache = {}
        super().__init__(block_id, facade, parent)
        self._pending_controller_init = False
        
        # Use the facade's cached settings manager so the Editor timeline's
        # sync-icon lookup (_check_live_sync_state) reads the SAME instance
        # that the SyncSystemManager writes to.  If the cache has no entry yet
        # we create one and register it; if it already exists we reuse it.
        cached = facade._show_manager_settings_cache.get(block_id)
        if cached:
            self._settings_manager = cached
        else:
            self._settings_manager = ShowManagerSettingsManager(facade, block_id)
            facade._show_manager_settings_cache[block_id] = self._settings_manager
        self._settings_manager.settings_changed.connect(self._on_setting_changed)
        # Connect to settings_loaded signal to refresh UI when settings are loaded from database
        self._settings_manager.settings_loaded.connect(self._on_settings_loaded)
        
        # Connection state (MA3 connection is tracked by SSM; panel reads via SSM property)
        self._was_fully_connected = False  # Track previous connection state for UI updates
        self._sync_list_updating: bool = False  # Guard flag for sync list updates
        
        # Reconcile state (must be set before _load_connection_state_from_service
        # which can trigger _start_sync_layer_reconcile)
        self._reconcile_in_progress = False
        self._reconcile_pending: Dict[str, Dict[str, Any]] = {}
        self._reconcile_results: List[Dict[str, Any]] = []
        self._reconcile_dialog_active = False
        self._reconcile_request_queue: List[str] = []
        self._reconcile_active_key: Optional[str] = None
        
        # Get ShowManager listener service (manages listener independently of panel lifecycle)
        self._listener_service = getattr(facade, 'show_manager_listener_service', None)
        
        # Get ShowManager state service (manages connection state independently of panel lifecycle)
        self._state_service = getattr(facade, 'show_manager_state_service', None)
        if self._state_service:
            # Connect to state service signals to update UI when state changes
            self._state_service.connection_state_changed.connect(self._on_connection_state_changed)
            # Load initial state
            self._load_connection_state_from_service()

        if hasattr(facade, "event_bus") and facade.event_bus:
            facade.event_bus.subscribe(MA3OscOutbound, self._on_ma3_osc_outbound)
            facade.event_bus.subscribe(MA3OscInbound, self._on_ma3_osc_inbound)
        
        # Register panel state provider with BlockStatusService
        if hasattr(facade, 'block_status_service') and facade.block_status_service:
            facade.block_status_service.register_panel_state_provider(block_id, self)
        
        # MA3 events cache
        self._ma3_events = []  # List of MA3Event objects
        
        # Guard flag to prevent auto-fetch loops during batch operations
        self._suppress_auto_fetch = False
        
        # Track last processed structure to avoid duplicate processing
        self._last_structure_signature = None
        
        # Pending actions (event-driven callbacks)
        self._pending_ma3_track_dialog = False  # Open dialog when structure is received
        self._pending_editor_track_select = None  # {"editor_layer_id": str}
        self._pending_track_groups_count = 0  # How many track groups we're waiting for
        self._received_track_groups_count = 0  # How many track groups we've received
        self._tracks_need_fetch = False  # Track groups loaded but tracks need fetching
        
        # Chunked message reassembly
        self._chunked_message_buffer = {}  # {message_id: {"chunks": {}, "total": 0, "received": 0, "started_at": timestamp, "chunk_indices_received": set()}}
        
        # Cleanup timer for incomplete chunked messages (5 second timeout)
        self._chunk_cleanup_timer = QTimer(self)
        self._chunk_cleanup_timer.timeout.connect(self._cleanup_incomplete_chunks)
        self._chunk_cleanup_timer.start(2000)  # Check every 2 seconds
        
        # Polling timer for change detection
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._on_poll_changes)
        self._polling_enabled = False
        
        # Connection health monitor (listener auto-restart and display updates)
        self._connection_monitor_timer = QTimer(self)
        self._connection_monitor_timer.timeout.connect(self._check_connection_health)
        self._connection_monitor_timer.start(10000)  # Check every 10 seconds
        self._listener_restart_cooldown_s = 5.0
        self._last_listener_restart_ts = 0.0
        
        # Connection monitoring is fully managed by SyncSystemManager.
        # Panel subscribes to ssm.connection_state_changed for UI updates.
        
        # Layer mapping service
        self._mapping_service = MA3LayerMappingService()
        self._ma3_tracks: List[MA3TrackInfo] = []
        self._ez_layers: List[str] = ["unassigned"]  # Initialize with default "unassigned" layer
        self._pending_track_create: Optional[Dict[str, Any]] = None
        self._pending_sequence_by_editor: Dict[str, int] = {}
        self._pending_track_group_by_editor: Dict[str, int] = {}
        # Pending config for available (not yet synced) editor layers
        self._available_editor_config: Dict[str, Dict[str, int]] = {}  # {layer_id: {"tc": int, "tg": int, "seq": int}}
        self._available_ma3_target: Dict[str, Optional[str]] = {}  # {ma3_coord: editor_layer_id or None for "Create New"}
        self._available_editor_target: Dict[str, Optional[str]] = {}  # {editor_layer_id: ma3_coord or None for "Create New"}
        self._last_ma3_refresh_at: float = 0.0
        self._ma3_refresh_min_interval: float = 2.0
        self._force_tracks_refresh: bool = False
        self._sync_list_pending: bool = False
        
        # Sync system manager (single source of truth for sync operations)
        # Use the facade's cached SSM so project-load monitoring shares the same instance
        self._sync_system_manager = facade.sync_system_manager(block_id)
        if self._sync_system_manager is None:
            # Fallback: create a new instance if the facade didn't have one
            self._sync_system_manager = SyncSystemManager(
                facade=facade,
                show_manager_block_id=block_id,
                settings_manager=self._settings_manager,
                parent=self,
            )
        self._sync_system_manager.entities_changed.connect(self._on_sync_system_entities_changed)
        self._sync_system_manager.entity_updated.connect(self._on_sync_system_entity_updated)
        self._sync_system_manager.sync_status_changed.connect(self._on_sync_system_status_changed)
        self._sync_system_manager.divergence_detected.connect(self._on_sync_system_divergence_detected)
        self._sync_system_manager.error_occurred.connect(self._on_sync_system_error)
        self._sync_system_manager.track_conflict_prompt.connect(self._on_track_conflict_prompt)
        self._sync_system_manager.connection_state_changed.connect(self._on_ssm_connection_state_changed)

        # Sync layer reconciliation
        from src.features.show_manager.application.sync_layer_manager import SyncLayerManager
        self._sync_layer_manager = SyncLayerManager()
        self._reconcile_pending: Dict[str, Dict[str, Any]] = {}
        self._reconcile_in_progress = False
        self._reconcile_results: List[Dict[str, Any]] = []
        self._reconcile_cache: Dict[str, Dict[str, Any]] = {}
        self._reconcile_dialog_active = False
        self._reconcile_request_queue: List[str] = []
        self._reconcile_active_key: Optional[str] = None
        self._last_sync_save_ts = 0.0
        self._sync_save_min_interval = 2.0
        self._ma3_status_summary: Dict[str, Any] = {}
        
        # Refresh will be triggered after widgets are created and settings are loaded
        # The settings_loaded signal will trigger refresh, but we also ensure refresh
        # happens after a delay to handle timing issues
        def delayed_refresh():
            if hasattr(self, 'ma3_ip_edit') and self.ma3_ip_edit and self._settings_manager.is_loaded():
                self.refresh()
                # Check if listener should be running and restore if needed
                self._check_and_restore_listener()
                # Load connection state from service and update UI (but don't check automatically)
                self._load_connection_state_from_service()
                # Populate available layers list (Editor layers + any cached MA3 tracks)
                # This must happen on init so layers are visible even when MA3 is offline.
                self._update_available_layers_list()
                self._update_synced_layers_list()
        QTimer.singleShot(300, delayed_refresh)
    
    def create_content_widget(self) -> QWidget:
        """Create ShowManager-specific UI with tabbed interface."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                border-top: 1px solid {Colors.BORDER.name()};
                background-color: {Colors.BG_DARK.name()};
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_SECONDARY.name()};
                padding: 6px 14px;
                margin-right: 1px;
                border: none;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QTabBar::tab:hover {{
                color: {Colors.TEXT_PRIMARY.name()};
            }}
        """)
        
        self.tab_widget.addTab(self._create_connection_tab(), "Connection")
        self.tab_widget.addTab(self._create_monitoring_tab(), "Monitoring")
        self.tab_widget.addTab(self._create_layer_sync_tab(), "Layer Sync")
        
        main_layout.addWidget(self.tab_widget)
        
        return widget
    
    def _create_connection_tab(self) -> QWidget:
        """Create Connection tab with status, settings, and controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        # === Connection Status (compact row) ===
        status_row = QHBoxLayout()
        status_row.setSpacing(Spacing.SM)
        
        self.connection_status_indicator = BlockStatusDot(self.block_id, self.facade, parent=tab)
        self.connection_status_indicator.setFixedSize(16, 16)
        status_row.addWidget(self.connection_status_indicator)
        
        self.connection_status_label = QLabel("Not Connected")
        self.connection_status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: bold;")
        status_row.addWidget(self.connection_status_label)
        
        self.connection_details_label = QLabel("")
        self.connection_details_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        status_row.addWidget(self.connection_details_label, stretch=1)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._check_connection_status)
        refresh_btn.setToolTip("Refresh connection status")
        status_row.addWidget(refresh_btn)
        
        layout.addLayout(status_row)
        
        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {Colors.BORDER.name()};")
        layout.addWidget(divider)
        
        # === Connection Settings (form) ===
        form = QFormLayout()
        form.setSpacing(Spacing.SM)
        form.setContentsMargins(0, 0, 0, 0)
        
        # Interface selection (listen address)
        listen_address = "127.0.0.1"
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            listen_address = self._settings_manager.listen_address or "127.0.0.1"
        self.listen_interface_combo = RefreshableComboBox(self._refresh_interfaces_on_open)
        self.listen_interface_combo.currentIndexChanged.connect(self._on_interface_changed)
        self.listen_interface_edit = QLineEdit(listen_address)
        self.listen_interface_edit.setPlaceholderText("Custom IP")
        self.listen_interface_edit.editingFinished.connect(self._on_interface_custom_changed)
        interface_widget = QWidget()
        interface_layout = QHBoxLayout(interface_widget)
        interface_layout.setContentsMargins(0, 0, 0, 0)
        interface_layout.setSpacing(Spacing.XS)
        interface_layout.addWidget(self.listen_interface_combo)
        interface_layout.addWidget(self.listen_interface_edit)
        form.addRow("Interface:", interface_widget)
        self._populate_interface_combo(listen_address)
        
        # MA3 Address
        initial_ip = "127.0.0.1"
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            initial_ip = self._settings_manager.ma3_ip
        self.ma3_ip_edit = QLineEdit(initial_ip)
        self.ma3_ip_edit.setPlaceholderText("e.g., 127.0.0.1")
        self.ma3_ip_edit.editingFinished.connect(self._on_ip_changed)
        form.addRow("MA3 Address:", self.ma3_ip_edit)
        
        # Ports side by side
        ports_row = QHBoxLayout()
        ports_row.setSpacing(Spacing.SM)
        
        initial_port = 9001
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            initial_port = self._settings_manager.ma3_port
        self.ma3_port_spin = QSpinBox()
        self.ma3_port_spin.setRange(1, 65535)
        self.ma3_port_spin.setValue(initial_port)
        self.ma3_port_spin.valueChanged.connect(self._on_port_changed)
        ports_row.addWidget(QLabel("Send:"))
        ports_row.addWidget(self.ma3_port_spin)
        
        initial_listen_port = 9000
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            initial_listen_port = self._settings_manager.listen_port
        self.listen_port_spin = QSpinBox()
        self.listen_port_spin.setRange(1, 65535)
        self.listen_port_spin.setValue(initial_listen_port)
        self.listen_port_spin.valueChanged.connect(self._on_listen_port_changed)
        ports_row.addWidget(QLabel("Listen:"))
        ports_row.addWidget(self.listen_port_spin)
        ports_row.addStretch()
        form.addRow("Ports:", ports_row)
        
        # Listener + MA3 controls in one row
        controls_row = QHBoxLayout()
        controls_row.setSpacing(Spacing.XS)
        
        self.start_listening_btn = QPushButton("Start Listening")
        self.start_listening_btn.clicked.connect(self._on_start_listening)
        controls_row.addWidget(self.start_listening_btn)
        
        self.stop_listening_btn = QPushButton("Stop")
        self.stop_listening_btn.clicked.connect(self._on_stop_listening)
        self.stop_listening_btn.setEnabled(False)
        controls_row.addWidget(self.stop_listening_btn)
        
        self.configure_ma3_btn = QPushButton("Configure MA3")
        self.configure_ma3_btn.clicked.connect(self._on_configure_ma3)
        self.configure_ma3_btn.setEnabled(False)
        controls_row.addWidget(self.configure_ma3_btn)
        
        self.ping_btn = QPushButton("Ping")
        self.ping_btn.clicked.connect(self._on_ping)
        controls_row.addWidget(self.ping_btn)
        
        controls_row.addStretch()
        form.addRow("Controls:", controls_row)

        # Force send (testing override)
        self.force_send_checkbox = QCheckBox("Force send OSC (bypass readiness checks)")
        self.force_send_checkbox.setToolTip(
            "Bypass MA3 readiness checks and send OSC even if not connected."
        )
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            self.force_send_checkbox.setChecked(self._settings_manager.force_send_osc)
        self.force_send_checkbox.toggled.connect(self._on_force_send_toggled)
        form.addRow("", self.force_send_checkbox)
        
        layout.addLayout(form)
        layout.addStretch()
        return tab
    
    def _create_monitoring_tab(self) -> QWidget:
        """Create Monitoring tab with OSC log, packet viewer, and commands."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        mono_style = f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                font-family: monospace;
                font-size: 11px;
                border: 1px solid {Colors.BORDER.name()};
            }}
        """
        
        # === Live Log ===
        log_bar = QHBoxLayout()
        log_bar.setSpacing(Spacing.SM)
        log_label = QLabel("Live OSC Log")
        log_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: bold;")
        log_bar.addWidget(log_label)
        
        initial_listen_port_str = "127.0.0.1:9000"
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            initial_listen_port_str = self._format_listen_status()
        self.listen_status_label = QLabel(initial_listen_port_str)
        self.listen_status_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN.name()}; font-size: 11px;")
        log_bar.addWidget(self.listen_status_label)
        
        log_bar.addStretch()
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())
        log_bar.addWidget(clear_log_btn)
        layout.addLayout(log_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setStyleSheet(mono_style)
        layout.addWidget(self.log_text, stretch=2)
        
        # === OSC Packet Viewer (side by side) ===
        packet_splitter = QSplitter(Qt.Orientation.Horizontal)
        packet_splitter.setHandleWidth(1)
        
        self.osc_raw_text = QTextEdit()
        self.osc_raw_text.setReadOnly(True)
        self.osc_raw_text.setPlaceholderText("Raw packets (hex)")
        self.osc_raw_text.setMaximumHeight(100)
        self.osc_raw_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.STATUS_INFO.name()};
                font-family: monospace;
                font-size: 10px;
                border: 1px solid {Colors.BORDER.name()};
            }}
        """)
        
        self.osc_parsed_text = QTextEdit()
        self.osc_parsed_text.setReadOnly(True)
        self.osc_parsed_text.setPlaceholderText("Interpreted response")
        self.osc_parsed_text.setMaximumHeight(100)
        self.osc_parsed_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.STATUS_SUCCESS.name()};
                font-family: monospace;
                font-size: 11px;
                border: 1px solid {Colors.BORDER.name()};
            }}
        """)
        
        packet_splitter.addWidget(self.osc_raw_text)
        packet_splitter.addWidget(self.osc_parsed_text)
        layout.addWidget(packet_splitter)
        
        # === Commands (compact) ===
        cmd_row = QHBoxLayout()
        cmd_row.setSpacing(Spacing.XS)
        self.custom_cmd_input = QLineEdit()
        self.custom_cmd_input.setPlaceholderText("Lua command (e.g., EZ.Ping())")
        self.custom_cmd_input.returnPressed.connect(self._on_send_command)
        cmd_row.addWidget(self.custom_cmd_input, stretch=1)
        
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._on_send_command)
        cmd_row.addWidget(send_btn)
        layout.addLayout(cmd_row)
        
        # Quick command buttons (single row)
        quick_row = QHBoxLayout()
        quick_row.setSpacing(Spacing.XS)
        
        quick_commands = [
            ("Ping", "EZ.Ping()"),
            ("Status", "EZ.Status()"),
            ("Timecodes", "EZ.GetTimecodes()"),
        ]
        for name, cmd in quick_commands:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, c=cmd: self._send_lua_command(c))
            quick_row.addWidget(btn)
        
        def get_tc():
            if hasattr(self, '_settings_manager') and self._settings_manager:
                return self._settings_manager.target_timecode or 101
            return 101
        
        tc_commands = [
            ("TrackGroups", lambda: f"EZ.GetTrackGroups({get_tc()})"),
            ("Tracks 1", lambda: f"EZ.GetTracks({get_tc()}, 1)"),
            ("Events 1.1", lambda: f"EZ.GetEvents({get_tc()}, 1, 1)"),
            ("All Events", lambda: f"EZ.GetAllEvents({get_tc()})"),
        ]
        for name, cmd_fn in tc_commands:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, fn=cmd_fn: self._send_lua_command(fn()))
            quick_row.addWidget(btn)
        
        quick_row.addStretch()
        layout.addLayout(quick_row)
        
        # === Events Display ===
        events_bar = QHBoxLayout()
        events_bar.setSpacing(Spacing.SM)
        events_label = QLabel("MA3 Events")
        events_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-weight: bold;")
        events_bar.addWidget(events_label)
        
        self.events_count_label = QLabel("0 events")
        self.events_count_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        events_bar.addWidget(self.events_count_label)
        events_bar.addStretch()
        
        self.watch_btn = QPushButton("Start Watching")
        self.watch_btn.clicked.connect(self._on_toggle_watching)
        events_bar.addWidget(self.watch_btn)
        
        clear_events_btn = QPushButton("Clear")
        clear_events_btn.clicked.connect(lambda: self.events_text.clear())
        events_bar.addWidget(clear_events_btn)
        layout.addLayout(events_bar)
        
        self.events_text = QTextEdit()
        self.events_text.setReadOnly(True)
        self.events_text.setMinimumHeight(80)
        self.events_text.setStyleSheet(mono_style)
        layout.addWidget(self.events_text, stretch=1)
        
        # === Manual Event Creation (compact row) ===
        manual_row = QHBoxLayout()
        manual_row.setSpacing(Spacing.SM)
        manual_row.addWidget(QLabel("Layer:"))
        self.manual_layer_combo = QComboBox()
        self.manual_layer_combo.setEditable(False)
        self.manual_layer_combo.addItem("(Select layer...)")
        manual_row.addWidget(self.manual_layer_combo)
        
        manual_row.addWidget(QLabel("Time:"))
        self.manual_time_input = QDoubleSpinBox()
        self.manual_time_input.setMinimum(0.0)
        self.manual_time_input.setMaximum(999999.0)
        self.manual_time_input.setSingleStep(0.1)
        self.manual_time_input.setDecimals(3)
        self.manual_time_input.setValue(0.0)
        manual_row.addWidget(self.manual_time_input)
        
        add_event_btn = QPushButton("Add Event")
        add_event_btn.clicked.connect(self._on_add_manual_event)
        manual_row.addWidget(add_event_btn)
        manual_row.addStretch()
        layout.addLayout(manual_row)
        
        return tab
    
    def _create_layer_sync_tab(self) -> QWidget:
        """Create Layer Sync tab with side-by-side Available + Synced tables."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(Spacing.XS)
        layout.setContentsMargins(Spacing.SM, Spacing.SM, Spacing.SM, Spacing.SM)
        
        # === Toolbar row: Timecode + Sync Tools (compact single row) ===
        toolbar = QHBoxLayout()
        toolbar.setSpacing(Spacing.SM)
        
        tc_label = QLabel("Timecode:")
        tc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        toolbar.addWidget(tc_label)
        
        self.target_timecode_edit = QLineEdit()
        self.target_timecode_edit.setPlaceholderText("e.g. 101")
        self.target_timecode_edit.setText("1")
        self.target_timecode_edit.setFixedWidth(60)
        toolbar.addWidget(self.target_timecode_edit)
        
        self.load_timecode_btn = QPushButton("Switch")
        self.load_timecode_btn.setToolTip("Clear old data and load tracks from this timecode")
        self.load_timecode_btn.clicked.connect(self._on_load_timecode_clicked)
        toolbar.addWidget(self.load_timecode_btn)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {Colors.BORDER.name()};")
        toolbar.addWidget(sep)
        
        self.hook_status_label = QLabel("Hooks: 0/0")
        self.hook_status_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        toolbar.addWidget(self.hook_status_label)
        
        self.refresh_layers_btn = QPushButton("Refresh")
        self.refresh_layers_btn.setToolTip("Fetch MA3 tracks and refresh lists")
        self.refresh_layers_btn.clicked.connect(self._on_manual_layer_refresh)
        toolbar.addWidget(self.refresh_layers_btn)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Shared table stylesheet
        table_style = f"""
            QTableWidget {{
                background-color: {Colors.BG_MEDIUM.name()};
                gridline-color: {Colors.BORDER.name()};
                border: none;
                border-top: 1px solid {Colors.BORDER.name()};
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                font-weight: bold;
                padding: 4px 6px;
                border: none;
                border-right: 1px solid {Colors.BORDER.name()};
                border-bottom: 1px solid {Colors.BORDER.name()};
            }}
        """
        
        # === Split View: Available (left) | Synced (right) via QSplitter ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER.name()};
            }}
        """)
        
        # --- Available Layers Panel ---
        available_panel = QWidget()
        available_layout = QVBoxLayout(available_panel)
        available_layout.setContentsMargins(0, 0, 0, 0)
        available_layout.setSpacing(0)
        
        avail_header_label = QLabel("Available Layers")
        avail_header_label.setStyleSheet(f"""
            background-color: {Colors.BG_DARK.name()};
            color: {Colors.TEXT_PRIMARY.name()};
            font-weight: bold;
            padding: 4px 8px;
            font-size: 12px;
        """)
        available_layout.addWidget(avail_header_label)
        
        self.available_table = QTableWidget()
        self.available_table.setColumnCount(7)
        self.available_table.setHorizontalHeaderLabels(["Sync", "Type", "Name", "Target", "TC", "TG", "Seq"])
        self.available_table.verticalHeader().setVisible(False)
        self.available_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.available_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.available_table.setAlternatingRowColors(True)
        self.available_table.setShowGrid(False)
        self.available_table.setWordWrap(False)
        
        avail_header = self.available_table.horizontalHeader()
        avail_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        avail_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        avail_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        avail_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        avail_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        avail_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        avail_header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        avail_header.setDefaultSectionSize(50)
        self.available_table.setStyleSheet(table_style)
        available_layout.addWidget(self.available_table)
        
        # --- Synced Layers Panel ---
        synced_panel = QWidget()
        synced_layout = QVBoxLayout(synced_panel)
        synced_layout.setContentsMargins(0, 0, 0, 0)
        synced_layout.setSpacing(0)
        
        synced_header_label = QLabel("Synced Layers")
        synced_header_label.setStyleSheet(f"""
            background-color: {Colors.BG_DARK.name()};
            color: {Colors.TEXT_PRIMARY.name()};
            font-weight: bold;
            padding: 4px 8px;
            font-size: 12px;
        """)
        synced_layout.addWidget(synced_header_label)
        
        # --- Batch actions toolbar for synced layers ---
        batch_toolbar = QHBoxLayout()
        batch_toolbar.setSpacing(Spacing.XS)
        batch_toolbar.setContentsMargins(4, 2, 4, 2)
        
        batch_label = QLabel("Batch:")
        batch_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()}; font-size: 11px;")
        batch_toolbar.addWidget(batch_label)
        
        self.batch_keep_ez_btn = QPushButton("All Keep EZ")
        self.batch_keep_ez_btn.setToolTip(
            "Keep Editor events for ALL synced layers and push them to MA3"
        )
        self.batch_keep_ez_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 3px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_DISABLED.name()};
            }}
        """)
        self.batch_keep_ez_btn.clicked.connect(self._on_batch_keep_ez)
        batch_toolbar.addWidget(self.batch_keep_ez_btn)
        
        self.batch_keep_ma3_btn = QPushButton("All Keep MA3")
        self.batch_keep_ma3_btn.setToolTip(
            "Keep MA3 events for ALL synced layers and apply them to the Editor"
        )
        self.batch_keep_ma3_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_GREEN.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 3px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_GREEN.lighter(110).name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_DISABLED.name()};
            }}
        """)
        self.batch_keep_ma3_btn.clicked.connect(self._on_batch_keep_ma3)
        batch_toolbar.addWidget(self.batch_keep_ma3_btn)
        
        batch_toolbar.addStretch()
        synced_layout.addLayout(batch_toolbar)
        
        self.layers_table = QTableWidget()
        self.layers_table.setColumnCount(10)
        self.layers_table.setHorizontalHeaderLabels([
            "Sync", "Type", "Name", "Status", "TG", "Seq", "Resync", "Keep EZ", "Keep MA3", "Delete"
        ])
        self.layers_table.verticalHeader().setVisible(False)
        self.layers_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.layers_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.layers_table.setAlternatingRowColors(True)
        self.layers_table.setShowGrid(False)
        self.layers_table.setWordWrap(False)
        
        header = self.layers_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.ResizeToContents)
        header.setDefaultSectionSize(50)
        self.layers_table.setStyleSheet(table_style)
        synced_layout.addWidget(self.layers_table)
        
        splitter.addWidget(available_panel)
        splitter.addWidget(synced_panel)
        splitter.setSizes([500, 500])
        
        layout.addWidget(splitter, stretch=1)
        
        # Initial update (after widgets are created)
        QTimer.singleShot(0, self._update_synced_layers_list)
        
        return tab
    
    def _log(self, message: str):
        """Add message to live log view."""
        # Check if log_text widget exists (might not be created yet during initialization)
        if not hasattr(self, 'log_text') or not self.log_text:
            return
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_ma3_osc_outbound(self, event):
        """Display outbound OSC in Monitoring tab."""
        data = getattr(event, "data", {}) or {}
        ip = data.get("ip")
        port = data.get("port")
        lua_code = data.get("lua_code")
        success = data.get("success")
        error = data.get("error")
        osc_len = data.get("osc_len")
        status = "OK" if success else "ERROR"
        parts = [f"OSC OUT [{status}]"]
        if ip and port:
            parts.append(f"{ip}:{port}")
        if osc_len is not None:
            parts.append(f"bytes={osc_len}")
        if lua_code:
            parts.append(lua_code)
        if error:
            parts.append(f"err={error}")
        self._log(" | ".join(parts))

    def _on_force_send_toggled(self, enabled: bool) -> None:
        """Persist force-send testing override."""
        if hasattr(self, '_settings_manager') and self._settings_manager:
            self._settings_manager.force_send_osc = bool(enabled)
            state = "enabled" if enabled else "disabled"
            self._log(f"Force send OSC: {state}")
    
    def _apply_local_styles(self):
        """Re-apply variant button styles on theme change."""
        # Helper for accent-colored buttons
        def _accent_btn_style(accent_color):
            return f"""
                QPushButton {{
                    background-color: {accent_color.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 3px 10px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    background-color: {accent_color.lighter(110).name()};
                }}
                QPushButton:disabled {{
                    background-color: {Colors.BG_MEDIUM.name()};
                    color: {Colors.TEXT_DISABLED.name()};
                }}
            """
        
        # Batch toolbar buttons
        if hasattr(self, 'batch_keep_ez_btn'):
            self.batch_keep_ez_btn.setStyleSheet(_accent_btn_style(Colors.ACCENT_BLUE))
        if hasattr(self, 'batch_keep_ma3_btn'):
            self.batch_keep_ma3_btn.setStyleSheet(_accent_btn_style(Colors.ACCENT_GREEN))
    
    def refresh(self):
        """Refresh UI with current block data."""
        if not self.block:
            return
        
        # Ensure settings manager exists and is loaded
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            Log.debug("ShowManagerPanel: Settings manager not available yet")
            return
        
        # Ensure settings manager has finished loading
        if not self._settings_manager.is_loaded():
            Log.debug("ShowManagerPanel: Settings manager not loaded yet, waiting...")
            # Settings will be loaded and settings_loaded signal will trigger refresh
            return
        
        # Ensure UI widgets exist (they might not be created yet on first call)
        if not hasattr(self, 'ma3_ip_edit') or not self.ma3_ip_edit:
            Log.debug("ShowManagerPanel: UI widgets not created yet")
            return
        
        try:
            # Connection settings
            self.ma3_ip_edit.setText(self._settings_manager.ma3_ip)
            self.ma3_port_spin.setValue(self._settings_manager.ma3_port)
            if hasattr(self, 'listen_interface_combo'):
                self._populate_interface_combo(self._settings_manager.listen_address)
            self.listen_port_spin.setValue(self._settings_manager.listen_port)
            if hasattr(self, 'target_tc_spin'):
                self.target_tc_spin.setValue(self._settings_manager.target_timecode)
            # Update Layer Sync tab textbox (but not during timecode switch)
            if hasattr(self, 'target_timecode_edit'):
                if not getattr(self, '_switching_timecode', False):
                    self.target_timecode_edit.blockSignals(True)
                    self.target_timecode_edit.setText(str(self._settings_manager.target_timecode))
                    self.target_timecode_edit.blockSignals(False)
            
            # Sync settings
            if hasattr(self, 'mapping_combo') and self.mapping_combo:
                mapping = self._settings_manager.mapping_template.value
                idx = self.mapping_combo.findData(mapping)
                if idx >= 0:
                    self.mapping_combo.setCurrentIndex(idx)
            
            if hasattr(self, 'safety_combo') and self.safety_combo:
                safety = self._settings_manager.safety_level.value
                idx = self.safety_combo.findData(safety)
                if idx >= 0:
                    self.safety_combo.setCurrentIndex(idx)
            
            if hasattr(self, 'auto_sync_check') and self.auto_sync_check:
                self.auto_sync_check.setChecked(self._settings_manager.auto_sync_enabled)
            
            # Conflict resolution strategy
            if hasattr(self, 'conflict_strategy_combo') and self.conflict_strategy_combo:
                strategy = self._settings_manager.conflict_resolution_strategy
                idx = self.conflict_strategy_combo.findData(strategy)
                if idx >= 0:
                    self.conflict_strategy_combo.setCurrentIndex(idx)
            
            # Update listen status
            self.listen_status_label.setText(self._format_listen_status())
            
            # Update connection status labels (status indicator now uses BlockStatusDot which auto-updates)
            # Load state from service (state persists independently of panel) - but don't check automatically
            if hasattr(self, 'connection_status_label') and self.connection_status_label:
                self._load_connection_state_from_service()
            
            Log.debug(
                f"ShowManagerPanel: UI refreshed with settings from block {self.block_id}. "
                f"Values: MA3={self._settings_manager.ma3_ip}:{self._settings_manager.ma3_port}, "
                f"Listen={self._settings_manager.listen_address}:{self._settings_manager.listen_port}, "
                f"TC={self._settings_manager.target_timecode}"
            )
        except Exception as e:
            import traceback
            Log.error(f"ShowManagerPanel: Error refreshing UI: {e}\n{traceback.format_exc()}")
            traceback.print_exc()
    
    # === Connection Handling ===
    
    def _attempt_auto_connect(self):
        """Attempt to automatically connect ShowManager to an Editor block via manipulator port."""
        # Auto-connect removed - user must manually connect
        pass
    
    def _check_connection_status(self):
        """Check current connection status and update display (delegates to state service)."""
        if not self._state_service:
            return
        
        # State service handles polling - just load current state
        self._load_connection_state_from_service()
    
    def _set_connection_status(self, status: str, editor_id: Optional[str], error: Optional[str]):
        """Set connection status and update UI display."""
        # Update state service (which persists state independently of panel)
        if self._state_service:
            self._state_service.update_connection_state(self.block_id, status, editor_id, error)
        
        # Update UI (service will emit signal, but we update immediately for responsiveness)
        self._update_connection_status_ui(status, editor_id, error)
    
    def _update_connection_status_ui(self, status: str, editor_id: Optional[str], error: Optional[str]):
        """Update UI display for connection status (does not update state service).
        
        Combines Editor connection state (from state service) with MA3 connection
        state (from SSM) to produce a single unified status display.
        
        Args:
            status: Editor connection status ("connected", "disconnected", "failed")
            editor_id: Connected editor block ID, or None
            error: Error message if status is "failed"
        """
        if not hasattr(self, 'connection_status_label') or not self.connection_status_label:
            return
        
        # --- Gather state from all sources ---
        
        # Editor connection
        has_editor = (status == "connected" and editor_id)
        editor_name = "Unknown"
        if has_editor:
            editor_result = self.facade.describe_block(editor_id)
            if editor_result.success and editor_result.data:
                editor_name = editor_result.data.name
        
        # MA3 connection state -- SSM is the single source of truth
        ma3_state = "disconnected"
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            ma3_state = self._sync_system_manager.connection_state
        
        # Listener and config state (for diagnosing why MA3 is disconnected)
        is_listening = False
        if hasattr(self, '_listener_service') and self._listener_service:
            is_listening = self._listener_service.is_listening(self.block_id)
        ma3_configured = False
        if hasattr(self, '_settings_manager') and self._settings_manager:
            ma3_ip = (self._settings_manager.ma3_ip or "").strip()
            ma3_port = self._settings_manager.ma3_port or 0
            ma3_configured = bool(ma3_ip and ma3_port > 0)
        
        is_fully_connected = (has_editor and ma3_state == "connected")
        
        
        # --- Update status label and details ---
        
        if status == "failed":
            error_text = error or "Unknown error"
            self.connection_status_label.setText("Connection Failed")
            self.connection_status_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText(f"Error: {error_text}")
            self.set_status_message(f"Connection failed: {error_text}", error=True)
        elif not has_editor:
            self.connection_status_label.setText("Not Connected")
            self.connection_status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText("No Editor block connected via manipulator port")
            self.set_status_message("Not connected to Editor block", error=False)
        elif is_fully_connected:
            # Editor + MA3 both active
            self.connection_status_label.setText("Connected: Editor + MA3")
            self.connection_status_label.setStyleSheet(f"color: {Colors.ACCENT_GREEN.name()}; font-size: 14px; font-weight: bold;")
            listen_port = self._settings_manager.listen_port if hasattr(self, '_settings_manager') else '?'
            self.connection_details_label.setText(f"Editor: {editor_name} | MA3: Active (port {listen_port})")
            self.set_status_message(f"Fully connected: Editor '{editor_name}' + MA3", error=False)
        elif ma3_state == "stale":
            # Editor connected, MA3 is going stale
            self.connection_status_label.setText("Editor Connected (MA3 stale)")
            self.connection_status_label.setStyleSheet(f"color: {Colors.ACCENT_ORANGE.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText(f"Editor: {editor_name} | MA3: Waiting for response...")
            self.set_status_message(f"Editor connected, waiting for MA3 response...", error=False)
        elif is_listening and ma3_state == "disconnected":
            # Editor connected, listener running, but MA3 not responding
            self.connection_status_label.setText("Editor Connected (MA3 not responding)")
            self.connection_status_label.setStyleSheet(f"color: {Colors.ACCENT_RED.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText(f"Editor: {editor_name} | MA3: No response (check MA3 OSC output)")
            self.set_status_message(f"MA3 not responding -- check MA3 OSC output is enabled", error=True)
        elif is_listening and not ma3_configured:
            self.connection_status_label.setText("Editor Connected (MA3 not configured)")
            self.connection_status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText(f"Editor: {editor_name} | Set MA3 IP/Port to enable sync")
            self.set_status_message(f"Editor connected: {editor_name} (configure MA3 target)", error=False)
        else:
            # Editor connected, listener not running
            self.connection_status_label.setText("Editor Connected (MA3 OSC not listening)")
            self.connection_status_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()}; font-size: 14px; font-weight: bold;")
            self.connection_details_label.setText(f"Editor: {editor_name} | Start listener to enable MA3 connection")
            self.set_status_message(f"Editor connected: {editor_name} (start listener for MA3)", error=False)
        
        # Note: connection_status_indicator (BlockStatusDot) auto-updates via BlockStatusService

        # Trigger divergence check when we transition to fully connected
        if is_fully_connected and not self._was_fully_connected:
            self._start_sync_layer_reconcile()
        self._was_fully_connected = is_fully_connected
    
    # === Event Handlers ===
    
    def _on_setting_changed(self, setting_name: str):
        """Handle settings changes."""
        self._log(f"Setting changed: {setting_name}")
        
        # Update target timecode textbox if it exists and setting changed
        # But skip during timecode switch to prevent race conditions
        if setting_name == "target_timecode" and hasattr(self, 'target_timecode_edit'):
            if not getattr(self, '_switching_timecode', False):
                if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
                    current_tc = self._settings_manager.target_timecode
                    # Only update if value actually changed (prevents feedback loop)
                    if self.target_timecode_edit.text() != str(current_tc):
                        self.target_timecode_edit.blockSignals(True)
                        self.target_timecode_edit.setText(str(current_tc))
                        self.target_timecode_edit.blockSignals(False)
        
        # If synced_layers changed, refresh controller
        if setting_name == "synced_layers":
            if getattr(self, "_refreshing_controller", False):
                return
            self._refresh_controller_from_settings()
    
    def _refresh_controller_from_settings(self):
        """Refresh sync state from current settings (after settings change).
        
        NOTE: Do NOT call reload_from_storage() here. This method is called
        from _on_setting_changed() which fires BEFORE the data is persisted
        to DB. The in-memory settings are already correct (set by the setter).
        Reloading from DB would read stale data and overwrite the new values.
        """
        if not hasattr(self, '_settings_manager') or not self._settings_manager.is_loaded():
            Log.debug("ShowManagerPanel: Settings manager not loaded, skipping refresh")
            return
        
        try:
            
            # Get current synced_layers from settings manager (already correct in-memory)
            synced_layers = self._settings_manager.synced_layers
            Log.debug(f"ShowManagerPanel: Refreshing with {len(synced_layers)} synced layer(s)")
            
            # Update UI list
            self._update_synced_layers_list()
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Failed to refresh from settings: {e}")
            import traceback
            Log.debug(traceback.format_exc())
    
    def _on_settings_loaded(self):
        """Handle settings loaded signal - refresh UI with loaded settings."""
        if self._pending_controller_init:
            self._pending_controller_init = False
            self._refresh_controller_from_settings()

        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            self._run_naming_migration()
        
        # Update target timecode textbox if it exists
        # BUT NOT if we're in the middle of switching timecodes (prevents race condition)
        if hasattr(self, 'target_timecode_edit') and hasattr(self, '_settings_manager'):
            if self._settings_manager.is_loaded():
                # Skip update if actively switching timecodes
                if getattr(self, '_switching_timecode', False):
                    Log.debug("ShowManagerPanel: Skipping target_timecode_edit update during switch")
                else:
                    current_tc = self._settings_manager.target_timecode
                    self.target_timecode_edit.blockSignals(True)
                    self.target_timecode_edit.setText(str(current_tc))
                    self.target_timecode_edit.blockSignals(False)
                    Log.debug(f"ShowManagerPanel: Updated target_timecode_edit to {current_tc} on settings load")

        self._update_hook_status_label()
        
        # Reload SyncSystemManager when settings are loaded (e.g., after project load or undo).
        # Skip when this load was triggered by BlockUpdated: SSM often caused that update
        # via _save_to_settings(), so reloading into SSM would be redundant and cause
        # repeated "Loaded target_timecode" / "Loaded synced layers" log spam and re-init.
        skip_ssm_load = getattr(self, '_skip_ssm_load_after_block_updated', False)
        if skip_ssm_load:
            self._skip_ssm_load_after_block_updated = False
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager and not skip_ssm_load:
            # Reload synced layers from settings
            self._sync_system_manager._load_from_settings()
        # Always refresh UI from current SSM state (whether we reloaded SSM or not)
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            self._update_synced_layers_list()
        
        # NOTE: Removed auto-fetch on settings load - user must explicitly click to fetch
        # This prevents the cascade of unnecessary MA3 commands
        
        # Refresh UI when settings are loaded from database
        # Use QTimer to ensure widgets are created before refreshing
        if self.block:
            def do_refresh():
                if hasattr(self, 'ma3_ip_edit') and self.ma3_ip_edit:
                    Log.debug(f"ShowManagerPanel: Refreshing UI after settings loaded for block {self.block_id}")
                    self.refresh()
                    # Ensure service uses current settings (important for IP/port changes)
                    self._apply_ma3_target_to_service()
                else:
                    # Widgets not ready yet, try again after a delay
                    Log.debug("ShowManagerPanel: Widgets not ready, retrying refresh...")
                    QTimer.singleShot(100, do_refresh)
            QTimer.singleShot(50, do_refresh)

    def _run_naming_migration(self) -> None:
        """Migrate naming to prefixed conventions for sync layers/tracks.
        
        NOTE: This migration has been deprecated as part of the sync system refactor.
        The sync_registry module was removed. Mark as migrated to skip.
        """
        if not hasattr(self, "_settings_manager") or not self._settings_manager.is_loaded():
            return
        if self._settings_manager.naming_migrated:
            return
        
        # Skip legacy migration - sync system has been refactored
        Log.info("ShowManagerPanel: Skipping legacy naming migration (sync system refactored)")
        self._settings_manager.naming_migrated = True
    
    def _validate_ip_address(self, ip: str) -> bool:
        """
        Validate IP address format.
        
        Args:
            ip: IP address string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not ip or not ip.strip():
            return False
        
        ip = ip.strip()
        
        # Basic validation: check for IPv4 format (x.x.x.x)
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        
        try:
            for part in parts:
                num = int(part)
                if num < 0 or num > 255:
                    return False
        except ValueError:
            return False
        
        return True

    def _refresh_interfaces_on_open(self) -> None:
        """Refresh interface list when dropdown opens, preserving selection if possible."""
        if not hasattr(self, "listen_interface_combo"):
            return
        # Remember current selection
        current_addr = self.listen_interface_combo.currentData()
        if current_addr == "custom":
            current_addr = self.listen_interface_edit.text().strip() or "127.0.0.1"
        
        # Repopulate with fresh interface list
        self.listen_interface_combo.blockSignals(True)
        self.listen_interface_combo.clear()
        candidates = self._get_interface_candidates()
        available_addrs = set()
        for name, addr in candidates:
            self.listen_interface_combo.addItem(f"{name} ({addr})", addr)
            available_addrs.add(addr)
        self.listen_interface_combo.addItem("Custom IP...", "custom")
        self.listen_interface_combo.blockSignals(False)
        
        # Try to restore previous selection
        idx = self.listen_interface_combo.findData(current_addr)
        if idx >= 0:
            # Interface still available
            self.listen_interface_combo.blockSignals(True)
            self.listen_interface_combo.setCurrentIndex(idx)
            self.listen_interface_combo.blockSignals(False)
        elif current_addr and current_addr not in available_addrs:
            # Interface disappeared - log warning and switch to custom
            self._log(f"Interface {current_addr} no longer available")
            custom_idx = self.listen_interface_combo.findData("custom")
            if custom_idx >= 0:
                self.listen_interface_combo.blockSignals(True)
                self.listen_interface_combo.setCurrentIndex(custom_idx)
                self.listen_interface_combo.blockSignals(False)
                self.listen_interface_edit.setEnabled(True)
                self.listen_interface_edit.setText(current_addr)

    def _populate_interface_combo(self, listen_address: str) -> None:
        """Populate interface combo with available interfaces."""
        if not hasattr(self, "listen_interface_combo"):
            return
        self.listen_interface_combo.blockSignals(True)
        self.listen_interface_combo.clear()
        candidates = self._get_interface_candidates()
        for name, addr in candidates:
            self.listen_interface_combo.addItem(f"{name} ({addr})", addr)
        self.listen_interface_combo.addItem("Custom IP...", "custom")
        self.listen_interface_combo.blockSignals(False)
        self._update_interface_controls(listen_address)

    def _get_interface_candidates(self) -> List[Tuple[str, str]]:
        """Return list of (interface_name, ip) for IPv4 interfaces using netifaces."""
        candidates: List[Tuple[str, str]] = []
        # Always include all-interfaces and loopback
        candidates.append(("All Interfaces", "0.0.0.0"))
        candidates.append(("Loopback", "127.0.0.1"))
        try:
            import netifaces
            for iface in netifaces.interfaces():
                addrs = netifaces.ifaddresses(iface)
                ipv4_list = addrs.get(netifaces.AF_INET, [])
                for entry in ipv4_list:
                    ip = entry.get('addr')
                    if ip and ip != "127.0.0.1" and (iface, ip) not in candidates:
                        candidates.append((iface, ip))
        except ImportError:
            logger.warning("netifaces not installed; interface discovery limited")
        except Exception as e:
            logger.debug(f"Interface discovery error: {e}")
        return candidates

    def _update_interface_controls(self, listen_address: str) -> None:
        """Update interface selection and custom input."""
        if not hasattr(self, "listen_interface_combo") or not hasattr(self, "listen_interface_edit"):
            return
        addr = (listen_address or "127.0.0.1").strip()
        idx = self.listen_interface_combo.findData(addr)
        # Block signals to prevent triggering _on_interface_changed during UI sync
        self.listen_interface_combo.blockSignals(True)
        if idx >= 0:
            self.listen_interface_combo.setCurrentIndex(idx)
            self.listen_interface_combo.blockSignals(False)
            self.listen_interface_edit.setEnabled(False)
            self.listen_interface_edit.setText(addr)
            return
        custom_idx = self.listen_interface_combo.findData("custom")
        if custom_idx >= 0:
            self.listen_interface_combo.setCurrentIndex(custom_idx)
        self.listen_interface_combo.blockSignals(False)
        self.listen_interface_edit.setEnabled(True)
        self.listen_interface_edit.setText(addr)

    def _on_interface_changed(self, _index: int) -> None:
        """Handle interface selection change."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        selection = self.listen_interface_combo.currentData()
        if selection == "custom":
            self.listen_interface_edit.setEnabled(True)
            return
        self.listen_interface_edit.setEnabled(False)
        self.listen_interface_edit.setText(selection)
        self._settings_manager.listen_address = selection
        self.listen_status_label.setText(self._format_listen_status())
        self._restart_listener_if_active()

    def _on_interface_custom_changed(self) -> None:
        """Handle custom interface IP input."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        if self.listen_interface_combo.currentData() != "custom":
            return
        new_addr = self.listen_interface_edit.text().strip()
        old_addr = self._settings_manager.listen_address
        if not new_addr:
            self._log("ERROR: Interface address cannot be empty")
            self.listen_interface_edit.setText(old_addr)
            return
        if not self._validate_ip_address(new_addr):
            self._log(f"ERROR: Invalid interface address format: {new_addr}")
            self.listen_interface_edit.setText(old_addr)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid Interface Address",
                f"'{new_addr}' is not a valid IPv4 address.\n\n"
                "Please enter an IPv4 address in the format: x.x.x.x"
            )
            return
        self._settings_manager.listen_address = new_addr
        self.listen_status_label.setText(self._format_listen_status())
        self._restart_listener_if_active()

    def _format_listen_status(self) -> str:
        """Format listen address/port display for monitoring tab."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return "9000"
        addr = self._settings_manager.listen_address or "127.0.0.1"
        port = self._settings_manager.listen_port or 9000
        return f"{addr}:{port}"

    def _restart_listener_if_active(self) -> None:
        """Restart listener if currently active (after config change)."""
        if self._listener_service and self._listener_service.is_listening(self.block_id):
            self._listener_service.stop_listener(self.block_id)
            QTimer.singleShot(100, self._on_start_listening)

    def _resolve_listen_target_ip(self) -> str:
        """
        Resolve the IP to give MA3 for callbacks.
        
        If listening on 0.0.0.0, determine the local IP used to reach MA3.
        """
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return "127.0.0.1"
        listen_address = (self._settings_manager.listen_address or "127.0.0.1").strip()
        if listen_address not in ("0.0.0.0", ""):
            return listen_address
        target_ip = self._settings_manager.ma3_ip or "127.0.0.1"
        target_port = self._settings_manager.ma3_port or 9001
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.connect((target_ip, target_port))
                return sock.getsockname()[0]
            finally:
                sock.close()
        except Exception:
            return "127.0.0.1"
    
    def _on_ip_changed(self):
        """Handle MA3 address change."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        new_ip = self.ma3_ip_edit.text().strip()
        old_ip = self._settings_manager.ma3_ip
        if not new_ip:
            self._log("ERROR: MA3 address cannot be empty")
            self.ma3_ip_edit.setText(old_ip)
            return
        if not self._validate_ip_address(new_ip):
            self._log(f"ERROR: Invalid MA3 address format: {new_ip}")
            self.ma3_ip_edit.setText(old_ip)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Invalid MA3 Address",
                f"'{new_ip}' is not a valid IP address.\n\n"
                "Please enter an IPv4 address in the format: x.x.x.x"
            )
            return
        self._settings_manager.ma3_ip = new_ip
        self._apply_ma3_target_to_service()
        self._log(f"MA3 Address updated: {old_ip} -> {new_ip}")
        Log.info(f"ShowManagerPanel: MA3 address changed from {old_ip} to {new_ip} for block {self.block_id}")

    def _on_port_changed(self, value: int):
        """Handle MA3 port change."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        if value < 1 or value > 65535:
            self._log("ERROR: Port must be between 1 and 65535")
            self.ma3_port_spin.setValue(self._settings_manager.ma3_port)
            return
        self._settings_manager.ma3_port = value
        self._apply_ma3_target_to_service()
        self._log(f"MA3 Port updated to: {value}")
        Log.info(f"ShowManagerPanel: MA3 port changed to {value} for block {self.block_id}")
    
    def _on_listen_port_changed(self, value: int):
        if hasattr(self, '_settings_manager'):
            self._settings_manager.listen_port = value
            self.listen_status_label.setText(self._format_listen_status())
        self._restart_listener_if_active()
    
    def _on_target_tc_changed(self, value: int):
        if hasattr(self, '_settings_manager'):
            self._settings_manager.target_timecode = value
        
        # Also update SyncSystemManager's configured timecode
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            self._sync_system_manager.configured_timecode = value
        
        # Update any pending editor configs to use new timecode
        for layer_id in self._available_editor_config:
            self._available_editor_config[layer_id]["tc"] = value
        
        # Refresh available layers to show updated TC
        self._update_available_layers_list()
    
    def _on_mapping_changed(self, index: int):
        if hasattr(self, '_settings_manager'):
            value = self.mapping_combo.currentData()
            self._settings_manager.mapping_template = MappingTemplate(value)
    
    def _on_safety_changed(self, index: int):
        if hasattr(self, '_settings_manager'):
            value = self.safety_combo.currentData()
            self._settings_manager.safety_level = SafetyLevel(value)
    
    def _on_auto_sync_changed(self, state: int):
        if hasattr(self, '_settings_manager'):
            self._settings_manager.auto_sync_enabled = (state == Qt.CheckState.Checked.value)
    
    def _on_conflict_strategy_changed(self, index: int):
        """Handle conflict resolution strategy selection."""
        if not hasattr(self, '_settings_manager'):
            return
        
        strategy = self.conflict_strategy_combo.currentData()
        self._settings_manager.conflict_resolution_strategy = strategy
        
        self._log(f"Conflict resolution strategy: {strategy}")
    
    def _handle_quick_command(self, lua_code: str):
        """Handle quick command button clicks with error handling."""
        try:
            self._send_lua_command(lua_code)
        except RuntimeError as e:
            self._log(f"ERROR: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MA3 Connection Error", str(e))
    
    def _send_lua_command(self, lua_code: str) -> bool:
        """
        Send a Lua command to MA3 via OSC /cmd.
        
        Raises:
            RuntimeError: If MA3 is not ready for communication
        """
        # Ensure MA3 is ready before sending unless forced
        force_send = bool(getattr(self, "force_send_checkbox", None) and self.force_send_checkbox.isChecked())
        if not force_send:
            self._ensure_ma3_ready()
        else:
            self._log("WARNING: Forcing OSC send without MA3 readiness checks")
        
        ma3_ip = self._settings_manager.ma3_ip
        ma3_port = self._settings_manager.ma3_port
        
        try:
            ma3_comm = getattr(self.facade, "ma3_communication_service", None) if self.facade else None
            if not ma3_comm:
                raise RuntimeError("MA3 communication service not available")
            self._apply_ma3_target_to_service()
            success = ma3_comm.send_lua_command(lua_code, ma3_ip, ma3_port)
            if not success:
                raise RuntimeError("Failed to send command to MA3")
            
            

            self._log(f">>> Sending: {lua_code}")
            return True
        except RuntimeError:
            # Re-raise connection errors
            raise
        except Exception as e:
            error_msg = f"Failed to send command to MA3: {e}"
            self._log(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    def _apply_ma3_target_to_service(self) -> None:
        """
        Ensure MA3CommunicationService uses ShowManager target.
        
        This updates the service's default send_address and send_port so that
        callers that don't pass explicit IP/port will use the current settings.
        """
        try:
            if not self.facade or not hasattr(self, '_settings_manager') or not self._settings_manager:
                return
            
            ma3_comm = getattr(self.facade, "ma3_communication_service", None)
            if not ma3_comm:
                Log.warning("ShowManagerPanel: MA3 communication service not available")
                return
            
            ma3_ip = self._settings_manager.ma3_ip
            ma3_port = self._settings_manager.ma3_port
            
            # Update service defaults
            ma3_comm.set_target(ma3_ip, ma3_port)
            
            Log.debug(f"ShowManagerPanel: Updated MA3 service target to {ma3_ip}:{ma3_port}")
            
        except Exception as e:
            Log.error(f"ShowManagerPanel: Failed to update MA3 service target: {e}")
            import traceback
            traceback.print_exc()
    
    def send_ma3_event(self, ma3_event: MA3Event) -> bool:
        """
        Send a single MA3 event to MA3 via OSC.
        
        Constructs the appropriate EZ.AddEvent() call and sends it.
        """
        # Build Lua command
        # EZ.AddEvent(tc_no, tg_idx, track_idx, time_secs, classification, event_type, properties)
        lua_cmd = (
            f"EZ.AddEvent({ma3_event.timecode_no}, {ma3_event.track_group}, "
            f"{ma3_event.track}, {ma3_event.time:.3f}, '{ma3_event.name}', "
            f"'{ma3_event.event_type}')"
        )
        
        return self._send_lua_command(lua_cmd)
    
    def send_ma3_events_batch(self, ma3_events: List[MA3Event]) -> int:
        """
        Send multiple MA3 events to MA3.
        
        Returns:
            Number of events successfully sent
        """
        sent_count = 0
        for event in ma3_events:
            if self.send_ma3_event(event):
                sent_count += 1
        
        return sent_count
    
    def _on_send_command(self):
        """Handle send command button/enter."""
        cmd = self.custom_cmd_input.text().strip()
        if cmd:
            try:
                self._send_lua_command(cmd)
                self.custom_cmd_input.clear()
            except RuntimeError as e:
                self._log(f"ERROR: {e}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "MA3 Connection Error", str(e))
    
    def _on_start_listening(self):
        """Handle start listening button click."""
        
        # Try to get service from facade if panel doesn't have it
        if not self._listener_service and self.facade:
            self._listener_service = getattr(self.facade, 'show_manager_listener_service', None)
        
        if not self._listener_service:
            # More informative error message
            self._log("ERROR: Listener service not available. The service may not be initialized yet. Please try again or restart the application.")
            Log.warning(f"ShowManagerPanel: Listener service not available for block {self.block_id}. Service may not be initialized yet.")
            return
        
        if self._listener_service.is_listening(self.block_id):
            self._log("Already listening")
            return
        
        listen_port = 9000
        listen_address = "127.0.0.1"
        if hasattr(self, '_settings_manager') and self._settings_manager:
            listen_port = self._settings_manager.listen_port
            listen_address = self._settings_manager.listen_address or "127.0.0.1"
        
        self._log("Starting listener...")
        success, error_message = self._listener_service.start_listener(
            block_id=self.block_id,
            listen_port=listen_port,
            listen_address=listen_address,
            message_handler=self._handle_osc_message
        )
        
        if success:
            # Verify listener is actually running after a brief delay
            # This catches cases where the thread dies immediately after starting
            QTimer.singleShot(500, lambda: self._verify_listener_after_start())
            # Save state to metadata optimistically (will be corrected by verification if needed)
            self._save_listener_state(True)
            # Update UI optimistically (will be corrected by verification if needed)
            self.start_listening_btn.setEnabled(False)
            self.stop_listening_btn.setEnabled(True)
            self.configure_ma3_btn.setEnabled(True)
            # Update connection status display (load from service)
            self._load_connection_state_from_service()
            # Trigger status update with small delay to ensure service state has propagated
            QTimer.singleShot(100, self._trigger_status_update)
            self._log("Listener started successfully")
            # Delegate connection monitoring and reconnect to SSM
            if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                self._sync_system_manager.start_connection_monitoring()
        else:
            error_msg = error_message or "Unknown error"
            self._log(f"ERROR: Failed to start listener: {error_msg}")
            Log.error(f"ShowManagerPanel: Failed to start listener for block {self.block_id}: {error_msg}")
            # Ensure UI reflects actual state
            if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                self.start_listening_btn.setEnabled(True)
                self.stop_listening_btn.setEnabled(False)
                if hasattr(self, 'configure_ma3_btn'):
                    self.configure_ma3_btn.setEnabled(False)
            # Update connection status display (editor state + listener state)
            self._load_connection_state_from_service()
            self._trigger_status_update()
            self.set_status_message(f"Listener failed: {error_msg}", error=True)
    
    def _on_stop_listening(self):
        """Handle stop listening button click."""
        if not self._listener_service:
            return
        
        self._log("Stopping listener...")
        self._listener_service.stop_listener(self.block_id)
        # Delegate connection teardown to SSM
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            self._sync_system_manager.stop_connection_monitoring()
        # Save state to metadata
        self._save_listener_state(False)
        # Update UI
        self.start_listening_btn.setEnabled(True)
        self.stop_listening_btn.setEnabled(False)
        self.configure_ma3_btn.setEnabled(False)
        # Update connection status display (load from service)
        self._load_connection_state_from_service()
        # Trigger status update with small delay to ensure service state has propagated
        QTimer.singleShot(100, self._trigger_status_update)
    
    def _handle_osc_message(self, address: str, args: list, addr: tuple, data: bytes) -> None:
        """Handle OSC message from service (called in listener thread)."""
        # This is called by the service's listener thread
        # We can process it here or let it go through the message queue
        # For now, let it go through the queue for consistency
        pass
    
    def _on_configure_ma3(self):
        """Configure MA3 to send responses to our listener port."""
        if not hasattr(self, '_settings_manager'):
            self._log("ERROR: Settings manager not available")
            return
        
        listen_port = self._settings_manager.listen_port
        echozero_ip = self._resolve_listen_target_ip()
        
        self._log(f"Configuring MA3 to send to {echozero_ip}:{listen_port}...")
        try:
            if self._send_lua_command(f'EZ.SetTarget("{echozero_ip}", {listen_port})'):
                self._log(f"✓ MA3 configured to send to {echozero_ip}:{listen_port}")
            else:
                self._log("ERROR: Failed to send SetTarget command")
        except RuntimeError as e:
            self._log(f"ERROR: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MA3 Connection Error", str(e))
    
    def _on_ping(self):
        """Handle ping button click."""
        self._log("Sending ping...")
        try:
            self._send_lua_command("EZ.Ping()")
        except RuntimeError as e:
            self._log(f"ERROR: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MA3 Connection Error", str(e))
    
    def _save_listener_state(self, listening: bool):
        """Save listener state to block metadata so it persists across panel close/open."""
        if not self.block_id or not self.facade:
            return
        
        try:
            # Use the base class method to update metadata (proper API)
            self.set_multiple_metadata(
                {"osc_listener_active": listening},
                success_message="",
                description="Update OSC listener state"
            )
            Log.debug(f"ShowManagerPanel: Saved listener state to metadata: listening={listening}")
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Failed to save listener state: {e}")
    
    def _check_and_restore_listener(self):
        """Check if listener should be running and restore if needed."""
        if not self.block_id or not self.facade or not self._listener_service:
            return
        
        try:
            result = self.facade.describe_block(self.block_id)
            if not result.success or not result.data:
                return
            
            metadata = result.data.metadata or {}
            should_be_listening = metadata.get("osc_listener_active", False)
            is_listening = self._listener_service.is_listening(self.block_id)
            
            # If metadata says listening but service says not, start it
            if should_be_listening and not is_listening:
                listen_port = 9000
                listen_address = "127.0.0.1"
                if hasattr(self, '_settings_manager') and self._settings_manager:
                    listen_port = self._settings_manager.listen_port
                    listen_address = self._settings_manager.listen_address or "127.0.0.1"
                
                Log.info("ShowManagerPanel: Restoring listener from metadata")
                success, error_message = self._listener_service.start_listener(
                    block_id=self.block_id,
                    listen_port=listen_port,
                    listen_address=listen_address,
                    message_handler=self._handle_osc_message
                )
                
                if success:
                    # Verify listener is actually running after a brief delay
                    # This catches cases where the thread dies immediately after starting
                    QTimer.singleShot(500, lambda: self._verify_listener_after_start())
                    # Update UI optimistically (will be corrected by verification)
                    if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                        self.start_listening_btn.setEnabled(False)
                        self.stop_listening_btn.setEnabled(True)
                        if hasattr(self, 'configure_ma3_btn'):
                            self.configure_ma3_btn.setEnabled(True)
                    # Delegate connection monitoring to SSM (handles pings, structure fetch, reconnect)
                    if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                        self._sync_system_manager.start_connection_monitoring()
                else:
                    # Failed to restore - clear metadata state to prevent retry loop
                    error_msg = error_message or "Unknown error"
                    Log.warning(f"ShowManagerPanel: Failed to restore listener from metadata: {error_msg}")
                    self._save_listener_state(False)
                    # Update UI to reflect actual state
                    if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                        self.start_listening_btn.setEnabled(True)
                        self.stop_listening_btn.setEnabled(False)
                        if hasattr(self, 'configure_ma3_btn'):
                            self.configure_ma3_btn.setEnabled(False)
                    # Update connection status display (editor state + listener state)
                    self._load_connection_state_from_service()
                    self._trigger_status_update()
            elif is_listening:
                # Listener is running - update UI to reflect state
                # Also ensure metadata matches actual state
                if not should_be_listening:
                    Log.info(f"ShowManagerPanel: Listener is running but metadata says not listening - syncing metadata")
                    self._save_listener_state(True)
                if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                    self.start_listening_btn.setEnabled(False)
                    self.stop_listening_btn.setEnabled(True)
                    if hasattr(self, 'configure_ma3_btn'):
                        self.configure_ma3_btn.setEnabled(True)
                # Delegate connection monitoring to SSM (handles pings, structure fetch, reconnect)
                if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                    self._sync_system_manager.start_connection_monitoring()
            else:
                # Neither listening nor should be - ensure metadata matches
                if should_be_listening:
                    Log.warning(f"ShowManagerPanel: Metadata says listening but service says not - clearing stale metadata")
                    self._save_listener_state(False)
                # Update UI to reflect actual state
                if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                    self.start_listening_btn.setEnabled(True)
                    self.stop_listening_btn.setEnabled(False)
                    if hasattr(self, 'configure_ma3_btn'):
                        self.configure_ma3_btn.setEnabled(False)
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Failed to check listener state: {e}")
    
    def _verify_listener_after_start(self):
        """
        Verify that the listener is actually running after a start attempt.
        
        This catches cases where start_listener() returns success=True but the
        listener thread dies immediately due to socket errors or other issues.
        If the listener is not actually running, update metadata and UI to reflect
        the actual state.
        """
        if not self.block_id or not self._listener_service:
            return
        
        try:
            is_listening = self._listener_service.is_listening(self.block_id)
            
            if not is_listening:
                # Listener failed silently - update metadata and UI to reflect actual state
                Log.warning(f"ShowManagerPanel: Listener verification failed for block {self.block_id} - listener is not actually running")
                self._save_listener_state(False)
                
                # Update UI to reflect actual state
                if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
                    self.start_listening_btn.setEnabled(True)
                    self.stop_listening_btn.setEnabled(False)
                    if hasattr(self, 'configure_ma3_btn'):
                        self.configure_ma3_btn.setEnabled(False)
                
                self._log("WARNING: Listener failed to start properly. Please check port configuration and try again.")
            else:
                # Listener is running - ensure metadata matches
                Log.debug(f"ShowManagerPanel: Listener verification passed for block {self.block_id}")
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Error verifying listener state: {e}")
    
    # Listener management is handled by ShowManagerListenerService.
    
    def _on_ma3_osc_inbound(self, event) -> None:
        """Handle inbound OSC packets published by MA3CommunicationService."""
        data = getattr(event, "data", {}) or {}
        address = data.get("address")
        args = data.get("args") or []
        addr = data.get("addr") or ("unknown", 0)
        raw_data = data.get("raw_data")
        if address:
            # Notify SSM that a message arrived (connection health tracking)
            if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                self._sync_system_manager.on_ma3_message_received()
            self._log_received(address, args, addr, raw_data)
    
    def _log_received(self, address: str, args: list, addr: tuple, raw_data: bytes = None):
        """Log received OSC message (called from main thread)."""
        self._log(f"<<< RECEIVED from {addr[0]}:{addr[1]}")
        self._log(f"    Address: {address}")
        if args:
            self._log(f"    Args: {args}")
        
        # Update OSC Packet Viewer (wrapped in try/except to not block handler processing)
        try:
            # Raw hex view
            if raw_data is not None and hasattr(self, 'osc_raw_text'):
                import time
                timestamp = time.strftime("%H:%M:%S")
                hex_lines = []
                
                for i in range(0, len(raw_data), 16):
                    chunk = raw_data[i:i+16]
                    hex_part = ' '.join(f'{b:02x}' for b in chunk)
                    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                    hex_lines.append(f"{i:04x}  {hex_part:<48}  {ascii_part}")
                
                hex_display = f"[{timestamp}] {addr[0]}:{addr[1]} ({len(raw_data)} bytes)\n"
                hex_display += '\n'.join(hex_lines)
                hex_display += '\n' + '-' * 60 + '\n'
                
                from PyQt6.QtGui import QTextCursor
                self.osc_raw_text.moveCursor(QTextCursor.MoveOperation.Start)
                self.osc_raw_text.insertPlainText(hex_display)
                
                if self.osc_raw_text.document().blockCount() > 200:
                    cursor = self.osc_raw_text.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                    for _ in range(50):
                        cursor.movePosition(QTextCursor.MoveOperation.Up, QTextCursor.MoveMode.KeepAnchor)
                    cursor.removeSelectedText()
            
            # Interpreted view
            if hasattr(self, 'osc_parsed_text'):
                import time
                timestamp = time.strftime("%H:%M:%S")
                parsed_lines = [f"[{timestamp}] {address}"]
                
                if args:
                    for i, arg in enumerate(args):
                        arg_type = type(arg).__name__
                        if isinstance(arg, str) and len(arg) > 100:
                            display = arg[:100] + '...'
                        else:
                            display = str(arg)
                        parsed_lines.append(f"  [{i}] ({arg_type}) {display}")
                    
                    if args and isinstance(args[0], str) and '|' in args[0]:
                        parsed_lines.append("  --- Parsed Fields ---")
                        parts = args[0].split('|')
                        for part in parts:
                            if '=' in part:
                                key, val = part.split('=', 1)
                                parsed_lines.append(f"    {key}: {val[:60]}{'...' if len(val) > 60 else ''}")
                
                parsed_lines.append('-' * 40)
                
                from PyQt6.QtGui import QTextCursor
                self.osc_parsed_text.moveCursor(QTextCursor.MoveOperation.Start)
                self.osc_parsed_text.insertPlainText('\n'.join(parsed_lines) + '\n')
                
                if self.osc_parsed_text.document().blockCount() > 150:
                    cursor = self.osc_parsed_text.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                    for _ in range(30):
                        cursor.movePosition(QTextCursor.MoveOperation.Up, QTextCursor.MoveMode.KeepAnchor)
                    cursor.removeSelectedText()
            
            # Legacy log
            if raw_data is not None:
                hex_str = ' '.join(f'{b:02x}' for b in raw_data[:64])
                if len(raw_data) > 64:
                    hex_str += f'... (+{len(raw_data) - 64} bytes)'
                self._log(f"    Raw (hex): {hex_str}")
                self._log(f"    Raw length: {len(raw_data)} bytes")
        except Exception as e:
            Log.error(f"ShowManagerPanel: Error updating OSC viewer: {e}")
        
        # Parse MA3-specific messages
        if address == "/ma3/debug":
            debug_msg = args[0] if args and len(args) > 0 else "Unknown debug message"
            # Display debug message prominently in the log
            self._log(f"🔍 [MA3 DEBUG] {debug_msg}")
        elif address == "/ma3/error":
            error_msg = args[0] if args and len(args) > 0 else "Unknown error"
            self._log(f"❌ MA3 Error: {error_msg}")
            Log.error(f"ShowManagerPanel: MA3 error - {error_msg}")
        elif address == "/ma3/all_events_start":
            # Start of chunked message
            if args and len(args) >= 3:
                timecode_no = args[0]
                total_chunks = args[1]
                total_events = args[2]
                message_id = f"all_events_{timecode_no}"
                import time
                self._chunked_message_buffer[message_id] = {
                    "chunks": {},
                    "total": total_chunks,
                    "received": 0,
                    "timecode_no": timecode_no,
                    "total_events": total_events,
                    "started_at": time.time(),
                    "chunk_indices_received": set()  # Track which chunks we've received (for duplicate detection)
                }
                self._log(f"📦 Starting chunked message: {total_chunks} chunks, {total_events} events")
        elif address == "/ma3/all_events_chunk":
            # Chunk of message
            if args and len(args) >= 3:
                chunk_idx = args[0]
                total_chunks = args[1]
                chunk_data = args[2]
                # Find message ID by matching total_chunks (should only be one active)
                message_id = None
                for mid, buf in self._chunked_message_buffer.items():
                    if buf["total"] == total_chunks and buf["received"] < buf["total"]:
                        message_id = mid
                        break
                if message_id:
                    buf = self._chunked_message_buffer[message_id]
                    # Check for duplicate chunk (already received this index)
                    if chunk_idx in buf["chunk_indices_received"]:
                        self._log(f"⚠️ Duplicate chunk {chunk_idx}/{total_chunks} ignored")
                        return
                    
                    # Store chunk
                    buf["chunks"][chunk_idx] = chunk_data
                    buf["chunk_indices_received"].add(chunk_idx)
                    buf["received"] = len(buf["chunk_indices_received"])  # Count unique chunks
                    
                    self._log(f"📦 Received chunk {chunk_idx}/{total_chunks} ({buf['received']}/{total_chunks})")
                    
                    # Check if all chunks received (handle out-of-order delivery)
                    if buf["received"] == buf["total"]:
                        # Reassemble chunks in order
                        chunks = []
                        missing_chunks = []
                        for i in range(1, buf["total"] + 1):
                            if i in buf["chunks"]:
                                chunks.append(buf["chunks"][i])
                            else:
                                missing_chunks.append(i)
                        
                        if missing_chunks:
                            self._log(f"❌ ERROR: Missing chunks: {missing_chunks}")
                            Log.error(f"ShowManagerPanel: Missing chunks {missing_chunks} in reassembly")
                            del self._chunked_message_buffer[message_id]
                            return
                        
                        json_str = "".join(chunks)
                        self._log(f"📦 Reassembled {len(json_str)} bytes from {buf['total']} chunks")
                        # Parse
                        self._parse_all_events(json_str)
                        # Clean up
                        del self._chunked_message_buffer[message_id]
                else:
                    self._log(f"⚠️ Received chunk {chunk_idx}/{total_chunks} but no matching buffer found")
        elif address == "/ma3/all_events_end":
            # End marker (redundant, but useful for confirmation)
            if args and len(args) > 0:
                timecode_no = args[0]
                message_id = f"all_events_{timecode_no}"
                if message_id in self._chunked_message_buffer:
                    self._log(f"📦 End marker received for timecode {timecode_no}")
        elif address == "/ma3/all_events":
            # Single-packet message (for small payloads)
            if args and len(args) > 0:
                json_str = args[0]
                self._log(f"Parsing /ma3/all_events with {len(args)} args, JSON length: {len(json_str) if json_str else 0}")
                if json_str:
                    # Log first 200 chars of JSON for debugging
                    preview = json_str[:200] if len(json_str) > 200 else json_str
                    self._log(f"JSON preview: {preview}...")
                self._parse_all_events(json_str)
            else:
                self._log("ERROR: /ma3/all_events received but no args!")
                Log.error("ShowManagerPanel: /ma3/all_events received but args is empty")
        elif address == "/ma3/structure" and args:
            self._handle_ma3_structure(args[0])
        elif address == "/ma3/event_added" and args:
            self._handle_event_added(args)
        elif address == "/ma3/event_modified" and args:
            self._handle_event_modified(args)
        elif address == "/ma3/event_deleted" and args:
            self._handle_event_deleted(args)
        # Structure change notifications from hooks
        elif address == "/ma3/structure/track_added" and args:
            self._handle_structure_track_added(args)
        elif address == "/ma3/structure/track_removed" and args:
            self._handle_structure_track_removed(args)
        elif address == "/ma3/structure/track_changed" and args:
            self._handle_structure_track_changed(args)
        elif address == "/ma3/structure/track_group_added" and args:
            self._handle_structure_track_group_added(args)
        elif address == "/ma3/structure/track_group_removed" and args:
            self._handle_structure_track_group_removed(args)
        elif address == "/ma3/structure/timecode_added" and args:
            self._handle_structure_timecode_added(args)
        elif address == "/ez/message" and args:
            # Handle pipe-delimited EchoZero messages from MA3 Lua plugin
            self._handle_ez_message(args[0] if args else "")
    
    def _handle_ez_message(self, message_str: str):
        """
        Handle pipe-delimited message from MA3 Lua plugin.
        
        Format: type=X|change=Y|timestamp=Z|...data fields...
        Uses the new OSCParser for consistent parsing.
        """
        if not message_str:
            return
        
        
        try:
            # Use the new OSC parser
            parser = get_osc_parser()
            message = parser.parse_message(message_str)
            
            if not message:
                self._log(f"Failed to parse message: {message_str[:50]}...")
                return
            
            
            self._log(f"EZ Message: {message.type_key}")
            
            # Route based on type key
            type_key = message.type_key
            
            if type_key == "trackgroups.list":
                self._handle_trackgroups_list_v2(message)
            elif type_key == "tracks.list":
                self._handle_tracks_list_v2(message)
            elif type_key == "events.list":
                self._handle_events_list(message.data)
            elif type_key == "events.all":
                self._handle_all_events(message.data)
            # Hook notifications
            elif type_key == "track.hooked":
                self._handle_track_hooked(message)
            elif type_key == "subtrack.hooked":
                # HookCmdSubTrack sends this - same handler as track.hooked
                self._handle_track_hooked(message)
            elif type_key == "track.unhooked":
                self._handle_track_unhooked(message)
            elif type_key == "track.changed":
                self._handle_track_changed(message)
            elif type_key.startswith("event."):
                pass  # No handler yet
            elif type_key.startswith("track.") or type_key.startswith("subtrack."):
                pass  # No handler for generic track/subtrack messages
            elif type_key == "tracks.unhooked_all":
                self._handle_tracks_unhooked_all(message)
            # Track group hooks
            elif type_key == "trackgroup.hooked":
                count = message.get('count', 0)
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                self._log(f"Track group hooked: TC{tc}.TG{tg} ({count} tracks)")
            elif type_key == "trackgroup.unhooked":
                count = message.get('count', 0)
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                self._log(f"Track group unhooked: TC{tc}.TG{tg} ({count} tracks)")
            # Hook test results
            elif type_key == "hook_test.success":
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                track = message.get('track', 0)
                msg = message.get('message', 'Hook created')
                self._log(f"HOOK TEST SUCCESS: TC{tc}.TG{tg}.TR{track}")
                self._log(f"  {msg}")
            elif type_key == "hook_test.failed":
                tc = message.get('tc', 0)
                tg = message.get('tg', 0)
                track = message.get('track', 0)
                self._log(f"HOOK TEST FAILED: TC{tc}.TG{tg}.TR{track}")
            elif type_key == "hook_test.error":
                error = message.get('error', 'Unknown error')
                self._log(f"HOOK TEST ERROR: {error}")
            # Hook listing
            elif type_key == "hooks.list":
                count = message.get('count', 0)
                hooks = message.get('hooks', [])
                self._log(f"Active hooks: {count}")
                for h in hooks:
                    self._log(f"  - {h.get('key', '?')}")
            elif type_key == "hooks.trace" or (message.message_type == MessageType.HOOKS and message.change_type == ChangeType.UNKNOWN):
                data = message.data if hasattr(message, "data") else None
                return
            # Connection messages
            elif message.message_type == MessageType.CONNECTION:
                # Notify SSM (connection health tracking)
                if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                    self._sync_system_manager.on_ma3_message_received()
                if message.change_type == ChangeType.PING:
                    self._log("Received ping response from MA3")
                elif message.change_type == ChangeType.STATUS:
                    self._ma3_status_summary = message.data if isinstance(message.data, dict) else {}
                    self._log(f"MA3 Status: {message.data}")
            elif message.change_type == ChangeType.ERROR:
                error_msg = message.get('error', 'Unknown error')
                self._log(f"MA3 Error ({message.message_type.value}): {error_msg}")
                
        except Exception as e:
            Log.error(f"ShowManagerPanel: Error handling EZ message: {e}")
            self._log(f"ERROR handling EZ message: {e}")
    
    # NOTE: Old _handle_trackgroups_list and _handle_tracks_list removed
    # Now using _handle_trackgroups_list_v2 and _handle_tracks_list_v2
    
    def _handle_events_list(self, data: dict):
        """Handle events.list message - update events for a specific track."""
        tc = data.get('tc', 0)
        tg = data.get('tg', 0)
        track = data.get('track', 0)
        count = data.get('count', 0)
        events = data.get('events', [])
        offset = data.get('offset', None)
        chunk_index = data.get('chunk_index', None)
        total_chunks = data.get('total_chunks', None)
        request_id = data.get('request_id', None)
        checksum = data.get('checksum', None)
        
        
        # Chunked events support (merge before processing)
        coord = f"tc{tc}_tg{tg}_tr{track}"
        Log.info(f"[MULTITRACK-DIAG] events.list arrived: coord={coord}, chunk={chunk_index}/{total_chunks}, request_id={request_id}, event_count={len(events) if events else 0}")
        if total_chunks or chunk_index or offset or request_id is not None:
            if not hasattr(self, "_pending_events_chunks"):
                self._pending_events_chunks = {}
            if hasattr(self, "_sync_system_manager") and self._sync_system_manager:
                latest_request_id = self._sync_system_manager.get_latest_events_request_id(coord)
                if latest_request_id is not None and request_id is not None and int(request_id) != int(latest_request_id):
                    Log.debug(f"Ignoring stale events.list for {coord} (request_id={request_id}, latest={latest_request_id})")
                    return
            entry = self._pending_events_chunks.get(coord)
            if not entry:
                entry = {
                    "total": count,
                    "total_chunks": total_chunks or 1,
                    "chunks": {},
                    "request_id": request_id,
                    "checksum": checksum
                }
                self._pending_events_chunks[coord] = entry
            if entry.get("checksum") is not None and checksum is not None and int(entry.get("checksum")) != int(checksum):
                Log.warning(f"ShowManagerPanel: Checksum mismatch for {coord} (got={checksum}, expected={entry.get('checksum')})")
                return
            if entry.get("request_id") is not None and request_id is not None and int(entry.get("request_id")) != int(request_id):
                # New request started; reset entry for this coord
                entry = {
                    "total": count,
                    "total_chunks": total_chunks or 1,
                    "chunks": {},
                    "request_id": request_id,
                    "checksum": checksum
                }
                self._pending_events_chunks[coord] = entry
            entry["chunks"][int(chunk_index or 1)] = {
                "offset": int(offset or 1),
                "events": events or []
            }
            if len(entry["chunks"]) < int(entry["total_chunks"] or 1):
                Log.info(f"========== RECEIVED: events.list chunk {chunk_index}/{total_chunks} for {coord} ==========")
                self._log(f"=== Received events.list chunk {chunk_index}/{total_chunks} for {coord} ===")
                return
            # Merge chunks by offset
            events = _merge_event_chunks(entry)
            count = entry["total"]
            checksum = entry.get("checksum")
            del self._pending_events_chunks[coord]
        
        if checksum is not None:
            try:
                expected_checksum = int(checksum)
            except (ValueError, TypeError):
                expected_checksum = None
            if expected_checksum is not None:
                actual_checksum = _compute_events_checksum(events or [])
                if actual_checksum != expected_checksum:
                    Log.warning(f"ShowManagerPanel: Events checksum mismatch for {coord} (actual={actual_checksum}, expected={expected_checksum})")
                    return
        
        Log.info(f"")
        Log.info(f"========== RECEIVED: events.list ==========")
        Log.info(f"  Track: TC{tc}.TG{tg}.TR{track}")
        Log.info(f"  Event count: {count}")
        
        self._log(f"=== Received events.list for TC{tc}.TG{tg}.TR{track} ===")
        self._log(f"  Event count: {count}")
        
        # Print each event
        for i, evt in enumerate(events):
            event_time = evt.get('time', 0)
            if isinstance(event_time, str):
                event_time = float(event_time)
            evt_name = evt.get('name', '')
            evt_cmd = evt.get('cmd', '')
            evt_idx = evt.get('idx', i+1)
            
            log_line = f"  [{evt_idx}] time={event_time:.6f}s"
            if evt_name:
                log_line += f", name='{evt_name}'"
            if evt_cmd:
                log_line += f", cmd='{evt_cmd}'"
            
            Log.info(log_line)
            self._log(log_line)
        
        Log.info(f"============================================")
        self._log(f"============================================")
        
        # Route to SyncSystemManager for sync layer handling
        # NOTE: Route even if events is empty (0-event layers need sync too)
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            self._sync_system_manager.on_track_events_received(coord, events or [])
            Log.info(f"SyncSystemManager: Routed {len(events)} events for {coord}")
        
        # Trigger reconcile flow when requested events arrive
        self._handle_reconcile_events(tc, tg, track, events)
    
    def _handle_all_events(self, data: dict):
        """Handle events.all message - all events from a timecode."""
        tc = data.get('tc', 0)
        count = data.get('count', 0)
        tracks = data.get('tracks', [])  # List of track entries with events
        
        Log.info(f"")
        Log.info(f"========== RECEIVED: events.all ==========")
        Log.info(f"  Timecode: {tc}")
        Log.info(f"  Track entries: {count}")
        
        self._log(f"=== Received events.all for TC{tc} ===")
        self._log(f"  Track entries: {count}")
        
        # Print events for each track
        total_events = 0
        for track_entry in tracks:
            track_tc = track_entry.get('tc', tc)
            track_tg = track_entry.get('tg', 0)
            track_no = track_entry.get('track', 0)
            track_name = track_entry.get('track_name', '')
            track_events = track_entry.get('events', [])
            
            Log.info(f"  --- Track TC{track_tc}.TG{track_tg}.TR{track_no} ({track_name}) ---")
            self._log(f"  --- Track TC{track_tc}.TG{track_tg}.TR{track_no} ({track_name}) ---")
            
            for i, evt in enumerate(track_events):
                event_time = evt.get('time', 0)
                if isinstance(event_time, str):
                    event_time = float(event_time)
                evt_name = evt.get('name', '')
                evt_cmd = evt.get('cmd', '')
                evt_idx = evt.get('idx', i+1)
                
                log_line = f"    [{evt_idx}] time={event_time:.6f}s"
                if evt_name:
                    log_line += f", name='{evt_name}'"
                if evt_cmd:
                    log_line += f", cmd='{evt_cmd}'"
                
                Log.info(log_line)
                self._log(log_line)
                total_events += 1
        
        Log.info(f"  Total events across all tracks: {total_events}")
        Log.info(f"==========================================")
        self._log(f"  Total events: {total_events}")
        self._log(f"==========================================")

    def _start_sync_layer_reconcile(self):
        """Start reconciliation for all synced MA3 layers on reconnect."""
        if self._reconcile_in_progress:
            return
        if not hasattr(self, '_settings_manager') or not self._settings_manager.is_loaded():
            return
        if not self._is_ma3_ready():
            return

        ma3_entities = self._settings_manager.get_synced_ma3_tracks()
        if not ma3_entities:
            return

        self._reconcile_pending.clear()
        self._reconcile_results = []
        self._reconcile_dialog_active = False
        self._reconcile_request_queue = []
        self._reconcile_active_key = None
        for entity in ma3_entities:
            coord = entity.get("coord")
            mapped_layer_id = entity.get("mapped_editor_layer_id")
            if not coord or not mapped_layer_id:
                continue

            parsed = self._parse_ma3_coord(coord)
            if not parsed:
                continue
            tc, tg, track = parsed
            key = f"{tc}.{tg}.{track}"
            if key in self._reconcile_pending:
                continue

            editor_block_id = None
            editor_entity = self._settings_manager.get_synced_layer("editor", mapped_layer_id)
            if editor_entity:
                editor_block_id = editor_entity.get("block_id")
            if not editor_block_id:
                editor_block_id = self._find_connected_editor()
            if not editor_block_id:
                continue

            self._reconcile_pending[key] = {
                "ma3_coord_key": coord,
                "mapped_layer_id": mapped_layer_id,
                "editor_block_id": editor_block_id,
                "tc": tc,
                "tg": tg,
                "track": track,
            }
            self._reconcile_request_queue.append(key)

        self._reconcile_in_progress = bool(self._reconcile_pending)
        self._send_next_reconcile_request()
        if self._reconcile_in_progress:
            Log.info(f"ShowManagerPanel: Reconciling {len(self._reconcile_pending)} sync layer(s)")

    def _build_reconcile_info(
        self,
        tc: int,
        tg: int,
        track: int,
        ma3_coord_key: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Build reconcile context for a MA3 track."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager.is_loaded():
            return None

        coord_key = f"{tc}.{tg}.{track}"
        ma3_coord_key = ma3_coord_key or f"tc{tc}_tg{tg}_tr{track}"
        mapped_layer_id = self._find_mapped_editor_layer(ma3_coord_key)
        if not mapped_layer_id:
            return None

        editor_block_id = None
        editor_entity = self._settings_manager.get_synced_layer("editor", mapped_layer_id)
        if editor_entity:
            editor_block_id = editor_entity.get("block_id")
        if not editor_block_id:
            editor_block_id = self._find_connected_editor()
        if not editor_block_id:
            return None

        config = self._get_sync_config_for_coord(ma3_coord_key)
        return {
            "coord_key": coord_key,
            "ma3_coord_key": ma3_coord_key,
            "mapped_layer_id": mapped_layer_id,
            "editor_block_id": editor_block_id,
            "tc": tc,
            "tg": tg,
            "track": track,
            **config,
        }

    def _start_reconcile_for_coord(self, ma3_coord_key: str) -> None:
        """Request reconciliation for a single MA3 coord."""
        parsed = self._parse_ma3_coord(ma3_coord_key)
        if not parsed:
            return
        tc, tg, track = parsed
        info = self._build_reconcile_info(tc, tg, track, ma3_coord_key=ma3_coord_key)
        if not info:
            return
        info["auto_apply"] = False
        key = f"{tc}.{tg}.{track}"
        if key in self._reconcile_pending:
            return
        self._reconcile_pending[key] = info
        self._reconcile_request_queue.append(key)
        self._reconcile_in_progress = True
        self._send_next_reconcile_request()

    def _store_reconcile_cache_entry(
        self,
        info: Dict[str, Any],
        ma3_events: List[Dict[str, Any]],
        editor_events: List[Dict[str, Any]],
        comparison: Any,
    ) -> Dict[str, Any]:
        """Store a reconcile entry in the always-on cache."""
        import time

        apply_enabled = self._apply_updates_enabled_for_coord(info["ma3_coord_key"])
        entry = {
            "coord_key": info["coord_key"],
            "ma3_coord_key": info["ma3_coord_key"],
            "mapped_layer_id": info["mapped_layer_id"],
            "editor_block_id": info["editor_block_id"],
            "tc": info["tc"],
            "tg": info["tg"],
            "track": info["track"],
            "ma3_events": ma3_events,
            "editor_events": editor_events,
            "comparison": comparison,
            "apply_enabled": apply_enabled,
            "last_updated": time.time(),
        }
        self._reconcile_cache[info["coord_key"]] = entry
        return entry

    def _refresh_reconcile_cache_from_ma3_events(
        self,
        tc: int,
        tg: int,
        track: int,
        ma3_events: List[Dict[str, Any]],
    ) -> None:
        """Refresh reconcile cache from MA3 events updates."""
        from src.features.blocks.application.editor_api import create_editor_api

        info = self._build_reconcile_info(tc, tg, track)
        if not info:
            return
        editor_api = create_editor_api(self.facade, info["editor_block_id"])
        if not editor_api:
            return
        editor_events = editor_api.get_events_in_layer(info["mapped_layer_id"])
        comparison = self._sync_layer_manager.compare_events(editor_events, ma3_events)
        self._store_reconcile_cache_entry(info, ma3_events, editor_events, comparison)

    def _refresh_reconcile_cache_from_editor_layer(self, layer_id: str) -> None:
        """Refresh reconcile cache from Editor events updates."""
        from src.features.blocks.application.editor_api import create_editor_api

        if not hasattr(self, "_settings_manager") or not self._settings_manager.is_loaded():
            return

        editor_entity = self._settings_manager.get_synced_layer("editor", layer_id)
        if not editor_entity:
            return
        ma3_coord_key = editor_entity.get("mapped_ma3_track_id")
        if not ma3_coord_key:
            return

        parsed = self._parse_ma3_coord(ma3_coord_key)
        if not parsed:
            return
        tc, tg, track = parsed

        # Get MA3 events from sync system manager
        coord = f"tc{tc}_tg{tg}_tr{track}"
        ma3_events = self._sync_system_manager._get_ma3_events(coord)
        if not ma3_events:
            return

        info = self._build_reconcile_info(tc, tg, track, ma3_coord_key=ma3_coord_key)
        if not info:
            return
        editor_api = create_editor_api(self.facade, info["editor_block_id"])
        if not editor_api:
            return
        editor_events = editor_api.get_events_in_layer(info["mapped_layer_id"])
        comparison = self._sync_layer_manager.compare_events(editor_events, ma3_events)
        self._store_reconcile_cache_entry(info, ma3_events, editor_events, comparison)

    def _send_next_reconcile_request(self) -> None:
        """Send the next MA3 events request sequentially."""
        if self._reconcile_active_key:
            return
        while self._reconcile_request_queue:
            key = self._reconcile_request_queue.pop(0)
            if key not in self._reconcile_pending:
                continue
            info = self._reconcile_pending.get(key)
            if not info:
                continue
            tc = info.get("tc")
            tg = info.get("tg")
            track = info.get("track")
            coord = info.get("ma3_coord_key", key)
            if tc is None or tg is None or track is None:
                self._reconcile_pending.pop(key, None)
                continue
            self._reconcile_active_key = key
            # Get cached events and hook status from sync system manager
            norm_coord = f"tc{tc}_tg{tg}_tr{track}"
            cached_events = self._sync_system_manager._get_ma3_events(norm_coord) or None
            hooked = self._sync_system_manager.is_track_hooked(norm_coord)
            try:
                if not hooked:
                    self._send_lua_command(f"EZ.HookCmdSubTrack({tc}, {tg}, {track})")
                else:
                    self._send_lua_command(f"EZ.GetEvents({tc}, {tg}, {track})")
            except RuntimeError as e:
                Log.warning(f"ShowManagerPanel: Reconcile request failed for {coord}: {e}")
                self._reconcile_pending.pop(key, None)
                self._reconcile_active_key = None
                continue
            return

    def _handle_reconcile_events(self, tc: int, tg: int, track: int, ma3_events: list):
        """Handle reconciliation when events.list response arrives."""
        key = f"{tc}.{tg}.{track}"
        info = self._reconcile_pending.get(key)
        if not info:
            return

        from src.features.blocks.application.editor_api import create_editor_api

        editor_block_id = info.get("editor_block_id")
        mapped_layer_id = info.get("mapped_layer_id")
        if not editor_block_id or not mapped_layer_id:
            self._reconcile_pending.pop(key, None)
            return

        editor_api = create_editor_api(self.facade, editor_block_id)
        if not editor_api:
            self._reconcile_pending.pop(key, None)
            return

        editor_events = editor_api.get_events_in_layer(mapped_layer_id)
        comparison = self._sync_layer_manager.compare_events(editor_events, ma3_events)

        info = {
            **info,
            "coord_key": key,
            "ma3_coord_key": info.get("ma3_coord_key", f"tc{tc}_tg{tg}_tr{track}"),
        }
        cache_entry = self._store_reconcile_cache_entry(info, ma3_events, editor_events, comparison)
        auto_apply = bool(info.get("auto_apply"))
        if auto_apply and comparison.diverged:
            conflict_strategy = info.get("conflict_strategy")
            action = self._resolve_auto_apply_action(conflict_strategy)
            if action and cache_entry.get("apply_enabled", True):
                self._apply_reconcile_action(cache_entry, action)
        if comparison.diverged and not auto_apply:
            self._reconcile_results.append(cache_entry)

        self._reconcile_pending.pop(key, None)
        if self._reconcile_active_key == key:
            self._reconcile_active_key = None
        if not self._reconcile_pending:
            self._reconcile_in_progress = False
            if self._reconcile_results:
                self._show_reconcile_table_dialog()
        else:
            self._send_next_reconcile_request()

    def _show_reconcile_table_dialog(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        title: str = "Sync Divergence Resolution",
        info_text: str = "Resolve each out-of-sync layer/track pair below.",
        allow_actions_for_synced: bool = True,
    ) -> None:
        """Show a table dialog to resolve or review sync status."""
        if self._reconcile_dialog_active:
            return
        self._reconcile_dialog_active = True

        clear_results = results is None
        results = list(self._reconcile_results) if results is None else list(results)
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setSizeGripEnabled(True)
        
        layout = QVBoxLayout(dialog)
        info_label = QLabel(info_text)
        info_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
        layout.addWidget(info_label)
        
        allow_paused_checkbox = QCheckBox("Allow actions when apply is paused")
        allow_paused_checkbox.setChecked(True)
        layout.addWidget(allow_paused_checkbox)
        
        table = QTableWidget(dialog)
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            "Layer",
            "MA3 Track",
            "Status",
            "MA3 Events",
            "Editor Events",
            "Matched",
            "Apply Paused",
            "Action"
        ])
        table.setRowCount(len(results))
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        
        action_widgets: List[QComboBox] = []
        paused_rows: List[bool] = []
        action_allowed_rows: List[bool] = []
        for row, result in enumerate(results):
            comparison = result["comparison"]
            table.setItem(row, 0, QTableWidgetItem(result["mapped_layer_id"]))
            table.setItem(row, 1, QTableWidgetItem(result["ma3_coord_key"]))
            diverged = bool(comparison.diverged)
            status_text = "Diverged" if diverged else "In Sync"
            table.setItem(row, 2, QTableWidgetItem(status_text))
            table.setItem(row, 3, QTableWidgetItem(str(comparison.ma3_count)))
            table.setItem(row, 4, QTableWidgetItem(str(comparison.editor_count)))
            table.setItem(row, 5, QTableWidgetItem(str(comparison.matched_count)))
            
            paused = not result["apply_enabled"]
            paused_rows.append(paused)
            table.setItem(row, 6, QTableWidgetItem("Yes" if paused else "No"))
            action_allowed = allow_actions_for_synced or diverged
            action_allowed_rows.append(action_allowed)
            
            action_combo = QComboBox(table)
            action_combo.addItem("Overwrite Editor", "overwrite")
            action_combo.addItem("Merge into Editor", "merge")
            action_combo.addItem("Skip", "skip")
            action_combo.setCurrentIndex(2)
            if not action_allowed:
                action_combo.setEnabled(False)
            table.setCellWidget(row, 7, action_combo)
            action_widgets.append(action_combo)
        
        def _update_action_availability():
            allow_paused = allow_paused_checkbox.isChecked()
            for idx, combo in enumerate(action_widgets):
                if not action_allowed_rows[idx]:
                    combo.setEnabled(False)
                    continue
                if paused_rows[idx] and not allow_paused:
                    combo.setCurrentIndex(2)
                    combo.setEnabled(False)
                else:
                    combo.setEnabled(True)
        
        allow_paused_checkbox.stateChanged.connect(_update_action_availability)
        _update_action_availability()
        
        layout.addWidget(table)

        table.resizeColumnsToContents()
        total_width = table.verticalHeader().width() + table.horizontalHeader().length() + table.frameWidth() * 2
        total_width += table.verticalScrollBar().sizeHint().width()
        screen = dialog.screen()
        max_width = int(screen.availableGeometry().width() * 0.9) if screen else total_width
        dialog.setMinimumWidth(min(total_width + 40, max_width))
        
        button_row = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        close_btn = QPushButton("Close")
        button_row.addStretch()
        button_row.addWidget(apply_btn)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        
        
        def _apply_actions():
            applied_any = False
            allow_paused = allow_paused_checkbox.isChecked()
            for idx, result in enumerate(results):
                if not action_allowed_rows[idx]:
                    continue
                action = action_widgets[idx].currentData()
                if action == "skip":
                    continue
                if not result["apply_enabled"] and not allow_paused:
                    continue
                self._apply_reconcile_action(result, action)
                applied_any = True
            if applied_any:
                self._maybe_persist_sync_changes()
            dialog.accept()
        
        def _close_dialog():
            dialog.reject()
        
        apply_btn.clicked.connect(_apply_actions)
        close_btn.clicked.connect(_close_dialog)
        
        dialog.exec()
        if clear_results:
            self._reconcile_results = []
        self._reconcile_dialog_active = False

    def _apply_reconcile_action(self, result: Dict[str, Any], action: str) -> None:
        """Apply a reconcile action for a single row."""
        from src.features.blocks.application.editor_api import create_editor_api
        
        tc = result["tc"]
        tg = result["tg"]
        track = result["track"]
        mapped_layer_id = result["mapped_layer_id"]
        ma3_events = result["ma3_events"]
        editor_events = result["editor_events"]
        ma3_coord_key = result["ma3_coord_key"]
        
        if not result["apply_enabled"]:
            Log.info(
                f"ShowManagerPanel: Apply paused for {ma3_coord_key}, "
                f"but user chose {action} during divergence resolution"
            )
        
        editor_api = create_editor_api(self.facade, result["editor_block_id"])
        if not editor_api:
            Log.warning(f"ShowManagerPanel: EditorAPI unavailable for {mapped_layer_id}")
            return
        
        if action == "overwrite":
            ma3_coord = f"{tc}.{tg}.{track}"
            events_to_add = self._build_editor_sync_events(
                mapped_layer_id,
                ma3_coord,
                ma3_events
            )
            editor_api.sync_layer_replace(
                layer_name=mapped_layer_id,
                events=events_to_add,
                source="ma3_sync"
            )
        elif action == "merge":
            merged_events = self._sync_layer_manager.merge_events(editor_events, ma3_events)
            ma3_coord = f"{tc}.{tg}.{track}"
            events_to_add = self._build_editor_sync_events(
                mapped_layer_id,
                ma3_coord,
                merged_events
            )
            editor_api.sync_layer_replace(
                layer_name=mapped_layer_id,
                events=events_to_add,
                source="ma3_sync"
            )
        

    def _parse_ma3_coord(self, coord: str) -> Optional[Tuple[int, int, int]]:
        """Parse coord like 'tc101_tg1_tr2' into (tc, tg, track)."""
        try:
            if not coord.startswith("tc"):
                return None
            parts = coord.split("_")
            tc = int(parts[0].replace("tc", ""))
            tg = int(parts[1].replace("tg", ""))
            track = int(parts[2].replace("tr", ""))
            return tc, tg, track
        except Exception:
            return None

    def _parse_coord_key(self, coord: str) -> Optional[Tuple[int, int, int]]:
        """Parse coord like '101.1.2' into (tc, tg, track)."""
        try:
            parts = coord.split(".")
            if len(parts) != 3:
                return None
            return int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            return None

    def _build_editor_sync_events(
        self,
        layer_name: str,
        ma3_coord: str,
        ma3_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build editor event dicts for sync_layer_replace."""
        from src.features.show_manager.domain.sync_state import compute_fingerprint

        events_to_add: List[Dict[str, Any]] = []
        for evt in ma3_events:
            if not isinstance(evt, dict):
                continue
            ma3_idx = evt.get("idx") or evt.get("no") or evt.get("index")
            fingerprint = compute_fingerprint(evt)
            events_to_add.append({
                "time": float(evt.get("time", 0.0)),
                "duration": float(evt.get("duration", 0.0)),
                "classification": layer_name,
                "metadata": {
                    "ma3_idx": ma3_idx,
                    "ma3_fingerprint": fingerprint,
                    "ma3_coord": ma3_coord,
                    "ma3_name": evt.get("name", ""),
                    "ma3_cmd": evt.get("cmd", ""),
                    "source": "ma3_sync",
                },
            })
        return events_to_add

    def _maybe_persist_sync_changes(self) -> None:
        """Persist sync changes to project file if possible (debounced)."""
        try:
            import time
            now = time.time()
            if now - self._last_sync_save_ts < self._sync_save_min_interval:
                return

            if not self.facade or not self.facade.current_project_id:
                return

            project_result = self.facade.describe_project(self.facade.current_project_id)
            if not project_result.success or not project_result.data:
                return

            project = project_result.data
            if not getattr(project, "save_directory", None):
                return

            self.facade.save_project()
            self._last_sync_save_ts = now
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Failed to persist sync changes: {e}")

    def _handle_trackgroups_list_v2(self, message: OSCMessage):
        """
        Handle trackgroups.list message using new OSCMessage format.
        
        Uses the OSCParser for structured data access.
        NOTE: Does NOT auto-fetch tracks - that's done on-demand when dialog opens.
        """
        parser = get_osc_parser()
        
        tc = message.get('tc', 0)
        count = message.get('count', 0)
        trackgroups = parser.parse_trackgroups(message)
        
        self._log(f"Received {count} track groups for timecode {tc}")
        
        # Route to SyncSystemManager (single source of truth)
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            groups_data = [
                {'no': tg.no, 'name': tg.name, 'track_count': tg.track_count}
                for tg in trackgroups
            ]
            self._sync_system_manager.on_track_groups_received(tc, groups_data)
        
        # Legacy: Store track groups info for later use (to be removed)
        if not hasattr(self, '_ma3_track_groups'):
            self._ma3_track_groups = {}
        self._ma3_track_groups[tc] = [
            {'no': tg.no, 'name': tg.name, 'track_count': tg.track_count}
            for tg in trackgroups
        ]
        
        # Log track groups
        for tg in trackgroups:
            self._log(f"  TG[{tg.no}]: '{tg.name}' ({tg.track_count} tracks)")

        # If a create request is pending, create track now (no need to fetch tracks first)
        if getattr(self, "_pending_track_create", None):
            info = self._pending_track_create
            if not info.get("create_attempted"):
                ma3_comm = getattr(self.facade, "ma3_communication_service", None)
                if ma3_comm:
                    chosen_tg = info.get("track_group") or (trackgroups[0].no if trackgroups else 1)
                    info["track_group"] = chosen_tg
                    info["create_attempted"] = True
                    ma3_comm.create_track(info["timecode_no"], chosen_tg, info["track_name"])
                    # After creating the track, fetch tracks so we can resolve pending link
                    try:
                        self._send_lua_command(f"EZ.GetTracks({tc}, {chosen_tg})")
                    except Exception as e:
                        Log.warning(f"Failed to fetch tracks after create for TG {chosen_tg}: {e}")
                    self._request_ma3_structure_refresh("track_create")
        
        # If dialog is pending, fetch tracks for all track groups
        if getattr(self, '_force_tracks_refresh', False):
            self._force_tracks_refresh = False
            self._tracks_need_fetch = False
            self._pending_track_groups_count = len(trackgroups)
            self._received_track_groups_count = 0
            for tg in trackgroups:
                cmd = f"EZ.GetTracks({tc}, {tg.no})"
                try:
                    self._send_lua_command(cmd)
                except Exception as e:
                    Log.warning(f"Failed to fetch tracks for TG {tg.no}: {e}")
                    self._received_track_groups_count += 1
        elif getattr(self, '_pending_ma3_track_dialog', False):
            from src.utils.message import Log
            Log.info(f"ShowManagerPanel: Dialog pending, fetching tracks for {len(trackgroups)} track groups...")
            # Track how many track groups we're waiting for
            self._pending_track_groups_count = len(trackgroups)
            self._received_track_groups_count = 0
            for tg in trackgroups:
                cmd = f"EZ.GetTracks({tc}, {tg.no})"
                try:
                    self._send_lua_command(cmd)
                except Exception as e:
                    Log.warning(f"Failed to fetch tracks for TG {tg.no}: {e}")
                    self._received_track_groups_count += 1  # Count as received (failed)
        else:
            # Mark that we need to fetch tracks when dialog opens
            self._tracks_need_fetch = True

    def _handle_tracks_list_v2(self, message: OSCMessage):
        """
        Handle tracks.list message using new OSCMessage format.
        
        Uses the OSCParser for structured data access.
        """
        parser = get_osc_parser()
        
        tc = message.get('tc', 0)
        tg = message.get('tg', 0)
        count = message.get('count', 0)
        tracks = parser.parse_tracks(message)
        
        self._log(f"Received {count} tracks for TC{tc}.TG{tg}")
        
        # Route to SyncSystemManager (single source of truth)
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            tracks_data = [
                {'no': track.no, 'name': track.name or f"Track {track.no}", 'event_count': 0}
                for track in tracks
            ]
            self._sync_system_manager.on_tracks_received(tc, tg, tracks_data)
        
        # Legacy: Create MA3TrackInfo for each track (to be removed)
        for track in tracks:
            test_coord = f"tc{tc}_tg{tg}_tr{track.no}"
            existing = next((t for t in self._ma3_tracks if t.coord == test_coord), None)
            
            if not existing:
                track_info = MA3TrackInfo(
                    timecode_no=tc,
                    track_group=tg,
                    track=track.no,
                    name=track.name or f"Track {track.no}",
                    event_count=0
                )
                self._ma3_tracks.append(track_info)
                self._log(f"  Added: {track_info.name} ({track_info.coord})")
            else:
                self._log(f"  Track already exists: {test_coord}")
        
        # Refresh UI with new tracks
        self._update_synced_layers_list()
        self._resolve_pending_track_create()
        
        # Track received track groups count for pending dialog
        if hasattr(self, '_pending_track_groups_count') and self._pending_track_groups_count > 0:
            self._received_track_groups_count = getattr(self, '_received_track_groups_count', 0) + 1
            from src.utils.message import Log
            Log.debug(f"ShowManagerPanel: Received tracks for TG {tg}, {self._received_track_groups_count}/{self._pending_track_groups_count} track groups")
            
            # Open dialog when all track groups received - NO event fetch needed yet
            if self._received_track_groups_count >= self._pending_track_groups_count:
                self._pending_track_groups_count = 0
                self._received_track_groups_count = 0
                if self._pending_editor_track_select:
                    pending = self._pending_editor_track_select
                    self._pending_editor_track_select = None
                    editor_layer_id = pending.get("editor_layer_id")
                    Log.info("ShowManagerPanel: All tracks received, opening editor track dialog")
                    self._open_ma3_track_dialog_for_editor(editor_layer_id)
                elif self._pending_ma3_track_dialog:
                    self._pending_ma3_track_dialog = False
                    Log.info("ShowManagerPanel: All tracks received, opening dialog")
                    self._open_ma3_track_dialog()
        elif self._pending_editor_track_select:
            pending = self._pending_editor_track_select
            self._pending_editor_track_select = None
            from src.utils.message import Log
            editor_layer_id = pending.get("editor_layer_id")
            Log.info("ShowManagerPanel: Tracks received, opening editor track dialog")
            self._open_ma3_track_dialog_for_editor(editor_layer_id)
        elif self._pending_ma3_track_dialog:
            # Fallback: only one track group - open dialog immediately
            self._pending_ma3_track_dialog = False
            from src.utils.message import Log
            Log.info("ShowManagerPanel: Tracks received, opening dialog")
            self._open_ma3_track_dialog()

    def _handle_track_hooked(self, message: OSCMessage):
        """
        Handle track.hooked / subtrack.hooked message - MA3 confirmed hook registered.
        
        This message now includes initial events from the track, so we can sync them
        immediately without needing a separate fetch.
        """
        tc = message.get('tc', 0)
        tg = message.get('tg', 0)
        track = message.get('track', 0)
        subtrack = message.get('subtrack', 0)
        events = message.get('events', [])
        event_count = message.get('event_count', len(events))
        coord = f"tc{tc}_tg{tg}_tr{track}"
        
        Log.info(f"")
        Log.info(f"========== RECEIVED: subtrack.hooked ==========")
        Log.info(f"  Track: {coord}")
        Log.info(f"  Initial events: {event_count}")
        
        self._log(f"=== Track hooked: {coord} ===")
        self._log(f"  Initial events: {event_count}")
        
        # Print each initial event
        for i, evt in enumerate(events):
            event_time = evt.get('time', 0)
            if isinstance(event_time, str):
                event_time = float(event_time)
            evt_name = evt.get('name', '')
            evt_cmd = evt.get('cmd', '')
            evt_idx = evt.get('idx', i+1)
            
            log_line = f"  [{evt_idx}] time={event_time:.6f}s"
            if evt_name:
                log_line += f", name='{evt_name}'"
            if evt_cmd:
                log_line += f", cmd='{evt_cmd}'"
            
            Log.info(log_line)
            self._log(log_line)
        
        self._log(f"===============================")
        
        # Update hooked tracks set
        if not hasattr(self, '_hooked_tracks'):
            self._hooked_tracks = set()
        self._hooked_tracks.add(coord)

        pending_key = f"{tc}.{tg}.{track}"
        if pending_key in getattr(self, "_reconcile_pending", {}):
            self._handle_reconcile_events(tc, tg, track, events)
            return

        if events is not None:
            self._refresh_reconcile_cache_from_ma3_events(tc, tg, track, events)
        
        # If events were included, add them to the Editor layer
        if events and len(events) > 0:
            if not self._apply_updates_enabled_for_coord(coord):
                Log.info(f"  Apply paused for {coord} - skipping initial sync")
                self._update_synced_layers_list()
                return
            Log.info(f"  Adding {len(events)} initial events to Editor...")
            
            # Use SyncSystemManager to handle initial sync
            self._sync_system_manager.on_track_events_received(coord, events)
            Log.info(f"  Routed {len(events)} events to SyncSystemManager")
        else:
            Log.info(f"  No initial events to add")
        
        Log.info(f"==============================================")
        Log.info(f"")
        
        # Update UI to show hook status
        self._update_synced_layers_list()

    def _handle_track_unhooked(self, message: OSCMessage):
        """Handle track.unhooked message - MA3 confirmed hook removed."""
        tc = message.get('tc', 0)
        tg = message.get('tg', 0)
        track = message.get('track', 0)
        coord = f"tc{tc}_tg{tg}_tr{track}"
        
        self._log(f"Track unhooked: {coord}")
        
        # Update hooked tracks set
        if hasattr(self, '_hooked_tracks') and coord in self._hooked_tracks:
            self._hooked_tracks.remove(coord)
        
        # Update UI
        self._update_synced_layers_list()

    def _handle_track_changed(self, message: OSCMessage):
        """
        Handle track.changed message - track events were modified in MA3.
        
        This is called when the Lua plugin detects changes on a hooked CmdSubTrack.
        The message contains:
        - tc, tg, track: Track coordinates
        
        NOTE: Lua only sends the track identity. EchoZero requests full events via EZ.GetEvents.
        """
        tc = message.get('tc', 0)
        tg = message.get('tg', 0)
        track_no = message.get('track', 0)
        
        coord = f"tc{tc}_tg{tg}_tr{track_no}"
        
        
        import time as _time_mod
        _now = _time_mod.time()
        Log.info(f"[MULTITRACK-DIAG] track.changed arrived: coord={coord}, timestamp={_now:.4f}")
        
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            # Record arrival timestamp for round-trip measurement
            self._sync_system_manager.record_track_changed_timestamp(coord, _now)
            self._sync_system_manager.request_ma3_events(coord)

    def _handle_moved_events_direct(self, tc: int, tg: int, track_no: int, moved: list):
        """
        Handle moved events by updating positions directly.
        
        This bypasses the full sync path for better performance and to avoid
        sync protection issues.
        
        Matches events by fingerprint (old_time + name) rather than index,
        since indices can shift when events are added/deleted.
        
        Note: Events in EchoZero don't have IDs, so we find them by fingerprint
        and directly modify the Event objects, then save the data item.
        """
        coord = f"tc{tc}_tg{tg}_tr{track_no}"
        
        # Find the mapped Editor layer for this track
        mapped_layer_name = self._find_mapped_editor_layer(coord)
        if not mapped_layer_name:
            Log.warning(f"ShowManagerPanel: No layer mapped to {coord}, falling back to full sync")
            return
        
        # Find connected Editor block
        editor_block_id = self._find_connected_editor()
        if not editor_block_id:
            Log.warning(f"ShowManagerPanel: No Editor connected, cannot move events")
            return
        
        # Get EditorAPI with timeline widget for direct visual updates
        editor_api = self._get_editor_api_with_timeline(editor_block_id)
        
        # Get ALL EventDataItems for this Editor block and find events in our layer
        from src.shared.domain.entities import EventDataItem
        data_items = self.facade.data_item_repo.list_by_block(editor_block_id)
        event_data_items = [item for item in data_items if isinstance(item, EventDataItem)]
        
        if not event_data_items:
            Log.warning(f"ShowManagerPanel: No EventDataItems found for Editor block")
            return
        
        # Build lookups for events:
        # 1. By ma3_idx (most reliable for moves)
        # 2. By fingerprint (time + name) as fallback
        # Event objects don't have IDs, so we store the object and its containing data item
        event_lookup_by_idx = {}  # ma3_idx -> (event_obj, data_item)
        event_lookup_by_time = {}  # time (6 decimals) -> (event_obj, data_item)
        matched_count = 0
        for item in event_data_items:
            for event in item.get_events():
                if event.classification == mapped_layer_name:
                    matched_count += 1
                    metadata = event.metadata or {}
                    
                    # Index by ma3_idx if available
                    ma3_idx = metadata.get('ma3_idx')
                    if ma3_idx is not None:
                        event_lookup_by_idx[ma3_idx] = (event, item)
                    
                    # Also index by time for fallback matching
                    event_time = event.time
                    time_key = f"{event_time:.6f}"
                    event_lookup_by_time[time_key] = (event, item)
        
        Log.info(f"  Matched {matched_count} events with classification '{mapped_layer_name}'")
        
        Log.info(f"  Built lookup with {len(event_lookup_by_idx)} by idx, {len(event_lookup_by_time)} by time")
        
        # Track which data items were modified so we can save them
        # Using dict with data_item.id as key since EventDataItem is not hashable
        modified_data_items = {}  # data_item.id -> data_item
        
        # Process each moved event
        moved_count = 0
        for move_info in moved:
            new_time = move_info.get('new_time', 0)
            old_time = move_info.get('old_time', 0)
            event_data = move_info.get('event', {})
            
            # Get ma3_idx for lookup (most reliable)
            ma3_idx = event_data.get('idx') or event_data.get('no') or event_data.get('index')
            
            # Find the event - try ma3_idx first, then old_time
            event_obj = None
            data_item = None
            
            if ma3_idx is not None and ma3_idx in event_lookup_by_idx:
                event_obj, data_item = event_lookup_by_idx[ma3_idx]
                Log.info(f"  Found event by ma3_idx={ma3_idx}")
            else:
                # Fallback: try old_time
                old_time_key = f"{old_time:.6f}"
                if old_time_key in event_lookup_by_time:
                    event_obj, data_item = event_lookup_by_time[old_time_key]
                    Log.info(f"  Found event by old_time={old_time}")
            
            if event_obj and data_item:
                # Directly modify the event's time
                old_event_time = event_obj.time
                event_obj.time = new_time
                
                # Update metadata fingerprint
                if event_obj.metadata and 'ma3_fingerprint' in event_obj.metadata:
                    ma3_name = event_obj.metadata.get('ma3_name', '')
                    event_obj.metadata['ma3_fingerprint'] = f"{new_time:.6f}||{ma3_name}"
                
                # Track that this data item was modified
                modified_data_items[data_item.id] = data_item
                
                # Update lookups for subsequent moves in this batch
                if ma3_idx is not None:
                    event_lookup_by_idx[ma3_idx] = (event_obj, data_item)
                old_time_key = f"{old_event_time:.6f}"
                new_time_key = f"{new_time:.6f}"
                if old_time_key in event_lookup_by_time:
                    del event_lookup_by_time[old_time_key]
                event_lookup_by_time[new_time_key] = (event_obj, data_item)
                
                # Update the visual in the timeline widget
                # Find the timeline event by old time + layer (since we don't have stable IDs)
                if editor_api and editor_api._timeline_widget:
                    try:
                        scene = editor_api._timeline_widget._scene
                        time_tolerance = 0.001  # 1ms tolerance for time matching
                        
                        # Find timeline event that matches old time and layer
                        for timeline_event_id, item in scene._event_items.items():
                            if item.layer_id == mapped_layer_name:
                                item_time = item.start_time
                                if abs(item_time - old_event_time) < time_tolerance:
                                    # Found the event - update its time
                                    scene.update_event(timeline_event_id, start_time=new_time)
                                    Log.info(f"  Updated timeline visual: event {timeline_event_id}")
                                    break
                    except Exception as e:
                        Log.warning(f"  Could not update timeline visual: {e}")
                
                moved_count += 1
                Log.info(f"  Moved event from {old_event_time:.3f}s to {new_time:.3f}s")
            else:
                Log.warning(f"  Could not find event to move (ma3_idx={ma3_idx}, old_time={old_time})")
        
        # Save all modified data items to repository
        for data_item in modified_data_items.values():
            # Update the event count (in case it changed)
            data_item.event_count = len(data_item.get_events())
            self.facade.data_item_repo.update(data_item)
            Log.info(f"  Saved modified data item: {data_item.name}")
        
        Log.info(f"ShowManagerPanel: Moved {moved_count}/{len(moved)} events directly")
    
    def _add_events_direct(self, tc: int, tg: int, track_no: int, events: list):
        """
        Add events directly via EditorAPI.
        
        This bypasses the full sync path for immediate visual feedback.
        """
        coord = f"tc{tc}_tg{tg}_tr{track_no}"
        Log.info(f"ShowManagerPanel._add_events_direct: {coord} with {len(events)} events")
        
        # Find the mapped Editor layer for this track
        mapped_layer_name = self._find_mapped_editor_layer(coord)
        Log.info(f"  Mapped layer name: {mapped_layer_name}")
        if not mapped_layer_name:
            Log.warning(f"ShowManagerPanel: No layer mapped to {coord}, cannot add events")
            return
        
        # Find connected Editor block
        editor_block_id = self._find_connected_editor()
        Log.info(f"  Connected Editor block: {editor_block_id}")
        if not editor_block_id:
            Log.warning(f"ShowManagerPanel: No Editor connected, cannot add events")
            return
        
        # Get EditorAPI with timeline widget for direct visual updates
        editor_api = self._get_editor_api_with_timeline(editor_block_id)
        Log.info(f"  EditorAPI: {'got it' if editor_api else 'None'}")
        if not editor_api:
            Log.warning(f"ShowManagerPanel: Could not get EditorAPI")
            return
        
        # Convert MA3 events to EZ format
        events_data = []
        for evt in events:
            event_time = evt.get('time', 0)
            if isinstance(event_time, str):
                event_time = float(event_time)
            
            events_data.append({
                'time': event_time,
                'duration': 0.0,
                'classification': mapped_layer_name,
                'metadata': {
                    'source': 'ma3_sync',
                    'ma3_coord': coord,
                    'name': evt.get('name', ''),
                    'cmd': evt.get('cmd', ''),
                    'timecode_no': tc,
                    'track_group': tg,
                    'track': track_no,
                }
            })
        
        # Add events via EditorAPI (updates both repo and visual)
        added_count = editor_api.add_events(events_data, source="ma3_sync")
        Log.info(f"ShowManagerPanel: Added {added_count} events directly to layer '{mapped_layer_name}'")

    def _handle_tracks_unhooked_all(self, message: OSCMessage):
        """Handle tracks.unhooked_all message - all hooks removed."""
        count = message.get('count', 0)
        self._log(f"All tracks unhooked ({count} tracks)")
        Log.info(f"ShowManagerPanel: All tracks unhooked ({count} tracks)")
        
        # Clear hooked tracks
        if hasattr(self, '_hooked_tracks'):
            self._hooked_tracks.clear()
        
        # Update UI
        self._update_synced_layers_list()

    def _sync_events_to_editor(self, tc: int, tg: int, track_no: int, events: list):
        """
        Sync events from MA3 to the Editor block.
        
        Simple approach: Full replace of events for this layer on every change.
        This is robust and guarantees Editor always matches MA3 state.
        
        Args:
            tc: Timecode number
            tg: Track group number
            track_no: Track number (user-visible, 1-based)
            events: List of event dicts from MA3 with time, cmd, name, etc.
        """
        import json
        
        coord = f"tc{tc}_tg{tg}_tr{track_no}"
        Log.info(f"ShowManagerPanel: _sync_events_to_editor() for {coord}")
        
        
        # Step 1: Find the mapped Editor layer for this track
        mapped_layer_id = self._find_mapped_editor_layer(coord)
        
        if not mapped_layer_id:
            Log.warning(f"ShowManagerPanel: No Editor layer mapped to track {coord} - cannot sync")
            self._log(f"  No Editor layer mapped to {coord}")
            return
        
        Log.info(f"ShowManagerPanel: Found mapped layer: '{mapped_layer_id}'")
        
        # Step 2: Find connected Editor block
        editor_block_id = self._find_connected_editor()
        if not editor_block_id:
            Log.warning(f"ShowManagerPanel: No Editor block connected - cannot sync")
            self._log(f"  No Editor block connected")
            return
        
        Log.info(f"ShowManagerPanel: Found Editor block: {editor_block_id}")
        
        # Step 3: Find or create EventDataItem for this timecode
        from src.shared.domain.entities import EventDataItem
        
        event_data_item_name = f"tc_{tc}_events"
        data_items = self.facade.data_item_repo.list_by_block(editor_block_id)
        event_data_item = None
        for item in data_items:
            if item.name == event_data_item_name and isinstance(item, EventDataItem):
                event_data_item = item
                break
        
        if not event_data_item:
            Log.info(f"ShowManagerPanel: EventDataItem '{event_data_item_name}' not found - creating...")
            try:
                event_data_item = EventDataItem(
                    name=event_data_item_name,
                    block_id=editor_block_id
                )
                event_data_item = self.facade.data_item_repo.create(event_data_item)
                Log.info(f"ShowManagerPanel: Created EventDataItem: {event_data_item.id}")
            except Exception as e:
                Log.error(f"ShowManagerPanel: Failed to create EventDataItem: {e}")
                self._log(f"  Failed to create EventDataItem: {e}")
                return
        
        Log.info(f"ShowManagerPanel: Found EventDataItem: {event_data_item.id}")
        
        # Step 4: Full replace - clear this layer's events, keep other layers
        try:
            existing_events = event_data_item.events if hasattr(event_data_item, 'events') else []
            
            # Keep events from OTHER layers (different classification)
            other_layer_events = [
                e for e in existing_events 
                if e.get('classification') != mapped_layer_id and 
                   e.get('metadata', {}).get('_visual_layer_name') != mapped_layer_id
            ]
            
            Log.info(f"ShowManagerPanel: Keeping {len(other_layer_events)} events from other layers")
            
            # Convert MA3 events to EZ event format
            self._log(f"=== Syncing {len(events)} events from MA3 ({coord}) ===")
            new_events = []
            for i, evt in enumerate(events):
                event_time = evt.get('time', 0)
                if isinstance(event_time, str):
                    event_time = float(event_time)
                
                evt_name = evt.get('name', '')
                evt_cmd = evt.get('cmd', '')
                self._log(f"  [{i+1}] time={event_time:.3f}s, name='{evt_name}'")
                
                new_events.append({
                    'time': event_time,
                    'duration': 0.0,
                    'classification': mapped_layer_id,
                    'metadata': {
                        'source': 'ma3_sync',
                        'ma3_coord': coord,
                        'name': evt_name,
                        'cmd': evt_cmd,
                        '_visual_layer_name': mapped_layer_id,
                        'timecode_no': tc,
                        'track_group': tg,
                        'track': track_no,
                    }
                })
            self._log(f"===============================")
            
            # Combine: other layers + new events for this layer
            all_events = other_layer_events + new_events
            
            # Update the EventDataItem
            event_data_item.clear_events()
            for evt in all_events:
                event_data_item.add_event(
                    time=evt['time'],
                    duration=evt.get('duration', 0.0),
                    classification=evt.get('classification', ''),
                    metadata=evt.get('metadata', {})
                )
            
            
            # Save to repo
            
            self.facade.data_item_repo.update(event_data_item)
            
            
            Log.info(f"ShowManagerPanel: Saved {len(all_events)} total events to repo")
            
            # Trigger UI refresh via BlockUpdated event
            from src.application.events.events import BlockUpdated
            event_bus = self.facade.event_bus  # Use application's event bus, not a new instance
            
            
            event_bus.publish(BlockUpdated(
                project_id=self.facade.current_project_id,
                data={
                    "id": editor_block_id,
                    "name": "Editor",
                    "events_updated": True,
                    "events_count": len(new_events),
                    "layer_name": mapped_layer_id,
                    "source": "ma3_sync"
                }
            ))
            
            
            self._log(f"  Synced {len(new_events)} events to layer '{mapped_layer_id}'")
            Log.info(f"ShowManagerPanel: Sync complete - {len(new_events)} events in '{mapped_layer_id}'")
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            Log.error(f"ShowManagerPanel: Error syncing events to Editor: {e}")
            Log.error(tb)
            self._log(f"  ERROR: Failed to sync events: {e}")
    
    def _find_mapped_editor_layer(self, ma3_coord: str) -> Optional[str]:
        """
        Find the Editor layer ID that is mapped to an MA3 track.
        
        Args:
            ma3_coord: MA3 track coordinate (e.g., "tc101_tg1_tr1")
            
        Returns:
            Editor layer ID/name if found, None otherwise
        """
        # Check settings manager for mappings
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            synced_layer = self._settings_manager.get_synced_layer("ma3", ma3_coord)
            if synced_layer:
                return synced_layer.get('mapped_editor_layer_id')
        
        return None

    def _check_connection_health(self):
        """Check listener health and handle auto-restart if the thread died.
        
        Called every 10s. Connection monitoring (pings, disconnect detection)
        is handled by SSM.start_connection_monitoring(), not here.
        """
        if not self._listener_service:
            return
        if not self._listener_service.is_listening(self.block_id):
            if self._should_restore_listener():
                now = time.time()
                if now - self._last_listener_restart_ts >= self._listener_restart_cooldown_s:
                    self._last_listener_restart_ts = now
                    self._log("Listener stopped unexpectedly. Attempting to restart...")
                    self._on_start_listening()
            return

    def _update_sync_buttons_for_connection(self, connected: bool) -> None:
        """Enable or disable sync-related UI controls based on MA3 connection state.
        
        Buttons that send commands to MA3 are disabled when disconnected.
        Local-only actions (delete, config spinboxes, start/stop listener) remain enabled.
        
        Args:
            connected: True if MA3 is actively connected, False otherwise.
        """
        # Toolbar-level buttons
        if hasattr(self, 'refresh_layers_btn'):
            self.refresh_layers_btn.setEnabled(connected)
        
        # Per-row buttons in the synced layers table
        if hasattr(self, 'layers_table'):
            table = self.layers_table
            for row_idx in range(table.rowCount()):
                # Column 6: Resync button
                resync_widget = table.cellWidget(row_idx, 6)
                if resync_widget:
                    btn = resync_widget.findChild(QPushButton)
                    if btn:
                        btn.setEnabled(connected)
                # Column 7: Keep EZ button
                keep_ez_widget = table.cellWidget(row_idx, 7)
                if keep_ez_widget:
                    btn = keep_ez_widget.findChild(QPushButton)
                    if btn:
                        btn.setEnabled(connected)
                # Column 8: Keep MA3 button
                keep_ma3_widget = table.cellWidget(row_idx, 8)
                if keep_ma3_widget:
                    btn = keep_ma3_widget.findChild(QPushButton)
                    if btn:
                        btn.setEnabled(connected)
    
    def _should_restore_listener(self) -> bool:
        """Return True if metadata indicates listener should be active."""
        try:
            result = self.facade.describe_block(self.block_id)
            if not result.success or not result.data:
                return False
            metadata = result.data.metadata or {}
            return bool(metadata.get("osc_listener_active"))
        except Exception:
            return False

    def _parse_all_events(self, json_str: str):
        """Parse all events JSON from MA3."""
        if not json_str:
            self._log("ERROR: Empty JSON string received")
            Log.error("ShowManagerPanel: Empty JSON string in _parse_all_events")
            return
            
        try:
            self._log(f"Received events JSON, length: {len(json_str)}")
            data = json.loads(json_str)
            self._log(f"JSON parsed successfully, keys: {list(data.keys())}")
            self._ma3_events.clear()
            
            timecode_no = data.get('timecode_no', 0)
            self._log(f"Parsing events for timecode {timecode_no}")
            
            total_events_found = 0
            track_groups_data = data.get('track_groups', [])
                # Parse track groups
            for tg_data in track_groups_data:
                tg_idx = tg_data.get('index', 0)
                tracks_data = tg_data.get('tracks', [])
                                
                # Parse tracks
                for track_data in tracks_data:
                    track_idx = track_data.get('index', 0)
                    events_in_track = track_data.get('events', [])
                    total_events_found += len(events_in_track)
                    self._log(f"Track {tg_idx}.{track_idx} has {len(events_in_track)} events")
                    
                    # Parse events
                    events_parsed_in_track = 0
                    for event_data in events_in_track:
                        try:
                            # Map new structure (time_range, sub_track) to old structure (event_layer)
                            # For backward compatibility, use sub_track as event_layer if available
                            event_layer = event_data.get('event_layer') or event_data.get('sub_track', 1)
                            
                            # Determine event type from sub_track_class if available
                            # MA3 sends type as integer (0=cmd, 1=fader) or string
                            event_type_raw = event_data.get('type', 'cmd')
                            
                            # Convert integer type to string
                            if isinstance(event_type_raw, int):
                                event_type = 'fader' if event_type_raw == 1 else 'cmd'
                            elif isinstance(event_type_raw, str):
                                event_type = event_type_raw.lower()
                            else:
                                event_type = 'cmd'  # Default
                            
                            # If type is unknown or empty, try to determine from sub_track_class
                            if event_type == 'unknown' or not event_type:
                                sub_track_class = event_data.get('sub_track_class', '')
                                if sub_track_class == 'FaderSubTrack':
                                    event_type = 'fader'
                                elif sub_track_class == 'CmdSubTrack':
                                    event_type = 'cmd'
                                else:
                                    event_type = 'cmd'  # Default
                            
                        # Create MA3Event from the data
                        # Convert numeric fields from string to float (MA3 sends them as strings)
                            def to_float(value, default=0.0):
                                """Convert value to float, handling string inputs."""
                                if value is None:
                                    return None
                                if isinstance(value, (int, float)):
                                    return float(value)
                                if isinstance(value, str):
                                    try:
                                        return float(value)
                                    except (ValueError, TypeError):
                                        return default
                                return default
                            
                            # Convert time value - Lua plugin should have converted to seconds,
                            # but we validate and convert if needed as a safety measure
                            from src.features.ma3.domain.ma3_event import ma3_time_to_seconds
                            raw_time_value = to_float(event_data.get('time', 0.0), 0.0)
                            time_value = ma3_time_to_seconds(raw_time_value)
                            
                            fade_value = to_float(event_data.get('fade'), None)
                            delay_value = to_float(event_data.get('delay'), None)
                            
                            ma3_event = MA3Event(
                                timecode_no=timecode_no,
                                track_group=event_data.get('track_group', tg_idx),
                                track=event_data.get('track', track_idx),
                                event_layer=event_layer,
                                event_index=event_data.get('event_index', 0),
                                time=time_value,
                                event_type=event_type,
                                name=event_data.get('name', ''),
                                cmd=event_data.get('cmd'),
                                fade=fade_value,
                                delay=delay_value,
                            )
                            self._ma3_events.append(ma3_event)
                            events_parsed_in_track += 1
                        except Exception as e:
                            import traceback
                            Log.error(f"Failed to parse MA3 event: {e}\n{traceback.format_exc()}")
                            Log.error(f"Event data: {event_data}")
            
            self._log(f"Found {total_events_found} events in JSON, parsed {len(self._ma3_events)} successfully")
            
            
            # Update display
            self._update_events_display()
            self._log(f"Parsed {len(self._ma3_events)} events from MA3")
            
        except json.JSONDecodeError as e:
            Log.error(f"Failed to parse MA3 events JSON: {e}")
            self._log(f"Error parsing events: {e}")
            import traceback
            Log.error(f"JSON string (first 500 chars): {json_str[:500] if json_str else 'None'}")
        except Exception as e:
            import traceback
            Log.error(f"Failed to parse MA3 events: {e}\n{traceback.format_exc()}")
            self._log(f"Error parsing events: {e}")
    
    def _update_events_display(self):
        """Update the events display text."""
        if not hasattr(self, 'events_text') or not self.events_text:
            Log.warning("events_text widget not available")
            return
            
        if not self._ma3_events:
            self.events_text.setPlainText("No events loaded")
            if hasattr(self, 'events_count_label'):
                self.events_count_label.setText("0 events")
            return
        
        # Group events by track
        events_by_track = {}
        for event in self._ma3_events:
            key = (event.track_group, event.track)
            if key not in events_by_track:
                events_by_track[key] = []
            events_by_track[key].append(event)
        
        # Format display
        lines = []
        lines.append(f"Timecode {self._ma3_events[0].timecode_no} - {len(self._ma3_events)} events\n")
        
        for (tg, tr), events in sorted(events_by_track.items()):
            lines.append(f"\nTrack Group {tg}, Track {tr} ({len(events)} events):")
            for event in sorted(events, key=lambda e: e.time):
                time_str = f"{event.time:7.3f}s"
                type_str = event.event_type.upper()[:3]
                name_str = event.name[:20] if event.name else "(unnamed)"
                lines.append(f"  {time_str}  [{type_str}]  {name_str}")
        
                self.events_text.setPlainText("\n".join(lines))
                self.events_count_label.setText(f"{len(self._ma3_events)} events")
            
    def _send_events_to_connected_editors(self):
        """Send MA3 events to any connected Editor blocks via manipulator port.
        
        Events are routed to the correct layers based on layer mappings.
        Events use the mapped layer name as their classification so they appear on the correct layer.
        """
        try:
            # Find Editor blocks connected via manipulator port
            connected_editors = self._find_connected_editors()
            
            if not connected_editors:
                Log.debug("No Editor blocks connected to ShowManager")
                return
            
            # Get synced layer mappings from SyncSystemManager
            synced_ma3_tracks = {}
            for entity in self._sync_system_manager.get_synced_layers():
                if entity.ma3_coord and entity.editor_layer_id:
                    synced_ma3_tracks[entity.ma3_coord] = entity.editor_layer_id
            
            # Convert MA3Events to dict format for command, using synced layer mappings
            events_data = []
            for ma3_event in self._ma3_events:
                # Build MA3 track coordinate from event
                ma3_coord = f"tc{ma3_event.timecode_no}_tg{ma3_event.track_group}_tr{ma3_event.track}"
                
                # Look up synced Editor layer name
                mapped_layer_id = synced_ma3_tracks.get(ma3_coord)
                if not mapped_layer_id:
                    # Not synced, skip this event (only sync events for synced tracks)
                    continue
                else:
                    # Use mapped layer name as classification (events appear on this layer)
                    classification = mapped_layer_id
                
                events_data.append({
                    "time": ma3_event.time,
                    "duration": 0.0,  # MA3 events are one-shot (instant)
                    "classification": classification,
                    "metadata": {
                        "name": ma3_event.name,
                        "cmd": ma3_event.cmd,
                        "fade": ma3_event.fade,
                        "delay": ma3_event.delay,
                        "ma3_timecode_no": ma3_event.timecode_no,
                        "ma3_track_group": ma3_event.track_group,
                        "ma3_track": ma3_event.track,
                        "ma3_event_layer": ma3_event.event_layer,
                        "ma3_event_index": ma3_event.event_index,
                        "ma3_coord": ma3_coord,  # Track coordinate for reference
                    }
                })
            
            # Send to each connected Editor using Editor API command
            from src.application.commands.editor_commands import EditorAddEventsCommand
            
            for editor_id in connected_editors:
                cmd = EditorAddEventsCommand(
                    facade=self.facade,
                    block_id=editor_id,
                    events=events_data,
                    source="ma3"
                )
                # Execute through command bus (undoable)
                success = self.facade.command_bus.execute(cmd)
                
                if success:
                    self._log(f"✓ Sent {len(events_data)} events to Editor block (from synced MA3 tracks)")
                else:
                    self._log(f"✗ Failed to send events to Editor block")
                    
        except Exception as e:
            Log.error(f"Failed to send events to connected editors: {e}")
            import traceback
            traceback.print_exc()
    
    def _find_connected_editors(self) -> list:
        """Find Editor blocks connected to this ShowManager via manipulator port."""
        try:
            # Get all connections
            connections_result = self.facade.list_connections()
            if not connections_result.success or not connections_result.data:
                return []
            
            connected_editors = []
            
            # Look for connections where ShowManager's manipulator port is involved
            for conn in connections_result.data:
                # Check if this ShowManager is the source
                if conn.source_block_id == self.block.id:
                    # Check if source port is manipulator
                    source_block_result = self.facade.describe_block(conn.source_block_id)
                    if source_block_result.success and source_block_result.data:
                        source_block = source_block_result.data
                        bidirectional_ports = source_block.get_bidirectional()
                        if conn.source_output_name in bidirectional_ports:
                            # Check if target is an Editor
                            target_block_result = self.facade.describe_block(conn.target_block_id)
                            if target_block_result.success and target_block_result.data:
                                target_block = target_block_result.data
                                if target_block.type == "Editor":
                                    connected_editors.append(target_block.id)
                
                # Check if this ShowManager is the target
                if conn.target_block_id == self.block.id:
                    # Check if target port is manipulator
                    target_block_result = self.facade.describe_block(conn.target_block_id)
                    if target_block_result.success and target_block_result.data:
                        target_block = target_block_result.data
                        bidirectional_ports = target_block.get_bidirectional()
                        if conn.target_input_name in bidirectional_ports:
                            # Check if source is an Editor
                            source_block_result = self.facade.describe_block(conn.source_block_id)
                            if source_block_result.success and source_block_result.data:
                                source_block = source_block_result.data
                                if source_block.type == "Editor":
                                    connected_editors.append(source_block.id)
            
            return connected_editors
            
        except Exception as e:
            Log.error(f"Failed to find connected editors: {e}")
            return []
    
    def _handle_event_added(self, args: list):
        """Handle event_added message from MA3."""
        # TODO: Parse and add single event
        self._log(f"Event added: {args}")
    
    def _handle_event_modified(self, args: list):
        """Handle event_modified message from MA3."""
        # TODO: Parse and update single event
        self._log(f"Event modified: {args}")
    
    def _handle_event_deleted(self, args: list):
        """Handle event_deleted message from MA3."""
        # TODO: Parse and remove single event
        self._log(f"Event deleted: {args}")
    
    def _cleanup_incomplete_chunks(self):
        """Clean up incomplete chunked messages that have timed out (5 seconds)."""
        import time
        current_time = time.time()
        timeout_seconds = 5.0
        
        incomplete_messages = []
        for message_id, buf in self._chunked_message_buffer.items():
            elapsed = current_time - buf.get("started_at", current_time)
            if elapsed > timeout_seconds:
                incomplete_messages.append((message_id, buf))
        
        for message_id, buf in incomplete_messages:
            received_count = buf.get("received", 0)
            total_count = buf.get("total", 0)
            self._log(f"⏱️ Timeout: Cleaning up incomplete chunked message {message_id} ({received_count}/{total_count} chunks received)")
            Log.warning(f"ShowManagerPanel: Chunked message {message_id} timed out after {timeout_seconds}s ({received_count}/{total_count} chunks)")
            del self._chunked_message_buffer[message_id]
    
    def _on_get_all_events(self):
        """Request all events from MA3."""
        
        try:
            self._auto_fetch_ma3_events()
        except RuntimeError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MA3 Connection Error", str(e))
            self._log(f"ERROR: {e}")
    
    def _is_ma3_ready(self) -> bool:
        """Check if MA3 is ready for communication (configured and listening)."""
        # Check settings manager exists
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return False
        
        # Check MA3 IP and port are configured
        ma3_ip = self._settings_manager.ma3_ip
        ma3_port = self._settings_manager.ma3_port
        if not ma3_ip or not ma3_port:
            return False
        
        # Check listener is active
        is_listening = False
        if hasattr(self, '_listener_service') and self._listener_service:
            is_listening = self._listener_service.is_listening(self.block_id)
        return is_listening
    
    def _ensure_ma3_ready(self) -> None:
        """
        Ensure MA3 is ready for communication, raise exception if not.
        
        Raises:
            RuntimeError: If MA3 is not configured or listener is not active
        """
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            raise RuntimeError("ShowManager settings not available")
        
        ma3_ip = self._settings_manager.ma3_ip
        ma3_port = self._settings_manager.ma3_port
        
        if not ma3_ip:
            raise RuntimeError("MA3 IP address is not configured. Please set MA3 IP in ShowManager settings.")
        
        if not ma3_port:
            raise RuntimeError("MA3 port is not configured. Please set MA3 port in ShowManager settings.")
        
        is_listening = False
        if hasattr(self, '_listener_service') and self._listener_service:
            is_listening = self._listener_service.is_listening(self.block_id)
        
        if not is_listening:
            # Check if metadata says it should be listening (indicates silent failure)
            error_message = "MA3 listener is not active. Please start the MA3 listener first."
            try:
                result = self.facade.describe_block(self.block_id)
                if result.success and result.data:
                    metadata = result.data.metadata or {}
                    should_be_listening = metadata.get("osc_listener_active", False)
                    if should_be_listening:
                        # Metadata says listening but service says not - listener failed silently
                        Log.warning(f"ShowManagerPanel: Listener state mismatch detected - metadata says listening but service says not. Clearing stale metadata.")
                        self._save_listener_state(False)
                        error_message = "MA3 listener failed silently. The listener was started but is no longer running. Please check the port configuration and try starting the listener again."
            except Exception as e:
                # If we can't check metadata, just provide the standard error
                Log.debug(f"ShowManagerPanel: Could not check metadata for listener state: {e}")
            
            raise RuntimeError(error_message)
    
    def _auto_fetch_ma3_data(self):
        """Automatically fetch MA3 structure (tracks) and events when ready."""
        from src.utils.message import Log
        
        # Skip if suppressed (prevents loops during batch operations)
        if self._suppress_auto_fetch:
            Log.debug("ShowManagerPanel: Auto-fetch suppressed (batch operation in progress)")
            return
        
        # Only fetch if MA3 is ready
        if not self._is_ma3_ready():
            Log.debug("ShowManagerPanel: MA3 not ready (not listening or not configured), skipping auto-fetch")
            return
        
        Log.debug("ShowManagerPanel: MA3 ready, auto-fetching structure and events")
        
        # Fetch structure first (tracks)
        self._auto_fetch_ma3_structure()
        
        # Events will be auto-fetched after structure is received
        # (see _handle_ma3_structure)
    
    def _auto_fetch_ma3_structure(self):
        """
        Automatically fetch MA3 structure (tracks) for the current target timecode.
        
        Note: This will refetch even if tracks are already loaded, to ensure
        we have data for the correct timecode.
        
        Raises:
            RuntimeError: If MA3 is not ready or fetch fails
        """
        from src.utils.message import Log
        
        # Ensure MA3 is ready (will raise if not)
        self._ensure_ma3_ready()
        
        # Get timecode from settings
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            raise RuntimeError("ShowManager settings not available")
        
        timecode_no = self._settings_manager.target_timecode
        if not timecode_no:
            raise RuntimeError("Timecode number is not set. Please configure target timecode in ShowManager settings.")
        
        # Fetch track groups (which will trigger tracks/events fetch cascade)
        cmd = f"EZ.GetTrackGroups({timecode_no})"
        try:
            self._send_lua_command(cmd)
            Log.info(f"Auto-fetching MA3 track groups for timecode {timecode_no}...")
        except RuntimeError as e:
            Log.error(f"Failed to fetch MA3 track groups: {e}")
            raise

    def _request_ma3_structure_refresh(self, reason: str) -> None:
        """Refresh MA3 structure when connected, with throttling."""
        now = time.time()
        if now - self._last_ma3_refresh_at < self._ma3_refresh_min_interval:
            return
        if not self._is_ma3_ready():
            return
        self._force_tracks_refresh = True
        try:
            self._last_ma3_refresh_at = now
            self._auto_fetch_ma3_structure()
            Log.debug(f"ShowManagerPanel: Auto-fetch structure ({reason})")
        except RuntimeError as e:
            Log.warning(f"ShowManagerPanel: Auto-fetch failed ({reason}): {e}")
    
    def _auto_fetch_ma3_events(self):
        """
        Automatically fetch MA3 events if not already loaded.
        
        Ensures structure (tracks) is loaded first before fetching events.
        
        Raises:
            RuntimeError: If MA3 is not ready or fetch fails
        """
        from src.utils.message import Log
        
        # Ensure MA3 is ready (will raise if not)
        self._ensure_ma3_ready()
        
        # Ensure structure (tracks) is loaded before fetching events
        if not hasattr(self, '_ma3_tracks') or not self._ma3_tracks:
            Log.debug("ShowManagerPanel: Structure not loaded, fetching structure before events")
            self._auto_fetch_ma3_structure()
            # Note: Events will be auto-fetched by _handle_ma3_structure after structure loads
            return
        
        # Get timecode from settings
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            raise RuntimeError("ShowManager settings not available")
        
        timecode_no = self._settings_manager.target_timecode
        if not timecode_no:
            raise RuntimeError("Timecode number is not set. Please configure target timecode in ShowManager settings.")
        
        cmd = f"EZ.GetAllEvents({timecode_no})"
        try:
            self._send_lua_command(cmd)
            Log.info(f"Auto-fetching MA3 events for timecode {timecode_no}...")
        except RuntimeError as e:
            Log.error(f"Failed to fetch MA3 events: {e}")
            raise
    
    def _on_refresh_manual_layers(self):
        """Refresh the layer dropdown with layers from connected Editor blocks."""
        try:
            # Find connected Editor blocks
            connected_editors = self._find_connected_editors()
            if not connected_editors:
                self._log("No Editor blocks connected to ShowManager")
                self.manual_layer_combo.clear()
                self.manual_layer_combo.addItem("(No Editor connected)")
                return
            
            # Collect layers from all connected Editor blocks
            all_layers = []
            for editor_id in connected_editors:
                result = self.facade.describe_block(editor_id)
                if not result.success or not result.data:
                    continue
                
                editor_block = result.data
                if editor_block.type != "Editor":
                    continue
                
                # Get layer state from UI state repository
                try:
                    layer_result = self.facade.get_ui_state(
                        state_type='editor_layers',
                        entity_id=editor_id
                    )
                    
                    if layer_result.success and layer_result.data:
                        layers_data = layer_result.data.get('layers', [])
                        for layer in layers_data:
                            layer_name = layer.get('name')
                            if layer_name and layer_name not in all_layers:
                                all_layers.append(layer_name)
                except Exception as e:
                    Log.warning(f"Failed to get layers from Editor {editor_id}: {e}")
            
            # Update combo box
            self.manual_layer_combo.clear()
            if all_layers:
                self.manual_layer_combo.addItem("(Select layer...)")
                for layer_name in sorted(all_layers):
                    self.manual_layer_combo.addItem(layer_name)
                self._log(f"✓ Loaded {len(all_layers)} layers for manual event creation")
            else:
                self.manual_layer_combo.addItem("(No layers found)")
                self._log("⚠️ No layers found in connected Editor blocks")
                
        except Exception as e:
            Log.error(f"Failed to refresh manual layers: {e}")
            self._log(f"✗ Failed to refresh layers: {e}")
    
    def _on_add_manual_event(self):
        """Add a manual event to connected Editor blocks."""
        try:
            # Get selected layer
            selected_index = self.manual_layer_combo.currentIndex()
            if selected_index <= 0:  # First item is "(Select layer...)" or "(No layers found)"
                self._log("Please select a layer first")
                return
            
            layer_name = self.manual_layer_combo.currentText()
            
            # Get time
            time_value = self.manual_time_input.value()
            
            # Find connected Editor blocks
            connected_editors = self._find_connected_editors()
            if not connected_editors:
                self._log("No Editor blocks connected to ShowManager")
                return
            
            # Create event data
            events_data = [{
                "time": time_value,
                "duration": 0.0,  # One-shot event
                "classification": layer_name,  # Use layer name as classification
                "metadata": {
                    "name": f"Manual Event at {time_value:.3f}s",
                    "source": "manual",
                    "created_by": "ShowManager"
                }
            }]
            
            # Send to each connected Editor
            from src.application.commands.editor_commands import EditorAddEventsCommand
            
            for editor_id in connected_editors:
                cmd = EditorAddEventsCommand(
                    facade=self.facade,
                    block_id=editor_id,
                    events=events_data,
                    source="manual"
                )
                # Execute through command bus (undoable)
                success = self.facade.command_bus.execute(cmd)
                
                if success:
                    self._log(f"✓ Added manual event at {time_value:.3f}s to layer '{layer_name}' in Editor")
                else:
                    self._log(f"✗ Failed to add event to Editor block")
                    
        except Exception as e:
            Log.error(f"Failed to add manual event: {e}")
            import traceback
            traceback.print_exc()
            self._log(f"✗ Failed to add manual event: {e}")
    
    def _on_toggle_watching(self):
        """Toggle change detection watching."""
        if not hasattr(self, '_settings_manager'):
            timecode_no = 101
            poll_interval = 5
        else:
            timecode_no = self._settings_manager.target_timecode
            poll_interval = self._settings_manager.sync_interval
        
        if not self._polling_enabled:
            # Start watching
            cmd = f"EZ.WatchTimecode({timecode_no})"
            try:
                self._send_lua_command(cmd)
            except RuntimeError as e:
                self._log(f"ERROR: {e}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "MA3 Connection Error", str(e))
                return
            
            # Start polling timer
            self._poll_timer.start(poll_interval * 1000)  # Convert to milliseconds
            self._polling_enabled = True
            
            self.watch_btn.setText("Stop Watching")
            self._log(f"Started watching timecode {timecode_no} (polling every {poll_interval}s)")
        else:
            # Stop watching
            cmd = f"EZ.StopWatching({timecode_no})"
            try:
                self._send_lua_command(cmd)
            except RuntimeError as e:
                self._log(f"ERROR: {e}")
                # Don't show error for stop operations - just log it
                return
            
            # Stop polling timer
            self._poll_timer.stop()
            self._polling_enabled = False
            
            self.watch_btn.setText("Start Watching")
            self._log(f"Stopped watching timecode {timecode_no}")
    
    def _on_poll_changes(self):
        """Poll MA3 for changes (called by timer)."""
        if not self._polling_enabled:
            return
        
        if not hasattr(self, '_settings_manager'):
            timecode_no = 101
        else:
            timecode_no = self._settings_manager.target_timecode
        
        # Send poll command
        cmd = f"EZ.PollForChanges({timecode_no})"
        try:
            self._send_lua_command(cmd)
        except RuntimeError as e:
            # Don't show error for polling - just log it
            self._log(f"ERROR: Poll failed: {e}")
    
    def refresh_for_undo(self):
        """
        Refresh UI after undo/redo operations.
        
        Called by BlockPanelBase when undo/redo affects this block's settings.
        Reloads settings from database and updates UI.
        """
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        # Reload settings from database (single source of truth)
        self._settings_manager.reload_from_storage()
        
        # Refresh UI with current settings
        self.refresh()
    
    def _subscribe_to_events(self):
        """Subscribe to block events including BlockRemoved for cleanup."""
        super()._subscribe_to_events()
        # Subscribe to BlockRemoved to clean up listener when block is deleted
        self.facade.event_bus.subscribe("BlockRemoved", self._on_block_removed)
    
    def _on_block_removed(self, event):
        """
        Handle block removal event - clean up OSC listener and resources.
        
        This is called when any block is removed, so we check if it's our block.
        All cleanup is defensive to handle cases where resources may already be cleaned up.
        """
        if not hasattr(event, 'data'):
            return
        
        removed_block_id = event.data.get('id')
        if removed_block_id == self.block_id:
            Log.info(f"ShowManagerPanel: Block {self.block_id} was deleted, cleaning up OSC listener and resources")
            
            try:
                # Stop listener via service (service also handles block removal events)
                if self._listener_service:
                    self._listener_service.stop_listener(self.block_id)
            except Exception as e:
                Log.warning(f"ShowManagerPanel: Error stopping listener during block deletion: {e}")
            
            # Stop all timers (defensive - check if they exist)
            try:
                if hasattr(self, '_process_timer') and self._process_timer:
                    self._process_timer.stop()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping process timer: {e}")
            
            try:
                if hasattr(self, '_chunk_cleanup_timer') and self._chunk_cleanup_timer:
                    self._chunk_cleanup_timer.stop()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping chunk cleanup timer: {e}")
            
            try:
                if hasattr(self, '_connection_status_timer') and self._connection_status_timer:
                    self._connection_status_timer.stop()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping connection status timer: {e}")
            
            try:
                if hasattr(self, '_connection_monitor_timer') and self._connection_monitor_timer:
                    self._connection_monitor_timer.stop()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping connection monitor timer: {e}")
            
            try:
                if hasattr(self, '_poll_timer') and self._poll_timer:
                    self._poll_timer.stop()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping poll timer: {e}")
            
            # Ping timer is managed by SSM; stop SSM connection monitoring on cleanup
            try:
                if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
                    self._sync_system_manager.stop_connection_monitoring()
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error stopping SSM connection monitoring: {e}")
            
            # Wait for listener thread to finish (with timeout)
            try:
                if hasattr(self, '_listener_thread') and self._listener_thread and self._listener_thread.is_alive():
                    # Give thread a moment to finish
                    self._listener_thread.join(timeout=1.0)
                    if self._listener_thread.is_alive():
                        Log.warning(f"ShowManagerPanel: Listener thread did not finish within timeout")
            except Exception as e:
                Log.warning(f"ShowManagerPanel: Error waiting for listener thread: {e}")
            
            # Unsubscribe from events (defensive - may already be unsubscribed)
            try:
                if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'event_bus'):
                    self.facade.event_bus.unsubscribe("BlockRemoved", self._on_block_removed)
                    self.facade.event_bus.unsubscribe(MA3OscOutbound, self._on_ma3_osc_outbound)
                    self.facade.event_bus.unsubscribe(MA3OscInbound, self._on_ma3_osc_inbound)
            except Exception as e:
                Log.debug(f"ShowManagerPanel: Error unsubscribing from BlockRemoved (may already be unsubscribed): {e}")
            
            Log.info(f"ShowManagerPanel: Cleanup complete for deleted block {self.block_id}")
    
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
                Log.debug(f"ShowManagerPanel: Skipping refresh during save for {self.block_id}")
                return
            
            # Skip if settings manager hasn't finished initial loading yet
            # This prevents reloading during project load before metadata is fully set
            if hasattr(self, '_settings_manager') and self._settings_manager:
                if not self._settings_manager.is_loaded():
                    Log.debug(f"ShowManagerPanel: Skipping reload - settings manager still loading for {self.block_id}")
                    return
            else:
                Log.debug(f"ShowManagerPanel: Settings manager not available yet for {self.block_id}")
                return
            
            Log.debug(f"ShowManagerPanel: Block {self.block_id} updated externally, refreshing UI")
            
            # Reload block data from database (ensures self.block is current)
            result = self.facade.describe_block(self.block_id)
            if result.success:
                self.block = result.data
                self._update_header()
                
                # Only reload settings if block metadata actually has settings
                # Skip if metadata is empty (might be during project load)
                if self.block.metadata and len(self.block.metadata) > 0:
                    # Skip pushing loaded settings into SSM in _on_settings_loaded:
                    # this update is often caused by SSM's own _save_to_settings(), so
                    # SSM already has the canonical state; reloading would spam logs and re-init.
                    self._skip_ssm_load_after_block_updated = True
                    # Reload settings from database (single source of truth)
                    self._settings_manager.reload_from_storage()
                    Log.debug(f"ShowManagerPanel: Settings manager reloaded from database")
                else:
                    Log.debug(f"ShowManagerPanel: Block metadata is empty, skipping settings reload for {self.block_id}")
                    return
            else:
                Log.warning(f"ShowManagerPanel: Failed to reload block {self.block_id}")
                return
            
            # Refresh UI to reflect changes (now that both block and settings are reloaded)
            # Use QTimer.singleShot to ensure refresh happens after event processing
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, self.refresh)
    
    def showEvent(self, event):
        """Register panel state provider when panel becomes visible."""
        super().showEvent(event) if hasattr(super(), 'showEvent') else None
        # Register panel state provider with BlockStatusService
        if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'block_status_service') and self.facade.block_status_service:
            self.facade.block_status_service.register_panel_state_provider(self.block_id, self)
        
        # NOTE: Removed auto-fetch on panel show - user must explicitly click to fetch
        
        # Update UI buttons to reflect actual listener state from service
        if hasattr(self, 'start_listening_btn') and hasattr(self, 'stop_listening_btn'):
            listening = False
            if self._listener_service:
                listening = self._listener_service.is_listening(self.block_id)
            self.start_listening_btn.setEnabled(not listening)
            self.stop_listening_btn.setEnabled(listening)
            if hasattr(self, 'configure_ma3_btn'):
                self.configure_ma3_btn.setEnabled(listening)
    
    def get_panel_state(self) -> Dict[str, Any]:
        """
        Get current panel state for status evaluation.
        
        Returns panel internal state that isn't stored in block metadata.
        This allows BlockStatusService to access panel state (like listener status)
        when evaluating status conditions.
        
        Returns:
            Dictionary of panel state values
        """
        # Read listener state from service (not panel instance state)
        listening = False
        if self._listener_service:
            listening = self._listener_service.is_listening(self.block_id)
        
        return {
            "listening": listening,
            "connected": hasattr(self, '_sync_system_manager') and self._sync_system_manager and self._sync_system_manager.connection_state == "connected",
            "connection_status": getattr(self, '_connection_status', 'unknown'),
            "connected_editor_id": getattr(self, '_connected_editor_id', None),
        }
    
    def closeEvent(self, event):
        """Handle panel close - listener is managed by service, persists independently."""
        # Force save any pending settings before closing
        if hasattr(self, '_settings_manager') and self._settings_manager:
            # Check if there are synced layers to save
            synced_layers = getattr(self._settings_manager, 'synced_layers', [])
            if synced_layers:
                self._settings_manager.force_save()
            else:
                self._settings_manager.force_save()  # Always save to ensure any changes are persisted
        
        # Check if the application is closing (not just the panel)
        app_closing = False
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for widget in app.topLevelWidgets():
                    if hasattr(widget, '_closing') and widget._closing:
                        app_closing = True
                        break
        except Exception:
            pass
        
        
        # Only stop listener if app is closing (service handles cleanup on app close)
        if app_closing and self._listener_service:
            self._listener_service.stop_listener(self.block_id)
            self._save_listener_state(False)
        # Otherwise, listener persists in service - panel just closes
        
        # Unregister panel state provider (status check now uses listener service, not panel state)
        if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'block_status_service') and self.facade.block_status_service:
            self.facade.block_status_service.unregister_panel_state_provider(self.block_id)
            # Trigger status recalculation (status check now uses listener service, should stay correct)
            self._trigger_status_update()
        
        # Unsubscribe from BlockRemoved event
        try:
            if hasattr(self, 'facade') and self.facade and hasattr(self.facade, 'event_bus'):
                self.facade.event_bus.unsubscribe("BlockRemoved", self._on_block_removed)
                self.facade.event_bus.unsubscribe(MA3OscOutbound, self._on_ma3_osc_outbound)
                self.facade.event_bus.unsubscribe(MA3OscInbound, self._on_ma3_osc_inbound)
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Error unsubscribing from BlockRemoved in closeEvent: {e}")
        
        # Cleanup sync system manager (unhook all MA3 tracks)
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            try:
                self._sync_system_manager.cleanup()
            except Exception as e:
                Log.warning(f"ShowManagerPanel: Error cleaning up sync system manager: {e}")
        
        super().closeEvent(event) if hasattr(super(), 'closeEvent') else event.accept()
    
    def _load_connection_state_from_service(self):
        """Load connection state from service and update UI."""
        if not self._state_service:
            return
        
        state = self._state_service.get_connection_state(self.block_id)
        # Reconcile cached state with actual connections (prevents stale static text)
        actual_editor_id = None
        try:
            if hasattr(self, 'block') and self.block:
                connected_editors = self._find_connected_editors()
                actual_editor_id = connected_editors[0] if connected_editors else None
        except Exception as e:
            Log.debug(f"ShowManagerPanel: Failed to resolve connected editors for state refresh: {e}")

        if actual_editor_id:
            if state.connection_status != "connected" or state.connected_editor_id != actual_editor_id:
                self._state_service.update_connection_state(
                    self.block_id,
                    "connected",
                    actual_editor_id,
                    None
                )
                state = self._state_service.get_connection_state(self.block_id)
        elif state.connection_status == "connected":
            self._state_service.update_connection_state(self.block_id, "disconnected", None, None)
            state = self._state_service.get_connection_state(self.block_id)

        self._update_connection_status_ui(state.connection_status, state.connected_editor_id, state.connection_error)
    
    def _on_connection_state_changed(self, block_id: str, status: str, editor_id: str, error: str):
        """Handle connection state changed signal from state service."""
        if block_id != self.block_id:
            return
        
        # Convert empty strings back to None (PyQt signals don't support None)
        editor_id = editor_id if editor_id else None
        error = error if error else None
        
        # Update UI when state changes
        self._update_connection_status_ui(status, editor_id, error)
    
    def _handle_ma3_structure(self, json_str: str):
        """Handle /ma3/structure response."""
        from src.utils.message import Log
        
        try:
            # First principles: Skip processing if structure hasn't changed
            # Use a simple signature (hash of JSON string) to detect duplicates
            structure_signature = hash(json_str)
            if structure_signature == getattr(self, '_last_structure_signature', None):
                Log.debug("ShowManagerPanel: Structure unchanged (same signature), skipping processing")
                # Still need to handle pending dialog even if structure unchanged
                if self._pending_editor_track_select:
                    pending = self._pending_editor_track_select
                    self._pending_editor_track_select = None
                    editor_layer_id = pending.get("editor_layer_id")
                    self._open_ma3_track_dialog_for_editor(editor_layer_id)
                elif getattr(self, '_pending_ma3_track_dialog', False):
                    self._pending_ma3_track_dialog = False
                    self._open_ma3_track_dialog()
                return
            
            data = json.loads(json_str)
            
            # Parse tracks
            new_tracks = self._mapping_service.parse_ma3_structure(data)
            
            # First principles: Only process if tracks actually changed
            # Compare track count and coordinates to detect real changes
            old_track_coords = {track.coord for track in getattr(self, '_ma3_tracks', [])}
            new_track_coords = {track.coord for track in new_tracks}
            tracks_changed = (len(new_tracks) != len(getattr(self, '_ma3_tracks', [])) or 
                            old_track_coords != new_track_coords)
            
            # Update tracks and signature
            self._ma3_tracks = new_tracks
            self._last_structure_signature = structure_signature
            
            # Only log and update UI if tracks actually changed
            if tracks_changed:
                self._log(f"Loaded {len(self._ma3_tracks)} MA3 tracks")
                
                # Update UI with new tracks
                self._update_synced_layers_list()
                self._update_available_layers_list()
                self._resolve_pending_track_create()
            else:
                # Tracks didn't change - skip expensive refresh operations
                Log.debug(f"ShowManagerPanel: Structure unchanged ({len(self._ma3_tracks)} tracks), skipping refresh")
            
            # Handle pending dialog (user clicked "Add MA3 Track" but tracks weren't loaded)
            if self._pending_editor_track_select:
                pending = self._pending_editor_track_select
                self._pending_editor_track_select = None
                editor_layer_id = pending.get("editor_layer_id")
                self._open_ma3_track_dialog_for_editor(editor_layer_id)
            elif getattr(self, '_pending_ma3_track_dialog', False):
                self._pending_ma3_track_dialog = False
                self._open_ma3_track_dialog()
            
            # NOTE: Removed auto-fetch of all events here.
            # Events are fetched per-track when the track is added and hooked.
            # The hook confirmation includes initial events, no separate fetch needed.
            
        except json.JSONDecodeError as e:
            Log.error(f"Failed to parse MA3 structure: {e}")
            self._log(f"Error parsing MA3 structure: {e}")
    
    def _handle_structure_track_added(self, args: list):
        """Handle /ma3/structure/track_added notification."""
        
        if len(args) < 4:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        tg_idx = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1])
        track_idx = int(args[2]) if isinstance(args[2], (int, float)) else int(args[2])
        track_name = str(args[3]) if len(args) > 3 else ""
        
        coord = f"{tc_no}.{tg_idx}.{track_idx}"
        self._log(f"Structure change: Track added - {coord} ({track_name})")
        
        
        self._request_ma3_structure_refresh("structure_track_added")
        
        # Auto-assign to "unassigned" layer
        self._handle_new_track(coord, track_name)
    
    def _handle_structure_track_removed(self, args: list):
        """Handle /ma3/structure/track_removed notification."""
        
        if len(args) < 4:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        tg_idx = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1])
        track_idx = int(args[2]) if isinstance(args[2], (int, float)) else int(args[2])
        track_name = str(args[3]) if len(args) > 3 else ""
        
        coord = f"{tc_no}.{tg_idx}.{track_idx}"
        self._log(f"Structure change: Track removed - {coord} ({track_name})")
        
        
        self._request_ma3_structure_refresh("structure_track_removed")
        self._handle_deleted_track(coord, track_name)
    
    def _handle_structure_track_changed(self, args: list):
        """Handle /ma3/structure/track_changed notification."""
        if len(args) < 4:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        tg_idx = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1])
        track_idx = int(args[2]) if isinstance(args[2], (int, float)) else int(args[2])
        track_name = str(args[3]) if len(args) > 3 else ""
        
        coord = f"{tc_no}.{tg_idx}.{track_idx}"
        self._log(f"Structure change: Track changed - {coord} ({track_name})")
        
        self._request_ma3_structure_refresh("structure_track_changed")
    
    def _handle_structure_track_group_added(self, args: list):
        """Handle /ma3/structure/track_group_added notification."""
        if len(args) < 2:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        tg_idx = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1])
        
        self._log(f"Structure change: Track group added - TC {tc_no}, TG {tg_idx}")
        
        self._request_ma3_structure_refresh("structure_track_group_added")
    
    def _handle_structure_track_group_removed(self, args: list):
        """Handle /ma3/structure/track_group_removed notification."""
        if len(args) < 2:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        tg_idx = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1])
        
        self._log(f"Structure change: Track group removed - TC {tc_no}, TG {tg_idx}")
        
        self._request_ma3_structure_refresh("structure_track_group_removed")
    
    def _handle_structure_timecode_added(self, args: list):
        """Handle /ma3/structure/timecode_added notification."""
        if len(args) < 1:
            return
        
        tc_no = int(args[0]) if isinstance(args[0], (int, float)) else int(args[0])
        
        self._log(f"Structure change: Timecode added - {tc_no}")
        
        self._request_ma3_structure_refresh("structure_timecode_added")
    
    def _handle_new_track(self, coord: str, track_name: str):
        """Handle a new track being added to MA3."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        # Ensure "unassigned" layer exists
        if "unassigned" not in self._ez_layers:
            self._ez_layers.insert(0, "unassigned")
            self._mapping_service.set_ez_layers(self._ez_layers)
        
        # Check if this track is already synced
        if self._sync_system_manager.is_synced("ma3", coord):
            # Already synced, no action needed
            return
        
        # Track is not synced - user must explicitly add it via Layer Sync tab
        self._log(f"New MA3 track '{track_name}' ({coord}) detected. Add it via Layer Sync tab to sync.")
    
    def _handle_deleted_track(self, coord: str, track_name: str):
        """Handle a track being deleted from MA3."""
        if not hasattr(self, '_settings_manager') or not self._settings_manager:
            return
        
        # Check if this track is synced
        if self._sync_system_manager.is_synced("ma3", coord):
            entity = self._sync_system_manager.get_synced_layer_by_ma3_coord(coord)
            if entity and entity.editor_layer_id:
                # Remove the sync relationship
                self._sync_system_manager.remove_synced_layer(entity.id)
                self._log(f"Track '{track_name}' ({coord}) deleted from MA3, sync relationship removed")
            else:
                self._log(f"Track '{track_name}' ({coord}) deleted from MA3 (was synced but no Editor layer mapped)")
        else:
            self._log(f"Track '{track_name}' ({coord}) deleted from MA3 (was not synced)")
    
    def _remove_ez_layer(self, layer_name: str):
        """Remove a layer from the connected Editor block."""
        # Get connected editor ID from state service
        if not self._state_service:
            return
        
        state = self._state_service.get_connection_state(self.block_id)
        if not state.connected_editor_id:
            return
        
        try:
            # Send command to Editor to delete layer
            result = self.facade.send_command(
                state.connected_editor_id,
                "delete_layer",
                {"layer_name": layer_name}
            )
            
            if result.success:
                self._log(f"Removed layer '{layer_name}' from Editor")
            else:
                self._log(f"Failed to remove layer '{layer_name}': {result.error}")
        except Exception as e:
            import traceback
            Log.error(f"ShowManagerPanel: Error removing EZ layer: {e}\n{traceback.format_exc()}")
            self._log(f"Error removing layer '{layer_name}': {e}")
    
    # === Layer Sync Tab Handlers ===
    
    def _on_target_timecode_changed(self):
        """Handle target timecode text change (undoable via settings manager)."""
        try:
            value = int(self.target_timecode_edit.text().strip())
        except ValueError:
            # Invalid input, restore previous value
            if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
                current_tc = self._settings_manager.target_timecode
                self.target_timecode_edit.blockSignals(True)
                self.target_timecode_edit.setText(str(current_tc))
                self.target_timecode_edit.blockSignals(False)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the timecode.")
            return
        
        # Validate range
        if value < 1:
            # Invalid range, restore previous value
            if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
                current_tc = self._settings_manager.target_timecode
                self.target_timecode_edit.blockSignals(True)
                self.target_timecode_edit.setText(str(current_tc))
                self.target_timecode_edit.blockSignals(False)
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Timecode", "Timecode number must be >= 1")
            return
        
        # Store old timecode for comparison
        old_timecode = getattr(self._settings_manager, 'target_timecode', None) if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded() else None
        
        # Update via settings manager (single pathway, auto-saves, follows standard pattern)
        try:
            self._settings_manager.target_timecode = value
            Log.info(f"Target timecode set to {value}")
        except ValueError as e:
            Log.error(f"Failed to set target timecode: {e}")
            # Restore previous value
            if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
                current_tc = self._settings_manager.target_timecode
                self.target_timecode_edit.blockSignals(True)
                self.target_timecode_edit.setText(str(current_tc))
                self.target_timecode_edit.blockSignals(False)
            return
        
        # If timecode actually changed, clear cached MA3 data and refetch for new timecode
        if old_timecode is not None and old_timecode != value:
            Log.info(f"Timecode changed from {old_timecode} to {value}. Clearing cached MA3 data and refetching...")
            
            # Clear cached data (reset to initial state)
            if hasattr(self, '_ma3_tracks'):
                self._ma3_tracks = []
            if hasattr(self, '_ma3_events'):
                self._ma3_events = []
            
            # Trigger refetch of structure (which will auto-fetch events)
            try:
                self._auto_fetch_ma3_structure()
            except RuntimeError as e:
                Log.warning(f"Could not auto-fetch MA3 structure after timecode change: {e}")
                # Don't show error dialog - user might not have MA3 connected yet

    def _on_load_timecode_clicked(self) -> None:
        """
        Handle Load Timecode button click.
        
        This:
        1. Validates the new timecode number
        2. Clears all non-synced MA3 tracks from the previous timecode
        3. Clears local MA3 state in SyncSystemManager
        4. Fetches track groups and tracks for the new timecode
        """
        # Validate input
        try:
            new_timecode = int(self.target_timecode_edit.text().strip())
        except ValueError:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the timecode.")
            return
        
        if new_timecode < 1:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Timecode", "Timecode number must be >= 1")
            return
        
        # Get old timecode for comparison
        old_timecode = self._settings_manager.target_timecode if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded() else None
        
        Log.info(f"Switching to timecode {new_timecode} (previous: {old_timecode})")
        
        # Set flag to prevent UI refresh from overwriting the text box during switch
        self._switching_timecode = True
        
        # Step 1: Clear non-synced MA3 tracks from SyncSystemManager
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            # Clear all MA3 track state (available tracks list)
            self._sync_system_manager.clear_ma3_state()
            Log.info("Cleared MA3 state from SyncSystemManager")
        
        # Step 2: Clear local cached data
        if hasattr(self, '_ma3_tracks'):
            self._ma3_tracks = []
        if hasattr(self, '_ma3_events'):
            self._ma3_events = []
        
        # Step 3: Update the timecode setting
        try:
            self._settings_manager.target_timecode = new_timecode
            # Force immediate save to prevent race condition with reload_from_storage
            # (undo stack changes trigger reloads before debounced save completes)
            self._settings_manager.force_save()
            Log.info(f"Target timecode set to {new_timecode}")
        except ValueError as e:
            Log.error(f"Failed to set target timecode: {e}")
            self._switching_timecode = False
            return
        
        # Step 3.5: Re-initialize layers for new timecode
        if hasattr(self, '_sync_system_manager') and self._sync_system_manager:
            self._sync_system_manager._configured_timecode = new_timecode
            self._sync_system_manager._reinitialize_for_current_timecode()
            Log.info(f"Re-initialized layers for timecode {new_timecode}")
        
        # Step 4: Fetch new timecode data from MA3
        try:
            # Force tracks to be fetched when track groups arrive
            self._force_tracks_refresh = True
            self._auto_fetch_ma3_structure()
            Log.info(f"Fetching track groups for timecode {new_timecode}...")
        except RuntimeError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                "MA3 Not Ready", 
                f"Could not fetch MA3 data: {e}\n\nMake sure MA3 is connected and the listener is active."
            )
            self._switching_timecode = False
            return
        
        # Step 5: Refresh the available layers list
        self._update_available_layers_list()
        
        # Step 6: Ensure the text box shows the correct value (prevent race conditions)
        self.target_timecode_edit.blockSignals(True)
        self.target_timecode_edit.setText(str(new_timecode))
        self.target_timecode_edit.blockSignals(False)
        
        # Clear the switching flag
        self._switching_timecode = False
        
        Log.info(f"Timecode {new_timecode} switched successfully")

    def _on_manual_layer_refresh(self) -> None:
        """Manual refresh for MA3 tracks and layer list."""
        self._request_ma3_structure_refresh("manual")
        self._update_synced_layers_list()
        self._update_available_layers_list()
    
    def _on_layer_sync_entities_changed(self):
        """Handle entities changed signal from controller"""
        self._update_synced_layers_list()
    
    def _on_layer_sync_entity_added(self, entity_id: str, entity_type: str):
        """Handle entity added signal from controller"""
        self._update_synced_layers_list()
    
    def _on_layer_sync_entity_updated(self, entity_id: str, entity_type: str):
        """Handle entity updated signal from SyncSystemManager"""
        self._update_synced_layers_list()

        if entity_type == "ma3_events":
            parsed = self._parse_coord_key(entity_id)
            if not parsed:
                return
            tc, tg, track = parsed
            # Get events from SyncSystemManager
            coord = f"tc{tc}_tg{tg}_tr{track}"
            cached_events = self._sync_system_manager._get_ma3_events(coord)
            if not cached_events:
                return
            if self._reconcile_pending and entity_id in self._reconcile_pending:
                self._handle_reconcile_events(tc, tg, track, cached_events)
            else:
                self._refresh_reconcile_cache_from_ma3_events(tc, tg, track, cached_events)
            return

        if entity_type == "editor_events":
            self._refresh_reconcile_cache_from_editor_layer(entity_id)
            return

    def _on_layer_sync_entity_removed(self, entity_id: str, entity_type: str):
        """Handle entity removed signal from controller"""
        self._update_synced_layers_list()

    def _on_editor_layer_deleted(self, layer_id: str, ma3_track_coord: str) -> None:
        """Prompt to clear MA3 track when a synced Editor layer is deleted."""
        if not ma3_track_coord:
            return
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Clear MA3 Track?",
            (
                f"Editor layer '{layer_id}' was deleted.\n\n"
                f"Do you want to clear the MA3 track '{ma3_track_coord}'?\n"
                "This removes all events from that track but keeps the track itself."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        parsed = self._parse_ma3_coord(ma3_track_coord)
        if not parsed:
            self._log(f"Could not parse MA3 coord: {ma3_track_coord}")
            return

        tc, tg, track = parsed
        try:
            self._send_lua_command(f"EZ.ClearTrack({tc}, {tg}, {track})")
            self._log(f"Cleared MA3 track {ma3_track_coord}")
        except Exception as exc:
            self._log(f"Failed to clear MA3 track {ma3_track_coord}: {exc}")
    
    # =========================================================================
    # SyncSystemManager Signal Handlers (New Architecture)
    # =========================================================================
    
    def _on_sync_system_entities_changed(self) -> None:
        """Handle entities changed signal from SyncSystemManager.
        
        Updates local sync tables.  Sync icon state is pushed directly
        by SyncSystemManager._push_sync_state_to_all_layers (no panel involvement).
        """
        self._update_synced_layers_list()
        self._update_available_layers_list()
    
    def _on_sync_system_entity_updated(self, entity_id: str) -> None:
        """Handle entity updated signal from SyncSystemManager."""
        self._update_synced_layers_list()
    
    def _on_sync_system_status_changed(self, entity_id: str, status: str) -> None:
        """Handle sync status changed signal from SyncSystemManager."""
        self._update_synced_layers_list()
    
    def _on_sync_system_divergence_detected(self, entity_id: str, comparison: Any) -> None:
        """Update UI to show divergence -- no popup.
        
        Resolution is deferred to the synced layers table where the user
        can click 'Keep EZ' or 'Keep MA3' on the DIVERGED row.
        """
        self._update_synced_layers_list()
    
    def _on_sync_system_error(self, entity_id: str, error_message: str) -> None:
        """Handle error signal from SyncSystemManager."""
        self._log(f"Sync error for {entity_id}: {error_message}")
    
    def _on_ssm_connection_state_changed(self, state: str) -> None:
        """Handle connection state changes from SyncSystemManager.
        
        SSM is the single source of truth for MA3 connection state.
        This handler refreshes the full connection status display (which
        now reads SSM state directly) and updates sync button states.
        
        Args:
            state: "connected", "disconnected", or "stale"
        """
        # Refresh the full connection status display (reads SSM state internally)
        self._load_connection_state_from_service()
        
        # Enable/disable MA3-dependent buttons
        if state == "connected":
            self._update_sync_buttons_for_connection(True)
        elif state == "disconnected":
            self._update_sync_buttons_for_connection(False)
        
        # Refresh synced layers table (status column may change, e.g. awaiting_connection -> synced)
        self._update_synced_layers_list()
        self._update_available_layers_list()
    
    def _on_track_conflict_prompt(self, editor_layer_id: str, track_name: str, 
                                   editor_event_count: int, ma3_event_count: int) -> None:
        """Handle track conflict - prompt user for resolution."""
        from PyQt6.QtWidgets import QMessageBox
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Track Already Exists")
        msg.setText(f"Track '{track_name}' already exists in MA3.")
        msg.setInformativeText(
            f"Editor layer has {editor_event_count} events.\n"
            f"MA3 track has {ma3_event_count} events.\n\n"
            "How would you like to proceed?"
        )
        
        overwrite_btn = msg.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        merge_btn = msg.addButton("Merge", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        
        msg.setDefaultButton(cancel_btn)
        msg.exec()
        
        clicked = msg.clickedButton()
        
        if clicked == overwrite_btn:
            strategy = "overwrite"
        elif clicked == merge_btn:
            strategy = "merge"
        else:
            strategy = "cancel"
        
        if strategy != "cancel":
            # Get the config for this layer and retry with the strategy
            config = self._available_editor_config.get(editor_layer_id, {})
            target_tc = config.get("tc", self._sync_system_manager.configured_timecode)
            target_tg = config.get("tg", 1)
            target_seq = config.get("seq", 1)
            
            entity_id = self._sync_system_manager.sync_editor_to_ma3(
                editor_layer_id=editor_layer_id,
                target_timecode=target_tc,
                target_track_group=target_tg,
                target_sequence=target_seq,
                conflict_strategy=strategy,
            )
            
            if entity_id:
                self._log(f"Synced Editor layer to MA3 ({strategy}): {editor_layer_id}")
                # Clean up pending config
                if editor_layer_id in self._available_editor_config:
                    del self._available_editor_config[editor_layer_id]
            else:
                self._log(f"Failed to sync Editor layer to MA3: {editor_layer_id}")
        else:
            self._log(f"Sync cancelled for layer: {editor_layer_id}")
    
    def _update_available_layers_list(self) -> None:
        """Update the available layers list (MA3 tracks and Editor layers not synced)."""
        if not hasattr(self, 'available_table') or not self.available_table:
            return
        
        self.available_table.setRowCount(0)
        
        # Get available items from manager
        available_ma3 = self._sync_system_manager.get_available_ma3_tracks()
        available_editor = self._sync_system_manager.get_available_editor_layers()
        
        # Combine and sort
        all_items = []
        for track in available_ma3:
            all_items.append({
                "type": "ma3",
                "coord": track.get("coord"),  # MA3 track coordinate, there should be no default fail loud, since ma3 tracks names can change, we can use this for fingerprinting
                "index": track.get("index"),  # MA3 track index, there should be no default fail loud
                "name": track.get("name"),
            })
        for layer in available_editor:
            all_items.append({
                "type": "editor",
                "id": layer.get("layer_id"), #editor layer id, there should be no default fail loud
                "name": layer.get("name"),  #editor layer name, there should be no default fail loud
            })
        
        all_items.sort(key=lambda x: (x["type"], x["name"]))
        
        self.available_table.setRowCount(len(all_items))
        
        # Get current configured timecode
        current_tc = self._sync_system_manager.configured_timecode if hasattr(self, '_sync_system_manager') and self._sync_system_manager else 1
        
        for row, item in enumerate(all_items):
            # Get the identifier based on type
            # MA3 tracks use coord, Editor layers use id
            item_identifier = item.get("coord") if item["type"] == "ma3" else item.get("id")
            
            # Sync checkbox
            sync_check = QCheckBox()
            sync_check.setChecked(False)
            sync_check.stateChanged.connect(
                lambda state, t=item["type"], i=item_identifier: self._on_available_sync_toggled(t, i, state)
            )
            self.available_table.setCellWidget(row, 0, self._wrap_table_cell_widget(sync_check))
            
            # Type
            type_label = QLabel("MA3" if item["type"] == "ma3" else "Editor")
            type_color = Colors.ACCENT_GREEN.name() if item["type"] == "ma3" else Colors.ACCENT_BLUE.name()
            type_label.setStyleSheet(f"color: {type_color}; font-weight: bold;")
            type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.available_table.setCellWidget(row, 1, self._wrap_table_cell_widget(type_label))
            
            # Name
            name_item = QTableWidgetItem(item.get("name", "Unknown"))
            name_item.setForeground(QBrush(Colors.TEXT_PRIMARY))
            self.available_table.setItem(row, 2, name_item)
            
            # Target dropdown + TC/TG/Seq columns
            if item["type"] == "ma3":
                # Target dropdown: "Create New" or existing Editor layer
                target_combo = QComboBox()
                target_combo.addItem("Create New", None)
                for editor_layer in available_editor:
                    editor_name = editor_layer.get("name", "?")
                    editor_id = editor_layer.get("layer_id", "")
                    target_combo.addItem(f"Map to: {editor_name}", editor_id)
                
                # Restore previous selection if any
                ma3_coord = item.get("coord")
                prev_target = self._available_ma3_target.get(ma3_coord)
                if prev_target:
                    for idx in range(target_combo.count()):
                        if target_combo.itemData(idx) == prev_target:
                            target_combo.setCurrentIndex(idx)
                            break
                
                target_combo.currentIndexChanged.connect(
                    lambda index, combo=target_combo, coord=ma3_coord: self._on_available_ma3_target_changed(coord, combo.itemData(index))
                )
                self.available_table.setCellWidget(row, 3, self._wrap_table_cell_widget(target_combo))
                
                # MA3 tracks don't need TC/TG/Seq config
                for col in (4, 5, 6):
                    empty_item = QTableWidgetItem("-")
                    empty_item.setForeground(QBrush(Colors.TEXT_SECONDARY))
                    empty_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.available_table.setItem(row, col, empty_item)
            elif item["type"] == "editor":
                layer_id = item.get("id")
                
                # Target dropdown: "Create New" or existing MA3 track
                target_combo = QComboBox()
                target_combo.addItem("Create New", None)
                for ma3_track in available_ma3:
                    track_name = ma3_track.get("name", "?")
                    track_coord = ma3_track.get("coord", "")
                    target_combo.addItem(f"Map to: {track_name}", track_coord)
                
                # Restore previous selection if any
                prev_target = self._available_editor_target.get(layer_id)
                if prev_target:
                    for idx in range(target_combo.count()):
                        if target_combo.itemData(idx) == prev_target:
                            target_combo.setCurrentIndex(idx)
                            break
                
                target_combo.currentIndexChanged.connect(
                    lambda index, combo=target_combo, lid=layer_id: self._on_available_editor_target_changed(lid, combo.itemData(index))
                )
                self.available_table.setCellWidget(row, 3, self._wrap_table_cell_widget(target_combo))
                
                # Get or initialize config for this layer
                if layer_id not in self._available_editor_config:
                    self._available_editor_config[layer_id] = {
                        "tc": current_tc,
                        "tg": 1,
                        "seq": 1,
                    }
                config = self._available_editor_config[layer_id]
                # Always update TC to current configured timecode
                config["tc"] = current_tc
                
                # TC - read-only label showing current timecode
                tc_label = QLabel(str(current_tc))
                tc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                tc_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY.name()};")
                tc_label.setToolTip("Timecode is set to the Layer Sync's current timecode")
                self.available_table.setCellWidget(row, 4, self._wrap_table_cell_widget(tc_label))
                
                # TG - editable spinbox
                tg_spin = QSpinBox()
                tg_spin.setRange(1, 99999)
                tg_spin.setValue(config.get("tg", 1))
                tg_spin.setToolTip("MA3 Track Group number (1-99999)")
                tg_spin.valueChanged.connect(
                    lambda value, lid=layer_id: self._on_available_editor_tg_changed(lid, value)
                )
                self.available_table.setCellWidget(row, 5, self._wrap_table_cell_widget(tg_spin))
                
                # Seq - editable spinbox
                seq_spin = QSpinBox()
                seq_spin.setRange(1, 99999)
                seq_spin.setValue(config.get("seq", 1))
                seq_spin.setToolTip("MA3 Sequence number (1-99999)")
                seq_spin.valueChanged.connect(
                    lambda value, lid=layer_id: self._on_available_editor_seq_changed(lid, value)
                )
                self.available_table.setCellWidget(row, 6, self._wrap_table_cell_widget(seq_spin))
    
    def _on_available_sync_toggled(self, item_type: str, item_id: str, state: int) -> None:
        """Handle sync checkbox toggle in the Available tab.
        
        For Editor->MA3 sync, this method handles all dialog logic BEFORE calling
        sync_editor_to_ma3, which is now an atomic operation with no prompts.
        """
        
        if state != Qt.CheckState.Checked.value:
            return  # Only handle checking, not unchecking
        
        Log.info(f"ShowManagerPanel: Sync checkbox checked for {item_type} layer: {item_id}")
        
        if item_type == "editor":
            # Editor->MA3 sync: check the Target dropdown for "Create New" vs existing MA3 track
            target_ma3_coord = self._available_editor_target.get(item_id)
            
            if target_ma3_coord:
                # Map to existing MA3 track - run reconciliation first
                Log.info(f"ShowManagerPanel: Editor sync with existing MA3 target: {target_ma3_coord}")
                authority = self._reconcile_existing_target(item_id, target_ma3_coord)
                if authority is None:
                    self._log(f"Sync cancelled for Editor layer: {item_id}")
                    self._update_available_layers_list()
                    return
                
                entity_id = self._sync_system_manager.sync_layer(
                    source="editor",
                    source_id=item_id,
                    target_id=target_ma3_coord,
                    event_authority=authority,
                )
                
                if entity_id:
                    self._available_editor_target.pop(item_id, None)
                    self._available_editor_config.pop(item_id, None)
                    authority_desc = "pushed Editor events" if authority == "editor" else "receiving MA3 events"
                    self._log(f"Synced Editor layer to existing MA3 track ({authority_desc}): {item_id}")
                    self._update_available_layers_list()
                    self._update_synced_layers_list()
                else:
                    self._log(f"Failed to sync Editor layer to MA3 track: {item_id}")
            else:
                # Create New: use the dialog flow for creating new MA3 track
                self._sync_editor_layer_with_dialogs(item_id)
        else:
            # MA3->Editor sync: check the Target dropdown for "Create New" vs existing layer
            target_editor_layer_id = self._available_ma3_target.get(item_id)
            
            if target_editor_layer_id:
                # Map to existing Editor layer - run reconciliation first
                Log.info(f"ShowManagerPanel: MA3 sync with existing target: {target_editor_layer_id}")
                authority = self._reconcile_existing_target(target_editor_layer_id, item_id)
                if authority is None:
                    self._log(f"Sync cancelled for MA3 track: {item_id}")
                    self._update_available_layers_list()
                    return
                
                entity_id = self._sync_system_manager.sync_layer(
                    source=item_type,
                    source_id=item_id,
                    target_id=target_editor_layer_id,
                    event_authority=authority,
                )
            else:
                # Create new Editor layer (default behavior)
                entity_id = self._sync_system_manager.sync_layer(
                    source=item_type,
                    source_id=item_id,
                    auto_create=True,
                )
            
            if entity_id:
                # Clean up the target selection
                self._available_ma3_target.pop(item_id, None)
                if target_editor_layer_id:
                    authority_desc = "pushed Editor events" if authority == "editor" else "receiving MA3 events"
                    self._log(f"Synced MA3 track to existing Editor layer ({authority_desc}): {item_id}")
                else:
                    self._log(f"Started syncing {item_type} layer: {item_id}")
            else:
                self._log(f"Failed to sync {item_type} layer: {item_id}")
    
    def _sync_editor_layer_with_dialogs(self, editor_layer_id: str) -> None:
        """
        Sync an Editor layer to MA3 with dialog prompts for conflicts.
        
        SIMPLIFIED FLOW:
        1. Editor layer selected to add
        2. Check if track exists in MA3
        3. If exists: prompt "Use Existing" or "Create New"
        4. If "Use Existing": fetch MA3 events, compare, show reconciliation dialog
        5. Execute chosen action
        """
        from PyQt6.QtWidgets import QMessageBox
        import time
        
        
        Log.info(f"ShowManagerPanel: _sync_editor_layer_with_dialogs called for: {editor_layer_id}")
        
        # Get config from table
        config = self._available_editor_config.get(editor_layer_id, {})
        target_tc = config.get("tc", self._sync_system_manager.configured_timecode)
        target_tg = config.get("tg", 1)
        target_seq = config.get("seq", 1)
        
        
        # Get layer info
        layer_info = self._sync_system_manager._get_editor_layer_info(editor_layer_id)
        if not layer_info:
            self._log(f"Editor layer not found: {editor_layer_id}")
            return
        
        raw_name = layer_info.get("name", editor_layer_id)
        base_track_name = raw_name if raw_name.startswith("ez_") else f"ez_{raw_name}"
        
        # Get Editor events
        editor_events = self._sync_system_manager._get_editor_events(
            editor_layer_id, layer_info.get("block_id")
        )
        editor_event_count = len(editor_events) if editor_events else 0
        Log.info(f"ShowManagerPanel: Editor layer '{raw_name}' has {editor_event_count} events")
        
        # =========================================================================
        # STEP 1: Check if track exists in MA3
        # =========================================================================
        # CRITICAL: Refresh tracks from MA3 before checking to ensure we have current data
        # This handles cases where tracks were deleted in MA3 but cache is stale
        Log.info(f"ShowManagerPanel: Refreshing tracks from MA3 (TC{target_tc}, TG{target_tg}) before checking for existing track")
        self._sync_system_manager._refresh_ma3_tracks_sync(target_tc, target_tg)
        
        existing_track_no = None
        existing_track_name = None
        
        if self._sync_system_manager._check_ma3_track_name_exists(target_tc, target_tg, base_track_name):
            existing_track_name = base_track_name
            existing_track_no = self._sync_system_manager._find_ma3_track_by_name(
                target_tc, target_tg, base_track_name
            )
            Log.info(f"ShowManagerPanel: Found existing track '{existing_track_name}' (track_no={existing_track_no})")
        
        # Determine action
        action = "create_new"
        track_no = None
        
        # =========================================================================
        # STEP 2: If track exists, prompt user
        # =========================================================================
        if existing_track_no is not None:
            msg = QMessageBox(self)
            msg.setWindowTitle("Existing Track Found")
            msg.setText(f"A track named '{existing_track_name}' already exists in MA3.")
            msg.setInformativeText(
                f"You are syncing Editor layer '{raw_name}'.\n\n"
                "Do you want to use this existing track or create a new one?"
            )
            
            use_existing_btn = msg.addButton("Use Existing Track", QMessageBox.ButtonRole.AcceptRole)
            create_new_btn = msg.addButton("Create New Track", QMessageBox.ButtonRole.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(use_existing_btn)
            msg.exec()
            
            clicked = msg.clickedButton()
            
            if clicked == cancel_btn:
                self._log(f"Sync cancelled for layer: {editor_layer_id}")
                self._update_available_layers_list()
                return
            elif clicked == create_new_btn:
                action = "create_new"
                track_no = None
                Log.info(f"ShowManagerPanel: User chose to create new track")
            elif clicked == use_existing_btn:
                # =========================================================================
                # STEP 3: Fetch MA3 events for comparison
                # =========================================================================
                Log.info(f"ShowManagerPanel: User chose to use existing track, fetching MA3 events...")
                
                coord = f"tc{target_tc}_tg{target_tg}_tr{existing_track_no}"
                
                # Clear old cached events to ensure we get fresh data
                if coord in self._sync_system_manager._ma3_track_events:
                    del self._sync_system_manager._ma3_track_events[coord]
                
                # Request events from MA3
                if self._sync_system_manager.ma3_comm_service:
                    lua_cmd = f"EZ.GetEvents({target_tc}, {target_tg}, {existing_track_no})"
                    self._sync_system_manager.ma3_comm_service.send_lua_command(lua_cmd)
                    Log.info(f"ShowManagerPanel: Sent {lua_cmd}, waiting for response...")
                    
                    # Wait for response, processing Qt events to allow signal delivery
                    from PyQt6.QtCore import QCoreApplication
                    for i in range(20):  # Wait up to 2 seconds
                        QCoreApplication.processEvents()
                        time.sleep(0.1)
                        # Check if events arrived (cache was cleared, so any events means response arrived)
                        if self._sync_system_manager._get_ma3_events(coord):
                            Log.info(f"ShowManagerPanel: Events received after {(i+1)*100}ms")
                            break
                    else:
                        Log.warning(f"ShowManagerPanel: Timeout waiting for events from MA3")
                
                # Get events from cache (now populated by GetEvents response)
                ma3_events = self._sync_system_manager._get_ma3_events(coord)
                ma3_event_count = len(ma3_events) if ma3_events else 0
                Log.info(f"ShowManagerPanel: MA3 track has {ma3_event_count} events")
                
                # =========================================================================
                # STEP 4: Compare and show reconciliation dialog if needed
                # =========================================================================
                if ma3_event_count > 0 and editor_event_count > 0:
                    # Both have events - need reconciliation
                    recon_msg = QMessageBox(self)
                    recon_msg.setWindowTitle("Events Differ")
                    recon_msg.setText("Both Editor and MA3 have events.")
                    recon_msg.setInformativeText(
                        f"Editor layer '{raw_name}': {editor_event_count} events\n"
                        f"MA3 track '{existing_track_name}': {ma3_event_count} events\n\n"
                        "Which events do you want to keep?"
                    )
                    
                    keep_editor_btn = recon_msg.addButton("Keep Editor Events", QMessageBox.ButtonRole.AcceptRole)
                    keep_ma3_btn = recon_msg.addButton("Keep MA3 Events", QMessageBox.ButtonRole.ActionRole)
                    recon_cancel_btn = recon_msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
                    recon_msg.setDefaultButton(keep_editor_btn)
                    recon_msg.exec()
                    
                    recon_clicked = recon_msg.clickedButton()
                    
                    if recon_clicked == recon_cancel_btn:
                        self._log(f"Sync cancelled for layer: {editor_layer_id}")
                        self._update_available_layers_list()
                        return
                    elif recon_clicked == keep_editor_btn:
                        action = "use_existing_keep_editor"
                        track_no = existing_track_no
                        Log.info(f"ShowManagerPanel: User chose to keep Editor events")
                    elif recon_clicked == keep_ma3_btn:
                        action = "use_existing_keep_ma3"
                        track_no = existing_track_no
                        Log.info(f"ShowManagerPanel: User chose to keep MA3 events")
                elif ma3_event_count == 0:
                    # MA3 empty, push Editor events
                    action = "use_existing_keep_editor"
                    track_no = existing_track_no
                    Log.info(f"ShowManagerPanel: MA3 track empty, will push Editor events")
                else:
                    # Editor empty, receive MA3 events
                    action = "use_existing_keep_ma3"
                    track_no = existing_track_no
                    Log.info(f"ShowManagerPanel: Editor layer empty, will receive MA3 events")
        
        # =========================================================================
        # STEP 5: Execute sync
        # =========================================================================
        Log.info(f"ShowManagerPanel: Executing sync with action='{action}'")
        
        entity_id = self._sync_system_manager.sync_editor_to_ma3(
            editor_layer_id=editor_layer_id,
            target_timecode=target_tc,
            target_track_group=target_tg,
            target_sequence=target_seq,
            action=action,
            existing_track_no=track_no,
        )
        
        if entity_id:
            action_desc = {
                "create_new": "created new track",
                "use_existing_keep_editor": "linked to existing (pushed Editor events)",
                "use_existing_keep_ma3": "linked to existing (receiving MA3 events)",
            }.get(action, action)
            self._log(f"Synced Editor layer ({action_desc}): {editor_layer_id}")
            
            # Clean up the pending config
            if editor_layer_id in self._available_editor_config:
                del self._available_editor_config[editor_layer_id]
            
            # Refresh UI
            self._update_available_layers_list()
            self._update_synced_layers_list()
        else:
            self._log(f"Failed to sync Editor layer: {editor_layer_id}")
            self._update_available_layers_list()
    
    def _on_available_ma3_target_changed(self, ma3_coord: str, editor_layer_id: Optional[str]) -> None:
        """Handle Target dropdown change for an available MA3 track.
        
        Args:
            ma3_coord: The MA3 track coordinate
            editor_layer_id: The selected Editor layer ID, or None for 'Create New'
        """
        self._available_ma3_target[ma3_coord] = editor_layer_id
    
    def _on_available_editor_target_changed(self, editor_layer_id: str, ma3_coord: Optional[str]) -> None:
        """Handle Target dropdown change for an available Editor layer.
        
        Args:
            editor_layer_id: The Editor layer ID
            ma3_coord: The selected MA3 track coordinate, or None for 'Create New'
        """
        self._available_editor_target[editor_layer_id] = ma3_coord
    
    def _reconcile_existing_target(
        self,
        editor_layer_id: str,
        ma3_coord: str,
    ) -> Optional[str]:
        """Run event diff and prompt user when mapping to an existing target.
        
        Fetches event counts from both the Editor layer and the MA3 track,
        then shows a reconciliation dialog if both sides have events.
        
        Args:
            editor_layer_id: The Editor layer name/ID
            ma3_coord: The MA3 track coordinate (e.g. 'tc1_tg1_tr3')
            
        Returns:
            "editor" - keep Editor events (push Editor -> MA3)
            "ma3"    - keep MA3 events (push MA3 -> Editor)
            None     - user cancelled
        """
        from PyQt6.QtWidgets import QMessageBox
        from PyQt6.QtCore import QCoreApplication
        import time
        
        # -- Get Editor event count --
        editor_events = self._sync_system_manager._get_editor_events(
            editor_layer_id, self._sync_system_manager._get_editor_block_id()
        )
        editor_count = len(editor_events) if editor_events else 0
        
        # -- Get MA3 event count (fresh fetch) --
        # Clear old cache to ensure fresh data
        if ma3_coord in self._sync_system_manager._ma3_track_events:
            del self._sync_system_manager._ma3_track_events[ma3_coord]
        
        # Parse coord to get tc/tg/tr numbers for the Lua command
        parts = self._sync_system_manager._parse_ma3_coord(ma3_coord)
        ma3_count = 0
        if parts and self._sync_system_manager.ma3_comm_service:
            tc_no = parts["timecode_no"]
            tg_no = parts["track_group"]
            tr_no = parts["track"]
            lua_cmd = f"EZ.GetEvents({tc_no}, {tg_no}, {tr_no})"
            self._sync_system_manager.ma3_comm_service.send_lua_command(lua_cmd)
            Log.info(f"ShowManagerPanel: Sent {lua_cmd}, waiting for response...")
            
            # Wait for response (up to 2 seconds)
            for i in range(20):
                QCoreApplication.processEvents()
                time.sleep(0.1)
                if self._sync_system_manager._get_ma3_events(ma3_coord):
                    Log.info(f"ShowManagerPanel: MA3 events received after {(i+1)*100}ms")
                    break
            else:
                Log.warning(f"ShowManagerPanel: Timeout waiting for MA3 events")
            
            ma3_events = self._sync_system_manager._get_ma3_events(ma3_coord)
            ma3_count = len(ma3_events) if ma3_events else 0
        
        Log.info(f"ShowManagerPanel: Reconcile - Editor: {editor_count} events, MA3: {ma3_count} events")
        
        # -- Determine result based on event counts --
        if editor_count == 0 and ma3_count == 0:
            # Neither has events, no conflict - default to editor authority
            return "editor"
        
        if editor_count > 0 and ma3_count == 0:
            # Only Editor has events, auto-select
            return "editor"
        
        if editor_count == 0 and ma3_count > 0:
            # Only MA3 has events, auto-select
            return "ma3"
        
        # Both have events - prompt user
        # Get display names
        track_info = self._sync_system_manager._get_ma3_track_info(ma3_coord)
        ma3_name = track_info.get("name", ma3_coord) if track_info else ma3_coord
        
        msg = QMessageBox(self)
        msg.setWindowTitle("Events Differ")
        msg.setText("Both Editor and MA3 have events.")
        msg.setInformativeText(
            f"Editor layer '{editor_layer_id}': {editor_count} events\n"
            f"MA3 track '{ma3_name}': {ma3_count} events\n\n"
            "Which events do you want to keep?"
        )
        
        keep_editor_btn = msg.addButton("Keep Editor Events", QMessageBox.ButtonRole.AcceptRole)
        keep_ma3_btn = msg.addButton("Keep MA3 Events", QMessageBox.ButtonRole.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(keep_editor_btn)
        msg.exec()
        
        clicked = msg.clickedButton()
        if clicked == keep_editor_btn:
            Log.info(f"ShowManagerPanel: User chose to keep Editor events")
            return "editor"
        elif clicked == keep_ma3_btn:
            Log.info(f"ShowManagerPanel: User chose to keep MA3 events")
            return "ma3"
        else:
            Log.info(f"ShowManagerPanel: User cancelled reconciliation")
            return None
    
    def _on_available_editor_tg_changed(self, layer_id: str, value: int) -> None:
        """Handle Track Group change for an available editor layer."""
        if layer_id not in self._available_editor_config:
            self._available_editor_config[layer_id] = {
                "tc": self._sync_system_manager.configured_timecode if hasattr(self, '_sync_system_manager') and self._sync_system_manager else 1,
                "tg": 1,
                "seq": 1,
            }
        self._available_editor_config[layer_id]["tg"] = value
    
    def _on_available_editor_seq_changed(self, layer_id: str, value: int) -> None:
        """Handle Sequence change for an available editor layer."""
        if layer_id not in self._available_editor_config:
            self._available_editor_config[layer_id] = {
                "tc": self._sync_system_manager.configured_timecode if hasattr(self, '_sync_system_manager') and self._sync_system_manager else 1,
                "tg": 1,
                "seq": 1,
            }
        self._available_editor_config[layer_id]["seq"] = value
    
    def _show_editor_to_ma3_config_dialog(self, editor_layer_id: str) -> None:
        """Show configuration dialog for Editor->MA3 sync."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QFormLayout
        from PyQt6.QtWidgets import QLabel, QSpinBox, QComboBox, QPushButton, QGroupBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Sync Editor Layer to MA3")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Get layer name for display
        layer_info = self._sync_system_manager._get_editor_layer_info(editor_layer_id)
        layer_name = layer_info.get("name", editor_layer_id) if layer_info else editor_layer_id
        
        # Header
        header = QLabel(f"<b>Sync '{layer_name}' to MA3</b>")
        layout.addWidget(header)
        
        # Config group
        config_group = QGroupBox("MA3 Target Configuration")
        config_layout = QFormLayout(config_group)
        
        # Timecode selection (default to Layer Sync's current timecode)
        tc_spin = QSpinBox()
        tc_spin.setRange(1, 99999)
        tc_spin.setValue(self._sync_system_manager.configured_timecode)
        config_layout.addRow("Timecode:", tc_spin)
        
        # Track Group selection (spinbox with 1-99999 range)
        tg_spin = QSpinBox()
        tg_spin.setRange(1, 99999)
        tg_spin.setValue(1)
        tg_spin.setToolTip("MA3 Track Group number (1-99999)")
        # Try to get default from available track groups
        tc_no = tc_spin.value()
        if tc_no in self._sync_system_manager._ma3_track_groups:
            track_groups = self._sync_system_manager._ma3_track_groups[tc_no]
            if track_groups:
                # Use first available track group as default
                tg_spin.setValue(track_groups[0].track_group_no)
        config_layout.addRow("Track Group:", tg_spin)
        
        # Sequence selection
        seq_spin = QSpinBox()
        seq_spin.setRange(1, 99999)
        seq_spin.setValue(1)
        seq_spin.setToolTip("MA3 Sequence number (1-99999)")
        config_layout.addRow("Sequence:", seq_spin)
        
        # Auto-create sequence checkbox
        from PyQt6.QtWidgets import QCheckBox
        auto_seq_check = QCheckBox("Auto-create sequence if needed")
        auto_seq_check.setChecked(True)
        config_layout.addRow("", auto_seq_check)
        
        layout.addWidget(config_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        sync_btn = QPushButton("Sync to MA3")
        sync_btn.setDefault(True)
        
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(sync_btn)
        layout.addLayout(btn_layout)
        
        # Connect signals
        cancel_btn.clicked.connect(dialog.reject)
        
        def on_sync():
            target_tc = tc_spin.value()
            target_tg = tg_spin.value()
            target_seq = seq_spin.value()
            
            # Call the new Editor->MA3 sync method
            entity_id = self._sync_system_manager.sync_editor_to_ma3(
                editor_layer_id=editor_layer_id,
                target_timecode=target_tc,
                target_track_group=target_tg,
                target_sequence=target_seq,
                conflict_strategy=None,  # Fail on conflict
            )
            
            if entity_id:
                self._log(f"Synced Editor layer '{layer_name}' to MA3")
                dialog.accept()
            else:
                self._log(f"Failed to sync Editor layer to MA3")
                # Don't close dialog on failure - let user try again
        
        sync_btn.clicked.connect(on_sync)
        
        dialog.exec()
    
    def _update_synced_layers_list(self):
        """Update the synced layers list using SyncSystemManager."""
        if not hasattr(self, 'layers_table') or not self.layers_table:
            return
        if self._sync_list_updating:
            self._sync_list_pending = True
            return
        self._sync_list_updating = True
        
        # Clear existing rows
        self.layers_table.setRowCount(0)
        
        # Use new SyncSystemManager for synced layers
        if not hasattr(self, '_sync_system_manager') or not self._sync_system_manager:
            self._set_layers_table_message("Sync system not ready.")
            self._sync_list_updating = False
            return
        
        # Get current timecode and filter layers (only show layers matching current timecode)
        current_tc = self._settings_manager.target_timecode if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded() else 1
        all_synced_layers = self._sync_system_manager.get_synced_layers()
        synced_layers = [
            e for e in all_synced_layers 
            if e.ma3_timecode_no == current_tc
        ]
        
        if not synced_layers:
            self._set_layers_table_message("No synced layers. Check the 'Available Layers' tab to add layers.")
            self._sync_list_updating = False
            if self._sync_list_pending:
                self._sync_list_pending = False
                QTimer.singleShot(0, self._update_synced_layers_list)
            return
        
        # Convert SyncLayerEntity to display items
        ma3_items = []
        editor_items = []
        
        for entity in synced_layers:
            # Create display item from SyncLayerEntity
            item = self._entity_to_display_item(entity)
            if entity.source == SyncSource.MA3:
                ma3_items.append(item)
            else:
                editor_items.append(item)
        
        # Sort by name
        ma3_items.sort(key=lambda x: x.name)
        editor_items.sort(key=lambda x: x.name)
        
        # Add sections
        if ma3_items:
            self._add_layer_section("MA3-Sourced Layers", ma3_items)
        if editor_items:
            self._add_layer_section("Editor-Sourced Layers", editor_items)
        
        if not ma3_items and not editor_items:
            self._set_layers_table_message("No synced layers.")

        self._update_hook_status_label()
        self._sync_list_updating = False
        if self._sync_list_pending:
            self._sync_list_pending = False
            QTimer.singleShot(0, self._update_synced_layers_list)
    
    def _entity_to_display_item(self, entity: SyncLayerEntity):
        """Convert SyncLayerEntity to a display item for the table."""
        from dataclasses import dataclass
        
        @dataclass
        class DisplayItem:
            entity_id: str
            item_type: str  # "ma3" or "editor"
            item_id: str
            name: str
            status: str
            sync_enabled: bool
            track_group_no: int
            sequence_no: int
            layer_id: str | None = None
            mapped_id: str | None = None
        
        # Determine item_id: use primary identity from source, fall back to name
        if entity.source == SyncSource.MA3:
            item_id = entity.ma3_coord or entity.name
        else:
            item_id = entity.editor_layer_id or entity.name
        
        return DisplayItem(
            entity_id=entity.id,
            item_type=entity.source.value,  # "ma3" or "editor"
            item_id=item_id,
            name=entity.name,
            status=entity.sync_status.value,
            sync_enabled=True,  # All items in synced list are synced
            track_group_no=entity.settings.track_group_no,
            sequence_no=entity.settings.sequence_no,
            layer_id=entity.editor_layer_id,
            mapped_id=entity.ma3_coord if entity.source == SyncSource.EDITOR else entity.editor_layer_id,
        )

    def _add_layer_section(self, title: str, items: List[Any]) -> None:
        """Add a titled section with layer rows in the table."""
        self._add_layers_table_section_row(title)
        if not items:
            self._set_layers_table_message("No layers found.", within_section=True)
            return

        for item in items:
            self._add_layers_table_item_row(item)

    def _set_layers_table_message(self, message: str, within_section: bool = False) -> None:
        """Render a single message row in the layers table."""
        row = self.layers_table.rowCount()
        self.layers_table.insertRow(row)
        item = QTableWidgetItem(message)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        item.setForeground(QBrush(Colors.TEXT_SECONDARY))
        if within_section:
            item.setText(f"  {message}")
        self.layers_table.setItem(row, 0, item)
        self.layers_table.setSpan(row, 0, 1, self.layers_table.columnCount())

    def _add_layers_table_section_row(self, title: str) -> None:
        """Add a section header row to the layers table."""
        row = self.layers_table.rowCount()
        self.layers_table.insertRow(row)
        header_item = QTableWidgetItem(title)
        header_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        header_item.setBackground(QBrush(Colors.BG_DARK))
        header_item.setForeground(QBrush(Colors.TEXT_PRIMARY))
        header_font = header_item.font()
        header_font.setBold(True)
        header_item.setFont(header_font)
        self.layers_table.setItem(row, 0, header_item)
        self.layers_table.setSpan(row, 0, 1, self.layers_table.columnCount())

    def _add_layers_table_item_row(self, item) -> None:
        """Add a worksheet-style row for a layer item."""
        row = self.layers_table.rowCount()
        self.layers_table.insertRow(row)

        sync_check = QCheckBox()
        sync_check.setToolTip("Sync/unsync this layer")
        sync_check.blockSignals(True)
        sync_check.setChecked(bool(getattr(item, "sync_enabled", False)))
        sync_check.blockSignals(False)
        sync_check.stateChanged.connect(lambda state, row=item: self._on_layer_sync_toggled(row, state))
        self.layers_table.setCellWidget(row, 0, self._wrap_table_cell_widget(sync_check))

        type_label = QLabel("MA3" if item.item_type == "ma3" else "Editor")
        type_color = Colors.ACCENT_GREEN.name() if item.item_type == "ma3" else Colors.ACCENT_BLUE.name()
        type_label.setStyleSheet(f"color: {type_color}; font-weight: bold;")
        type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layers_table.setCellWidget(row, 1, self._wrap_table_cell_widget(type_label))

        name_item = QTableWidgetItem(getattr(item, "name", item.item_id))
        name_item.setForeground(QBrush(Colors.TEXT_PRIMARY))
        name_item.setToolTip(getattr(item, "name", item.item_id))
        self.layers_table.setItem(row, 2, name_item)

        status_text = self._format_sync_status(getattr(item, "status", "unmapped"))
        status_item = QTableWidgetItem(status_text)
        raw_status = getattr(item, "status", "unmapped").lower()
        if raw_status == "diverged":
            status_item.setForeground(QBrush(Colors.STATUS_WARNING))
        elif raw_status == "synced":
            status_item.setForeground(QBrush(Colors.STATUS_SUCCESS))
        elif raw_status == "awaiting_connection":
            status_item.setForeground(QBrush(Colors.ACCENT_ORANGE))
        elif raw_status == "pending":
            status_item.setForeground(QBrush(Colors.ACCENT_ORANGE))
        elif raw_status == "error":
            status_item.setForeground(QBrush(Colors.ACCENT_RED))
        else:
            status_item.setForeground(QBrush(Colors.TEXT_SECONDARY))
        self.layers_table.setItem(row, 3, status_item)

        # Track Group spinbox (column 4)
        tg_spin = QSpinBox()
        tg_spin.setMinimum(1)
        tg_spin.setMaximum(99999)
        tg_value = int(getattr(item, "track_group_no", 1) or 1)
        if item.item_type == "editor":
            key = getattr(item, "layer_id", None) or item.item_id
            tg_value = int(self._pending_track_group_by_editor.get(key, tg_value) or 1)
        tg_spin.setValue(tg_value)
        tg_spin.setToolTip("MA3 Track Group number for this layer (1-99999)")
        tg_spin.valueChanged.connect(lambda value, row=item: self._on_layer_track_group_changed(row, value))
        self.layers_table.setCellWidget(row, 4, self._wrap_table_cell_widget(tg_spin))

        # Sequence spinbox (column 5)
        seq_spin = QSpinBox()
        seq_spin.setMinimum(1)
        seq_spin.setMaximum(99999)
        seq_value = int(getattr(item, "sequence_no", 1) or 1)
        if item.item_type == "editor":
            key = getattr(item, "layer_id", None) or item.item_id
            seq_value = int(self._pending_sequence_by_editor.get(key, seq_value) or 1)
        seq_spin.setValue(seq_value)
        seq_spin.setToolTip("MA3 Sequence number for this layer (1-99999)")
        seq_spin.valueChanged.connect(lambda value, row=item: self._on_layer_sequence_changed(row, value))
        self.layers_table.setCellWidget(row, 5, self._wrap_table_cell_widget(seq_spin))

        entity = None
        allow_apply = False
        if hasattr(self, "_sync_system_manager") and self._sync_system_manager:
            entity = self._sync_system_manager.get_synced_layer(getattr(item, "entity_id", None))
        if entity and entity.ma3_coord and entity.editor_layer_id and entity.sync_status in (SyncStatus.SYNCED, SyncStatus.DIVERGED):
            allow_apply = self._sync_system_manager.is_track_hooked(entity.ma3_coord)

        resync_btn = QPushButton("Resync")
        resync_btn.setToolTip("Re-sync this layer (re-hook, diff calculation, and update)")
        resync_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
            }}
        """)
        resync_btn.clicked.connect(lambda _, row=item: self._on_layer_resync_requested(row))
        self.layers_table.setCellWidget(row, 6, self._wrap_table_cell_widget(resync_btn))

        if allow_apply:
            apply_ma3_btn = QPushButton("Keep EZ")
            apply_ma3_btn.setToolTip("Keep Editor events and push them to the MA3 track")
            apply_ma3_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_BLUE.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 4px 10px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_BLUE.lighter(110).name()};
                }}
            """)
            apply_ma3_btn.clicked.connect(lambda _, row=item: self._on_apply_to_ma3_requested(row))
            self.layers_table.setCellWidget(row, 7, self._wrap_table_cell_widget(apply_ma3_btn))
        else:
            self.layers_table.setCellWidget(row, 7, self._wrap_table_cell_widget(QLabel("-")))

        if allow_apply:
            apply_ez_btn = QPushButton("Keep MA3")
            apply_ez_btn.setToolTip("Keep MA3 events and apply them to the Editor layer")
            apply_ez_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {Colors.ACCENT_GREEN.name()};
                    color: {Colors.TEXT_PRIMARY.name()};
                    border: none;
                    border-radius: {border_radius(4)};
                    padding: 4px 10px;
                }}
                QPushButton:hover {{
                    background-color: {Colors.ACCENT_GREEN.lighter(110).name()};
                }}
            """)
            apply_ez_btn.clicked.connect(lambda _, row=item: self._on_apply_to_ez_requested(row))
            self.layers_table.setCellWidget(row, 8, self._wrap_table_cell_widget(apply_ez_btn))
        else:
            self.layers_table.setCellWidget(row, 8, self._wrap_table_cell_widget(QLabel("-")))

        delete_btn = QPushButton("Delete")
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {Colors.ACCENT_RED.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 4px 10px;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_RED.lighter(110).name()};
            }}
        """)
        delete_btn.clicked.connect(lambda _, row=item: self._on_layer_delete_requested(row))
        self.layers_table.setCellWidget(row, 9, self._wrap_table_cell_widget(delete_btn))
        
        # Disable MA3-dependent buttons if MA3 is not connected
        ma3_connected = hasattr(self, '_sync_system_manager') and self._sync_system_manager and self._sync_system_manager.connection_state == "connected"
        if not ma3_connected:
            resync_btn.setEnabled(False)
            if allow_apply:
                apply_ma3_btn.setEnabled(False)
                apply_ez_btn.setEnabled(False)

    def _wrap_table_cell_widget(self, widget: QWidget) -> QWidget:
        """Center a widget within a table cell."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(widget)
        layout.addStretch()
        return container

    def _format_sync_status(self, status: str) -> str:
        """Format status text for display."""
        if not status:
            return "unsynced"
        # Map specific statuses to clearer display text
        status_lower = str(status).strip().lower()
        display_map = {
            "awaiting_connection": "awaiting MA3",
            "awaiting connection": "awaiting MA3",
        }
        if status_lower in display_map:
            return display_map[status_lower]
        normalized = status_lower.replace("_", " ")
        return normalized

    def _on_layer_sync_toggled(self, item, state: int) -> None:
        """Handle sync checkbox toggles in the Synced Layers tab."""
        enabled = state == Qt.CheckState.Checked.value
        
        # In the Synced tab, items have entity_id from SyncSystemManager
        entity_id = getattr(item, "entity_id", None)
        
        if entity_id and hasattr(self, "_sync_system_manager") and self._sync_system_manager:
            # Use new SyncSystemManager
            if not enabled:
                # Unchecking = unsync with asymmetric behavior
                success = self._sync_system_manager.unsync_layer(entity_id)
                if success:
                    self._log(f"Unsynced layer: {item.name}")
                else:
                    self._log(f"Failed to unsync layer: {item.name}")
            # Note: In the Synced tab, checking doesn't make sense (already synced)
            # Re-checking after uncheck would need to go through Available tab
        else:
            # Fallback to legacy behavior
            if enabled:
                if item.item_type == "editor":
                    self._sync_editor_layer(item)
                else:
                    self._sync_ma3_track(item)
            else:
                self._unsync_layer(item)

    def _on_layer_track_group_changed(self, item, track_group_no: int) -> None:
        """Handle per-layer track group changes."""
        tg_value = int(track_group_no or 1)
        
        # Use new SyncSystemManager if available
        entity_id = getattr(item, "entity_id", None)
        if entity_id and hasattr(self, "_sync_system_manager") and self._sync_system_manager:
            self._sync_system_manager.set_track_group(entity_id, tg_value)
            return
        
        # Legacy fallback - store pending value
        key = getattr(item, "layer_id", None) or item.item_id
        self._pending_track_group_by_editor[key] = tg_value

    def _on_layer_sequence_changed(self, item, sequence_no: int) -> None:
        """Handle per-layer sequence changes."""
        seq_value = int(sequence_no or 1)
        
        # Use new SyncSystemManager if available
        entity_id = getattr(item, "entity_id", None)
        if entity_id and hasattr(self, "_sync_system_manager") and self._sync_system_manager:
            self._sync_system_manager.set_sequence(entity_id, seq_value)
            return
        
        # Legacy fallback
        if item.item_type == "ma3":
            self._on_track_sequence_changed(item.item_id, seq_value)
            return

        key = getattr(item, "layer_id", None) or item.item_id
        self._pending_sequence_by_editor[key] = seq_value
        if getattr(item, "mapped_id", None):
            self._on_track_sequence_changed(item.mapped_id, seq_value)

    def _on_layer_delete_requested(self, item) -> None:
        """Handle delete button clicks for layer rows."""
        from PyQt6.QtWidgets import QMessageBox
        
        # Use new SyncSystemManager if available
        entity_id = getattr(item, "entity_id", None)
        if entity_id and hasattr(self, "_sync_system_manager") and self._sync_system_manager:
            # Confirm deletion
            entity = self._sync_system_manager.get_synced_layer(entity_id)
            if not entity:
                return
            
            msg = f"Are you sure you want to delete the synced layer '{entity.name}'?"
            if entity.source == SyncSource.MA3:
                msg += "\n\nThis will remove the synced copy from the Editor."
            else:
                msg += "\n\nThe MA3 copy will remain (can be resynced later)."
            
            reply = QMessageBox.question(
                self,
                "Delete Synced Layer?",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                success = self._sync_system_manager.unsync_layer(entity_id)
                if success:
                    self._log(f"Deleted synced layer: {entity.name}")
                else:
                    self._log(f"Failed to delete synced layer: {entity.name}")
            return
        
        # Legacy fallback
        if item.item_type == "ma3":
            self._delete_ma3_track(item)
        else:
            self._delete_editor_layer(item)

    def _on_layer_resync_requested(self, item) -> None:
        """Handle resync button clicks for layer rows."""
        entity_id = getattr(item, "entity_id", None)
        if not entity_id:
            self._log("Cannot resync: no entity ID")
            return
        
        if not hasattr(self, "_sync_system_manager") or not self._sync_system_manager:
            self._log("Cannot resync: sync system not ready")
            return
        
        entity = self._sync_system_manager.get_synced_layer(entity_id)
        if not entity:
            self._log(f"Cannot resync: entity not found ({entity_id})")
            return
        
        self._log(f"Resyncing layer: {entity.name}")
        
        success = self._sync_system_manager.resync_layer(entity_id)
        if success:
            self._log(f"Resync complete: {entity.name}")
        else:
            self._log(f"Resync failed: {entity.name}")
    
    def _on_apply_to_ma3_requested(self, item) -> None:
        """Handle Keep EZ button clicks for layer rows.
        
        Keeps Editor events and pushes them to MA3.
        For DIVERGED entities, this resolves divergence (ez_wins).
        For SYNCED entities, this does a direct apply.
        """
        entity_id = getattr(item, "entity_id", None)
        if not entity_id:
            self._log("Cannot apply to MA3: no entity ID")
            return
        if not hasattr(self, "_sync_system_manager") or not self._sync_system_manager:
            self._log("Cannot apply to MA3: sync system not ready")
            return
        
        entity = self._sync_system_manager.get_synced_layer(entity_id)
        if not entity:
            self._log(f"Cannot apply to MA3: entity not found ({entity_id})")
            return
        
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        if entity.sync_status == SyncStatus.DIVERGED:
            self._log(f"Resolving divergence (EZ wins) for: {entity.name}")
            success = self._sync_system_manager.resolve_divergence(entity_id, "ez_wins")
        else:
            self._log(f"Applying to MA3: {entity.name}")
            success = self._sync_system_manager.apply_to_ma3(entity_id)
        
        if success:
            self._log(f"Apply to MA3 complete: {entity.name}")
            self._update_synced_layers_list()
        else:
            self._log(f"Apply to MA3 failed: {entity.name}")
    
    def _on_apply_to_ez_requested(self, item) -> None:
        """Handle Keep MA3 button clicks for layer rows.
        
        Keeps MA3 events and applies them to the Editor layer.
        For DIVERGED entities, this resolves divergence (ma3_wins).
        For SYNCED entities, this does a direct apply.
        """
        entity_id = getattr(item, "entity_id", None)
        if not entity_id:
            self._log("Cannot apply to EZ: no entity ID")
            return
        if not hasattr(self, "_sync_system_manager") or not self._sync_system_manager:
            self._log("Cannot apply to EZ: sync system not ready")
            return
        
        entity = self._sync_system_manager.get_synced_layer(entity_id)
        if not entity:
            self._log(f"Cannot apply to EZ: entity not found ({entity_id})")
            return
        
        
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        if entity.sync_status == SyncStatus.DIVERGED:
            self._log(f"Resolving divergence (MA3 wins) for: {entity.name}")
            success = self._sync_system_manager.resolve_divergence(entity_id, "ma3_wins")
        else:
            self._log(f"Applying to EZ: {entity.name}")
            success = self._sync_system_manager.apply_to_ez(entity_id)
        
        if success:
            self._log(f"Apply to EZ complete: {entity.name}")
            self._update_synced_layers_list()
        else:
            self._log(f"Apply to EZ failed: {entity.name}")

    def _get_applicable_synced_entities(self):
        """
        Get all synced layer entities that are eligible for Keep EZ / Keep MA3 actions.
        
        An entity is eligible when:
        - It has both an MA3 coord and an Editor layer ID
        - Its sync status is SYNCED or DIVERGED
        - Its MA3 track is currently hooked
        
        Returns:
            List of SyncLayerEntity objects that can be acted upon.
        """
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        
        if not hasattr(self, '_sync_system_manager') or not self._sync_system_manager:
            return []
        
        current_tc = (
            self._settings_manager.target_timecode
            if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded()
            else 1
        )
        
        entities = []
        for entity in self._sync_system_manager.get_synced_layers():
            if entity.ma3_timecode_no != current_tc:
                continue
            if not entity.ma3_coord or not entity.editor_layer_id:
                continue
            if entity.sync_status not in (SyncStatus.SYNCED, SyncStatus.DIVERGED):
                continue
            if not self._sync_system_manager.is_track_hooked(entity.ma3_coord):
                continue
            entities.append(entity)
        
        return entities

    def _on_batch_keep_ez(self) -> None:
        """
        Batch Keep EZ: keep Editor events for all applicable synced layers
        and push them to their corresponding MA3 tracks.
        
        For DIVERGED entities this resolves divergence with ez_wins.
        For SYNCED entities this does a direct apply.
        """
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        
        entities = self._get_applicable_synced_entities()
        if not entities:
            self._log("Batch Keep EZ: no applicable synced layers found")
            return
        
        self._log(f"Batch Keep EZ: applying to {len(entities)} synced layer(s)...")
        
        success_count = 0
        fail_count = 0
        for entity in entities:
            try:
                if entity.sync_status == SyncStatus.DIVERGED:
                    self._log(f"  Resolving divergence (EZ wins): {entity.name}")
                    ok = self._sync_system_manager.resolve_divergence(entity.id, "ez_wins")
                else:
                    self._log(f"  Applying to MA3: {entity.name}")
                    ok = self._sync_system_manager.apply_to_ma3(entity.id)
                
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    self._log(f"  Failed: {entity.name}")
            except Exception as e:
                fail_count += 1
                self._log(f"  Error on {entity.name}: {e}")
        
        self._log(
            f"Batch Keep EZ complete: {success_count} succeeded, {fail_count} failed"
        )
        self._update_synced_layers_list()

    def _on_batch_keep_ma3(self) -> None:
        """
        Batch Keep MA3: keep MA3 events for all applicable synced layers
        and apply them to their corresponding Editor layers.
        
        For DIVERGED entities this resolves divergence with ma3_wins.
        For SYNCED entities this does a direct apply.
        """
        from src.features.show_manager.domain.sync_layer_entity import SyncStatus
        
        entities = self._get_applicable_synced_entities()
        if not entities:
            self._log("Batch Keep MA3: no applicable synced layers found")
            return
        
        self._log(f"Batch Keep MA3: applying to {len(entities)} synced layer(s)...")
        
        success_count = 0
        fail_count = 0
        for entity in entities:
            try:
                if entity.sync_status == SyncStatus.DIVERGED:
                    self._log(f"  Resolving divergence (MA3 wins): {entity.name}")
                    ok = self._sync_system_manager.resolve_divergence(entity.id, "ma3_wins")
                else:
                    self._log(f"  Applying to EZ: {entity.name}")
                    ok = self._sync_system_manager.apply_to_ez(entity.id)
                
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                    self._log(f"  Failed: {entity.name}")
            except Exception as e:
                fail_count += 1
                self._log(f"  Error on {entity.name}: {e}")
        
        self._log(
            f"Batch Keep MA3 complete: {success_count} succeeded, {fail_count} failed"
        )
        self._update_synced_layers_list()

    def _sync_editor_layer(self, item) -> None:
        """Sync an Editor layer to MA3 (auto-create if missing)."""
        from src.features.show_manager.application.commands import SyncLayerPairCommand

        layer_id = getattr(item, "layer_id", None) or item.item_id
        if not item.block_id:
            item.block_id = self._find_connected_editor()
        if not item.block_id:
            self._log("No Editor block connected; cannot sync layer.")
            return

        cmd = SyncLayerPairCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            source="editor",
            editor_layer_id=layer_id,
            editor_block_id=item.block_id,
            ma3_events=getattr(self, "_ma3_events", None),
        )
        self.facade.command_bus.execute(cmd)
        self._refresh_controller_from_settings()

    def _sync_ma3_track(self, item) -> None:
        """Sync a MA3 track to Editor (auto-create Editor layer)."""
        from src.features.show_manager.application.commands import SyncLayerPairCommand

        if not item.timecode_no or not item.track_group or not item.track:
            self._log("MA3 track details missing; refresh MA3 tracks first.")
            return

        seq_value = int(item.sequence_no or 1)
        cmd = SyncLayerPairCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            source="ma3",
            ma3_coord=item.item_id,
            ma3_timecode_no=item.timecode_no,
            ma3_track_group=item.track_group,
            ma3_track=item.track,
            ma3_track_name=getattr(item, "track_name", None) or item.name,
            ma3_events=getattr(self, "_ma3_events", None),
            sequence_no=seq_value,
        )
        self.facade.command_bus.execute(cmd)
        self._refresh_controller_from_settings()

    def _unsync_layer(self, item) -> None:
        """Unsync a layer without deleting the source layer."""
        from src.application.commands.layer_sync import RemoveSyncedEntityCommand

        if not hasattr(self, "_settings_manager") or not self._settings_manager:
            return

        mapped_id = getattr(item, "mapped_id", None)
        if item.item_type == "editor":
            layer_id = getattr(item, "layer_id", None) or item.item_id
            if mapped_id and self._settings_manager.get_synced_layer("ma3", mapped_id):
                cmd = RemoveSyncedEntityCommand(
                    facade=self.facade,
                    show_manager_block_id=self.block_id,
                    entity_type="ma3",
                    entity_id=mapped_id,
                    delete_editor_layer=False,
                )
                self.facade.command_bus.execute(cmd)
            elif self._settings_manager.get_synced_layer("editor", layer_id):
                cmd = RemoveSyncedEntityCommand(
                    facade=self.facade,
                    show_manager_block_id=self.block_id,
                    entity_type="editor",
                    entity_id=layer_id,
                    delete_editor_layer=False,
                )
                self.facade.command_bus.execute(cmd)
            if mapped_id:
                self._settings_manager.remove_layer_mapping(mapped_id)
        elif item.item_type == "ma3":
            if self._settings_manager.get_synced_layer("ma3", item.item_id):
                cmd = RemoveSyncedEntityCommand(
                    facade=self.facade,
                    show_manager_block_id=self.block_id,
                    entity_type="ma3",
                    entity_id=item.item_id,
                    delete_editor_layer=False,
                )
                self.facade.command_bus.execute(cmd)
            self._settings_manager.remove_layer_mapping(item.item_id)

        self._refresh_controller_from_settings()

    def _delete_editor_layer(self, item) -> None:
        """Delete an Editor layer and remove sync metadata."""
        from PyQt6.QtWidgets import QMessageBox
        from src.application.commands.editor_commands import EditorDeleteLayerCommand

        layer_id = getattr(item, "layer_id", None) or item.item_id
        reply = QMessageBox.question(
            self,
            "Delete Editor Layer?",
            f"Delete Editor layer '{layer_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        editor_block_id = item.block_id or self._find_connected_editor()
        if not editor_block_id:
            self._log("No Editor block connected; cannot delete layer.")
            return

        delete_cmd = EditorDeleteLayerCommand(
            facade=self.facade,
            block_id=editor_block_id,
            layer_name=layer_id,
        )
        self.facade.command_bus.execute(delete_cmd)
        self._unsync_layer(item)

    def _delete_ma3_track(self, item) -> None:
        """Delete an MA3 track (clear contents) and remove sync metadata."""
        from PyQt6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Delete MA3 Track?",
            f"Delete MA3 track '{item.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        parsed = self._parse_ma3_coord(item.item_id) or self._parse_coord_key(item.item_id)
        if parsed:
            tc, tg, track = parsed
            try:
                self._send_lua_command(f"EZ.UnhookTrack({tc}, {tg}, {track})")
                self._send_lua_command(f"EZ.ClearTrack({tc}, {tg}, {track})")
            except Exception as exc:
                self._log(f"Failed to clear MA3 track {item.item_id}: {exc}")

        self._unsync_layer(item)
    
    def _on_add_editor_layer_clicked(self):
        """Handle 'Add Editor Layer' button click - open dialog"""
        # Get available Editor layers from SyncSystemManager
        available_layers = self._sync_system_manager.get_available_editor_layers()
        
        # Open dialog
        from ui.qt_gui.dialogs.add_editor_layer_dialog import AddEditorLayerDialog
        dialog = AddEditorLayerDialog(available_layers=available_layers, parent=self)
        dialog.layer_added.connect(self._on_editor_layer_added)
        dialog.exec()
    
    def _on_add_ma3_track_clicked(self):
        """Handle 'Add MA3 Track' button click - open dialog"""
        # Check if tracks are already loaded
        if hasattr(self, '_ma3_tracks') and self._ma3_tracks:
            # Tracks available - open dialog immediately
            self._open_ma3_track_dialog()
            return
        
        # Tracks not loaded - check if MA3 is ready
        if not self._is_ma3_ready():
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "MA3 Not Ready",
                "MA3 is not connected or listener is not active.\n\n"
                "Please:\n"
                "1. Ensure MA3 IP and port are configured\n"
                "2. Start the MA3 listener\n"
                "3. Verify MA3 is responding"
            )
            return
        
        # MA3 is ready but tracks not loaded - fetch and open dialog when received
        from src.utils.message import Log
        Log.info("ShowManagerPanel: No MA3 tracks loaded, auto-fetching...")
        self._pending_ma3_track_dialog = True  # Set flag to open dialog when structure is received
        try:
            self._auto_fetch_ma3_structure()
        except RuntimeError as e:
            self._pending_ma3_track_dialog = False  # Clear flag on error
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "MA3 Connection Error", str(e))
            self._log(f"ERROR: {e}")
    
    def _open_ma3_track_dialog(self):
        """Open the MA3 track selection dialog (helper method)."""
        if not hasattr(self, '_ma3_tracks') or not self._ma3_tracks:
            from src.utils.message import Log
            Log.warning("ShowManagerPanel: Cannot open dialog - no MA3 tracks loaded")
            return
        
        available_tracks = self._sync_system_manager.get_available_ma3_tracks()
        
        from ui.qt_gui.dialogs.add_ma3_track_dialog import AddMA3TrackDialog
        dialog = AddMA3TrackDialog(available_tracks=available_tracks, parent=self)
        dialog.track_added.connect(self._on_ma3_track_added)
        dialog.exec()

    def _open_ma3_track_dialog_for_editor(self, editor_layer_id: str):
        """Open MA3 track dialog to map an Editor layer to an existing track."""
        if not hasattr(self, '_ma3_tracks') or not self._ma3_tracks:
            from src.utils.message import Log
            Log.warning("ShowManagerPanel: Cannot open dialog - no MA3 tracks loaded")
            return
        available_tracks = self._sync_system_manager.get_available_ma3_tracks()
        from ui.qt_gui.dialogs.add_ma3_track_dialog import AddMA3TrackDialog
        dialog = AddMA3TrackDialog(available_tracks=available_tracks, parent=self)
        dialog.track_added.connect(
            lambda coord, tc, tg, track, name: self._on_editor_layer_track_selected(
                editor_layer_id,
                coord,
                tc,
                tg,
                track,
                name
            )
        )
        dialog.exec()

    def _on_editor_layer_track_selected(
        self,
        editor_layer_id: str,
        coord: str,
        timecode_no: int,
        track_group: int,
        track: int,
        name: str
    ) -> None:
        """Handle MA3 track selection for an Editor layer mapping."""
        track_info = MA3TrackInfo(
            timecode_no=timecode_no,
            track_group=track_group,
            track=track,
            name=name
        )
        self._link_editor_layer_to_ma3_track(
            editor_layer_id,
            track_info,
            create_ma3_entity=False,
            sequence_no=None,
            push_editor_events=False
        )
        self._start_reconcile_for_coord(coord)
    
    def _open_ma3_track_dialog_if_pending(self):
        """Open MA3 track dialog if still pending (called after events are fetched)."""
        if self._pending_editor_track_select:
            pending = self._pending_editor_track_select
            self._pending_editor_track_select = None
            from src.utils.message import Log
            editor_layer_id = pending.get("editor_layer_id")
            Log.info(f"ShowManagerPanel: Opening MA3 track dialog for editor layer {editor_layer_id}")
            self._open_ma3_track_dialog_for_editor(editor_layer_id)
            return
        if self._pending_ma3_track_dialog:
            self._pending_ma3_track_dialog = False
            from src.utils.message import Log
            event_count = len(getattr(self, '_ma3_events', []))
            Log.info(f"ShowManagerPanel: Opening MA3 track dialog ({event_count} events loaded)")
            self._open_ma3_track_dialog()
    
    def _on_editor_layer_added(self, layer_id: str, block_id: str):
        """Handle Editor layer added from dialog - execute command"""
        from src.application.commands.layer_sync import AddSyncedEditorLayerCommand
        from PyQt6.QtWidgets import QMessageBox
        
        # Pass MA3 events to command so it can add them to the layer
        ma3_events = getattr(self, '_ma3_events', []) or []
        
        cmd = AddSyncedEditorLayerCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            editor_layer_id=layer_id,
            editor_block_id=block_id,
            ma3_events=ma3_events  # Pass events so command can add them
        )
        result = self.facade.command_bus.execute(cmd)
        
        # Command now forces immediate save, so reload settings immediately
        # (force_save() is synchronous, so we can reload right away)
        self._refresh_controller_from_settings()

        # Get entity from SyncSystemManager
        entity = self._sync_system_manager.get_synced_layer_by_editor_layer_id(layer_id)
        if not entity:
            return

        reply = QMessageBox.question(
            self,
            "Select MA3 Track",
            (
                f"Editor layer '{entity.name}' added.\n\n"
                "Would you like to sync it to an existing MA3 track, or create a new MA3 track?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        choice = "cancel"
        if reply == QMessageBox.StandardButton.Yes:
            choice = "existing"
        elif reply == QMessageBox.StandardButton.No:
            choice = "new"
        if choice == "existing":
            if hasattr(self, '_ma3_tracks') and self._ma3_tracks:
                self._open_ma3_track_dialog_for_editor(editor_entity.layer_id)
                return
            if not self._is_ma3_ready():
                QMessageBox.information(
                    self,
                    "MA3 Not Ready",
                    "MA3 is not connected or listener is not active.\n\n"
                    "Please:\n"
                    "1. Ensure MA3 IP and port are configured\n"
                    "2. Start the MA3 listener\n"
                    "3. Verify MA3 is responding"
                )
                return
            self._pending_editor_track_select = {"editor_layer_id": editor_entity.layer_id}
            try:
                self._auto_fetch_ma3_structure()
            except RuntimeError as e:
                self._pending_editor_track_select = None
                QMessageBox.warning(self, "MA3 Connection Error", str(e))
                self._log(f"ERROR: {e}")
            return
        if choice == "new":
            self._ensure_ma3_track_for_editor_layer(editor_entity)
            return

        # UI will update via controller's entities_changed signal
    
    def _on_ma3_track_added(self, coord: str, timecode_no: int, track_group: int, track: int, name: str):
        """
        Handle MA3 track added from dialog - execute command.
        
        Simple workflow:
        1. Get timecode from settings
        2. Get events (should already be loaded from dialog flow)
        3. Execute AddSyncedMA3TrackCommand (which hooks the track for real-time sync)
        
        Args:
            coord: Track coordinate (e.g., "tc101_tg1_tr1")
            timecode_no: Timecode number from track data
            track_group: Track group number
            track: Track number (user-visible, 1-based)
            name: Track name
        """
        from src.application.commands.layer_sync import AddSyncedMA3TrackCommand
        from src.utils.message import Log
        
        Log.info(f"")
        Log.info(f"========== ADD MA3 TRACK FROM DIALOG ==========")
        Log.info(f"ShowManagerPanel: _on_ma3_track_added() called")
        Log.info(f"  coord={coord}, tc={timecode_no}, tg={track_group}, track={track}, name='{name}'")
        
        # Step 1: Use target_timecode from settings manager (not the timecode from track data)
        if hasattr(self, '_settings_manager') and self._settings_manager.is_loaded():
            target_tc = self._settings_manager.target_timecode
            if target_tc != timecode_no:
                Log.info(f"  Using target_timecode {target_tc} from settings (dialog had {timecode_no})")
                timecode_no = target_tc
        else:
            Log.warning("  Settings manager not loaded, using timecode from track data")
        
        # Step 2: Get events (should already be loaded from dialog flow)
        # Events are fetched when the dialog opens, so they should be available
        ma3_events = getattr(self, '_ma3_events', []) or []
        Log.info(f"  Events available: {len(ma3_events)}")
        
        if ma3_events:
            # Count events for this specific track
            matching_events = [e for e in ma3_events 
                             if (hasattr(e, 'timecode_no') and e.timecode_no == timecode_no and
                                 hasattr(e, 'track_group') and e.track_group == track_group and
                                 hasattr(e, 'track') and e.track == track)]
            Log.info(f"  Events matching this track: {len(matching_events)}")
        else:
            # Events not loaded - this shouldn't happen if dialog flow worked correctly
            # Don't trigger another fetch cascade - just proceed without events
            Log.warning("  No events loaded - proceeding without initial events")
            Log.warning("  (Hook will sync events on next MA3 track change)")
        
        # Step 3: Execute AddSyncedMA3TrackCommand
        # This command:
        # - Creates an Editor layer for the track
        # - Adds matching events to the layer
        # - Hooks the track's CmdSubTrack for real-time sync
        Log.info(f"  Executing AddSyncedMA3TrackCommand...")

        sequence_no = self._prompt_sequence_no(default_value=1)
        if sequence_no is None:
            Log.info("  Sequence prompt cancelled; aborting sync layer creation")
            return
        
        cmd = AddSyncedMA3TrackCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            coord=coord,
            timecode_no=timecode_no,
            track_group=track_group,
            track=track,
            name=name,
            ma3_events=ma3_events,
            sequence_no=sequence_no
        )
        result = self.facade.command_bus.execute(cmd)
        
        # Command forces immediate save, so reload settings immediately
        self._refresh_controller_from_settings()
        
        Log.info(f"================================================")
        Log.info(f"")
        
        # UI will update via controller's entities_changed signal
    
    def _on_remove_synced_entity(self, entity_type: str, entity_id: str):
        """Handle remove button click - execute command"""
        from src.application.commands.layer_sync import RemoveSyncedEntityCommand
        from PyQt6.QtWidgets import QMessageBox

        delete_editor_layer = True
        if entity_type == "ma3":
            mapped_editor_layer_id = None
            # Get mapping from SyncSystemManager
            entity = self._sync_system_manager.get_synced_layer_by_ma3_coord(entity_id)
            if entity:
                mapped_editor_layer_id = entity.editor_layer_id

            if mapped_editor_layer_id:
                reply = QMessageBox.question(
                    self,
                    "Delete Editor Layer?",
                    (
                        f"Remove MA3 track '{entity_id}' from synced layers.\n\n"
                        f"Also delete the linked Editor layer '{mapped_editor_layer_id}'?"
                    ),
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                delete_editor_layer = reply == QMessageBox.StandardButton.Yes

            parsed = self._parse_ma3_coord(entity_id) or self._parse_coord_key(entity_id)
            if parsed:
                tc, tg, track = parsed
                self._send_lua_command(f"EZ.UnhookTrack({tc}, {tg}, {track})")
        
        cmd = RemoveSyncedEntityCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            entity_type=entity_type,
            entity_id=entity_id,
            delete_editor_layer=delete_editor_layer,
        )
        result = self.facade.command_bus.execute(cmd)
        
        # Command now forces immediate save, so reload settings immediately
        # (force_save() is synchronous, so we can reload right away)
        self._refresh_controller_from_settings()
        
        # UI will update via controller's entities_changed signal
    
    def _on_sync_entity(self, entity_type: str, entity_id: str, entity):
        """
        Handle Sync button click - sync a single entity.
        
        LEGACY: This method used EditorLayerEntity/MA3TrackEntity which have been removed.
        Use the new SyncSystemManager via the Synced Layers tab instead.
        """
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Feature Moved",
            "This sync feature has been moved to the Synced Layers tab.\n\n"
            "Please use the new sync system for layer synchronization."
        )
        return
        # Legacy code below - kept for reference during migration
        from src.features.show_manager.application.commands import SyncLayerCommand
        
        # Find connected Editor block
        editor_block_id = self._find_connected_editor()
        if not editor_block_id:
            QMessageBox.warning(
                self,
                "No Editor Connected",
                "Please connect an Editor block to ShowManager to enable sync."
            )
            return
        
        if isinstance(entity, EditorLayerEntity):
            # Sync Editor layer to MA3 track
            mapped_ma3 = entity.mapped_ma3_track_id
            if not mapped_ma3 and hasattr(self, "_settings_manager") and self._settings_manager.is_loaded():
                settings_entity = self._settings_manager.get_synced_layer("editor", entity.layer_id)
                if settings_entity and isinstance(settings_entity, dict):
                    mapped_ma3 = settings_entity.get("mapped_ma3_track_id")
                    if mapped_ma3:
                        entity.mapped_ma3_track_id = mapped_ma3
            if not mapped_ma3:
                QMessageBox.information(
                    self,
                    "Not Mapped",
                    (
                        f"Editor layer '{entity.name}' is not mapped to an MA3 track.\n\n"
                        "Use 'Add MA3 Track' or map layers first."
                    ),
                )
                return
            
            cmd = SyncLayerCommand(
                facade=self.facade,
                show_manager_block_id=self.block_id,
                editor_block_id=editor_block_id,
                editor_layer_name=entity.layer_id,
                ma3_track_coord=entity.mapped_ma3_track_id,
                direction="editor_to_ma3",
                clear_target=True
            )
            
        elif isinstance(entity, MA3TrackEntity):
            # Sync MA3 track to Editor layer
            if not entity.mapped_editor_layer_id:
                QMessageBox.information(
                    self,
                    "Not Mapped",
                    f"MA3 track '{entity.display_name}' is not mapped to an Editor layer.\n\n"
                    "Use 'Auto Map' or manually map layers first."
                )
                return
            
            cmd = SyncLayerCommand(
                facade=self.facade,
                show_manager_block_id=self.block_id,
                editor_block_id=editor_block_id,
                editor_layer_name=entity.mapped_editor_layer_id,
                ma3_track_coord=entity.coord,
                direction="ma3_to_editor",
                clear_target=True
            )
        else:
            return
        
        # Execute sync command
        try:
            self.facade.command_bus.execute(cmd)
            Log.info(f"ShowManagerPanel: Synced {entity_type} entity '{entity_id}'")
        except Exception as e:
            Log.error(f"ShowManagerPanel: Failed to sync entity: {e}")
            QMessageBox.critical(
                self,
                "Sync Failed",
                f"Failed to sync: {str(e)}"
            )

    def _ensure_ma3_track_for_editor_layer(
        self,
        entity,  # Was: EditorLayerEntity (legacy type removed)
        sequence_no: Optional[int] = None
    ) -> None:
        """
        LEGACY: Ensure MA3 track exists for an Editor layer.
        This method used EditorLayerEntity which has been removed.
        Use SyncSystemManager instead.
        """
        Log.warning("_ensure_ma3_track_for_editor_layer is a legacy method - use SyncSystemManager")
        return
        # Legacy code below
        if not hasattr(self, "_settings_manager") or not self._settings_manager.is_loaded():
            self._log("Settings manager not loaded; cannot create MA3 track.")
            return
        if not self._is_ma3_ready():
            self._log("MA3 not ready; start listener and configure target first.")
            return

        if sequence_no is None:
            sequence_no = self._prompt_sequence_no(default_value=1)
            if sequence_no is None:
                return

        # Use entity name directly (legacy normalization removed with sync_registry)
        track_name = entity.name or entity.layer_id
        timecode_no = self._settings_manager.target_timecode
        track_group = 1
        if hasattr(self, "_ma3_track_groups") and self._ma3_track_groups.get(timecode_no):
            track_group = self._ma3_track_groups[timecode_no][0].get("no", 1) or 1

        if not hasattr(self, "_ma3_tracks") or not self._ma3_tracks:
            self._pending_track_create = {
                "editor_layer_id": entity.layer_id,
                "track_name": track_name,
                "timecode_no": timecode_no,
                "track_group": track_group,
                "sequence_no": sequence_no,
                "create_attempted": False,
            }
            self._request_ma3_structure_refresh("pending_track")
            return

        existing = self._find_ma3_track_by_name(track_name, timecode_no)
        if existing:
            self._link_editor_layer_to_ma3_track(
                entity.layer_id,
                existing,
                create_ma3_entity=False,
                sequence_no=sequence_no
            )
            return

        ma3_comm = getattr(self.facade, "ma3_communication_service", None)
        if not ma3_comm:
            self._log("MA3 communication service not available.")
            return

        self._pending_track_create = {
            "editor_layer_id": entity.layer_id,
            "track_name": track_name,
            "timecode_no": timecode_no,
            "track_group": track_group,
            "sequence_no": sequence_no,
            "create_attempted": True,
        }
        created = ma3_comm.create_track(timecode_no, track_group, track_name)
        if not created:
            self._log(f"Failed to create MA3 track '{track_name}'")
            self._pending_track_create = None
            return

        self._request_ma3_structure_refresh("pending_track")

    def _find_ma3_track_by_name(self, name: str, timecode_no: int) -> Optional[MA3TrackInfo]:
        """Find MA3 track by name for a specific timecode."""
        from src.features.show_manager.application.ma3_track_resolver import find_track_by_name

        if not hasattr(self, "_ma3_tracks"):
            return None
        candidates = [t for t in self._ma3_tracks if t.timecode_no == timecode_no]
        return find_track_by_name(candidates, name)

    def _link_editor_layer_to_ma3_track(
        self,
        editor_layer_id: str,
        track_info,  # MA3TrackInfo
        create_ma3_entity: bool = True,
        sequence_no: Optional[int] = None,
        push_editor_events: bool = True
    ) -> None:
        """
        LEGACY: Persist MA3 track entity and map it to the Editor layer.
        This method used MA3TrackEntity which has been removed.
        Use SyncSystemManager instead.
        """
        Log.warning("_link_editor_layer_to_ma3_track is a legacy method - use SyncSystemManager")
        return
        # Legacy code below
        from src.application.settings.show_manager_settings import ShowManagerSettingsManager
        from src.features.show_manager.domain.layer_sync_types import LayerSyncStatus, SyncType, SyncDirection
        from src.features.show_manager.application.commands import MapLayersCommand

        settings_manager = ShowManagerSettingsManager(self.facade, self.block_id)
        sync_settings = self._build_unified_sync_settings(
            SyncDirection.EZ_TO_MA3,
            auto_apply=True,
            apply_updates_enabled=settings_manager.apply_updates_enabled
        )
        if not create_ma3_entity:
            existing = settings_manager.get_synced_layer("ma3", track_info.coord)
            if existing:
                current_settings = existing.get("settings", {}) if isinstance(existing, dict) else {}
                settings_manager.update_synced_layer(
                    "ma3",
                    track_info.coord,
                    {
                        "settings": {
                            **current_settings,
                            **sync_settings,
                            "sequence_no": int(sequence_no or 1)
                        }
                    }
                )
            else:
                from src.features.show_manager.domain.ma3_track_entity import MA3TrackSettings
                ma3_entity = MA3TrackEntity(
                    coord=track_info.coord,
                    timecode_no=track_info.timecode_no,
                    track_group=track_info.track_group,
                    track=track_info.track,
                    name=track_info.name,
                    sync_type=SyncType.SHOWMANAGER_LAYER,
                    settings=MA3TrackSettings(
                        sync_direction=SyncDirection.EZ_TO_MA3,
                        conflict_resolution=self._resolve_conflict_resolution(),
                        auto_apply=True,
                        apply_updates_enabled=settings_manager.apply_updates_enabled,
                        sequence_no=int(sequence_no or 1)
                    ),
                )
                settings_manager.add_synced_layer(ma3_entity.to_dict())
                settings_manager.force_save()
            editor_entity = settings_manager.get_synced_layer("editor", editor_layer_id)
            editor_settings = editor_entity.get("settings", {}) if isinstance(editor_entity, dict) else {}
            editor_updated = settings_manager.update_synced_layer(
                "editor",
                editor_layer_id,
                {
                    "mapped_ma3_track_id": track_info.coord,
                    "sync_status": LayerSyncStatus.SYNCED.value,
                    "sync_type": SyncType.SHOWMANAGER_LAYER.value,
                    "settings": {**editor_settings, **sync_settings},
                }
            )
            if editor_updated:
                settings_manager.force_save()
                self._refresh_controller_from_settings()
            resolved_sequence_no = self._resolve_sequence_no(track_info.coord, sequence_no)
            self._assign_track_sequence(track_info, resolved_sequence_no)
            if push_editor_events:
                self._push_editor_events_to_ma3(editor_layer_id, track_info)
            return
        if settings_manager.get_synced_layer("ma3", track_info.coord):
            cmd = MapLayersCommand(
                facade=self.facade,
                show_manager_block_id=self.block_id,
                editor_layer_id=editor_layer_id,
                ma3_track_coord=track_info.coord
            )
            self.facade.command_bus.execute(cmd)
            self._refresh_controller_from_settings()
            existing = settings_manager.get_synced_layer("ma3", track_info.coord)
            current_settings = existing.get("settings", {}) if isinstance(existing, dict) else {}
            settings_manager.update_synced_layer(
                "ma3",
                track_info.coord,
                {
                    "settings": {
                        **current_settings,
                        **sync_settings,
                        "sequence_no": int(sequence_no or 1)
                    }
                }
            )
            editor_entity = settings_manager.get_synced_layer("editor", editor_layer_id)
            editor_settings = editor_entity.get("settings", {}) if isinstance(editor_entity, dict) else {}
            settings_manager.update_synced_layer(
                "editor",
                editor_layer_id,
                {"settings": {**editor_settings, **sync_settings}}
            )
            settings_manager.force_save()
            resolved_sequence_no = self._resolve_sequence_no(track_info.coord, sequence_no)
            self._assign_track_sequence(track_info, resolved_sequence_no)
            if push_editor_events:
                self._push_editor_events_to_ma3(editor_layer_id, track_info)
            return

        from src.features.show_manager.domain.ma3_track_entity import MA3TrackSettings
        ma3_entity = MA3TrackEntity(
            coord=track_info.coord,
            timecode_no=track_info.timecode_no,
            track_group=track_info.track_group,
            track=track_info.track,
            name=track_info.name,
            sync_type=SyncType.SHOWMANAGER_LAYER,
            settings=MA3TrackSettings(
                sync_direction=SyncDirection.EZ_TO_MA3,
                conflict_resolution=self._resolve_conflict_resolution(),
                auto_apply=True,
                apply_updates_enabled=settings_manager.apply_updates_enabled,
                sequence_no=int(sequence_no or 1)
            ),
        )
        settings_manager.add_synced_layer(ma3_entity.to_dict())
        settings_manager.force_save()

        cmd = MapLayersCommand(
            facade=self.facade,
            show_manager_block_id=self.block_id,
            editor_layer_id=editor_layer_id,
            ma3_track_coord=track_info.coord
        )
        self.facade.command_bus.execute(cmd)
        self._refresh_controller_from_settings()
        editor_entity = settings_manager.get_synced_layer("editor", editor_layer_id)
        editor_settings = editor_entity.get("settings", {}) if isinstance(editor_entity, dict) else {}
        settings_manager.update_synced_layer(
            "editor",
            editor_layer_id,
            {"settings": {**editor_settings, **sync_settings}}
        )
        settings_manager.force_save()
        resolved_sequence_no = self._resolve_sequence_no(track_info.coord, sequence_no)
        self._assign_track_sequence(track_info, resolved_sequence_no)
        if push_editor_events:
            self._push_editor_events_to_ma3(editor_layer_id, track_info)

    def _resolve_pending_track_create(self) -> None:
        """Resolve pending track creation by linking when track appears."""
        if not self._pending_track_create:
            return
        info = self._pending_track_create
        existing = self._find_ma3_track_by_name(info["track_name"], info["timecode_no"])
        if not existing:
            if info.get("create_attempted"):
                return
            ma3_comm = getattr(self.facade, "ma3_communication_service", None)
            if not ma3_comm:
                return
            info["create_attempted"] = True
            ma3_comm.create_track(info["timecode_no"], info["track_group"], info["track_name"])
            self._request_ma3_structure_refresh("pending_track")
            return
        self._pending_track_create = None
        self._link_editor_layer_to_ma3_track(
            info["editor_layer_id"],
            existing,
            create_ma3_entity=False,
            sequence_no=info.get("sequence_no")
        )

    def _prompt_sequence_no(self, default_value: int = 1) -> Optional[int]:
        """Prompt for MA3 sequence number (per-sync)."""
        sequence_no, ok = QInputDialog.getInt(
            self,
            "Assign MA3 Sequence",
            "Sequence number:",
            int(default_value or 1),
            1,
            9999,
            1
        )
        if not ok:
            return None
        return int(sequence_no)

    def _resolve_sequence_no(self, coord: str, sequence_no: Optional[int]) -> int:
        """Resolve sequence number from explicit value or saved settings."""
        if sequence_no is not None:
            return int(sequence_no or 1)
        # Check SyncSystemManager
        sync_entity = self._sync_system_manager.get_synced_layer_by_ma3_coord(coord)
        if sync_entity and sync_entity.settings:
            return int(getattr(sync_entity.settings, "sequence_no", 1) or 1)
        # Fallback to settings_manager
        if hasattr(self, "_settings_manager") and self._settings_manager.is_loaded():
            entity = self._settings_manager.get_synced_layer("ma3", coord)
            if entity and isinstance(entity, dict):
                settings = entity.get("settings") or {}
                return int(settings.get("sequence_no", 1) or 1)
        return 1

    def _resolve_conflict_resolution(self):
        """Map global conflict strategy to ConflictResolution."""
        from src.features.show_manager.domain.layer_sync_types import ConflictResolution

        strategy = None
        if hasattr(self, "_settings_manager") and self._settings_manager.is_loaded():
            strategy = self._settings_manager.conflict_resolution_strategy
        if not strategy:
            return ConflictResolution.PROMPT_USER
        normalized = str(strategy).strip().lower()
        mapping = {
            "ma3_wins": ConflictResolution.USE_MA3,
            "use_ma3": ConflictResolution.USE_MA3,
            "ez_wins": ConflictResolution.USE_EZ,
            "use_ez": ConflictResolution.USE_EZ,
            "merge": ConflictResolution.MERGE,
            "prompt_user": ConflictResolution.PROMPT_USER,
            "last_write_wins": ConflictResolution.PROMPT_USER,
            "skip": ConflictResolution.SKIP,
        }
        return mapping.get(normalized, ConflictResolution.PROMPT_USER)

    def _build_unified_sync_settings(
        self,
        direction,
        auto_apply: bool,
        apply_updates_enabled: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Build unified sync settings payload."""
        from src.features.show_manager.domain.layer_sync_types import SyncDirection

        if not isinstance(direction, SyncDirection):
            direction = SyncDirection.BIDIRECTIONAL
        if apply_updates_enabled is None:
            apply_updates_enabled = True
        conflict_resolution = self._resolve_conflict_resolution()
        return {
            "sync_direction": direction.name,
            "conflict_resolution": conflict_resolution.name,
            "direction": direction.name,
            "conflict_strategy": conflict_resolution.name,
            "auto_apply": bool(auto_apply),
            "apply_updates_enabled": bool(apply_updates_enabled),
        }

    def _get_sync_config_for_coord(self, ma3_coord_key: str) -> Dict[str, Any]:
        """Resolve unified sync config for a MA3 coord."""
        settings = {}
        # Check SyncSystemManager first
        sync_entity = self._sync_system_manager.get_synced_layer_by_ma3_coord(ma3_coord_key)
        if sync_entity and sync_entity.settings:
            from src.features.show_manager.domain.layer_sync_types import SyncDirection, ConflictResolution
            direction = getattr(sync_entity.settings, "sync_direction", SyncDirection.BIDIRECTIONAL)
            conflict = getattr(sync_entity.settings, "conflict_resolution", ConflictResolution.PROMPT_USER)
            settings = {
                "direction": getattr(direction, "name", "BIDIRECTIONAL"),
                "auto_apply": bool(getattr(sync_entity.settings, "auto_apply", False)),
                "conflict_strategy": getattr(conflict, "name", "PROMPT_USER"),
                "apply_updates_enabled": bool(getattr(sync_entity.settings, "apply_updates_enabled", True)),
            }
        # Fallback to settings_manager
        if not settings and hasattr(self, "_settings_manager") and self._settings_manager.is_loaded():
            entity = self._settings_manager.get_synced_layer("ma3", ma3_coord_key)
            if entity and isinstance(entity, dict):
                entity_settings = entity.get("settings") or {}
                settings = {
                    "direction": entity_settings.get("direction", entity_settings.get("sync_direction", "BIDIRECTIONAL")),
                    "auto_apply": bool(entity_settings.get("auto_apply", False)),
                    "conflict_strategy": entity_settings.get(
                        "conflict_strategy",
                        entity_settings.get("conflict_resolution", "PROMPT_USER")
                    ),
                    "apply_updates_enabled": bool(entity_settings.get("apply_updates_enabled", True)),
                }
        return settings

    def _resolve_auto_apply_action(self, conflict_strategy: Optional[str]) -> Optional[str]:
        """Resolve auto-apply action based on conflict strategy."""
        if not conflict_strategy:
            return "overwrite"
        normalized = str(conflict_strategy).strip().lower()
        if normalized in ("ma3_wins", "use_ma3"):
            return "overwrite"
        if normalized == "merge":
            return "merge"
        if normalized in ("ez_wins", "use_ez", "prompt_user", "last_write_wins", "skip"):
            return None
        return None

    def _assign_track_sequence(self, track_info: MA3TrackInfo, sequence_no: int) -> None:
        """Assign sequence number to MA3 track before pushing events."""
        ma3_comm = getattr(self.facade, "ma3_communication_service", None)
        if not ma3_comm:
            Log.warning("ShowManagerPanel: MA3 communication service not available for sequence assignment")
            return
        ma3_comm.assign_track_sequence(
            track_info.timecode_no,
            track_info.track_group,
            track_info.track,
            int(sequence_no or 1)
        )

    def _push_editor_events_to_ma3(self, editor_layer_id: str, track_info: MA3TrackInfo) -> None:
        """Push all Editor events to MA3 for a newly created track."""
        ma3_comm = getattr(self.facade, "ma3_communication_service", None)
        if not ma3_comm:
            Log.warning("ShowManagerPanel: MA3 communication service not available for event push")
            return
        coord = getattr(track_info, "coord", None)
        if not coord:
            coord = f"tc{track_info.timecode_no}_tg{track_info.track_group}_tr{track_info.track}"
        push_attempted = False
        try:
            from src.features.blocks.application.editor_api import create_editor_api
            editor_api = create_editor_api(self.facade, self._find_connected_editor())
            if not editor_api:
                Log.warning("ShowManagerPanel: EditorAPI not available for event push")
                return
            events = editor_api.get_events_in_layer(editor_layer_id) or []
            push_attempted = True
            for evt in events:
                evt_time = getattr(evt, "time", None)
                evt_meta = getattr(evt, "metadata", None) or {}
                evt_cmd = evt_meta.get("cmd") or ""
                ma3_comm.add_event(
                    track_info.timecode_no,
                    track_info.track_group,
                    track_info.track,
                    evt_time if evt_time is not None else 0.0,
                    evt_cmd
                )
        except Exception as exc:
            Log.warning(f"ShowManagerPanel: Failed to push events to MA3: {exc}")
        finally:
            if push_attempted:
                self._hook_and_sync_after_editor_push(editor_layer_id, track_info, coord)

    def _hook_and_sync_after_editor_push(
        self,
        editor_layer_id: str,
        track_info: MA3TrackInfo,
        coord: str
    ) -> None:
        """Hook the MA3 track and request events for immediate sync."""
        tc = track_info.timecode_no
        tg = track_info.track_group
        track = track_info.track
        try:
            info = self._build_reconcile_info(tc, tg, track, ma3_coord_key=coord)
            if not info:
                Log.warning(
                    f"ShowManagerPanel: Skipping hook+sync for {coord}; "
                    "no existing Editor↔MA3 mapping found"
                )
                return
            key = f"{tc}.{tg}.{track}"
            self._reconcile_pending[key] = info
            self._reconcile_active_key = key
            self._reconcile_in_progress = True
        except Exception:
            pass
        try:
            self._send_lua_command(f"EZ.HookCmdSubTrack({tc}, {tg}, {track})")
            self._send_lua_command(f"EZ.GetEvents({tc}, {tg}, {track})")
        except RuntimeError as exc:
            Log.warning(f"ShowManagerPanel: Hook/sync failed for {coord}: {exc}")
    
    def _find_connected_editor(self) -> str:
        """Find Editor block connected to this ShowManager."""
        try:
            connections_result = self.facade.list_connections()
            if not connections_result.success or not connections_result.data:
                return ""
            
            for conn in connections_result.data:
                if conn.source_block_id == self.block_id:
                    target_result = self.facade.describe_block(conn.target_block_id)
                    if target_result.success and target_result.data:
                        if target_result.data.type == "Editor":
                            return target_result.data.id
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Error finding connected Editor: {e}")
        
        return ""
    
    def _get_editor_timeline_widget(self, editor_block_id: str):
        """Get the TimelineWidget from the connected Editor panel."""
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if not app:
                return None
            
            # Find the main window
            for widget in app.topLevelWidgets():
                if hasattr(widget, 'open_panels'):
                    editor_panel = widget.open_panels.get(editor_block_id)
                    if editor_panel and hasattr(editor_panel, 'timeline_widget'):
                        return editor_panel.timeline_widget
            return None
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Error getting Editor timeline widget: {e}")
            return None
    
    def _get_editor_api_with_timeline(self, editor_block_id: str):
        """Get EditorAPI with timeline widget for direct visual updates."""
        from src.features.blocks.application.editor_api import create_editor_api
        
        timeline_widget = self._get_editor_timeline_widget(editor_block_id)
        return create_editor_api(self.facade, editor_block_id, timeline_widget=timeline_widget)
    
    def _execute_sync_tool_command(self, cmd) -> None:
        """Execute a sync tool command via command bus."""
        from PyQt6.QtWidgets import QMessageBox

        if not self.facade.command_bus:
            QMessageBox.warning(self, "Command Bus Missing", "Command bus is not available.")
            return
        try:
            self.facade.command_bus.execute(cmd)
        except Exception as exc:
            Log.error(f"ShowManagerPanel: Sync tool command failed: {exc}")
            QMessageBox.warning(self, "Sync Tool Failed", str(exc))

    def _update_hook_status_label(self) -> None:
        """Update hook status summary label."""
        if not hasattr(self, "hook_status_label") or not self.hook_status_label:
            return
        if not hasattr(self, "_sync_system_manager") or not self._sync_system_manager:
            return
        synced_layers = self._sync_system_manager.get_synced_layers()
        total = len([e for e in synced_layers if e.ma3_coord])
        hooked_count = sum(1 for e in synced_layers if e.ma3_coord and self._sync_system_manager.is_track_hooked(e.ma3_coord))
        self.hook_status_label.setText(f"Hooks: {hooked_count}/{total}")

    def _apply_updates_enabled_for_coord(self, coord: str) -> bool:
        """Check per-track apply setting for a MA3 coord."""
        if not hasattr(self, "_sync_system_manager") or not self._sync_system_manager:
            return True
        entity = self._sync_system_manager.get_synced_layer_by_ma3_coord(coord)
        if entity and entity.settings:
            return bool(getattr(entity.settings, "apply_updates_enabled", True))
        return True

    def _on_poll_sync_layers(self) -> None:
        """Poll all synced MA3 tracks."""
        from src.features.show_manager.application.commands import PollSyncedMA3TracksCommand

        cmd = PollSyncedMA3TracksCommand(self.facade, self.block_id)
        self._execute_sync_tool_command(cmd)
        self._update_hook_status_label()

    def _on_toggle_track_apply(self, coord: str, state: int) -> None:
        """Toggle per-track apply updates."""
        from src.features.show_manager.application.commands import UpdateEntitySettingsCommand

        enabled = state != Qt.CheckState.Checked.value
        cmd = UpdateEntitySettingsCommand(
            facade=self.facade,
            entity_type="ma3",
            entity_id=coord,
            show_manager_block_id=self.block_id,
            settings={"apply_updates_enabled": enabled}
        )
        self._execute_sync_tool_command(cmd)

    def _on_track_sequence_changed(self, coord: str, sequence_no: int) -> None:
        """Update per-track sequence assignment and apply to MA3."""
        from src.features.show_manager.application.commands import UpdateEntitySettingsCommand

        seq_value = int(sequence_no or 1)
        cmd = UpdateEntitySettingsCommand(
            facade=self.facade,
            entity_type="ma3",
            entity_id=coord,
            show_manager_block_id=self.block_id,
            settings={"sequence_no": seq_value}
        )
        self._execute_sync_tool_command(cmd)
        self._assign_track_sequence_by_coord(coord, seq_value)

    def _assign_track_sequence_by_coord(self, coord: str, sequence_no: int) -> None:
        """Assign sequence number using a coord string."""
        parsed = self._parse_ma3_coord(coord)
        if not parsed:
            return
        tc, tg, track = parsed
        ma3_comm = getattr(self.facade, "ma3_communication_service", None)
        if not ma3_comm:
            Log.warning("ShowManagerPanel: MA3 communication service not available for sequence assignment")
            return
        ma3_comm.assign_track_sequence(tc, tg, track, int(sequence_no or 1))

    def _on_test_sync_layer(self) -> None:
        """Rehook and poll a single MA3 track by coord."""
        from src.features.show_manager.application.commands import TestSyncedMA3TrackCommand
        from PyQt6.QtWidgets import QMessageBox

        coord, ok = QInputDialog.getText(
            self,
            "Test MA3 Track",
            "Enter MA3 coord (e.g., tc101_tg1_tr2 or 101.1.2):"
        )
        if not ok or not coord:
            return

        cmd = TestSyncedMA3TrackCommand(self.facade, self.block_id, coord.strip())
        self._execute_sync_tool_command(cmd)

    def _trigger_status_update(self):
        """Trigger a block status update by publishing BlockChanged event"""
        if not self.block_id or not self.facade or not hasattr(self.facade, 'event_bus'):
            return
        
        try:
            from src.application.events import BlockChanged
            # Get project_id from block
            project_id = self.block.project_id if hasattr(self, 'block') and self.block else None
            if not project_id:
                # Try to get from facade
                if hasattr(self.facade, 'project_repo') and self.facade.project_repo:
                    projects = self.facade.project_repo.list_all()
                    if projects:
                        project_id = projects[0].id
            
            if project_id:
                # Check current listener state before publishing
                is_listening = False
                if self._listener_service:
                    is_listening = self._listener_service.is_listening(self.block_id)
                
                
                self.facade.event_bus.publish(BlockChanged(
                    project_id=project_id,
                    data={
                        "block_id": self.block_id,
                        "change_type": "listener_state"
                    }
                ))
                Log.debug(f"ShowManagerPanel: Published BlockChanged event for status update (block {self.block_id}, listener={is_listening})")
        except Exception as e:
            Log.warning(f"ShowManagerPanel: Failed to trigger status update: {e}")
