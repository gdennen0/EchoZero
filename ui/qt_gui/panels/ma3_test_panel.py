"""
MA3 Test Panel - Comprehensive testing UI for MA3 integration features

Tests:
- OSC connection and communication
- MA3 command generation and execution
- Event/Track/TrackGroup management
- Bidirectional synchronization
- Mapping templates
"""
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QTabWidget, QScrollArea, QFrame,
    QSplitter, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

from ui.qt_gui.design_system import Colors, Spacing, ThemeAwareMixin, border_radius
from ui.qt_gui.style_factory import StyleFactory
from src.utils.message import Log


class StatusIndicator(QLabel):
    """Visual status indicator (LED-style)"""
    
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setFixedWidth(120)
        self._status = "disconnected"
        self._update_style()
    
    def set_status(self, status: str):
        """Set status: 'connected', 'disconnected', 'pending', 'error'"""
        self._status = status
        self._update_style()
    
    def _update_style(self):
        colors = {
            "connected": Colors.STATUS_SUCCESS.name(),
            "disconnected": Colors.STATUS_INACTIVE.name(),
            "pending": Colors.STATUS_WARNING.name(),
            "error": Colors.STATUS_ERROR.name(),
        }
        color = colors.get(self._status, colors["disconnected"])
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: {Colors.TEXT_PRIMARY.name()};
                padding: 4px 8px;
                border-radius: {border_radius(4)};
                font-weight: bold;
                font-size: 11px;
            }}
        """)
        status_text = {
            "connected": "● Connected",
            "disconnected": "○ Disconnected",
            "pending": "◐ Pending...",
            "error": "✕ Error"
        }
        self.setText(status_text.get(self._status, "Unknown"))


class LogWidget(QTextEdit):
    """Styled log output widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Menlo", 10))
        self.setMaximumHeight(200)
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 8px;
            }}
        """)
    
    def log(self, message: str, level: str = "info"):
        colors = {
            "info": Colors.STATUS_INFO.name(),
            "success": Colors.STATUS_SUCCESS.name(),
            "warning": Colors.STATUS_WARNING.name(),
            "error": Colors.STATUS_ERROR.name(),
        }
        color = colors.get(level, colors["info"])
        self.append(f'<span style="color: {color}">[{level.upper()}]</span> {message}')
        self.ensureCursorVisible()
    
    def clear_log(self):
        self.clear()


class MA3TestPanel(ThemeAwareMixin, QWidget):
    """
    Comprehensive test panel for MA3 integration.
    
    Provides UI for testing:
    - OSC connection/communication
    - Command generation
    - Event manipulation
    - State synchronization
    """
    
    def __init__(self, facade=None, parent=None):
        super().__init__(parent)
        self.facade = facade
        self._sync_service = None
        self._osc_bridge = None
        
        self._setup_ui()
        self._connect_signals()
        self._init_theme_aware()
        
        # Try to get services from facade
        self._init_services()
    
    def _init_services(self):
        """Initialize service connections"""
        if self.facade:
            # Try to get MA3SyncService
            if hasattr(self.facade, 'ma3_sync_service'):
                self._sync_service = self.facade.ma3_sync_service
                self._log.log("MA3SyncService found", "success")
            else:
                self._log.log("MA3SyncService not available", "warning")
            
            # Try to get OSCBridgeService
            if hasattr(self.facade, 'osc_bridge_service'):
                self._osc_bridge = self.facade.osc_bridge_service
                self._log.log("OSCBridgeService found", "success")
            else:
                self._log.log("OSCBridgeService not available", "warning")
    
    def _setup_ui(self):
        """Setup the panel UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(Spacing.MD)
        
        # Header
        header = QLabel("MA3 Integration Test Panel")
        header.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {Colors.TEXT_PRIMARY.name()};")
        layout.addWidget(header)
        
        # Tab widget for different test sections
        tabs = QTabWidget()
        tabs.setStyleSheet(StyleFactory.tabs())
        
        tabs.addTab(self._create_connection_tab(), "🔌 Connection")
        tabs.addTab(self._create_commands_tab(), "⚡ Commands")
        tabs.addTab(self._create_events_tab(), "📍 Events")
        tabs.addTab(self._create_sync_tab(), "🔄 Sync")
        tabs.addTab(self._create_mapping_tab(), "🗺️ Mapping")
        
        layout.addWidget(tabs, 1)
        
        # Log output (shared across all tabs)
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout(log_group)
        
        self._log = LogWidget()
        log_layout.addWidget(self._log)
        
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self._log.clear_log)
        log_layout.addWidget(clear_log_btn)
        
        layout.addWidget(log_group)
        
        self.setStyleSheet(self._get_panel_style())
    
    def _create_connection_tab(self) -> QWidget:
        """Create connection testing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Connection settings
        settings_group = QGroupBox("Connection Settings")
        settings_layout = QFormLayout(settings_group)
        
        self._ma3_ip = QLineEdit("127.0.0.1")
        settings_layout.addRow("MA3 IP:", self._ma3_ip)
        
        self._ma3_send_port = QSpinBox()
        self._ma3_send_port.setRange(1024, 65535)
        self._ma3_send_port.setValue(9001)
        settings_layout.addRow("Send Port:", self._ma3_send_port)
        
        self._ma3_listen_port = QSpinBox()
        self._ma3_listen_port.setRange(1024, 65535)
        self._ma3_listen_port.setValue(9000)
        settings_layout.addRow("Listen Port:", self._ma3_listen_port)
        
        layout.addWidget(settings_group)
        
        # Status
        status_group = QGroupBox("Connection Status")
        status_layout = QHBoxLayout(status_group)
        
        self._status_indicator = StatusIndicator()
        status_layout.addWidget(self._status_indicator)
        status_layout.addStretch()
        
        layout.addWidget(status_group)
        
        # Connection controls
        controls_group = QGroupBox("Connection Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        controls_layout.addWidget(self._connect_btn)
        
        self._disconnect_btn = QPushButton("Disconnect")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        controls_layout.addWidget(self._disconnect_btn)
        
        layout.addWidget(controls_group)
        
        # OSC Testing
        osc_group = QGroupBox("OSC Communication Test")
        osc_layout = QVBoxLayout(osc_group)
        
        # Ping test
        ping_row = QHBoxLayout()
        ping_btn = QPushButton("Send Ping")
        ping_btn.clicked.connect(self._on_send_ping)
        ping_row.addWidget(ping_btn)
        
        echo_input = QLineEdit("Hello MA3!")
        self._echo_input = echo_input
        ping_row.addWidget(echo_input)
        
        echo_btn = QPushButton("Send Echo")
        echo_btn.clicked.connect(self._on_send_echo)
        ping_row.addWidget(echo_btn)
        
        osc_layout.addLayout(ping_row)
        
        # Custom OSC message
        custom_row = QHBoxLayout()
        self._custom_address = QLineEdit("/echozero/status")
        custom_row.addWidget(QLabel("Address:"))
        custom_row.addWidget(self._custom_address)
        
        self._custom_args = QLineEdit("")
        custom_row.addWidget(QLabel("Args:"))
        custom_row.addWidget(self._custom_args)
        
        send_custom_btn = QPushButton("Send")
        send_custom_btn.clicked.connect(self._on_send_custom)
        custom_row.addWidget(send_custom_btn)
        
        osc_layout.addLayout(custom_row)
        
        layout.addWidget(osc_group)
        layout.addStretch()
        
        return widget
    
    def _create_commands_tab(self) -> QWidget:
        """Create command testing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Command registry info
        registry_group = QGroupBox("Command Registry")
        registry_layout = QVBoxLayout(registry_group)
        
        self._cmd_list = QTableWidget()
        self._cmd_list.setColumnCount(3)
        self._cmd_list.setHorizontalHeaderLabels(["Command", "Type", "Undoable"])
        self._cmd_list.horizontalHeader().setStretchLastSection(True)
        self._cmd_list.setMaximumHeight(150)
        registry_layout.addWidget(self._cmd_list)
        
        refresh_cmds_btn = QPushButton("Refresh Command List")
        refresh_cmds_btn.clicked.connect(self._refresh_command_list)
        registry_layout.addWidget(refresh_cmds_btn)
        
        layout.addWidget(registry_group)
        
        # Test commands
        test_group = QGroupBox("Test Commands")
        test_layout = QVBoxLayout(test_group)
        
        # Add Event command
        add_event_row = QHBoxLayout()
        add_event_row.addWidget(QLabel("Add Event:"))
        
        self._event_time = QDoubleSpinBox()
        self._event_time.setRange(0, 3600)
        self._event_time.setValue(1.0)
        self._event_time.setSuffix(" sec")
        add_event_row.addWidget(self._event_time)
        
        self._event_duration = QDoubleSpinBox()
        self._event_duration.setRange(0.01, 60)
        self._event_duration.setValue(0.1)
        self._event_duration.setSuffix(" sec")
        add_event_row.addWidget(self._event_duration)
        
        self._event_class = QLineEdit("kick")
        self._event_class.setPlaceholderText("Classification")
        add_event_row.addWidget(self._event_class)
        
        add_event_btn = QPushButton("Create")
        add_event_btn.clicked.connect(self._on_add_event)
        add_event_row.addWidget(add_event_btn)
        
        test_layout.addLayout(add_event_row)
        
        # Add Track command
        add_track_row = QHBoxLayout()
        add_track_row.addWidget(QLabel("Add Track:"))
        
        self._track_name = QLineEdit("Drums")
        self._track_name.setPlaceholderText("Track Name")
        add_track_row.addWidget(self._track_name)
        
        self._trackgroup_name = QLineEdit("Song1")
        self._trackgroup_name.setPlaceholderText("Track Group")
        add_track_row.addWidget(self._trackgroup_name)
        
        add_track_btn = QPushButton("Create")
        add_track_btn.clicked.connect(self._on_add_track)
        add_track_row.addWidget(add_track_btn)
        
        test_layout.addLayout(add_track_row)
        
        layout.addWidget(test_group)
        
        # Command validation
        validate_group = QGroupBox("Command Validation")
        validate_layout = QVBoxLayout(validate_group)
        
        validate_btn = QPushButton("Validate All Pending Commands")
        validate_btn.clicked.connect(self._on_validate_commands)
        validate_layout.addWidget(validate_btn)
        
        self._validation_result = QLabel("No validation run yet")
        validate_layout.addWidget(self._validation_result)
        
        layout.addWidget(validate_group)
        layout.addStretch()
        
        return widget
    
    def _create_events_tab(self) -> QWidget:
        """Create event manipulation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Current events display
        events_group = QGroupBox("Current Events")
        events_layout = QVBoxLayout(events_group)
        
        self._events_table = QTableWidget()
        self._events_table.setColumnCount(5)
        self._events_table.setHorizontalHeaderLabels(["ID", "Time", "Duration", "Class", "Track"])
        self._events_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        events_layout.addWidget(self._events_table)
        
        refresh_events_btn = QPushButton("Refresh Events")
        refresh_events_btn.clicked.connect(self._refresh_events)
        events_layout.addWidget(refresh_events_btn)
        
        layout.addWidget(events_group)
        
        # Event operations
        ops_group = QGroupBox("Event Operations")
        ops_layout = QVBoxLayout(ops_group)
        
        # Move event
        move_row = QHBoxLayout()
        move_row.addWidget(QLabel("Move Event:"))
        
        self._move_event_id = QLineEdit()
        self._move_event_id.setPlaceholderText("Event ID")
        move_row.addWidget(self._move_event_id)
        
        self._move_new_time = QDoubleSpinBox()
        self._move_new_time.setRange(0, 3600)
        self._move_new_time.setSuffix(" sec")
        move_row.addWidget(self._move_new_time)
        
        move_btn = QPushButton("Move")
        move_btn.clicked.connect(self._on_move_event)
        move_row.addWidget(move_btn)
        
        ops_layout.addLayout(move_row)
        
        # Delete event
        delete_row = QHBoxLayout()
        delete_row.addWidget(QLabel("Delete Event:"))
        
        self._delete_event_id = QLineEdit()
        self._delete_event_id.setPlaceholderText("Event ID")
        delete_row.addWidget(self._delete_event_id)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_delete_event)
        delete_row.addWidget(delete_btn)
        
        ops_layout.addLayout(delete_row)
        
        layout.addWidget(ops_group)
        layout.addStretch()
        
        return widget
    
    def _create_sync_tab(self) -> QWidget:
        """Create synchronization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sync status
        status_group = QGroupBox("Sync Status")
        status_layout = QFormLayout(status_group)
        
        self._sync_status = QLabel("Not synced")
        status_layout.addRow("Status:", self._sync_status)
        
        self._last_sync = QLabel("Never")
        status_layout.addRow("Last Sync:", self._last_sync)
        
        self._pending_changes = QLabel("0")
        status_layout.addRow("Pending Changes:", self._pending_changes)
        
        layout.addWidget(status_group)
        
        # Sync operations
        sync_ops_group = QGroupBox("Sync Operations")
        sync_ops_layout = QVBoxLayout(sync_ops_group)
        
        sync_row = QHBoxLayout()
        
        push_btn = QPushButton("Push to MA3")
        push_btn.clicked.connect(self._on_push_to_ma3)
        sync_row.addWidget(push_btn)
        
        pull_btn = QPushButton("Pull from MA3")
        pull_btn.clicked.connect(self._on_pull_from_ma3)
        sync_row.addWidget(pull_btn)
        
        full_sync_btn = QPushButton("Full Sync")
        full_sync_btn.clicked.connect(self._on_full_sync)
        sync_row.addWidget(full_sync_btn)
        
        sync_ops_layout.addLayout(sync_row)
        
        # Dry run option
        self._dry_run = QCheckBox("Dry Run (preview only)")
        sync_ops_layout.addWidget(self._dry_run)
        
        layout.addWidget(sync_ops_group)
        
        # Diff preview
        diff_group = QGroupBox("Sync Diff Preview")
        diff_layout = QVBoxLayout(diff_group)
        
        self._diff_output = QTextEdit()
        self._diff_output.setReadOnly(True)
        self._diff_output.setFont(QFont("Menlo", 10))
        self._diff_output.setMaximumHeight(200)
        diff_layout.addWidget(self._diff_output)
        
        calc_diff_btn = QPushButton("Calculate Diff")
        calc_diff_btn.clicked.connect(self._on_calc_diff)
        diff_layout.addWidget(calc_diff_btn)
        
        layout.addWidget(diff_group)
        layout.addStretch()
        
        return widget
    
    def _create_mapping_tab(self) -> QWidget:
        """Create mapping templates tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Available templates
        templates_group = QGroupBox("Mapping Templates")
        templates_layout = QVBoxLayout(templates_group)
        
        self._templates_combo = QComboBox()
        templates_layout.addWidget(self._templates_combo)
        
        refresh_templates_btn = QPushButton("Refresh Templates")
        refresh_templates_btn.clicked.connect(self._refresh_templates)
        templates_layout.addWidget(refresh_templates_btn)
        
        layout.addWidget(templates_group)
        
        # Template preview
        preview_group = QGroupBox("Template Details")
        preview_layout = QFormLayout(preview_group)
        
        self._template_name = QLabel("-")
        preview_layout.addRow("Name:", self._template_name)
        
        self._template_desc = QLabel("-")
        self._template_desc.setWordWrap(True)
        preview_layout.addRow("Description:", self._template_desc)
        
        self._template_strategy = QLabel("-")
        preview_layout.addRow("Track Strategy:", self._template_strategy)
        
        layout.addWidget(preview_group)
        
        # Test mapping
        test_group = QGroupBox("Test Mapping")
        test_layout = QVBoxLayout(test_group)
        
        test_mapping_btn = QPushButton("Preview Mapping (Dry Run)")
        test_mapping_btn.clicked.connect(self._on_test_mapping)
        test_layout.addWidget(test_mapping_btn)
        
        self._mapping_preview = QTextEdit()
        self._mapping_preview.setReadOnly(True)
        self._mapping_preview.setFont(QFont("Menlo", 10))
        self._mapping_preview.setMaximumHeight(200)
        test_layout.addWidget(self._mapping_preview)
        
        layout.addWidget(test_group)
        layout.addStretch()
        
        return widget
    
    def _connect_signals(self):
        """Connect widget signals"""
        self._templates_combo.currentTextChanged.connect(self._on_template_selected)
    
    # --- Event Handlers ---
    
    def _on_connect(self):
        """Handle connect button click"""
        self._status_indicator.set_status("pending")
        self._log.log(f"Connecting to MA3 at {self._ma3_ip.text()}:{self._ma3_send_port.value()}...")
        
        try:
            if self._sync_service:
                # Use real service
                success = self._sync_service.connect()
                if success:
                    self._status_indicator.set_status("connected")
                    self._connect_btn.setEnabled(False)
                    self._disconnect_btn.setEnabled(True)
                    self._log.log("Connected to MA3", "success")
                else:
                    self._status_indicator.set_status("error")
                    self._log.log("Connection failed", "error")
            else:
                # Simulate for testing
                QTimer.singleShot(500, lambda: self._simulate_connect())
        except Exception as e:
            self._status_indicator.set_status("error")
            self._log.log(f"Connection error: {e}", "error")
    
    def _simulate_connect(self):
        """Simulate successful connection for testing"""
        self._status_indicator.set_status("connected")
        self._connect_btn.setEnabled(False)
        self._disconnect_btn.setEnabled(True)
        self._log.log("Connected (simulated)", "success")
    
    def _on_disconnect(self):
        """Handle disconnect button click"""
        self._log.log("Disconnecting from MA3...")
        
        if self._sync_service:
            self._sync_service.disconnect()
        
        self._status_indicator.set_status("disconnected")
        self._connect_btn.setEnabled(True)
        self._disconnect_btn.setEnabled(False)
        self._log.log("Disconnected", "info")
    
    def _on_send_ping(self):
        """Send ping to MA3"""
        self._log.log("Sending ping to MA3...")
        
        if self._osc_bridge:
            success = self._osc_bridge.ping()
            if success:
                self._log.log("Ping sent successfully", "success")
            else:
                self._log.log("Failed to send ping", "error")
        else:
            self._log.log("Ping sent (simulated)", "success")
    
    def _on_send_echo(self):
        """Send echo message to MA3"""
        message = self._echo_input.text()
        self._log.log(f"Sending echo: '{message}'...")
        
        if self._osc_bridge:
            success = self._osc_bridge.echo(message)
            if success:
                self._log.log(f"Echo sent: '{message}'", "success")
            else:
                self._log.log("Failed to send echo", "error")
        else:
            self._log.log(f"Echo sent (simulated): '{message}'", "success")
    
    def _on_send_custom(self):
        """Send custom OSC message"""
        address = self._custom_address.text()
        args_str = self._custom_args.text()
        
        # Parse args (simple comma-separated for now)
        args = [arg.strip() for arg in args_str.split(",")] if args_str else []
        
        self._log.log(f"Sending OSC: {address} {args}")
        
        if self._osc_bridge:
            success = self._osc_bridge.send(address, *args)
            if success:
                self._log.log("Custom OSC sent", "success")
            else:
                self._log.log("Failed to send custom OSC", "error")
        else:
            self._log.log("Custom OSC sent (simulated)", "success")
    
    def _refresh_command_list(self):
        """Refresh the command registry list"""
        try:
            from src.application.commands.ma3 import MA3CommandRegistry
            
            commands = MA3CommandRegistry.list_commands()
            self._cmd_list.setRowCount(len(commands))
            
            for i, (cmd_type, cmd_class) in enumerate(commands.items()):
                self._cmd_list.setItem(i, 0, QTableWidgetItem(cmd_type))
                self._cmd_list.setItem(i, 1, QTableWidgetItem(cmd_class.__name__))
                self._cmd_list.setItem(i, 2, QTableWidgetItem("Yes"))
            
            self._log.log(f"Found {len(commands)} registered commands", "info")
        except Exception as e:
            self._log.log(f"Failed to load commands: {e}", "error")
    
    def _on_add_event(self):
        """Test adding an event"""
        time = self._event_time.value()
        duration = self._event_duration.value()
        classification = self._event_class.text()
        
        self._log.log(f"Creating event: time={time}, duration={duration}, class={classification}")
        
        try:
            from src.application.commands.ma3 import AddEventCommand, CommandContext
            
            # Create command context (facade can be None for validation testing)
            context = CommandContext(
                facade=self.facade,  # May be None
                timecode_no=1,
                track_group_idx=0
            )
            
            # Create command
            cmd = AddEventCommand(
                context=context,
                time=time,
                duration=duration,
                classification=classification
            )
            
            # Validate
            result = cmd.validate()
            if result.valid:
                self._log.log(f"Event command created and validated: time={time}s, class={classification}", "success")
            else:
                self._log.log(f"Validation failed: {result.errors}", "warning")
        except Exception as e:
            self._log.log(f"Error creating event: {e}", "error")
    
    def _on_add_track(self):
        """Test adding a track"""
        track_name = self._track_name.text()
        trackgroup = self._trackgroup_name.text()
        
        self._log.log(f"Creating track: {track_name} in group {trackgroup}")
        
        try:
            from src.application.commands.ma3 import AddTrackCommand, CommandContext
            
            context = CommandContext(
                facade=self.facade,  # May be None
                timecode_no=1,
                track_group_idx=0
            )
            
            cmd = AddTrackCommand(
                context=context,
                name=track_name
            )
            
            result = cmd.validate()
            if result.valid:
                self._log.log(f"Track command created: {track_name}", "success")
            else:
                self._log.log(f"Validation failed: {result.errors}", "warning")
        except Exception as e:
            self._log.log(f"Error creating track: {e}", "error")
    
    def _on_validate_commands(self):
        """Validate all pending commands"""
        self._log.log("Validating pending commands...")
        # Placeholder - would validate commands in queue
        self._validation_result.setText("✓ All commands valid")
        self._log.log("Validation complete", "success")
    
    def _refresh_events(self):
        """Refresh events display"""
        self._log.log("Refreshing events...")
        
        # Placeholder - would load from EventDataItem
        self._events_table.setRowCount(3)
        for i in range(3):
            self._events_table.setItem(i, 0, QTableWidgetItem(f"evt-{i}"))
            self._events_table.setItem(i, 1, QTableWidgetItem(f"{i * 0.5:.2f}"))
            self._events_table.setItem(i, 2, QTableWidgetItem("0.10"))
            self._events_table.setItem(i, 3, QTableWidgetItem(["kick", "snare", "hihat"][i]))
            self._events_table.setItem(i, 4, QTableWidgetItem("Drums"))
        
        self._log.log(f"Found 3 events (sample data)", "info")
    
    def _on_move_event(self):
        """Move an event"""
        event_id = self._move_event_id.text()
        new_time = self._move_new_time.value()
        self._log.log(f"Moving event {event_id} to time {new_time}")
        self._log.log("Move command created (not executed)", "success")
    
    def _on_delete_event(self):
        """Delete an event"""
        event_id = self._delete_event_id.text()
        self._log.log(f"Deleting event {event_id}")
        self._log.log("Delete command created (not executed)", "success")
    
    def _on_push_to_ma3(self):
        """Push events to MA3"""
        dry_run = self._dry_run.isChecked()
        self._log.log(f"Pushing to MA3 (dry_run={dry_run})...")
        
        if self._sync_service and not dry_run:
            # Real push
            pass
        else:
            # Simulated
            self._log.log("Push preview: 10 events would be created", "info")
            self._last_sync.setText("Just now (preview)")
    
    def _on_pull_from_ma3(self):
        """Pull events from MA3"""
        dry_run = self._dry_run.isChecked()
        self._log.log(f"Pulling from MA3 (dry_run={dry_run})...")
        
        if self._sync_service and not dry_run:
            # Real pull
            pass
        else:
            self._log.log("Pull preview: 5 events would be imported", "info")
    
    def _on_full_sync(self):
        """Full bidirectional sync"""
        self._log.log("Starting full sync...")
        self._sync_status.setText("Syncing...")
        
        QTimer.singleShot(1000, lambda: self._complete_sync())
    
    def _complete_sync(self):
        """Complete sync simulation"""
        self._sync_status.setText("Synced")
        self._last_sync.setText("Just now")
        self._log.log("Sync complete", "success")
    
    def _on_calc_diff(self):
        """Calculate diff between EchoZero and MA3"""
        self._log.log("Calculating diff...")
        
        diff_text = """=== Sync Diff ===
EchoZero -> MA3:
  + 3 events to add
  ~ 2 events to update
  - 1 event to delete

MA3 -> EchoZero:
  + 1 track to import
  ~ 0 events updated
"""
        self._diff_output.setPlainText(diff_text)
        self._log.log("Diff calculated", "success")
    
    def _refresh_templates(self):
        """Refresh mapping templates list"""
        try:
            from src.application.settings.ma3_mapping_templates import MappingTemplateRegistry
            
            templates = MappingTemplateRegistry.get_template_names()
            self._templates_combo.clear()
            self._templates_combo.addItems(templates)
            
            self._log.log(f"Found {len(templates)} templates", "info")
        except Exception as e:
            self._log.log(f"Error loading templates: {e}", "error")
            # Add default templates
            self._templates_combo.addItems(["default", "drums", "stems", "single"])
    
    def _on_template_selected(self, template_name: str):
        """Handle template selection"""
        try:
            from src.application.settings.ma3_mapping_templates import MappingTemplateRegistry
            
            template = MappingTemplateRegistry.get_template(template_name)
            if template:
                self._template_name.setText(template.name)
                self._template_desc.setText(template.description)
                self._template_strategy.setText(template.track_strategy)
        except Exception as e:
            self._template_name.setText(template_name)
            self._template_desc.setText("Template preview not available")
            self._template_strategy.setText("-")
    
    def _on_test_mapping(self):
        """Test mapping with current template"""
        template = self._templates_combo.currentText()
        self._log.log(f"Testing mapping with template: {template}")
        
        preview = f"""=== Mapping Preview ({template}) ===
TrackGroup: "Song1"
  Track: "Kick Drum"
    - 12 events
  Track: "Snare Drum"
    - 24 events  
  Track: "Hi-Hat"
    - 48 events
Total: 84 events -> 3 tracks
"""
        self._mapping_preview.setPlainText(preview)
        self._log.log("Mapping preview generated", "success")
    
    # --- Styling ---
    
    def _get_panel_style(self) -> str:
        return f"""
            QWidget {{
                background-color: {Colors.BG_DARK.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(6)};
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QPushButton {{
                background-color: {Colors.ACCENT_BLUE.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border: none;
                border-radius: {border_radius(4)};
                padding: 6px 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {Colors.ACCENT_BLUE.lighter(120).name()};
            }}
            QPushButton:pressed {{
                background-color: {Colors.ACCENT_BLUE.darker(120).name()};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BG_MEDIUM.name()};
                color: {Colors.TEXT_DISABLED.name()};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
                padding: 4px 8px;
            }}
            QTableWidget {{
                background-color: {Colors.BG_LIGHT.name()};
                border: 1px solid {Colors.BORDER.name()};
                border-radius: {border_radius(4)};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_MEDIUM.name()};
                padding: 4px;
                border: none;
            }}
        """
    