"""
Application Bootstrap

Centralized service initialization and dependency injection.
Provides a single point for setting up the application architecture.
"""
import atexit
import os
from typing import Dict, Any, Optional

from src.infrastructure.persistence.sqlite.database import Database
from src.features.projects.infrastructure import SQLiteProjectRepository
from src.features.blocks.infrastructure import SQLiteBlockRepository
from src.features.connections.infrastructure import SQLiteConnectionRepository
from src.shared.infrastructure.persistence import SQLiteDataItemRepository
from src.shared.infrastructure.persistence.layer_order_repository_impl import SQLiteLayerOrderRepository
from src.shared.infrastructure.persistence.block_local_state_repository_impl import SQLiteBlockLocalStateRepository
from src.infrastructure.persistence.sqlite.ui_state_repository_impl import UIStateRepository
from src.infrastructure.persistence.sqlite.preferences_repository_impl import PreferencesRepository
from src.infrastructure.persistence.sqlite.session_state_repository_impl import SessionStateRepository
from src.infrastructure.persistence.caching_repository import CachedBlockRepository
from src.application.events.event_bus import EventBus
from src.features.projects.application import ProjectService
from src.features.blocks.application import BlockService
from src.features.connections.application import ConnectionService
from src.features.execution.application import BlockExecutionEngine
from src.shared.domain.repositories import DataItemRepository
from src.application.api.application_facade import ApplicationFacade
from src.application.settings.app_settings import AppSettingsManager, init_app_settings_manager
from src.utils.message import Log
from src.utils.recent_projects import RecentProjectsStore
from src.application.bootstrap_loading_progress import LoadingProgressTracker
from src.application.services.layer_order_service import LayerOrderService


def _process_qt_events():
    """Process Qt events if QApplication exists (for splash screen updates)"""
    try:
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.processEvents()
    except ImportError:
        pass  # Qt not available (e.g., CLI mode)


class ServiceContainer:
    """Container for all application services"""
    
    def __init__(
        self,
        database: Database,
        event_bus: EventBus,
        project_repo: SQLiteProjectRepository,
        block_repo: SQLiteBlockRepository,
        connection_repo: SQLiteConnectionRepository,
        project_service: ProjectService,
        block_service: BlockService,
        connection_service: ConnectionService,
        execution_engine: BlockExecutionEngine,
        data_item_repo: DataItemRepository,
        block_local_state_repo: SQLiteBlockLocalStateRepository,
        ui_state_repo: UIStateRepository,
        layer_order_repo: SQLiteLayerOrderRepository,
        preferences_repo: PreferencesRepository,
        session_state_repo: SessionStateRepository,
        app_settings_manager: AppSettingsManager,
        facade: ApplicationFacade,
        recent_store: RecentProjectsStore,
        data_state_service=None,
        block_status_service=None,
        data_filter_manager=None,
        expected_outputs_service=None,
        setlist_service=None,
        action_set_repo=None,
        action_item_repo=None,
        ma3_communication_service=None,
        show_manager_listener_service=None,
        show_manager_state_service=None,
        sync_port=None,
        layer_group_order_service=None
    ):
        self.database = database
        self.event_bus = event_bus
        self.project_repo = project_repo
        self.block_repo = block_repo
        self.data_state_service = data_state_service
        self.block_status_service = block_status_service
        self.data_filter_manager = data_filter_manager
        self.expected_outputs_service = expected_outputs_service
        self.connection_repo = connection_repo
        self.data_item_repo = data_item_repo
        self.block_local_state_repo = block_local_state_repo
        self.ui_state_repo = ui_state_repo
        self.layer_order_repo = layer_order_repo
        self.layer_group_order_service = layer_group_order_service
        self.preferences_repo = preferences_repo
        self.session_state_repo = session_state_repo
        self.app_settings = app_settings_manager
        self.project_service = project_service
        self.block_service = block_service
        self.connection_service = connection_service
        self.execution_engine = execution_engine
        self.recent_store = recent_store
        self.layer_order_service = None
        self.facade = facade
        self.setlist_service = setlist_service
        self.action_set_repo = action_set_repo
        self.action_item_repo = action_item_repo
        self.ma3_communication_service = ma3_communication_service
        self.show_manager_listener_service = show_manager_listener_service
        self.show_manager_state_service = show_manager_state_service
        self.sync_port = sync_port
    
    def cleanup(self) -> None:
        """
        Clean up all resources in the service container.
        
        Follows cleanup patterns from AgentAssets/modules/commands/cleanup/:
        - Close database connections
        - Stop services with active resources (sockets, threads)
        - Clear runtime state
        """
        Log.info("ServiceContainer: Starting cleanup")
        
        # Stop MA3 communication service if running (closes socket and thread)
        if self.ma3_communication_service:
            try:
                self.ma3_communication_service.stop_listening()
            except Exception as e:
                Log.warning(f"ServiceContainer: Error stopping MA3 communication service: {e}")
        
        # Stop ShowManager listener service if running
        if self.show_manager_listener_service:
            try:
                self.show_manager_listener_service.cleanup_all()
            except Exception as e:
                Log.warning(f"ServiceContainer: Error stopping ShowManager listener service: {e}")
        
        if self.show_manager_state_service:
            try:
                self.show_manager_state_service.cleanup()
            except Exception as e:
                Log.warning(f"ServiceContainer: Error cleaning up ShowManager state service: {e}")
        
        # Close database connection
        if self.database:
            try:
                self.database.close()
            except Exception as e:
                Log.warning(f"ServiceContainer: Error closing database: {e}")
        
        Log.info("ServiceContainer: Cleanup complete")


def initialize_services(
    db_path: str = None,
    progress_tracker: Optional[LoadingProgressTracker] = None,
    clear_runtime_tables: bool = True,
) -> ServiceContainer:
    """
    Initialize all application services.

    Args:
        db_path: Path to SQLite database file.
                If None, uses default location in data directory.
        progress_tracker: Optional progress tracker for loading feedback
        clear_runtime_tables: If True, clear session tables on init (default).
                Pass False when the worker subprocess shares the DB with the UI
                process so it does not wipe existing project/block data.

    Returns:
        ServiceContainer with all initialized services
    """
    # Set total modules for progress tracking
    if progress_tracker:
        progress_tracker.set_total_modules(9)  # Foundation, Repositories, Settings, Services, Execution, Block Processors, Integration, Container, Qt GUI
    
    # Determine database path
    if db_path is None:
        from src.utils.paths import get_database_path
        # Use platform-specific user data directory
        db_path = str(get_database_path("ez"))
    
    Log.info(f"Initializing services with database: {db_path}")
    
    # Module 1: Foundation Layer
    if progress_tracker:
        progress_tracker.start_module("Foundation", "Database and event system", 2)
        _process_qt_events()
    
    # Initialize database
    if progress_tracker:
        progress_tracker.update_step("Initializing database")
        _process_qt_events()
    database = Database(db_path)
    if clear_runtime_tables:
        database.clear_runtime_tables()
    Log.info("Database initialized")
    
    # Initialize event bus
    if progress_tracker:
        progress_tracker.update_step("Initializing event bus")
        _process_qt_events()
    event_bus = EventBus()
    Log.info("Event bus initialized")
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 2: Persistence Layer
    if progress_tracker:
        progress_tracker.start_module("Repositories", "Data persistence layer", 12)
        _process_qt_events()
    
    # Initialize repositories
    if progress_tracker:
        progress_tracker.update_step("Project repository")
        _process_qt_events()
    project_repo = SQLiteProjectRepository(database)
    
    # Wrap block_repo with caching for performance
    if progress_tracker:
        progress_tracker.update_step("Block repository (with caching)")
        _process_qt_events()
    block_repo_inner = SQLiteBlockRepository(database)
    block_repo = CachedBlockRepository(block_repo_inner, ttl_seconds=60, max_size=1000)
    
    if progress_tracker:
        progress_tracker.update_step("Connection repository")
        _process_qt_events()
    connection_repo = SQLiteConnectionRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Data item repository")
        _process_qt_events()
    data_item_repo = SQLiteDataItemRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Block local state repository")
        _process_qt_events()
    block_local_state_repo = SQLiteBlockLocalStateRepository(database)

    if progress_tracker:
        progress_tracker.update_step("Layer order repository")
        _process_qt_events()
    layer_order_repo = SQLiteLayerOrderRepository(database)
    
    # Initialize UI state repositories (Phase A Foundation)
    if progress_tracker:
        progress_tracker.update_step("UI state repository")
        _process_qt_events()
    ui_state_repo = UIStateRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Preferences repository")
        _process_qt_events()
    preferences_repo = PreferencesRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Session state repository")
        _process_qt_events()
    session_state_repo = SessionStateRepository(database)
    
    # Initialize setlist repositories
    from src.features.setlists.infrastructure import SQLiteSetlistRepository
    from src.features.setlists.infrastructure import SQLiteSetlistSongRepository
    from src.features.projects.infrastructure import SQLiteActionSetRepository, SQLiteActionItemRepository
    
    if progress_tracker:
        progress_tracker.update_step("Setlist repository")
        _process_qt_events()
    setlist_repo = SQLiteSetlistRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Setlist song repository")
        _process_qt_events()
    setlist_song_repo = SQLiteSetlistSongRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Action set repository")
        _process_qt_events()
    action_set_repo = SQLiteActionSetRepository(database)
    
    if progress_tracker:
        progress_tracker.update_step("Action item repository")
        _process_qt_events()
    action_item_repo = SQLiteActionItemRepository(database)
    
    Log.info("Repositories initialized (with caching)")
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 3: Settings Layer
    if progress_tracker:
        progress_tracker.start_module("Settings", "Application configuration", 4)
        _process_qt_events()
    
    # Initialize application settings manager (standardized settings system)
    if progress_tracker:
        progress_tracker.update_step("Loading settings manager")
        _process_qt_events()
    app_settings_manager = init_app_settings_manager(preferences_repo)
    # Migration code removed - use one-time conversion script if needed
    Log.info("Application settings manager initialized")
    
    # Apply log level setting from preferences
    if progress_tracker:
        progress_tracker.update_step("Applying log level settings")
        _process_qt_events()
    log_level = app_settings_manager.log_level
    if log_level:
        Log.set_level(log_level)
        Log.info(f"Log level set to: {log_level}")
    
    # Apply repetitive log filter setting from preferences
    if progress_tracker:
        progress_tracker.update_step("Applying log filter settings")
        _process_qt_events()
    filter_repetitive = app_settings_manager.filter_repetitive_logs
    Log.enable_repetitive_filter(filter_repetitive)
    if filter_repetitive:
        Log.info("Repetitive log filtering enabled (cache hits, status checks filtered)")
    
    if progress_tracker:
        progress_tracker.update_step("Initializing recent projects store")
        _process_qt_events()
    recent_store = RecentProjectsStore()
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()

    # Module 4: Core Services Layer
    if progress_tracker:
        progress_tracker.start_module("Services", "Core business logic services", 10)
        _process_qt_events()
    
    # Initialize services
    # ProjectService needs access to other repos for exporting complete project data
    if progress_tracker:
        progress_tracker.update_step("Project service")
        _process_qt_events()
    project_service = ProjectService(
        project_repo, 
        event_bus,
        block_repo=block_repo,
        connection_repo=connection_repo,
        data_item_repo=data_item_repo,
        database=database,
        recent_store=recent_store,
        ui_state_repo=ui_state_repo,
        block_local_state_repo=block_local_state_repo,
        layer_order_repo=layer_order_repo,
        setlist_repo=setlist_repo,
        setlist_song_repo=setlist_song_repo,
        action_set_repo=action_set_repo,
        action_item_repo=action_item_repo
    )
    
    if progress_tracker:
        progress_tracker.update_step("Block service")
        _process_qt_events()
    block_service = BlockService(block_repo, project_repo, event_bus, connection_repo)
    
    if progress_tracker:
        progress_tracker.update_step("Connection service")
        _process_qt_events()
    connection_service = ConnectionService(connection_repo, block_repo, event_bus)
    
    # Initialize data state service (deprecated - will be removed after migration)
    if progress_tracker:
        progress_tracker.update_step("Data state service")
        _process_qt_events()
    from src.shared.application.services.data_state_service import DataStateService
    data_state_service = DataStateService(
        block_local_state_repo=block_local_state_repo,
        data_item_repo=data_item_repo,
        connection_repo=connection_repo,
        block_repo=block_repo
    )
    
    # Initialize block status service (new standardized status system)
    if progress_tracker:
        progress_tracker.update_step("Block status service")
        _process_qt_events()
    from src.features.blocks.application import BlockStatusService
    block_status_service = BlockStatusService(
        block_repo=block_repo,
        event_bus=event_bus,
        data_state_service=data_state_service  # For fallback when blocks don't implement get_status_levels()
    )
    
    # Initialize data filter manager
    if progress_tracker:
        progress_tracker.update_step("Data filter manager")
        _process_qt_events()
    from src.shared.application.services.data_filter_manager import DataFilterManager
    data_filter_manager = DataFilterManager(
        data_item_repo=data_item_repo,
        connection_repo=connection_repo
    )
    
    # Initialize expected outputs service
    if progress_tracker:
        progress_tracker.update_step("Expected outputs service")
        _process_qt_events()
    from src.features.blocks.application.expected_outputs_service import ExpectedOutputsService
    expected_outputs_service = ExpectedOutputsService(
        connection_repo=connection_repo,
        block_repo=block_repo
    )
    
    # Initialize snapshot service
    if progress_tracker:
        progress_tracker.update_step("Snapshot service")
        _process_qt_events()
    from src.features.projects.application import SnapshotService
    snapshot_service = SnapshotService(
        block_repo=block_repo,
        data_item_repo=data_item_repo,
        block_local_state_repo=block_local_state_repo,
        project_service=project_service
    )
    
    # Initialize setlist service
    from src.features.setlists.application import SetlistService
    setlist_service = None  # Will be set after facade is created
    
    # Initialize MA3 communication service (DISABLED - using ShowManager panel instead)
    # This service expects pipe-delimited format, but MA3 now sends OSC format
    # The ShowManager panel handles OSC communication directly
    if progress_tracker:
        progress_tracker.update_step("MA3 communication service")
        _process_qt_events()
    from src.features.ma3.application.ma3_communication_service import MA3CommunicationService
    ma3_communication_service = MA3CommunicationService(
        event_bus=event_bus,
        listen_port=app_settings_manager.ma3_listen_port,
        listen_address=app_settings_manager.ma3_listen_address,
        send_port=app_settings_manager.ma3_send_port,
        send_address=app_settings_manager.ma3_send_address
    )
    # DISABLED: Start listening automatically if enabled (can be stopped/started later)
    # if app_settings_manager.ma3_listen_enabled:
    #     ma3_communication_service.start_listening()
    
    # Initialize ShowManager listener service
    if progress_tracker:
        progress_tracker.update_step("ShowManager listener service")
        _process_qt_events()
    from src.features.show_manager.application.show_manager_listener_service import ShowManagerListenerService
    show_manager_listener_service = ShowManagerListenerService(
        event_bus=event_bus,
        ma3_comm=ma3_communication_service
    )
    Log.info("Bootstrap: ShowManager listener service initialized")
    
    # Initialize ShowManager state service
    if progress_tracker:
        progress_tracker.update_step("ShowManager state service")
        _process_qt_events()
    from src.features.show_manager.application.show_manager_state_service import ShowManagerStateService
    # Note: facade will be set after service container is created
    show_manager_state_service = None  # Will be initialized after facade creation
    
    Log.info("Services initialized")
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 5: Execution Layer
    if progress_tracker:
        progress_tracker.start_module("Execution", "Block execution engine", 1)
        _process_qt_events()
    
    # Initialize execution engine
    if progress_tracker:
        progress_tracker.update_step("Initializing execution engine")
        _process_qt_events()
    execution_engine = BlockExecutionEngine(
        block_repo=block_repo,
        connection_repo=connection_repo,
        data_item_repo=data_item_repo,
        event_bus=event_bus,
        block_local_state_repo=block_local_state_repo,
        data_filter_manager=data_filter_manager
    )
    Log.info("Execution engine initialized")
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 5b: Block Processors (separate module for visibility)
    # Auto-register all block processors
    # Import modules first to trigger registration, then count
    from src.application.blocks import register_all_processors, get_registered_processors
    # Import all processor modules to trigger their registration
    try:
        from src.application.blocks import (
            load_audio_block, setlist_audio_input_block, detect_onsets_block,
            learned_onset_detector_block,
            tensorflow_classify_block, pytorch_audio_trainer_block, learned_onset_trainer_block,
            pytorch_audio_classify_block, separator_block, editor_block,
            export_audio_block, export_clips_by_class_block, note_extractor_basicpitch_block,
            note_extractor_librosa_block, plot_events_block, show_manager_block
        )
    except ImportError:
        pass  # Some processors may not be available
    
    if progress_tracker:
        # Count processors after imports
        processor_classes = get_registered_processors()
        total_processors = len(processor_classes) if processor_classes else 15  # Default estimate
        progress_tracker.start_module("Block Processors", "Registering block processors", total_processors)
        _process_qt_events()
    
    register_all_processors(execution_engine, progress_tracker=progress_tracker)
    Log.info("Block processors auto-registered")
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 6: Integration Layer
    if progress_tracker:
        progress_tracker.start_module("Integration", "Application facade and integration", 4)
        _process_qt_events()
    
    # Initialize setlist service (needs facade, so create after facade)
    if progress_tracker:
        progress_tracker.update_step("Setlist service (partial)")
        _process_qt_events()
    setlist_service = SetlistService(
        setlist_repo=setlist_repo,
        setlist_song_repo=setlist_song_repo,
        block_repo=block_repo,
        data_item_repo=data_item_repo,
        block_local_state_repo=block_local_state_repo,
        snapshot_service=snapshot_service,
        project_service=project_service,
        execution_engine=execution_engine,
        facade=None  # Will be set after facade creation
    )

    layer_order_service = LayerOrderService(layer_order_repo)
    from src.application.services.layer_group_order_service import LayerGroupOrderService
    layer_group_order_service = LayerGroupOrderService(ui_state_repo)
    
    # Create temporary container for facade initialization
    if progress_tracker:
        progress_tracker.update_step("Creating service container")
        _process_qt_events()
    temp_container = type('TempContainer', (), {
        'database': database,
        'project_service': project_service,
        'block_service': block_service,
        'connection_service': connection_service,
        'execution_engine': execution_engine,
        'event_bus': event_bus,
        'data_item_repo': data_item_repo,
        'block_local_state_repo': block_local_state_repo,
        'layer_order_repo': layer_order_repo,
        'layer_order_service': layer_order_service,
        'layer_group_order_service': layer_group_order_service,
        'ui_state_repo': ui_state_repo,
        'preferences_repo': preferences_repo,
        'session_state_repo': session_state_repo,
        'app_settings': app_settings_manager,
        'recent_store': recent_store,
        'data_state_service': data_state_service,
        'block_status_service': block_status_service,
        'data_filter_manager': data_filter_manager,
        'expected_outputs_service': expected_outputs_service,
        'setlist_service': setlist_service,
        'action_set_repo': action_set_repo,
        'action_item_repo': action_item_repo
    })()
    
    # Initialize application facade (unified API for all interfaces)
    if progress_tracker:
        progress_tracker.update_step("Application facade")
        _process_qt_events()
    facade = ApplicationFacade(temp_container)
    from src.application.services.sync_port import SyncPort
    sync_port = SyncPort(facade)
    facade.sync_port = sync_port
    
    # Set facade in setlist service (circular dependency resolved)
    setlist_service._facade = facade
    
    # Set facade provider in block status service (for active status recalculation)
    # This allows BlockStatusService to recalculate status immediately when BlockChanged events are received
    if block_status_service:
        def get_facade():
            return facade
        block_status_service._facade_provider = get_facade
    
    # Set up pull_callback for execution engine (needs facade for pull_block_inputs_overwrite)
    # This allows execution to pull data from upstream blocks before processing
    if progress_tracker:
        progress_tracker.update_step("Configuring execution callbacks")
        _process_qt_events()
    def pull_callback(block_id: str):
        """Pull data from upstream connections for a block"""
        result = facade.pull_block_inputs_overwrite(block_id)
        if not result.success:
            # Build comprehensive error message with diagnostics
            error_msg = result.message or "Failed to pull data from upstream"
            
            # Extract diagnostic information if available
            diagnostics = None
            if hasattr(result, 'data') and result.data:
                diagnostics = result.data.get('diagnostics')
                missing_upstream = result.data.get('missing_upstream')
                block_name = result.data.get('block_name')
                
                if diagnostics:
                    error_msg = f"{error_msg}\n\nDetailed Diagnostics:\n{diagnostics}"
                elif missing_upstream:
                    # Fallback: format missing_upstream if diagnostics not available
                    error_msg = f"{error_msg}\n\nMissing upstream connections:"
                    for missing in missing_upstream:
                        error_msg += f"\n  - {missing.get('source_block_id', 'unknown')}.{missing.get('source_output_name', 'unknown')} -> {missing.get('target_input_name', 'unknown')}"
                        if 'reason' in missing:
                            error_msg += f" (reason: {missing['reason']})"
            
            # Log the full error for debugging
            Log.error(f"Bootstrap: Pull callback failed for block {block_id}:\n{error_msg}")
            
            # Raise exception so execution engine knows pull failed
            raise RuntimeError(error_msg)
        return result
    
    execution_engine._pull_callback = pull_callback
    Log.info("Execution engine pull_callback configured")
    
    Log.info("Application facade initialized")
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    # Module 7: Container Assembly
    if progress_tracker:
        progress_tracker.start_module("Container", "Assembling service container", 1)
        _process_qt_events()
    
    # Create and return container
    if progress_tracker:
        progress_tracker.update_step("Creating service container")
        _process_qt_events()
    container = ServiceContainer(
        database=database,
        event_bus=event_bus,
        project_repo=project_repo,
        block_repo=block_repo,
        connection_repo=connection_repo,
        project_service=project_service,
        block_service=block_service,
        connection_service=connection_service,
        execution_engine=execution_engine,
        data_item_repo=data_item_repo,
        block_local_state_repo=block_local_state_repo,
        ui_state_repo=ui_state_repo,
        layer_order_repo=layer_order_repo,
        layer_group_order_service=layer_group_order_service,
        preferences_repo=preferences_repo,
        session_state_repo=session_state_repo,
        app_settings_manager=app_settings_manager,
        facade=facade,
        recent_store=recent_store,
        data_state_service=data_state_service,
        block_status_service=block_status_service,
        data_filter_manager=data_filter_manager,
        expected_outputs_service=expected_outputs_service,
        setlist_service=setlist_service,
        action_set_repo=action_set_repo,
        action_item_repo=action_item_repo,
        ma3_communication_service=ma3_communication_service,
        show_manager_listener_service=show_manager_listener_service,
        show_manager_state_service=None,  # Will be initialized after facade
        sync_port=sync_port
    )

    container.layer_order_service = layer_order_service
    facade.layer_order_service = layer_order_service
    container.layer_group_order_service = layer_group_order_service
    facade.layer_group_order_service = layer_group_order_service
    
    Log.info("Service container created successfully")
    
    # Initialize ShowManager state service (needs facade)
    if progress_tracker:
        progress_tracker.update_step("ShowManager state service")
        _process_qt_events()
    from src.features.show_manager.application.show_manager_state_service import ShowManagerStateService
    show_manager_state_service = ShowManagerStateService(event_bus=event_bus, facade=facade)
    container.show_manager_state_service = show_manager_state_service
    # Update facade to reference both services (facade was created before services)
    facade.show_manager_state_service = show_manager_state_service
    facade.show_manager_listener_service = show_manager_listener_service
    facade.ma3_communication_service = ma3_communication_service
    facade.sync_port = sync_port
    Log.info("Bootstrap: ShowManager services attached to facade")
    
    # Register cleanup handler for graceful shutdown (atexit ensures cleanup even on unexpected exits)
    # Capture container in closure for atexit handler
    def register_cleanup(container_ref):
        def cleanup_handler():
            try:
                if container_ref:
                    container_ref.cleanup()
            except Exception as e:
                Log.warning(f"Bootstrap: Error in atexit cleanup handler: {e}")
        atexit.register(cleanup_handler)
    
    register_cleanup(container)
    
    if progress_tracker:
        progress_tracker.complete_module()
        _process_qt_events()
    
    return container


# Migration code removed - use one-time conversion script if needed

