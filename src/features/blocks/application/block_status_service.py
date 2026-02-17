"""
Block Status Service

Evaluates block status levels and conditions to determine current block status.
Blocks define their own status levels via BlockProcessor.get_status_levels().

Unified status service for all blocks. Publishes StatusChanged events when status
actually changes, ensuring all status indicators stay in sync.
"""
from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING
from src.features.blocks.domain.block import Block
from src.features.blocks.domain.block_status import BlockStatus, BlockStatusLevel
from src.features.blocks.domain.block_repository import BlockRepository
from src.application.events.event_bus import EventBus
from src.application.events.events import BlockChanged, StatusChanged
from src.utils.message import Log

if TYPE_CHECKING:
    from src.application.api.application_facade import ApplicationFacade
    from src.shared.application.services.panel_state_provider import PanelStateProvider


class BlockStatusService:
    """
    Service for evaluating block status levels.
    
    Evaluates status levels defined by blocks in priority order (ascending).
    The first level where ANY condition is False becomes the active status.
    If all levels pass, returns the highest priority level.
    """
    
    def __init__(
        self,
        block_repo: BlockRepository,
        event_bus: EventBus,
        data_state_service: Optional[Any] = None,
        facade_provider: Optional[Callable[[], "ApplicationFacade"]] = None
    ):
        """
        Initialize block status service.
        
        Args:
            block_repo: Repository for accessing blocks
            event_bus: Event bus for subscribing to block changes and publishing status changes
            data_state_service: Optional DataStateService for fallback when blocks don't implement get_status_levels()
            facade_provider: Optional callable that returns ApplicationFacade (for active status recalculation)
        """
        self._block_repo = block_repo
        self._event_bus = event_bus
        self._data_state_service = data_state_service
        self._facade_provider = facade_provider
        self._status_cache: Dict[str, BlockStatus] = {}
        self._previous_status_cache: Dict[str, BlockStatus] = {}  # Track previous status for change detection
        # Panel state providers: block_id -> PanelStateProvider instance
        self._panel_state_providers: Dict[str, "PanelStateProvider"] = {}
        
        # Subscribe to block changes to invalidate cache and recalculate status
        self._event_bus.subscribe("BlockChanged", self._on_block_changed)
        # Also subscribe to connection changes - connections affect block status (e.g., ShowManager)
        from src.application.events import ConnectionCreated, ConnectionRemoved
        self._event_bus.subscribe("ConnectionCreated", self._on_connection_changed)
        self._event_bus.subscribe("ConnectionRemoved", self._on_connection_changed)
        
        Log.info("BlockStatusService: Initialized")
    
    def register_panel_state_provider(self, block_id: str, provider: "PanelStateProvider"):
        """
        Register a panel state provider for a block.
        
        Panels call this when opened to make their state available to status conditions.
        
        Args:
            block_id: Block identifier
            provider: Panel instance that implements PanelStateProvider protocol
        """
        self._panel_state_providers[block_id] = provider
        Log.debug(f"BlockStatusService: Registered panel state provider for block {block_id}")
        
        # Invalidate cache and recalculate status since panel state is now available
        if block_id in self._status_cache:
            del self._status_cache[block_id]
        if self._facade_provider:
            try:
                facade = self._facade_provider()
                if facade:
                    self.get_block_status(block_id, facade, force_recalculate=True)
            except Exception as e:
                Log.debug(f"BlockStatusService: Could not recalculate status after panel registration: {e}")
    
    def unregister_panel_state_provider(self, block_id: str):
        """
        Unregister a panel state provider for a block.
        
        Panels call this when closed to clean up.
        
        Args:
            block_id: Block identifier
        """
        if block_id in self._panel_state_providers:
            del self._panel_state_providers[block_id]
            Log.debug(f"BlockStatusService: Unregistered panel state provider for block {block_id}")
            
            # Invalidate cache since panel state is no longer available
            if block_id in self._status_cache:
                del self._status_cache[block_id]
    
    def get_panel_state(self, block_id: str) -> Optional[Dict[str, Any]]:
        """
        Get panel state for a block.
        
        Status conditions can use this to access panel internal state
        (e.g., listener status, connection state) that isn't in block metadata.
        
        Args:
            block_id: Block identifier
            
        Returns:
            Dictionary of panel state, or None if no panel is registered
        """
        provider = self._panel_state_providers.get(block_id)
        if provider:
            try:
                return provider.get_panel_state()
            except Exception as e:
                Log.warning(f"BlockStatusService: Error getting panel state for block {block_id}: {e}")
                return None
        return None
    
    def get_block_status(
        self,
        block_id: str,
        facade: "ApplicationFacade",
        force_recalculate: bool = False
    ) -> BlockStatus:
        """
        Get current status for a block.
        
        Evaluates status levels in ascending priority order (0, 1, 2...).
        For each level, checks ALL conditions.
        Returns first level where ANY condition is False.
        If all levels pass, returns highest priority level.
        
        Args:
            block_id: Block identifier
            facade: ApplicationFacade for processor access
            force_recalculate: If True, bypass cache and recalculate (default: False)
            
        Returns:
            BlockStatus representing the current status
        """
        # Check cache first (unless forcing recalculation)
        if not force_recalculate and block_id in self._status_cache:
            return self._status_cache[block_id]
        
        # Get block
        block = self._block_repo.get_by_id(block_id)
        if not block:
            # Return default status for missing block
            default_level = BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[]
            )
            return BlockStatus(level=default_level, message="Block not found")
        
        # Get processor via facade's execution engine
        if not hasattr(facade, 'execution_engine') or not facade.execution_engine:
            # Return default status for blocks without execution engine
            default_level = BlockStatusLevel(
                priority=0,
                name="unknown",
                display_name="Unknown",
                color="#999999",
                conditions=[]
            )
            return BlockStatus(level=default_level, message="Execution engine not available")
        
        processor = facade.execution_engine.get_processor(block)
        if not processor:
            # Return default status for blocks without processors
            default_level = BlockStatusLevel(
                priority=0,
                name="unknown",
                display_name="Unknown",
                color="#999999",
                conditions=[]
            )
            return BlockStatus(level=default_level, message="No processor for block type")
        
        # Get status levels from processor
        try:
            status_levels = processor.get_status_levels(block, facade)
        except Exception as e:
            Log.error(f"BlockStatusService: Error getting status levels for block {block_id}: {e}")
            default_level = BlockStatusLevel(
                priority=0,
                name="error",
                display_name="Error",
                color="#ff6b6b",
                conditions=[]
            )
            status = BlockStatus(level=default_level, message=f"Error evaluating status: {e}")
            self._update_status_and_publish(block_id, status, facade)
            return status
        
        if not status_levels:
            # No status levels defined - try fallback to DataStateService if available
            if self._data_state_service:
                try:
                    # Get project_id from facade if available
                    project_id = getattr(facade, 'current_project_id', None) if hasattr(facade, 'current_project_id') else None
                    data_state = self._data_state_service.get_block_data_state(block_id, project_id)
                    
                    # Convert DataState to BlockStatus
                    status = self._data_state_to_status(data_state)
                    self._update_status_and_publish(block_id, status, facade)
                    return status
                except Exception as e:
                    Log.debug(f"BlockStatusService: Fallback to DataStateService failed for {block_id}: {e}")
            
            # No fallback available - return default ready status
            default_level = BlockStatusLevel(
                priority=0,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
            status = BlockStatus(level=default_level)
            self._update_status_and_publish(block_id, status, facade)
            return status
        
        # Sort by priority (ascending - lower priority numbers first)
        sorted_levels = sorted(status_levels, key=lambda level: level.priority)
        
        # Evaluate levels in priority order
        for level in sorted_levels:
            # Check ALL conditions for this level
            all_conditions_pass = True
            for condition in level.conditions:
                try:
                    if not condition(block, facade):
                        # Any condition fails - this level is active
                        status = BlockStatus(level=level)
                        self._update_status_and_publish(block_id, status, facade)
                        return status
                except Exception as e:
                    Log.warning(f"BlockStatusService: Error evaluating condition for level '{level.name}': {e}")
                    # Condition error - treat as failure
                    status = BlockStatus(level=level, message=f"Error evaluating condition: {e}")
                    self._update_status_and_publish(block_id, status, facade)
                    return status
            
            # All conditions passed for this level - continue to next
        
        # All levels passed - return highest priority (last level)
        highest_priority_level = sorted_levels[-1]
        status = BlockStatus(level=highest_priority_level)
        self._update_status_and_publish(block_id, status, facade)
        return status
    
    def get_block_status_diagnostics(
        self,
        block_id: str,
        facade: "ApplicationFacade"
    ) -> Dict[str, Any]:
        """
        Get detailed diagnostic information about block status evaluation.
        
        Returns information about which conditions passed/failed for each status level,
        helping to understand why a specific status level was chosen.
        
        Args:
            block_id: Block identifier
            facade: ApplicationFacade for processor access
            
        Returns:
            Dict with diagnostic information:
            - current_status: The active status level
            - levels: List of level evaluations with condition results
        """
        from typing import Dict, Any, List
        
        # Get block
        block = self._block_repo.get_by_id(block_id)
        if not block:
            return {
                "current_status": None,
                "error": "Block not found",
                "levels": []
            }
        
        # Get processor
        if not hasattr(facade, 'execution_engine') or not facade.execution_engine:
            return {
                "current_status": None,
                "error": "Execution engine not available",
                "levels": []
            }
        
        processor = facade.execution_engine.get_processor(block)
        if not processor:
            return {
                "current_status": None,
                "error": "No processor for block type",
                "levels": []
            }
        
        # Get status levels
        try:
            status_levels = processor.get_status_levels(block, facade)
        except Exception as e:
            return {
                "current_status": None,
                "error": f"Error getting status levels: {e}",
                "levels": []
            }
        
        if not status_levels:
            return {
                "current_status": None,
                "error": "No status levels defined",
                "levels": []
            }
        
        # Sort by priority
        sorted_levels = sorted(status_levels, key=lambda level: level.priority)
        
        # Evaluate each level and collect diagnostic info
        level_diagnostics: List[Dict[str, Any]] = []
        active_status = None
        
        for level in sorted_levels:
            condition_results = []
            all_conditions_pass = True
            failed_conditions = []
            
            for i, condition in enumerate(level.conditions):
                try:
                    result = condition(block, facade)
                    condition_name = getattr(condition, '__name__', f'Condition {i+1}')
                    
                    # Get detailed failure reason for specific conditions
                    failure_reason = None
                    if not result:
                        failure_reason = self._get_condition_failure_reason(
                            condition_name, block, facade
                        )
                    
                    condition_results.append({
                        "index": i,
                        "passed": result,
                        "error": None,
                        "failure_reason": failure_reason
                    })
                    if not result:
                        all_conditions_pass = False
                        # Include detailed reason in failed conditions list
                        if failure_reason:
                            failed_conditions.append(f"{condition_name}: {failure_reason}")
                        else:
                            failed_conditions.append(condition_name)
                except Exception as e:
                    condition_results.append({
                        "index": i,
                        "passed": False,
                        "error": str(e),
                        "failure_reason": None
                    })
                    all_conditions_pass = False
                    condition_name = getattr(condition, '__name__', f'Condition {i+1}')
                    failed_conditions.append(f"{condition_name} (error: {e})")
            
            # Determine if this level is active
            is_active = not all_conditions_pass
            
            # Get actionable guidance for failed conditions
            actionable_guidance = []
            if not all_conditions_pass and failed_conditions:
                actionable_guidance = self._get_actionable_guidance(block_id, level, failed_conditions, facade)
            
            level_diagnostics.append({
                "level": {
                    "priority": level.priority,
                    "name": level.name,
                    "display_name": level.display_name,
                    "color": level.color
                },
                "all_conditions_pass": all_conditions_pass,
                "is_active": is_active,
                "condition_count": len(level.conditions),
                "condition_results": condition_results,
                "failed_conditions": failed_conditions,
                "reason": self._get_level_reason(level, all_conditions_pass, failed_conditions),
                "actionable_guidance": actionable_guidance
            })
            
            # If this level is active, it's the current status
            if is_active and active_status is None:
                active_status = BlockStatus(level=level)
        
        # If no level was active, highest priority is active
        if active_status is None:
            highest_priority_level = sorted_levels[-1]
            active_status = BlockStatus(level=highest_priority_level)
        
        return {
            "current_status": {
                "priority": active_status.level.priority,
                "name": active_status.level.name,
                "display_name": active_status.level.display_name,
                "color": active_status.level.color,
                "message": active_status.message
            },
            "levels": level_diagnostics
        }
    
    def _get_level_reason(
        self,
        level: BlockStatusLevel,
        all_conditions_pass: bool,
        failed_conditions: List[str]
    ) -> str:
        """Generate human-readable reason for level evaluation"""
        if not level.conditions:
            return "No conditions (always passes)"
        
        if all_conditions_pass:
            return f"All {len(level.conditions)} condition(s) passed"
        else:
            if len(failed_conditions) == 1:
                return f"Failed: {failed_conditions[0]}"
            else:
                return f"Failed {len(failed_conditions)} condition(s): {', '.join(failed_conditions[:3])}" + \
                       (f" and {len(failed_conditions) - 3} more" if len(failed_conditions) > 3 else "")
    
    def _get_condition_failure_reason(
        self,
        condition_name: str,
        block: Block,
        facade: "ApplicationFacade"
    ) -> Optional[str]:
        """
        Get detailed failure reason for a specific condition.
        
        Returns a human-readable explanation of why the condition failed.
        """
        import os
        from src.shared.domain.data_state import DataState
        
        # ShowManager-specific condition
        if condition_name == "check_not_fully_connected" and block.type == "ShowManager":
            # Detailed check for ShowManager connection status
            has_editor_connection = False
            editor_block_name = None
            if hasattr(facade, 'connection_service'):
                connections = facade.connection_service.list_connections_by_block(block.id)
                for conn in connections:
                    if (conn.source_block_id == block.id and conn.source_output_name == "manipulator") or \
                       (conn.target_block_id == block.id and conn.target_input_name == "manipulator"):
                        other_block_id = conn.target_block_id if conn.source_block_id == block.id else conn.source_block_id
                        other_result = facade.describe_block(other_block_id)
                        if other_result.success and other_result.data:
                            if other_result.data.type == "Editor":
                                has_editor_connection = True
                                editor_block_name = other_result.data.name
                                break
            
            panel_state = facade.get_panel_state(block.id) if hasattr(facade, 'get_panel_state') else None
            metadata = block.metadata or {}
            ma3_ip = metadata.get("ma3_ip", "").strip()
            ma3_port = metadata.get("ma3_port", 0)
            
            failures = []
            
            if not has_editor_connection:
                failures.append("No Editor block connected via manipulator port")
            
            if panel_state:
                listening = panel_state.get("listening", False)
                ma3_configured = bool(ma3_ip and ma3_port > 0)
                
                if not ma3_configured:
                    if not ma3_ip:
                        failures.append("MA3 IP not configured")
                    if not ma3_port:
                        failures.append("MA3 Port not configured")
                
                if not listening:
                    failures.append("MA3 listener not started")
            else:
                listen_port = metadata.get("listen_port", 0)
                osc_listener_active = metadata.get("osc_listener_active", False)
                
                if not ma3_ip:
                    failures.append("MA3 IP not configured")
                if not ma3_port:
                    failures.append("MA3 Port not configured")
                if not listen_port and not osc_listener_active:
                    failures.append("MA3 listener not active (panel closed, check listen_port or osc_listener_active in metadata)")
            
            # Return the failures
            if failures:
                return "; ".join(failures)
            else:
                return "Unknown reason (all checks passed but condition still failed)"
        
        # Connection input checks
        if condition_name in ["check_audio_input", "check_events_input", "check_audio_or_events_input"]:
            if not hasattr(facade, 'connection_service') or not facade.connection_service:
                return "Connection service not available"
            
            connections = facade.connection_service.list_connections_by_block(block.id)
            incoming = [c for c in connections if c.target_block_id == block.id]
            
            if condition_name == "check_audio_input":
                audio_connections = [c for c in incoming if c.target_input_name == "audio"]
                if not audio_connections:
                    return "No audio input connected to 'audio' port"
                else:
                    return f"Audio input connected from {len(audio_connections)} source(s)"
            
            elif condition_name == "check_events_input":
                events_connections = [c for c in incoming if c.target_input_name == "events"]
                if not events_connections:
                    return "No events input connected to 'events' port"
                else:
                    return f"Events input connected from {len(events_connections)} source(s)"
            
            elif condition_name == "check_audio_or_events_input":
                audio_connections = [c for c in incoming if c.target_input_name == "audio"]
                events_connections = [c for c in incoming if c.target_input_name == "events"]
                if not audio_connections and not events_connections:
                    return "No audio or events input connected (connect at least one)"
                parts = []
                if audio_connections:
                    parts.append(f"audio: {len(audio_connections)} source(s)")
                if events_connections:
                    parts.append(f"events: {len(events_connections)} source(s)")
                return "; ".join(parts)
        
        # Generic input check
        if condition_name == "check_has_inputs":
            if not hasattr(facade, 'connection_service') or not facade.connection_service:
                return "Connection service not available"
            
            connections = facade.connection_service.list_connections_by_block(block.id)
            incoming = [c for c in connections if c.target_block_id == block.id]
            if not incoming:
                return "No inputs connected"
            else:
                input_names = [c.target_input_name for c in incoming]
                return f"Inputs connected: {', '.join(set(input_names))}"
        
        # File path checks
        if condition_name == "check_file_path":
            audio_path = block.metadata.get("audio_path")
            if not audio_path:
                return "Audio file path not configured in block settings"
            if not os.path.exists(audio_path):
                return f"Audio file does not exist: {audio_path}"
            return f"Audio file exists: {audio_path}"
        
        # Model path checks
        if condition_name == "check_model_path":
            model_path = block.metadata.get("model_path")
            if not model_path:
                return "Model path not configured in block settings"
            if not os.path.exists(model_path):
                return f"Model file does not exist: {model_path}"
            return f"Model file exists: {model_path}"
        
        # Output directory checks
        if condition_name == "check_output_dir":
            output_dir = block.metadata.get("output_dir")
            if not output_dir:
                return "Output directory not configured in block settings"
            if not os.path.exists(output_dir):
                return f"Output directory does not exist: {output_dir}"
            if not os.access(output_dir, os.W_OK):
                return f"Output directory is not writable: {output_dir}"
            return f"Output directory is valid: {output_dir}"
        
        # Data directory checks
        if condition_name == "check_data_dir":
            from src.utils.datasets import resolve_dataset_path

            data_dir = resolve_dataset_path(block.metadata.get("data_dir"))
            if not data_dir:
                return "Data directory not configured in block settings"
            if not os.path.exists(data_dir):
                return f"Data directory does not exist: {data_dir}"
            if not os.path.isdir(data_dir):
                return f"Data path is not a directory: {data_dir}"
            return f"Data directory exists: {data_dir}"
        
        # Library availability checks
        if condition_name == "check_librosa_available":
            try:
                import librosa
                return "librosa is available"
            except ImportError:
                return "librosa is not installed (install with: pip install librosa)"
        
        if condition_name == "check_demucs_available":
            try:
                import demucs
                return "demucs is available"
            except ImportError:
                return "demucs is not installed (install with: pip install demucs)"
        
        if condition_name == "check_basicpitch_available":
            try:
                import basic_pitch
                return "basic-pitch is available"
            except ImportError:
                return "basic-pitch is not installed (install with: pip install basic-pitch)"
        
        if condition_name == "check_pytorch_available":
            try:
                import torch
                return "PyTorch is available"
            except ImportError:
                return "PyTorch is not installed (install with: pip install torch)"
        
        if condition_name == "check_tensorflow_available":
            try:
                import tensorflow
                return "TensorFlow is available"
            except ImportError:
                return "TensorFlow is not installed (install with: pip install tensorflow)"
        
        # Data freshness checks
        if condition_name == "check_data_fresh":
            if not hasattr(facade, 'data_state_service') or not facade.data_state_service:
                return "Data state service not available (assuming fresh)"
            
            try:
                project_id = getattr(facade, 'current_project_id', None) if hasattr(facade, 'current_project_id') else None
                data_state = facade.data_state_service.get_block_data_state(block.id, project_id)
                if data_state == DataState.STALE:
                    return "Data is stale (needs re-execution of upstream blocks)"
                else:
                    return f"Data is {data_state.value}"
            except Exception as e:
                return f"Error checking data state: {e}"
        
        return None
    
    def _get_actionable_guidance(
        self,
        block_id: str,
        level: BlockStatusLevel,
        failed_conditions: List[str],
        facade: "ApplicationFacade"
    ) -> List[str]:
        """
        Get actionable guidance for fixing failed conditions.
        
        Returns a list of steps the user can take to fix the status.
        """
        guidance = []
        
        # Get block to check type
        block = self._block_repo.get_by_id(block_id)
        if not block:
            return guidance
        
        block_type = block.type
        
        # ShowManager-specific guidance
        if block_type == "ShowManager":
            if "check_not_fully_connected" in failed_conditions:
                # Check what's actually missing
                has_editor_connection = False
                if hasattr(facade, 'connection_service'):
                    connections = facade.connection_service.list_connections_by_block(block_id)
                    for conn in connections:
                        if (conn.source_block_id == block_id and conn.source_output_name == "manipulator") or \
                           (conn.target_block_id == block_id and conn.target_input_name == "manipulator"):
                            other_block_id = conn.target_block_id if conn.source_block_id == block_id else conn.source_block_id
                            other_result = facade.describe_block(other_block_id)
                            if other_result.success and other_result.data and other_result.data.type == "Editor":
                                has_editor_connection = True
                                break
                
                panel_state = facade.get_panel_state(block_id) if hasattr(facade, 'get_panel_state') else None
                metadata = block.metadata or {}
                ma3_ip = metadata.get("ma3_ip", "").strip()
                ma3_port = metadata.get("ma3_port", 0)
                
                if not has_editor_connection:
                    guidance.append("1. Connect an Editor block:")
                    guidance.append("   • In the node editor, connect an Editor block to this ShowManager")
                    guidance.append("   • Use the 'manipulator' port (bidirectional connection)")
                
                if not ma3_ip or not ma3_port:
                    guidance.append("2. Configure MA3 OSC settings:")
                    guidance.append("   • Open the ShowManager panel")
                    guidance.append("   • Set MA3 IP address (e.g., 127.0.0.1)")
                    guidance.append("   • Set MA3 Port (e.g., 8001)")
                    guidance.append("   • Save settings")
                
                if panel_state:
                    listening = panel_state.get("listening", False)
                    if not listening:
                        guidance.append("3. Start the MA3 listener:")
                        guidance.append("   • In the ShowManager panel, click 'Start Listening'")
                        guidance.append("   • The listener must be active for full connection")
                else:
                    listen_port = metadata.get("listen_port", 0)
                    if not listen_port:
                        guidance.append("3. Start the MA3 listener:")
                        guidance.append("   • Open the ShowManager panel")
                        guidance.append("   • Click 'Start Listening' to activate the listener")
        
        # Generic guidance for other conditions
        if not guidance:
            for failed_cond in failed_conditions[:3]:  # Limit to first 3
                cond_name = failed_cond.split("(")[0].strip() if "(" in failed_cond else failed_cond
                
                # Common condition patterns
                if "input" in cond_name.lower() or "connection" in cond_name.lower():
                    guidance.append(f"• Check that required inputs are connected")
                elif "file" in cond_name.lower() or "path" in cond_name.lower():
                    guidance.append(f"• Configure the file path in the block settings")
                elif "data" in cond_name.lower() or "stale" in cond_name.lower():
                    guidance.append(f"• Execute upstream blocks to refresh data")
                elif "configured" in cond_name.lower() or "settings" in cond_name.lower():
                    guidance.append(f"• Configure required settings in the block panel")
                else:
                    guidance.append(f"• Fix: {cond_name}")
        
        return guidance
    
    def _update_status_and_publish(
        self,
        block_id: str,
        new_status: BlockStatus,
        facade: "ApplicationFacade"
    ) -> None:
        """
        Update status cache and publish StatusChanged event if status actually changed.
        
        Args:
            block_id: Block identifier
            new_status: Newly calculated status
            facade: ApplicationFacade for project_id access
        """
        # Get previous status for comparison
        # Use _status_cache as the source of truth for "previous" status
        previous_status = self._status_cache.get(block_id)
        
        # Check if status actually changed
        status_changed = (
            previous_status is None or
            previous_status.level.name != new_status.level.name or
            previous_status.level.color != new_status.level.color or
            previous_status.message != new_status.message
        )
        
        # Update caches
        # Store old current status in previous cache for next comparison
        if previous_status:
            self._previous_status_cache[block_id] = previous_status
        # Store new status as current
        self._status_cache[block_id] = new_status
        
        # Publish StatusChanged event if status actually changed
        if status_changed:
            # Get project_id from facade if available
            project_id = getattr(facade, 'current_project_id', None) if hasattr(facade, 'current_project_id') else None
            
            # Serialize status for event (convert to dict)
            status_dict = {
                'level': {
                    'priority': new_status.level.priority,
                    'name': new_status.level.name,
                    'display_name': new_status.level.display_name,
                    'color': new_status.level.color,
                },
                'message': new_status.message
            }
            
            previous_status_dict = None
            if previous_status:
                previous_status_dict = {
                    'level': {
                        'priority': previous_status.level.priority,
                        'name': previous_status.level.name,
                        'display_name': previous_status.level.display_name,
                        'color': previous_status.level.color,
                    },
                    'message': previous_status.message
                }
            
            self._event_bus.publish(StatusChanged(
                project_id=project_id,
                data={
                    'block_id': block_id,
                    'status': status_dict,
                    'previous_status': previous_status_dict
                }
            ))
            Log.debug(
                f"BlockStatusService: Published StatusChanged for block {block_id}: "
                f"{previous_status.level.name if previous_status else 'None'} -> {new_status.level.name}"
            )
    
    def _data_state_to_status(self, data_state) -> BlockStatus:
        """
        Convert DataState to BlockStatus for fallback support.
        
        Args:
            data_state: DataState enum value
            
        Returns:
            BlockStatus equivalent
        """
        from src.shared.domain.data_state import DataState
        
        if data_state == DataState.NO_DATA:
            level = BlockStatusLevel(
                priority=0,
                name="no_data",
                display_name="No Data",
                color=data_state.color,
                conditions=[]
            )
        elif data_state == DataState.STALE:
            level = BlockStatusLevel(
                priority=1,
                name="stale",
                display_name="Stale",
                color=data_state.color,
                conditions=[]
            )
        else:  # FRESH
            level = BlockStatusLevel(
                priority=2,
                name="fresh",
                display_name="Fresh",
                color=data_state.color,
                conditions=[]
            )
        
        return BlockStatus(level=level)
    
    def _on_block_changed(self, event: BlockChanged):
        """
        Handle block changed event - invalidate cache and actively recalculate status.
        
        When a block changes, we need to recalculate its status and publish
        StatusChanged if it changed. This ensures status stays in sync.
        """
        if not event.data:
            return
            
        block_id = event.data.get('block_id') or event.data.get('id')
        if not block_id:
            return
        
        # Invalidate both caches to force full recalculation
        # This ensures next get_block_status() call will recalculate and compare properly
        old_status = self._status_cache.get(block_id)
        if block_id in self._status_cache:
            del self._status_cache[block_id]
        if block_id in self._previous_status_cache:
            del self._previous_status_cache[block_id]
        Log.debug(f"BlockStatusService: Invalidated status cache for block {block_id}")
        
        # If we had a previous status, restore it to previous cache so comparison works
        # This ensures we can detect if status actually changed
        if old_status:
            self._previous_status_cache[block_id] = old_status
        
        # Actively recalculate status if we have a facade provider
        # This ensures StatusChanged events are published immediately
        if self._facade_provider:
            try:
                facade = self._facade_provider()
                if facade:
                    # Recalculate status (this will publish StatusChanged if it changed)
                    status = self.get_block_status(block_id, facade)
            except Exception as e:
                Log.debug(f"BlockStatusService: Could not recalculate status for {block_id}: {e}")
        else:
            # No facade provider - status will be recalculated on next get_block_status() call
            # Status dots will trigger this via their BlockChanged handlers
            Log.debug(f"BlockStatusService: No facade provider, status will be recalculated on next get_block_status() call")
    
    def _on_connection_changed(self, event):
        """
        Handle connection change event - invalidate cache and recalculate status for affected blocks.
        
        Connections affect block status (e.g., ShowManager needs connections,
        other blocks may check if inputs are connected).
        """
        if not event.data:
            return
        
        # Invalidate cache for both source and target blocks (connections affect status)
        source_block_id = event.data.get('source_block_id')
        target_block_id = event.data.get('target_block_id')
        
        # Process source block
        if source_block_id:
            old_status = self._status_cache.get(source_block_id)
            if source_block_id in self._status_cache:
                del self._status_cache[source_block_id]
            if source_block_id in self._previous_status_cache:
                del self._previous_status_cache[source_block_id]
            # Preserve old status for comparison
            if old_status:
                self._previous_status_cache[source_block_id] = old_status
            Log.debug(f"BlockStatusService: Invalidated status cache for source block {source_block_id} (connection changed)")
            
            # Actively recalculate if facade provider available
            if self._facade_provider:
                try:
                    facade = self._facade_provider()
                    if facade:
                        self.get_block_status(source_block_id, facade)
                except Exception as e:
                    Log.debug(f"BlockStatusService: Could not recalculate status for source block {source_block_id}: {e}")
        
        # Process target block
        if target_block_id:
            old_status = self._status_cache.get(target_block_id)
            if target_block_id in self._status_cache:
                del self._status_cache[target_block_id]
            if target_block_id in self._previous_status_cache:
                del self._previous_status_cache[target_block_id]
            # Preserve old status for comparison
            if old_status:
                self._previous_status_cache[target_block_id] = old_status
            Log.debug(f"BlockStatusService: Invalidated status cache for target block {target_block_id} (connection changed)")
            
            # Actively recalculate if facade provider available
            if self._facade_provider:
                try:
                    facade = self._facade_provider()
                    if facade:
                        self.get_block_status(target_block_id, facade)
                except Exception as e:
                    Log.debug(f"BlockStatusService: Could not recalculate status for target block {target_block_id}: {e}")
