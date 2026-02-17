"""
Editor Block Processor

Processes Editor blocks - audio editor with timeline visualization.
This block takes audio and events as input, allows visualization and editing,
and passes through both the audio data and events.

Phase 3: Updated to use owned data (block_id = Editor block.id).
"""
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from src.application.processing.block_processor import BlockProcessor, ProcessingError
from src.features.blocks.domain import Block
from src.shared.domain.entities import DataItem, AudioDataItem
from src.shared.domain.entities import EventDataItem
from src.shared.domain.value_objects.port_type import MANIPULATOR_TYPE
from src.shared.application.services.event_filter_manager import EventFilterManager
from src.application.blocks import register_processor_class
from src.utils.message import Log

if TYPE_CHECKING:
    from src.features.blocks.domain import BlockStatusLevel
    from src.application.api.application_facade import ApplicationFacade


class EditorBlockProcessor(BlockProcessor):
    """
    Processor for Editor block type.
    
    The Editor block is a visualization/editing block that:
    - Takes audio and events as input
    - Displays them on a timeline for visualization
    - Allows event editing (move, resize, delete)
    - Passes through audio data and outputs EventDataItems owned by this block
    
    The processor creates owned EventDataItems (block_id = Editor block.id) from inputs.
    The execution engine ensures outputs have correct block_id. The UI loads these
    execution outputs to display and edit.
    """
    
    def can_process(self, block: Block) -> bool:
        return block.type == "Editor"
    
    def get_block_type(self) -> str:
        return "Editor"
    
    # =========================================================================
    # Layer Classification Helpers
    # =========================================================================
    
    def _is_sync_layer_item(self, item: EventDataItem) -> bool:
        """
        Check if EventDataItem is a MA3 sync layer.
        
        Sync layers are identified by:
        - Metadata flag: _synced_from_ma3 = True
        - Name pattern: contains "ma3_sync" (case-insensitive)
        
        Args:
            item: EventDataItem to check
            
        Returns:
            True if item is a sync layer, False otherwise
        """
        if not isinstance(item, EventDataItem):
            return False
        
        # Check explicit metadata flags
        if item.metadata.get("_synced_from_ma3") is True:
            return True
        if item.metadata.get("_synced_to_ma3") is True:
            return True
        if item.metadata.get("_sync_type") in {"showmanager_layer", "ez_layer"}:
            return True
        if item.metadata.get("sync_type") in {"showmanager_layer", "ez_layer"}:
            return True

        # Check source-based metadata
        if item.metadata.get("source") in {"ma3", "ma3_sync"}:
            return True
        
        # Check name pattern (legacy fallback)
        if item.name and "ma3_sync" in item.name.lower():
            return True
        
        return False
    
    def _is_ez_layer_item(self, item: EventDataItem) -> bool:
        """
        Check if EventDataItem is an EchoZero layer (from upstream blocks).
        
        EZ layers are all EventDataItems that are NOT sync layers.
        
        Args:
            item: EventDataItem to check
            
        Returns:
            True if item is an EZ layer, False otherwise
        """
        return not self._is_sync_layer_item(item)
    
    # =========================================================================
    # STEP 1: Clear Local Data (override default to preserve synced items)
    # =========================================================================
    
    def step_clear_local_data(
        self,
        block: Block,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        STEP 1: Clear old owned data and UI state.
        
        Editor-specific behavior:
        - Delete old owned EventDataItems (PRESERVE synced items from MA3)
        - Clear non-synced UI layer state
        - Store preserved state in block.metadata for later steps
        
        Args:
            block: Block entity being executed
            metadata: Processing metadata with repos/services
        """
        Log.info(f"EditorBlockProcessor: [STEP 1] Clearing old data for '{block.name}'")
        metadata = metadata or {}
        
        data_item_repo = metadata.get("data_item_repo")
        ui_state_service = metadata.get("ui_state_service")
        
        # Clear old owned EventDataItems (preserve synced)
        synced_layer_items = []
        if data_item_repo:
            try:
                owned_items = data_item_repo.list_by_block(block.id)
                deleted_count = 0
                for item in owned_items:
                    if isinstance(item, EventDataItem) and item.metadata.get("output_port") == "events":
                        # Use robust sync layer detection
                        if self._is_sync_layer_item(item):
                            synced_layer_items.append(item)
                            Log.debug(f"EditorBlockProcessor: Preserving synced layer '{item.name}'")
                            continue
                        # Delete EZ layer items
                        try:
                            data_item_repo.delete(item.id)
                            deleted_count += 1
                        except Exception as e:
                            Log.warning(f"EditorBlockProcessor: Failed to delete old item: {e}")
                if deleted_count > 0:
                    Log.info(f"EditorBlockProcessor: Deleted {deleted_count} old EZ layer EventDataItem(s)")
                if synced_layer_items:
                    Log.info(f"EditorBlockProcessor: Preserved {len(synced_layer_items)} sync layer EventDataItem(s)")
            except Exception as e:
                Log.warning(f"EditorBlockProcessor: Failed to clear old owned items: {e}")
        
        # Clear non-synced UI layer state (preserve sync layers with full state)
        synced_layers_to_preserve = []
        if ui_state_service:
            try:
                current_state = ui_state_service.get_state('editor_layers', block.id)
                if current_state and current_state.get('layers'):
                    all_layers = current_state.get('layers', [])
                    # Preserve all sync layer entries with their full state (visibility, locked, height, color, etc.)
                    synced_layers_to_preserve = [
                        layer.copy() for layer in all_layers  # Copy to preserve full state
                        if layer.get('is_synced', False)
                    ]
                    cleared_count = len(all_layers) - len(synced_layers_to_preserve)
                    if cleared_count > 0:
                        Log.info(f"EditorBlockProcessor: Clearing {cleared_count} non-synced layers from UI state")
                    if synced_layers_to_preserve:
                        Log.info(f"EditorBlockProcessor: Preserving {len(synced_layers_to_preserve)} sync layer(s) in UI state")
                    # Update UI state with preserved sync layers only
                    ui_state_service.set_state('editor_layers', block.id, {'layers': synced_layers_to_preserve})
            except Exception as e:
                Log.warning(f"EditorBlockProcessor: Failed to clear UI layer state: {e}")
        
        # Store preserved state for process() and step_post_process() to use
        block.metadata["_step_synced_layers"] = synced_layers_to_preserve
        block.metadata["_step_synced_items"] = [item.id for item in synced_layer_items]
    
    def get_status_levels(self, block: Block, facade: "ApplicationFacade") -> List["BlockStatusLevel"]:
        """
        Get status levels for Editor block.
        
        Status levels:
        - Warning (0): No inputs connected (editor can still open)
        - Ready (1): Inputs connected or editor ready
        
        Args:
            block: Block entity to get status levels for
            facade: ApplicationFacade for accessing services
            
        Returns:
            List of BlockStatusLevel instances in priority order
        """
        from src.features.blocks.domain import BlockStatusLevel
        
        def check_has_inputs(blk: Block, f: "ApplicationFacade") -> bool:
            """Check if any inputs are connected."""
            if not hasattr(f, 'connection_service'):
                return True  # Can't check, assume ready
            connections = f.connection_service.list_connections_by_block(blk.id)
            incoming = [c for c in connections if c.target_block_id == blk.id]
            return len(incoming) > 0
        
        return [
            BlockStatusLevel(
                priority=0,
                name="warning",
                display_name="Warning",
                color="#ffd43b",
                conditions=[check_has_inputs]
            ),
            BlockStatusLevel(
                priority=1,
                name="ready",
                display_name="Ready",
                color="#51cf66",
                conditions=[]
            )
        ]
    
    def get_expected_outputs(self, block: Block) -> Dict[str, List[str]]:
        """
        Get expected output names for editor block.
        
        Editor block can output both audio and events, depending on what's configured.
        """
        from src.application.processing.output_name_helpers import make_output_name
        
        outputs = {}
        output_ports = block.get_outputs()
        if "audio" in output_ports:
            outputs["audio"] = [make_output_name("audio", "main")]
        if "events" in output_ports:
            outputs["events"] = [make_output_name("events", "edited")]
        return outputs
    
    def process(
        self,
        block: Block,
        inputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        Process Editor block using STANDARDIZED EXECUTION FLOW.
        
        This is STEP 3 of the standardized flow (pre_process handles STEP 1):
        - Create owned copies of input EventDataItems
        - Pass through audio data
        - Update UI layer state with new layers
        
        The execution engine handles:
        - STEP 1: pre_process() clears old data
        - STEP 2: Pull data from upstream
        - STEP 3: This method - process data
        - STEP 4: Engine saves outputs
        - STEP 5: Engine publishes BlockUpdated
        
        Args:
            block: Block entity
            inputs: Dict with 'audio' and optionally 'events' keys
            metadata: Processing metadata (repos, services, etc.)
            
        Returns:
            Dict with 'audio' and/or 'events' outputs
        """
        Log.info(f"EditorBlockProcessor: Processing block '{block.name}' (standardized flow)")
        metadata = metadata or {}
        
        # Get preserved state from step_clear_local_data (don't pop yet - needed in step_post_process)
        synced_layers_to_preserve = block.metadata.get("_step_synced_layers", [])
        
        outputs = {}
        new_layer_names = set()
        
        # =====================================================================
        # STEP 3a: Process audio input (pass-through)
        # =====================================================================
        audio_input = inputs.get("audio")
        if audio_input is None:
            Log.debug(f"EditorBlockProcessor: No audio input for block '{block.name}'")
        else:
            if isinstance(audio_input, list):
                audio_items = audio_input
            else:
                audio_items = [audio_input]
            
            valid_audio_items = []
            for item in audio_items:
                if isinstance(item, AudioDataItem):
                    valid_audio_items.append(item)
                    Log.debug(f"EditorBlockProcessor: Audio input '{item.name}' - {item.length_ms:.2f}ms")
                else:
                    Log.warning(f"EditorBlockProcessor: Skipping non-AudioDataItem: {type(item)}")
            
            if valid_audio_items:
                from src.application.processing.output_name_helpers import make_output_name
                for item in valid_audio_items:
                    if 'output_name' not in item.metadata:
                        item.metadata['output_name'] = make_output_name("audio", "main")
                
                if len(valid_audio_items) == 1:
                    outputs["audio"] = valid_audio_items[0]
                else:
                    outputs["audio"] = valid_audio_items
                
                Log.info(f"EditorBlockProcessor: Passing through {len(valid_audio_items)} audio item(s)")
        
        # =====================================================================
        # STEP 3b: Process event input (create owned copies + preserve synced)
        # =====================================================================
        events_input = inputs.get("events")

        
        # Get preserved synced item IDs from step_clear_local_data
        synced_item_ids = block.metadata.get("_step_synced_items", [])
        data_item_repo = metadata.get("data_item_repo")
        
        # Collect all event items (new + synced)
        all_event_items = []
        
        if events_input is not None:
            # Create new owned items from inputs (fresh execution)
            owned_event_items = self._create_owned_event_items_from_inputs(block, inputs)

            if owned_event_items:
                # Track layer names for UI state recreation
                for item in owned_event_items:
                    if hasattr(item, '_layers') and item._layers:
                        for layer in item._layers:
                            if layer.name:
                                new_layer_names.add(layer.name)
                    else:
                        # Legacy: extract from event classifications
                        for event in item.get_events():
                            if event.classification:
                                new_layer_names.add(event.classification)
                
                # Apply event filters if configured
                try:
                    filter_manager = EventFilterManager()
                    filter = filter_manager.get_filter_for_block(block)
                    if filter and filter.enabled:
                        filtered_items = []
                        for item in owned_event_items:
                            filtered_events = [e for e in item.get_events() if filter.matches(e)]
                            filtered_item = EventDataItem(
                                id=item.id,
                                block_id=item.block_id,
                                name=item.name,
                                type=item.type,
                                created_at=item.created_at,
                                file_path=item.file_path,
                                metadata=item.metadata.copy(),
                                events=filtered_events
                            )
                            filtered_items.append(filtered_item)
                        owned_event_items = filtered_items
                        Log.info(f"EditorBlockProcessor: Applied event filter - {sum(len(i.get_events()) for i in owned_event_items)} events after filtering")
                except Exception as e:
                    Log.warning(f"EditorBlockProcessor: Failed to apply event filter: {e}")
                
                # Set output_name on event items
                from src.application.processing.output_name_helpers import make_output_name
                for item in owned_event_items:
                    if 'output_name' not in item.metadata:
                        item.metadata['output_name'] = make_output_name("events", "edited")
                
                all_event_items.extend(owned_event_items)
                
                total_events = sum(len(item.get_events()) for item in owned_event_items)
                Log.info(f"EditorBlockProcessor: Created {len(owned_event_items)} owned event item(s) with {total_events} events")
        else:
            Log.debug(f"EditorBlockProcessor: No events input for block '{block.name}'")
        
        # Include preserved synced items (MA3 synced layers)
        if synced_item_ids and data_item_repo:
            synced_items_included = 0
            for item_id in synced_item_ids:
                try:
                    synced_item = data_item_repo.get(item_id)
                    if synced_item and isinstance(synced_item, EventDataItem):
                        all_event_items.append(synced_item)
                        synced_items_included += 1
                        # Track synced layer names for UI state
                        if hasattr(synced_item, '_layers') and synced_item._layers:
                            for layer in synced_item._layers:
                                if layer.name:
                                    new_layer_names.add(layer.name)
                except Exception as e:
                    Log.warning(f"EditorBlockProcessor: Failed to include synced item {item_id}: {e}")
            
            if synced_items_included > 0:
                Log.info(f"EditorBlockProcessor: Included {synced_items_included} preserved synced MA3 item(s)")

        # Set events output if we have any items
        if all_event_items:
            if len(all_event_items) == 1:
                outputs["events"] = all_event_items[0]
            else:
                outputs["events"] = all_event_items
        
        # Store layer names for step_post_process to handle UI state update
        block.metadata["_step_new_layer_names"] = list(new_layer_names) if new_layer_names else []
        block.metadata["last_processed"] = True
        
        if not outputs:
            Log.info(f"EditorBlockProcessor: [STEP 4] Block '{block.name}' processed (no outputs)")
        else:
            output_summary = ", ".join([f"{port}: {len(items) if isinstance(items, list) else 1} item(s)" 
                                       for port, items in outputs.items()])
            Log.info(f"EditorBlockProcessor: [STEP 4] Block '{block.name}' processed - {output_summary}")

        
        return outputs
    
    # =========================================================================
    # STEP 5: Post-Process (update UI state)
    # =========================================================================
    
    def step_post_process(
        self,
        block: Block,
        outputs: Dict[str, DataItem],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, DataItem]:
        """
        STEP 5: Update UI state after processing.
        
        Editor-specific behavior:
        - Update UI layer state with new layers from outputs
        - Set execution flags for UI refresh
        - Clean up temporary metadata
        
        Args:
            block: Block entity being executed
            outputs: Outputs from process() step
            metadata: Processing metadata with repos/services
            
        Returns:
            Outputs unchanged
        """
        Log.info(f"EditorBlockProcessor: [STEP 5] Post-processing for '{block.name}'")
        metadata = metadata or {}
        
        ui_state_service = metadata.get("ui_state_service")
        
        # Get preserved and new layer data from earlier steps
        synced_layers_to_preserve = block.metadata.pop("_step_synced_layers", [])
        block.metadata.pop("_step_synced_items", None)
        new_layer_names = block.metadata.pop("_step_new_layer_names", [])
        
        # Update UI layer state: merge new EZ layers with preserved sync layers
        if ui_state_service:
            try:
                # Start with preserved sync layers (full state preserved)
                existing_layer_names = {l.get('name') for l in synced_layers_to_preserve}
                merged_layers = list(synced_layers_to_preserve)  # Copy preserved sync layers
                
                # Add new EZ layers (from execution)
                if new_layer_names:
                    for layer_name in sorted(new_layer_names):
                        if layer_name not in existing_layer_names:
                            merged_layers.append({
                                'name': layer_name,
                                'height': 40,
                                'color': None,
                                'visible': True,
                                'locked': False,
                                'is_synced': False,
                            })
                            Log.debug(f"EditorBlockProcessor: Added EZ layer UI state for '{layer_name}'")
                
                # Update UI state with merged layers (sync + new EZ)
                ui_state_service.set_state('editor_layers', block.id, {'layers': merged_layers})
                Log.info(
                    f"EditorBlockProcessor: Updated UI state with {len(merged_layers)} layer(s) "
                    f"({len(synced_layers_to_preserve)} sync, {len(merged_layers) - len(synced_layers_to_preserve)} EZ)"
                )
            except Exception as e:
                Log.warning(f"EditorBlockProcessor: Failed to update UI layer state: {e}")
        
        # Set execution flags for UI notification (engine reads these in BlockUpdated event)
        block.metadata["_execution_triggered"] = True
        block.metadata["_new_layer_names"] = new_layer_names
        
        return outputs
    
    def _create_owned_event_items_from_inputs(
        self, 
        block: Block, 
        inputs: Dict[str, DataItem]
    ) -> List[EventDataItem]:
        """
        Create owned EventDataItems from input data.
        
        VALIDATES that input EventDataItems have proper EventLayers structure.
        Structure MUST be: EventDataItem -> EventLayers -> Events
        
        FAILS LOUD if input doesn't have proper EventLayers - no silent conversion.
        
        Args:
            block: Block entity
            inputs: Input data items
            
        Returns:
            List of owned EventDataItems with preserved EventLayers
            
        Raises:
            ProcessingError: If input doesn't have proper EventLayers structure
        """
        from src.shared.domain.entities import EventLayer
        
        events_input = inputs.get("events")
        if events_input is None:
            return []
        
        # Handle single EventDataItem or list
        if isinstance(events_input, list):
            event_items = events_input
        else:
            event_items = [events_input]


        # Create new EventDataItems owned by this Editor block
        owned_event_items = []
        for item in event_items:
            if not isinstance(item, EventDataItem):
                Log.warning(f"EditorBlockProcessor: Skipping non-EventDataItem: {type(item)}")
                continue
            
            # VALIDATE: Input MUST have proper EventLayers
            has_valid_layers = (
                hasattr(item, '_layers') and 
                item._layers is not None and
                isinstance(item._layers, list)
            )
            
            if not has_valid_layers:
                # FAIL LOUD: Input must have EventLayers
                raise ProcessingError(
                    f"Editor block '{block.name}' received EventDataItem '{item.name}' "
                    f"WITHOUT proper EventLayers! "
                    f"Structure MUST be: EventDataItem -> EventLayers -> Events. "
                    f"The source block must output EventLayers explicitly. "
                    f"Re-execute the upstream block to generate proper EventLayers.",
                    block_id=block.id,
                    block_name=block.name
                )
            
            # Check if layers are empty but there are legacy events
            has_legacy_events = (
                hasattr(item, '_events') and 
                item._events and 
                len(item._events) > 0
            )
            
            if len(item._layers) == 0 and has_legacy_events:
                # FAIL LOUD: Has legacy events but no layers
                raise ProcessingError(
                    f"Editor block '{block.name}' received EventDataItem '{item.name}' "
                    f"with {len(item._events)} legacy flat events but NO EventLayers! "
                    f"Structure MUST be: EventDataItem -> EventLayers -> Events. "
                    f"The source block must output events in EventLayers, not as flat events. "
                    f"Re-execute the upstream block to generate proper EventLayers.",
                    block_id=block.id,
                    block_name=block.name
                )
            
            # Skip empty items
            if len(item._layers) == 0:
                Log.debug(f"EditorBlockProcessor: Skipping empty EventDataItem '{item.name}' (no layers)")
                continue
            
            # PRESERVE EventLayers - layers are the single source of truth
            input_layers = item._layers
            
            # Create new EventDataItem with preserved layers
            owned_layers = []
            for layer in input_layers:
                # Create new EventLayer with same name and events
                owned_layer = EventLayer(
                    name=layer.name,
                    events=layer.events.copy(),  # Copy events but keep same objects (preserve metadata)
                    metadata=layer.metadata.copy() if layer.metadata else {}
                )
                owned_layers.append(owned_layer)
            
            owned_item = EventDataItem(
                id="",  # Will be generated by EventDataItem.__init__
                block_id=block.id,  # Editor block owns this
                name=f"{block.name}_{item.name}_edited",
                type="Event",
                metadata={
                    # Preserve source information
                    "_source_item_id": item.id,
                    "_source_item_name": item.name,
                    "_source_block_id": item.block_id,
                    "output_port": "events",  # Mark as output port
                },
                layers=owned_layers  # PRESERVE LAYERS - single source of truth
            )
            
            total_events = sum(len(l.events) for l in owned_layers)
            Log.info(
                f"EditorBlockProcessor: Created owned EventDataItem '{owned_item.name}' "
                f"with {len(owned_layers)} layer(s) and {total_events} event(s) from '{item.name}'"
            )
            owned_event_items.append(owned_item)
        
        return owned_event_items


    def get_default_inputs(self) -> Dict[str, Any]:
        """Get default input ports for Editor block."""
        from src.shared.domain.value_objects.port_type import AUDIO_TYPE, EVENT_TYPE
        return {
            "audio": AUDIO_TYPE,
            "events": EVENT_TYPE,
        }
    
    def get_default_outputs(self) -> Dict[str, Any]:
        """Get default output ports for Editor block."""
        from src.shared.domain.value_objects.port_type import AUDIO_TYPE, EVENT_TYPE
        return {
            "audio": AUDIO_TYPE,
            "events": EVENT_TYPE,
        }
    
    def get_default_bidirectional(self) -> Dict[str, Any]:
        """Get default bidirectional ports for Editor block."""
        return {
            "manipulator": MANIPULATOR_TYPE,
        }

    def validate_configuration(
        self,
        block: Block,
        data_item_repo: Any = None,
        connection_repo: Any = None,
        block_registry: Any = None
    ) -> List[str]:
        """
        Validate Editor block configuration before execution.

        Args:
            block: Block to validate
            data_item_repo: Data item repository (for checking upstream data)
            connection_repo: Connection repository (for checking connections)
            block_registry: Block registry (for getting expected input types)

        Returns:
            List of error messages (empty if valid)
        """
        # EditorBlock doesn't have specific validation requirements
        return []


register_processor_class(EditorBlockProcessor)
