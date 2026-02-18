"""
Block Service

Orchestrates block-related use cases.
Handles business rules, generates unique names, and emits domain events.
"""
from typing import Optional, List, Dict
import uuid
from pathlib import Path

from src.features.blocks.domain.block import Block
from src.shared.domain.entities import BlockSummary
from src.features.projects.domain import Project, ProjectRepository
from src.features.blocks.domain.block_repository import BlockRepository
from src.features.connections.domain import ConnectionRepository
from src.shared.domain.value_objects.port_type import PortType, get_port_type
from src.features.blocks.domain.port_direction import PortDirection
from src.application.events.event_bus import EventBus
from src.application.block_registry import get_block_registry
from src.application.events import BlockAdded, BlockUpdated, BlockRemoved
from src.utils.message import Log


class BlockService:
    """
    Service for managing blocks.
    
    Orchestrates block operations:
    - Creating blocks with unique names
    - Removing blocks
    - Updating block properties (name, ports)
    - Managing port definitions
    
    Emits domain events for UI synchronization.
    """
    
    def __init__(
        self,
        block_repo: BlockRepository,
        project_repo: ProjectRepository,
        event_bus: EventBus,
        connection_repo: Optional[ConnectionRepository] = None
    ):
        """
        Initialize block service.
        
        Args:
            block_repo: Repository for block persistence
            project_repo: Repository for project validation
            event_bus: Event bus for publishing domain events
            connection_repo: Optional repository for connection management (for cascade operations)
        """
        self._block_repo = block_repo
        self._connection_repo = connection_repo
        self._project_repo = project_repo
        self._event_bus = event_bus
        Log.info("BlockService: Initialized")
    
    def add_block(
        self,
        project_id: str,
        block_type: str,
        name: Optional[str] = None
    ) -> Block:
        """
        Add a new block to a project.
        
        Args:
            project_id: Project identifier
            block_type: Type of block (e.g., "LoadAudio", "ProcessAudio")
            name: Optional block name (will be generated if not provided)
            
        Returns:
            Created Block entity
            
        Raises:
            ValueError: If project not found or block type invalid
        """
        # Validate project exists
        project = self._project_repo.get(project_id)
        if not project:
            raise ValueError(f"Project with id '{project_id}' not found")
        
        # Validate block type
        if not block_type or not block_type.strip():
            raise ValueError("Block type cannot be empty")
        
        # Get block type metadata from registry (case-insensitive)
        registry = get_block_registry()
        block_metadata = registry.get(block_type)
        
        # Normalize block type to correct case from registry
        if block_metadata:
            block_type = block_metadata.type_id
        else:
            # If block type not in registry, allow it but log warning
            Log.warning(f"BlockType '{block_type}' not found in registry - creating without port definitions")
        
        # Generate unique name if not provided
        if not name:
            name = self._generate_unique_name(project_id, block_type)
        else:
            name = name.strip()
        
        # Create block entity
        block = Block(
            id="",  # Will be generated
            project_id=project_id,
            name=name,
            type=block_type
        )
        
        # Add port definitions from registry if available
        if block_metadata:
            for port_name, port_type in block_metadata.inputs.items():
                block.add_port(port_name, port_type, PortDirection.INPUT)
            for port_name, port_type in block_metadata.outputs.items():
                block.add_port(port_name, port_type, PortDirection.OUTPUT)
            # Add bidirectional ports
            if hasattr(block_metadata, 'bidirectional') and block_metadata.bidirectional:
                Log.debug(f"BlockService: Adding {len(block_metadata.bidirectional)} bidirectional ports to {block_type}")
                for port_name, port_type in block_metadata.bidirectional.items():
                    block.add_port(port_name, port_type, PortDirection.BIDIRECTIONAL)
                    Log.debug(f"BlockService: Added bidirectional port '{port_name}' ({port_type.name}) to {block.name}")
        
        # Save to repository
        created_block = self._block_repo.create(block)
        
        # Emit event
        self._event_bus.publish(BlockAdded(
            project_id=project_id,
            data={
                "id": created_block.id,
                "name": created_block.name,
                "type": created_block.type
            }
        ))
        
        Log.info(f"BlockService: Added block '{created_block.name}' (id: {created_block.id}) to project '{project.name}'")
        return created_block
    
    def remove_block(self, project_id: str, block_id: str) -> None:
        """
        Remove a block from a project.
        Enhanced to mark downstream blocks when their input source is deleted.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            
        Raises:
            ValueError: If block not found
        """
        # Get block before deletion (for event)
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        block_name = block.name
        
        # Mark downstream blocks as having missing inputs
        affected_blocks = []
        if self._connection_repo:
            # Get all connections from this block
            all_connections = self._connection_repo.list_by_project(project_id)
            outgoing = [c for c in all_connections if c.source_block_id == block_id]
            
            # Mark downstream blocks
            for conn in outgoing:
                downstream_block = self._block_repo.get(project_id, conn.target_block_id)
                if downstream_block:
                    # Track which inputs are missing
                    if '_missing_inputs' not in downstream_block.metadata:
                        downstream_block.metadata['_missing_inputs'] = []
                    
                    if conn.target_input_name not in downstream_block.metadata['_missing_inputs']:
                        downstream_block.metadata['_missing_inputs'].append(conn.target_input_name)
                    
                    self._block_repo.update(downstream_block)
                    affected_blocks.append({
                        'block_id': downstream_block.id,
                        'block_name': downstream_block.name,
                        'missing_input': conn.target_input_name
                    })
                    
                    Log.info(
                        f"BlockService: Block '{downstream_block.name}' marked with missing input "
                        f"on port '{conn.target_input_name}' due to deletion of '{block_name}'"
                    )
        
        # Delete from repository (cascade will delete connections and data items via FK)
        self._block_repo.delete(project_id, block_id)
        
        # Emit event with affected blocks info
        self._event_bus.publish(BlockRemoved(
            project_id=project_id,
            data={
                "id": block_id,
                "name": block_name,
                "type": block.type,
                "affected_blocks": affected_blocks
            }
        ))
        
        if affected_blocks:
            Log.warning(
                f"BlockService: Removed block '{block_name}' (id: {block_id}). "
                f"{len(affected_blocks)} downstream block(s) affected and marked with missing inputs."
            )
        else:
            Log.info(f"BlockService: Removed block '{block_name}' (id: {block_id}) from project {project_id}")
    
    # update_block_position method removed - backend-only architecture doesn't need UI coordinates
    
    def rename_block(self, project_id: str, block_id: str, new_name: str) -> Block:
        """
        Rename a block.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            new_name: New block name
            
        Returns:
            Updated Block entity
            
        Raises:
            ValueError: If block not found or name validation fails
        """
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        old_name = block.name
        
        # Rename
        block.rename(new_name)
        self._block_repo.update(block)
        
        # Emit event
        self._event_bus.publish(BlockUpdated(
            project_id=project_id,
            data={
                "id": block_id,
                "name": new_name,
                "old_name": old_name
            }
        ))
        
        Log.info(f"BlockService: Renamed block from '{old_name}' to '{new_name}' (id: {block_id})")
        return block

    def update_block(
        self,
        project_id: str,
        block_id: str,
        block: Block,
        *,
        changed_keys: list[str] | None = None,
        update_source: str | None = None,
        origin_panel_id: str | None = None,
    ) -> Block:
        """
        Update a block in the repository.
        Generic update method for any block changes (including metadata).

        Args:
            project_id: Project identifier
            block_id: Block identifier
            block: Updated block entity

        Returns:
            Updated Block entity
        """
        self._block_repo.update(block)
        event_data = {"id": block_id, "name": block.name}
        if changed_keys:
            event_data["changed_keys"] = list(changed_keys)
        if update_source:
            event_data["update_source"] = update_source
        if origin_panel_id:
            event_data["origin_panel_id"] = origin_panel_id
        self._event_bus.publish(BlockUpdated(project_id=project_id, data=event_data))
        
        # Also emit BlockChanged since metadata changes (errors, filters) affect status
        # BlockItem will handle BlockUpdated to refresh block entity, but this ensures
        # status is recalculated with latest metadata
        from src.application.events import BlockChanged
        self._event_bus.publish(BlockChanged(
            project_id=project_id,
            data={
                "block_id": block_id,
                "change_type": "metadata"
            }
        ))
        
        Log.debug(f"BlockService: Updated block '{block.name}'")
        return block
    
    def update_expected_outputs(
        self, 
        block: Block, 
        processor=None,
        expected_outputs_service=None,
        facade=None
    ) -> None:
        """
        Calculate and cache expected outputs for a block.
        Called automatically when block configuration changes or connections change.
        
        Uses ExpectedOutputsService to handle both static and connection-based outputs.
        
        Args:
            block: Block to update expected outputs for
            processor: Optional BlockProcessor instance. If not provided, will try to get from registry.
                     Callers should provide processor if available to avoid dependency issues.
            expected_outputs_service: Optional ExpectedOutputsService. If not provided, will try to get from facade.
            facade: Optional ApplicationFacade for accessing services
        """
        # Try to get processor if not provided
        if processor is None:
            try:
                from src.features.execution.application import BlockExecutionEngine
                # This is a fallback - ideally callers should provide processor
                # We can't easily get processor here without circular dependencies
                Log.debug(f"BlockService: update_expected_outputs called without processor for {block.name}")
                return
            except ImportError:
                Log.debug(f"BlockService: Cannot get processor for {block.name}")
                return
        
        if not expected_outputs_service:
            Log.debug(f"BlockService: update_expected_outputs called without expected_outputs_service for {block.name}")
            return
        
        try:
            expected_outputs = expected_outputs_service.calculate_expected_outputs(
                block,
                processor,
                facade=facade
            )
            block.metadata['expected_outputs'] = expected_outputs
            self._block_repo.update(block)
            Log.debug(f"BlockService: Updated expected_outputs for block '{block.name}': {expected_outputs}")
        except Exception as e:
            Log.warning(f"BlockService: Failed to update expected_outputs for {block.name}: {e}")

    def execute_block_command(
        self,
        project_id: str,
        block_id: str,
        command_name: str,
        args: List[str],
        kwargs: Dict[str, str]
    ) -> Block:
        """
        Execute a custom command on a block (persist metadata).
        """
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")

        supported_formats = {".mp3", ".wav", ".flac", ".ogg", ".aif", ".aiff", ".m4a"}

        if command_name == "set_path":
            if not args:
                raise ValueError("set_path requires a file path argument")

            audio_path = Path(args[0]).expanduser()
            if not audio_path.is_file():
                raise ValueError(f"Audio path does not exist: {audio_path}")

            if audio_path.suffix.lower() not in supported_formats:
                raise ValueError(f"Unsupported audio format: {audio_path.suffix}")

            args = [str(audio_path)]
            block.metadata["file_path"] = str(audio_path)

        if command_name == "set_output_dir":
            if not args:
                raise ValueError("set_output_dir requires a directory path")

            output_dir = Path(args[0]).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            if not output_dir.is_dir():
                raise ValueError(f"Output directory is invalid: {output_dir}")

            block.metadata["output_dir"] = str(output_dir)
            Log.info(f"BlockService: Set export directory for '{block.name}' -> {output_dir}")
            self._block_repo.update(block)
            command_history = block.metadata.get("commands", {})
            command_history[command_name] = {"args": [str(output_dir)], "kwargs": kwargs}
            block.metadata["commands"] = command_history
            return block

        if command_name == "set_format":
            if not args:
                raise ValueError("set_format requires a format name")

            fmt = args[0].lower()
            fmt_ext = fmt if fmt.startswith(".") else f".{fmt}"
            if fmt_ext not in supported_formats:
                raise ValueError(f"Unsupported export format: {fmt}")

            block.metadata["audio_format"] = fmt.lstrip(".")
            Log.info(f"BlockService: Set export format for '{block.name}' -> {block.metadata['audio_format']}")
            self._block_repo.update(block)
            command_history = block.metadata.get("commands", {})
            command_history[command_name] = {"args": [settings["format"]], "kwargs": kwargs}
            block.metadata["commands"] = command_history
            return block

        if command_name == "set_model":
            if not args:
                raise ValueError("set_model requires a model name")

            model_name = args[0].strip()
            
            # Validate model for Separator blocks
            if block.type == "Separator":
                from src.application.blocks.separator_block import DEMUCS_MODELS, get_demucs_models_info
                if model_name not in DEMUCS_MODELS:
                    error_msg = (
                        f"Invalid Demucs model: '{model_name}'\n\n"
                        f"{get_demucs_models_info()}\n"
                        f"Or use: list_separator_models"
                    )
                    raise ValueError(error_msg)
                
                settings = block.metadata.setdefault("separator_settings", {})
                settings["model"] = model_name
                model_info = DEMUCS_MODELS[model_name]
                Log.info(
                    f"BlockService: Set separator model for '{block.name}' -> {model_name} "
                    f"({model_info['quality']} quality, {model_info['speed']} speed)"
                )
            else:
                # Generic set_model for other block types
                settings = block.metadata.setdefault("separator_settings", {})
                settings["model"] = model_name
                Log.info(f"BlockService: Set model for '{block.name}' -> {model_name}")
            
            self._block_repo.update(block)
            command_history = block.metadata.get("commands", {})
            command_history[command_name] = {"args": [model_name], "kwargs": kwargs}
            block.metadata["commands"] = command_history
            return block
        
        if command_name == "list_models":
            # Handle list_models command for Separator blocks
            if block.type == "Separator":
                from src.application.blocks.separator_block import get_demucs_models_info
                Log.info(get_demucs_models_info())
                return block
            else:
                raise ValueError(f"list_models command not available for block type '{block.type}'")
        
        if command_name == "set_two_stems":
            if block.type != "Separator":
                raise ValueError(f"set_two_stems command only available for Separator blocks")
            if not args:
                raise ValueError("set_two_stems requires a stem name (drums, vocals, bass, or other)")
            
            stem_name = args[0].lower()
            valid_stems = ["drums", "vocals", "bass", "other"]
            if stem_name not in valid_stems:
                raise ValueError(f"Invalid stem: '{stem_name}'. Must be one of: {', '.join(valid_stems)}")
            
            block.metadata["two_stems"] = stem_name
            self._block_repo.update(block)
            Log.info(f"BlockService: Enabled 2-stem mode for '{block.name}' -> isolating '{stem_name}' (outputs 2 files instead of 4)")
            return block
        
        if command_name == "clear_two_stems":
            if block.type != "Separator":
                raise ValueError(f"clear_two_stems command only available for Separator blocks")
            
            if "two_stems" in block.metadata:
                del block.metadata["two_stems"]
                self._block_repo.update(block)
                Log.info(f"BlockService: Disabled 2-stem mode for '{block.name}' -> returning to full 4-stem separation")
            else:
                Log.info(f"BlockService: 2-stem mode was not enabled for '{block.name}'")
            return block
        
        if command_name == "set_device":
            if block.type != "Separator":
                raise ValueError(f"set_device command only available for Separator blocks")
            if not args:
                raise ValueError("set_device requires a device name (auto, cpu, or cuda)")
            
            device = args[0].lower()
            valid_devices = ["auto", "cpu", "cuda"]
            if device not in valid_devices:
                raise ValueError(f"Invalid device: '{device}'. Must be one of: {', '.join(valid_devices)}")
            
            block.metadata["device"] = device
            self._block_repo.update(block)
            Log.info(f"BlockService: Set processing device for '{block.name}' -> {device}")
            return block
        
        if command_name == "set_output_format":
            if block.type != "Separator":
                raise ValueError(f"set_output_format command only available for Separator blocks")
            if not args:
                raise ValueError("set_output_format requires a format (wav or mp3)")
            
            output_format = args[0].lower()
            valid_formats = ["wav", "mp3"]
            if output_format not in valid_formats:
                raise ValueError(f"Invalid format: '{output_format}'. Must be one of: {', '.join(valid_formats)}")
            
            block.metadata["output_format"] = output_format
            self._block_repo.update(block)
            Log.info(f"BlockService: Set output format for '{block.name}' -> {output_format}")
            return block
        
        if command_name == "set_shifts":
            if block.type != "Separator":
                raise ValueError(f"set_shifts command only available for Separator blocks")
            if not args:
                raise ValueError("set_shifts requires a number (0=fastest, 1=default, 10=paper quality)")
            
            try:
                shifts = int(args[0])
            except ValueError:
                raise ValueError(f"Invalid shifts value: '{args[0]}'. Must be an integer.")
            
            if shifts < 0:
                raise ValueError("Shifts must be 0 or greater")
            
            block.metadata["shifts"] = shifts
            self._block_repo.update(block)
            quality_note = "fastest/lowest quality" if shifts == 0 else "default" if shifts == 1 else "higher quality/slower"
            Log.info(f"BlockService: Set shifts for '{block.name}' -> {shifts} ({quality_note})")
            return block
        
        # Handle note extractor commands
        if command_name == "set_onset_threshold":
            if not args:
                raise ValueError("set_onset_threshold requires a threshold value")
            threshold = float(args[0])
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Onset threshold must be between 0.0 and 1.0")
            block.metadata["onset_threshold"] = threshold
            self._block_repo.update(block)
            self._event_bus.publish(BlockUpdated(
                project_id=project_id,
                data={"id": block_id, "name": block.name, "command": command_name}
            ))
            Log.info(f"BlockService: Set onset threshold for '{block.name}' -> {threshold}")
            return block
        
        if command_name == "set_frame_threshold":
            if not args:
                raise ValueError("set_frame_threshold requires a threshold value")
            threshold = float(args[0])
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Frame threshold must be between 0.0 and 1.0")
            block.metadata["frame_threshold"] = threshold
            self._block_repo.update(block)
            Log.info(f"BlockService: Set frame threshold for '{block.name}' -> {threshold}")
            return block
        
        if command_name == "set_min_note_length" or command_name == "set_min_duration":
            if not args:
                raise ValueError(f"{command_name} requires a duration in seconds")
            duration = float(args[0])
            if duration < 0:
                raise ValueError("Duration must be positive")
            # Store both keys for compatibility
            block.metadata["minimum_note_length"] = duration
            block.metadata["min_note_duration"] = duration
            self._block_repo.update(block)
            Log.info(f"BlockService: Set minimum note duration for '{block.name}' -> {duration}s")
            return block
        
        if command_name == "set_hop_length":
            if not args:
                raise ValueError("set_hop_length requires a number of samples")
            hop_length = int(args[0])
            if hop_length < 1:
                raise ValueError("Hop length must be positive")
            block.metadata["hop_length"] = hop_length
            self._block_repo.update(block)
            Log.info(f"BlockService: Set hop length for '{block.name}' -> {hop_length} samples")
            return block
        
        if command_name == "set_frequency_range":
            if len(args) < 2:
                raise ValueError("set_frequency_range requires min_hz and max_hz arguments")
            min_hz = float(args[0])
            max_hz = float(args[1])
            if min_hz <= 0 or max_hz <= 0:
                raise ValueError("Frequencies must be positive")
            if min_hz >= max_hz:
                raise ValueError("Minimum frequency must be less than maximum frequency")
            block.metadata["minimum_frequency"] = min_hz
            block.metadata["maximum_frequency"] = max_hz
            block.metadata["fmin"] = min_hz
            block.metadata["fmax"] = max_hz
            self._block_repo.update(block)
            Log.info(f"BlockService: Set frequency range for '{block.name}' -> {min_hz}-{max_hz} Hz")
            return block
        
        if command_name == "save_to_file":
            if not args:
                raise ValueError("save_to_file requires true or false")
            enabled = args[0].lower() in ["true", "1", "yes", "on"]
            block.metadata["save_to_file"] = enabled
            self._block_repo.update(block)
            Log.info(f"BlockService: Set save_to_file for '{block.name}' -> {enabled}")
            return block
        
        command_history = block.metadata.get("commands", {})
        command_history[command_name] = {
            "args": args,
            "kwargs": kwargs
        }
        block.metadata["commands"] = command_history

        self._block_repo.update(block)

        self._event_bus.publish(BlockUpdated(
            project_id=project_id,
            data={
                "id": block_id,
                "name": block.name,
                "command": command_name,
                "args": args,
                "kwargs": kwargs
            }
        ))

        Log.info(f"BlockService: Executed command '{command_name}' on block '{block.name}'")
        return block
    
    def add_input_port(
        self,
        project_id: str,
        block_id: str,
        port_name: str,
        port_type: str
    ) -> None:
        """
        Add an input port to a block.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            port_name: Port name
            port_type: Port type name (e.g., "Audio", "Event")
            
        Raises:
            ValueError: If block not found or port already exists
        """
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        # Convert string to PortType
        port_type_obj = get_port_type(port_type)
        
        # Add port
        block.add_port(port_name, port_type_obj, PortDirection.INPUT)
        self._block_repo.update(block)
        
        # Emit event
        self._event_bus.publish(BlockUpdated(
            project_id=project_id,
            data={
                "id": block_id,
                "added_input_port": port_name,
                "port_type": port_type
            }
        ))
        
        Log.info(f"BlockService: Added input port '{port_name}' ({port_type}) to block '{block.name}'")
    
    def add_output_port(
        self,
        project_id: str,
        block_id: str,
        port_name: str,
        port_type: str
    ) -> None:
        """
        Add an output port to a block.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            port_name: Port name
            port_type: Port type name (e.g., "Audio", "Event")
            
        Raises:
            ValueError: If block not found or port already exists
        """
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        # Convert string to PortType
        port_type_obj = get_port_type(port_type)
        
        # Add port
        block.add_port(port_name, port_type_obj, PortDirection.OUTPUT)
        self._block_repo.update(block)
        
        # Emit event
        self._event_bus.publish(BlockUpdated(
            project_id=project_id,
            data={
                "id": block_id,
                "added_output_port": port_name,
                "port_type": port_type
            }
        ))
        
        Log.info(f"BlockService: Added output port '{port_name}' ({port_type}) to block '{block.name}'")
    
    def list_blocks(self, project_id: str) -> List[BlockSummary]:
        """
        List all blocks in a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of Block entities
        """
        return self._block_repo.list_block_summaries(project_id)

    def _ensure_default_ports(self, block: Block, skip_db_update: bool = False) -> bool:
        """
        Ensure block has all default ports according to block registry.
        Adds any missing ports and updates the block in the repository.
        
        This ensures that blocks created before certain ports were added to the registry
        will have those ports added when loaded.
        
        Args:
            block: Block entity to check and update
            skip_db_update: If True, don't write to database (for read-only operations like scene refresh)
            
        Returns:
            True if ports were added, False otherwise
        """
        registry = get_block_registry()
        block_metadata = registry.get(block.type)
        if not block_metadata:
            return False
        
        ports_added = False
        
        # Check and add input ports
        for port_name, port_type in block_metadata.inputs.items():
            if not block.get_port(port_name, PortDirection.INPUT):
                try:
                    block.add_port(port_name, port_type, PortDirection.INPUT)
                    ports_added = True
                except ValueError:
                    # Port already exists, skip
                    pass
        
        # Check and add output ports
        for port_name, port_type in block_metadata.outputs.items():
            if not block.get_port(port_name, PortDirection.OUTPUT):
                try:
                    block.add_port(port_name, port_type, PortDirection.OUTPUT)
                    ports_added = True
                except ValueError:
                    # Port already exists, skip
                    pass
        
        # Check and add bidirectional ports (including manipulator)
        # These are added last to match the order in add_block(), ensuring consistency
        if hasattr(block_metadata, 'bidirectional') and block_metadata.bidirectional:
            for port_name, port_type in block_metadata.bidirectional.items():
                if not block.get_port(port_name, PortDirection.BIDIRECTIONAL):
                    try:
                        block.add_port(port_name, port_type, PortDirection.BIDIRECTIONAL)
                        ports_added = True
                    except ValueError:
                        # Port already exists, skip
                        pass
        
        if ports_added:
            if not skip_db_update:
                # Update block in repository (only if not skipping)
                self._block_repo.update(block)
                Log.debug(f"BlockService: Added missing default ports to block '{block.name}' ({block.type})")
            else:
                # Just log that ports would be added (for read-only operations)
                Log.debug(f"BlockService: Block '{block.name}' ({block.type}) missing ports but skipping DB update (read-only)")
        
        return ports_added
    
    def get_block_detail(self, project_id: str, block_id: str, skip_port_check: bool = False) -> Optional[Block]:
        """
        Load the full block detail (inputs/outputs) when needed.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            skip_port_check: If True, skip ensuring default ports (for performance during scene refresh)
            
        Returns:
            Block entity or None if not found
        """
        block = self._block_repo.load_block_detail(project_id, block_id)
        if block and not skip_port_check:
            # Ensure default ports are present (but skip DB update during read-only operations)
            # This prevents blocking database writes during scene refresh
            self._ensure_default_ports(block, skip_db_update=True)
        return block
    
    def get_block(self, project_id: str, block_id: str, skip_port_check: bool = False) -> Optional[Block]:
        """
        Get a block by ID.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            skip_port_check: If True, skip ensuring default ports (for performance during scene refresh)
            
        Returns:
            Block entity or None if not found
        """
        block = self._block_repo.get(project_id, block_id)
        if block and not skip_port_check:
            # Ensure default ports are present (but skip DB update during read-only operations)
            # This prevents blocking database writes during scene refresh
            self._ensure_default_ports(block, skip_db_update=True)
        return block
    
    def find_by_name(self, project_id: str, name: str) -> Optional[Block]:
        """
        Find a block by name within a project.
        
        Args:
            project_id: Project identifier
            name: Block name
            
        Returns:
            Block entity or None if not found
        """
        return self._block_repo.find_by_name(project_id, name)
    
    def reset_block_state(
        self, 
        project_id: str, 
        block_id: str,
        data_item_repo=None,
        block_local_state_repo=None
    ) -> Block:
        """
        Reset a block's state to its initial state.
        
        This clears:
        - All metadata (configuration and settings)
        - All owned data items (EventDataItems, AudioDataItems, etc.)
        - All local state (input/output references)
        
        Returns the block to the state it had when first created.
        
        Args:
            project_id: Project identifier
            block_id: Block identifier
            data_item_repo: Optional data item repository for clearing owned data items
            block_local_state_repo: Optional local state repository for clearing inputs
            
        Returns:
            Updated Block entity with cleared state
            
        Raises:
            ValueError: If block not found
        """
        block = self._block_repo.get(project_id, block_id)
        if not block:
            raise ValueError(f"Block with id '{block_id}' not found in project {project_id}")
        
        # Clear owned data items (events, layers, audio, etc.)
        if data_item_repo:
            try:
                deleted_count = data_item_repo.delete_by_block(block_id)
                if deleted_count > 0:
                    Log.info(f"BlockService: Deleted {deleted_count} data item(s) for block '{block.name}'")
            except Exception as e:
                Log.warning(f"BlockService: Failed to delete data items for block '{block.name}': {e}")
        
        # Clear local state (input/output references)
        if block_local_state_repo:
            try:
                block_local_state_repo.clear_inputs(block_id)
                Log.info(f"BlockService: Cleared local state for block '{block.name}'")
            except Exception as e:
                Log.warning(f"BlockService: Failed to clear local state for block '{block.name}': {e}")
        
        # Reset metadata to empty dict (initial state)
        block.metadata = {}
        
        # Update in repository
        updated_block = self._block_repo.update(block)
        
        # Emit event
        self._event_bus.publish(BlockUpdated(
            project_id=project_id,
            data={
                "id": block_id,
                "name": block.name,
                "metadata_reset": True,
                "data_items_cleared": True,
                "local_state_cleared": True
            }
        ))
        
        Log.info(f"BlockService: Reset state for block '{block.name}' (id: {block_id})")
        return updated_block
    
    def _generate_unique_name(self, project_id: str, block_type: str) -> str:
        """
        Generate a unique name for a block.
        
        Args:
            project_id: Project identifier
            block_type: Block type
            
        Returns:
            Unique block name
        """
        # Get existing blocks
        existing_blocks = self._block_repo.list_block_summaries(project_id)
        existing_names = {block.name for block in existing_blocks}
        
        # Try simple name first
        base_name = f"{block_type}1"
        if base_name not in existing_names:
            return base_name
        
        # Find next available number
        counter = 2
        while True:
            name = f"{block_type}{counter}"
            if name not in existing_names:
                return name
            counter += 1

