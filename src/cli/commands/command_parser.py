"""
Command Parser

Parses CLI text commands and routes to ApplicationFacade.
CLI-specific adapter for the application facade.
"""
from typing import Dict, List, Optional

from src.cli.commands.command_registry import CommandDefinition, CommandRegistry
from src.utils.message import Log
from src.utils.tools import prompt_yes_no
from src.application.api.result_types import CommandResult, ResultStatus


class CommandParser:
    """
    Parses CLI text commands and routes to ApplicationFacade.
    
    This is a CLI-specific adapter that:
    1. Parses command strings ("add_block LoadAudio MyBlock")
    2. Routes to appropriate facade methods
    3. Formats results for CLI output (logs)
    4. Returns CommandResult objects
    """
    
    def __init__(self, facade):
        """
        Initialize command parser.
        
        Args:
            facade: ApplicationFacade instance
        """
        self.facade = facade
        self._command_registry = CommandRegistry()
        self._register_commands()
        self._last_result: Optional[CommandResult] = None
    
    def parse_and_execute(self, input_string: str) -> CommandResult:
        """
        Parse CLI string and execute via registry.
        
        Args:
            input_string: Command string (e.g., "add_block LoadAudio MyBlock")
            
        Returns:
            CommandResult with status, message, and data
        """
        Log.info(f"Received input: '{input_string}'")
        
        parts = self._parse_command_parts(input_string)
        if not parts:
            result = CommandResult.error_result("Empty command")
            self._log_result(result)
            return result
        
        command_name = parts[0].lower()
        args_preview = parts[1:4] if len(parts) > 1 else []
        args_display = ' '.join(args_preview) + ('...' if len(parts) > 4 else '')
        Log.info(f"Parsed command: '{command_name}' with {len(parts)-1} argument(s): {args_display if args_display else '(none)'}")
        
        definition = self._command_registry.get(command_name)

        # Try fuzzy matching if exact match not found
        if not definition:
            fuzzy_match = self._find_fuzzy_command_match(command_name)
            if fuzzy_match:
                Log.info(f"Command '{command_name}' not found. Did you mean '{fuzzy_match}'? Using that command...")
                command_name = fuzzy_match
                definition = self._command_registry.get(fuzzy_match)

        if definition:
            args, kwargs = self._parse_args(parts[1:])
            try:
                result = definition.handler(args, kwargs)
            except Exception as e:
                Log.error(f"Command '{command_name}' failed with error: {e}")
                result = CommandResult.error_result(
                    message=f"Command '{command_name}' failed: {e}",
                    errors=[
                        f"Command attempted: '{command_name}'",
                        f"Arguments: {args}",
                        f"Keyword arguments: {kwargs}",
                        f"Error: {str(e)}"
                    ]
                )
        else:
            if len(parts) < 2:
                # Suggest similar commands
                suggestion = self._suggest_similar_command(command_name)
                error_msg = [
                    f"Unknown command: '{command_name}'",
                    f"Input received: '{input_string}'"
                ]
                if suggestion:
                    error_msg.append(f"Did you mean '{suggestion}'?")
                error_msg.append("Type 'help' for available commands")
                
                result = CommandResult.error_result(
                    f"Unknown command: '{command_name}'",
                    errors=error_msg
                )
            else:
                # Treat as block-specific command: <block_name> <command> [args...]
                Log.info(f"Treating as block command: block='{command_name}', command='{parts[1]}'")
                block_command_args, block_command_kwargs = self._parse_args(parts[2:])
                result = self._handle_block_command_fallback(
                    block_identifier=command_name,
                command_name=parts[1].lower(),
                    args=block_command_args,
                    kwargs=block_command_kwargs
                )

        self._last_result = result
        self._log_result(result)
        return result

    def _handle_block_command_fallback(
        self,
        block_identifier: str,
        command_name: str,
        args: List[str],
        kwargs: Dict[str, str]
    ) -> CommandResult:
        """Attempt to run a block-specific command when no registry entry is found."""
        return self.facade.execute_block_command(block_identifier, command_name, args, kwargs)
    
    def _find_fuzzy_command_match(self, command: str) -> Optional[str]:
        """
        Find a fuzzy match for a command (simple matching without underscores).
        Returns the best match or None.
        """
        # Remove underscores and lowercase for comparison
        command_normalized = command.replace('_', '').replace('-', '').lower()
        
        all_commands = self._command_registry.list_commands()
        for cmd_def in all_commands:
            # Check main command name
            cmd_normalized = cmd_def.name.replace('_', '').replace('-', '').lower()
            if cmd_normalized == command_normalized:
                return cmd_def.name
            
            # Check aliases
            for alias in cmd_def.aliases:
                alias_normalized = alias.replace('_', '').replace('-', '').lower()
                if alias_normalized == command_normalized:
                    return cmd_def.name
        
        return None
    
    def _suggest_similar_command(self, command: str) -> Optional[str]:
        """
        Suggest a similar command based on simple string distance.
        """
        all_commands = self._command_registry.list_commands()
        best_match = None
        best_distance = float('inf')
        
        for cmd_def in all_commands:
            # Calculate simple Levenshtein distance
            distance = self._levenshtein_distance(command.lower(), cmd_def.name.lower())
            
            # Only suggest if within 2 edits
            if distance < best_distance and distance <= 2:
                best_distance = distance
                best_match = cmd_def.name
        
        return best_match
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings (simple implementation)."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _register_commands(self):
        """Register the CLI command metadata and handlers."""
        self._command_registry.register(CommandDefinition(
            name="new",
            aliases=["create_project"],
            usage="new [name] [directory]",
            description="Create a new project (untitled by default).",
            handler=self._cmd_new_project
        ))
        self._command_registry.register(CommandDefinition(
            name="load",
            aliases=["load_project"],
            usage="load <project_id|name>",
            description="Load an existing project by ID or name.",
            handler=self._cmd_load_project
        ))
        self._command_registry.register(CommandDefinition(
            name="save",
            aliases=["save_project"],
            usage="save",
            description="Save the currently loaded project.",
            handler=self._cmd_save_project
        ))
        self._command_registry.register(CommandDefinition(
            name="save_as",
            aliases=["saveas"],
            usage="save_as <directory> [name=<name>]",
            description="Save the current project to a new location.",
            handler=self._cmd_save_as_project
        ))
        self._command_registry.register(CommandDefinition(
            name="delete_project",
            usage="delete_project [project_id]",
            description="Delete a project (defaults to the active project).",
            handler=self._cmd_delete_project
        ))
        self._command_registry.register(CommandDefinition(
            name="reset_session",
            aliases=["reset_runtime", "clear_session"],
            usage="reset_session",
            description="Reset the in-memory runtime cache so the next load starts clean.",
            handler=self._cmd_reset_session
        ))
        self._command_registry.register(CommandDefinition(
            name="add_block",
            aliases=["addblock", "add"],
            usage="add_block <type> [name]",
            description="Add a block of the given type to the current project.",
            handler=self._cmd_add_block
        ))
        self._command_registry.register(CommandDefinition(
            name="delete_block",
            aliases=["deleteblock", "delete"],
            usage="delete_block <block_id|name>",
            description="Remove a block from the current project.",
            handler=self._cmd_delete_block
        ))
        self._command_registry.register(CommandDefinition(
            name="list_blocks",
            aliases=["listblocks"],
            usage="list_blocks",
            description="Show all blocks in the current project.",
            handler=self._cmd_list_blocks
        ))
        self._command_registry.register(CommandDefinition(
            name="list_block_types",
            aliases=["list_types", "blocktypes"],
            usage="list_block_types",
            description="List available block types.",
            handler=self._cmd_list_block_types
        ))
        self._command_registry.register(CommandDefinition(
            name="rename_block",
            usage="rename_block <block_id|name> <new_name>",
            description="Rename a block.",
            handler=self._cmd_rename_block
        ))
        self._command_registry.register(CommandDefinition(
            name="block_help",
            aliases=["block_info"],
            usage="block_help <block_id|name>",
            description="Show commands and port information for a block.",
            handler=self._cmd_block_help
        ))
        self._command_registry.register(CommandDefinition(
            name="connect",
            usage="connect <src_id> <src_output> <tgt_id> <tgt_input>",
            description="Connect two blocks.",
            handler=self._cmd_connect
        ))
        self._command_registry.register(CommandDefinition(
            name="disconnect",
            usage="disconnect <connection_id|block_name port_name|block_name all>",
            description="Disconnect connections (by ID, specific port, or all from a block).",
            handler=self._cmd_disconnect
        ))
        self._command_registry.register(CommandDefinition(
            name="list_connections",
            usage="list_connections",
            description="List all connections in the project.",
            handler=self._cmd_list_connections
        ))
        self._command_registry.register(CommandDefinition(
            name="execute_block",
            aliases=["run_block", "process_block"],
            usage="execute_block <block_id|name>",
            description="Execute a single block (useful for testing or debugging).",
            handler=self._cmd_execute_block
        ))
        self._command_registry.register(CommandDefinition(
            name="validate",
            usage="validate [project_id]",
            description="Validate the project graph.",
            handler=self._cmd_validate
        ))
        self._command_registry.register(CommandDefinition(
            name="batch",
            aliases=["batch_run", "run_batch"],
            usage="batch <project.ez> <input_pattern> <input_block> <output_dir> [--stop-on-error]",
            description="Run batch processing: execute pipeline for each input file.",
            handler=self._cmd_batch
        ))
        self._command_registry.register(CommandDefinition(
            name="validate_data",
            usage="validate_data [project_id]",
            description="Validate all data items and check file paths.",
            handler=self._cmd_validate_data
        ))
        self._command_registry.register(CommandDefinition(
            name="get_block_data",
            aliases=["block_data", "inspect_block"],
            usage="get_block_data <block_id|name>",
            description="Get all data items (outputs) for a specific block.",
            handler=self._cmd_get_block_data
        ))
        self._command_registry.register(CommandDefinition(
            name="get_data_item",
            aliases=["data_item_details", "inspect_data"],
            usage="get_data_item <data_item_id>",
            description="Get detailed information about a specific data item.",
            handler=self._cmd_get_data_item_details
        ))
        self._command_registry.register(CommandDefinition(
            name="get_port_data",
            aliases=["port_data", "inspect_port"],
            usage="get_port_data <block_id|name> <port_name>",
            description="Get summary of data items on a specific block output port.",
            handler=self._cmd_get_port_data_summary
        ))
        self._command_registry.register(CommandDefinition(
            name="list_separator_models",
            aliases=["separator_models", "demucs_models"],
            usage="list_separator_models",
            description="List available Demucs model types for the Separator block.",
            handler=self._cmd_list_separator_models
        ))
        self._command_registry.register(CommandDefinition(
            name="help",
            aliases=["?", "commands"],
            usage="help",
            description="Display the CLI command reference.",
            handler=self._cmd_help
        ))
    
    def _cmd_new_project(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        name = args[0] if args else kwargs.get("name", "Untitled")
        save_dir = kwargs.get("directory") or kwargs.get("dir")
        if not save_dir and len(args) > 1:
            save_dir = args[1]
        return self.facade.create_project(name, save_dir)

    def _cmd_load_project(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if not args:
            return CommandResult.error_result("Usage: load <project_id|name>")
        return self.facade.load_project(args[0])

    def _cmd_save_project(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        return self.facade.save_project()

    def _cmd_save_as_project(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        save_directory = kwargs.get("directory") or kwargs.get("dir") or (args[0] if args else None)
        name = kwargs.get("name") or (args[1] if len(args) > 1 else None)
        if not save_directory:
            return CommandResult.error_result("Usage: save_as <directory> [name=<name>]")
        return self.facade.save_project_as(save_directory, name)

    def _cmd_delete_project(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        project_id = args[0] if args else None
        return self.facade.delete_project(project_id)

    def _cmd_reset_session(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        result = self.facade.reset_session()
        if result.success:
            Log.info("Session reset; no project is currently loaded.")
        return result

    def _cmd_add_block(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if not args:
            return CommandResult.error_result(
                "Usage: add_block <type> [name]",
                errors=["Use 'list_block_types' to see available types"]
            )
        block_type = args[0]
        name = args[1] if len(args) > 1 else None
        return self.facade.add_block(block_type, name)

    def _cmd_delete_block(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if not args:
            return CommandResult.error_result("Usage: delete_block <block_id|name>")
        return self.facade.delete_block(args[0])

    def _cmd_list_blocks(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        result = self.facade.list_blocks()
        if result.success:
            self._format_block_list(result.data)
        return result

    def _cmd_list_block_types(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        result = self.facade.list_block_types()
        if result.success:
            self._format_block_types(result.data)
        return result

    def _cmd_list_separator_models(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """List available Demucs models for the Separator block."""
        models = {
            "htdemucs_ft": {
                "description": "Hybrid Transformer Demucs (Fine-Tuned)",
                "quality": "Best",
                "speed": "Slower",
                "stems": 4
            },
            "htdemucs": {
                "description": "Hybrid Transformer Demucs",
                "quality": "Good",
                "speed": "Fast",
                "stems": 4
            },
            "htdemucs_6s": {
                "description": "Hybrid Transformer Demucs (6-stem)",
                "quality": "Best",
                "speed": "Slowest",
                "stems": 6
            },
            "mdx_extra": {
                "description": "MDX Extra Quality",
                "quality": "Very Good",
                "speed": "Medium",
                "stems": 4
            },
            "mdx_extra_q": {
                "description": "MDX Extra Quality (Quantized)",
                "quality": "Very Good",
                "speed": "Fast",
                "stems": 4
            }
        }
        
        Log.info("=" * 80)
        Log.info("Available Demucs Models for Separator Block")
        Log.info("=" * 80)
        Log.info("")
        
        for model_name, info in models.items():
            Log.info(f"  {model_name}")
            Log.info(f"    Description: {info['description']}")
            Log.info(f"    Quality: {info['quality']} | Speed: {info['speed']} | Stems: {info['stems']}")
            Log.info("")
        
        Log.info("Usage:")
        Log.info("  Set model in block metadata: metadata={'model': 'htdemucs'}")
        Log.info("")
        Log.info("Optimization Options:")
        Log.info("  device: 'auto' (default), 'cpu', 'cuda' (NVIDIA GPU)")
        Log.info("  output_format: 'wav' (default), 'mp3' (faster, smaller)")
        Log.info("  mp3_bitrate: '320' (default), '192', '128'")
        Log.info("  two_stems: None (default 4-stem) or 'vocals', 'drums', 'bass', 'other' (outputs 2 files)")
        Log.info("=" * 80)
        
        return CommandResult.success_result(
            message="Listed available Demucs models",
            data=models
        )

    def _cmd_rename_block(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if len(args) < 2:
            return CommandResult.error_result("Usage: rename_block <block_id|name> <new_name>")
        return self.facade.rename_block(args[0], args[1])

    def _cmd_block_help(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if not args:
            return CommandResult.error_result("Usage: block_help <block_id|name>")
        result = self.facade.describe_block(args[0])
        if result.success:
            self._format_block_help(result.data)
        return result

    def _cmd_connect(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        if len(args) < 4:
            return CommandResult.error_result("Usage: connect <src_id> <src_output> <tgt_id> <tgt_input>")
        return self.facade.connect_blocks(args[0], args[1], args[2], args[3])

    def _cmd_disconnect(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """
        Disconnect blocks with multiple modes:
        - disconnect <connection_id>              # Disconnect by ID
        - disconnect <block_name> <port_name>     # Disconnect specific input port
        - disconnect <block_name> all             # Disconnect all connections from block
        """
        if not args:
            return CommandResult.error_result(
                "Usage:\n"
                "  disconnect <connection_id>\n"
                "  disconnect <block_name> <port_name>\n"
                "  disconnect <block_name> all"
            )
        
        # Mode 1: Single argument - assume connection ID
        if len(args) == 1:
            return self.facade.disconnect_blocks(args[0])
        
        # Mode 2 & 3: Two arguments - block and port/all
        elif len(args) == 2:
            block_identifier = args[0]
            port_or_all = args[1].lower()
            
            if port_or_all == "all":
                # Disconnect all connections from block
                return self.facade.disconnect_all_from_block(block_identifier)
            else:
                # Disconnect specific port
                return self.facade.disconnect_by_port(block_identifier, args[1])
        
        else:
            return CommandResult.error_result(
                "Too many arguments. Usage:\n"
                "  disconnect <connection_id>\n"
                "  disconnect <block_name> <port_name>\n"
                "  disconnect <block_name> all"
            )

    def _cmd_list_connections(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        result = self.facade.list_connections()
        if result.success:
            self._format_connection_list(result.data)
        return result

    def _cmd_execute_block(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """Execute a single block by ID or name"""
        if not args:
            return CommandResult.error_result(
                "Usage: execute_block <block_id|name>",
                errors=[
                    "Specify which block to execute",
                    "Example: execute_block MyBlock1",
                    "Use 'listblocks' to see available blocks"
                ]
            )
        
        block_identifier = args[0]
        Log.info(f"Attempting to execute single block: '{block_identifier}'")
        result = self.facade.execute_single_block(block_identifier)
        
        if result.success:
            Log.info(f"Successfully executed block '{block_identifier}'")
            # Show output if any
            if result.data and isinstance(result.data, dict):
                outputs = result.data.get('outputs', {})
                if outputs:
                    Log.info("Block outputs:")
                    for port_name, data_value in outputs.items():
                        if isinstance(data_value, list):
                            Log.info(f"  {port_name}: [{len(data_value)} items]")
                            for item in data_value:
                                Log.info(f"    - {item.name} ({item.type})")
                        else:
                            Log.info(f"  {port_name}: {data_value.name} ({data_value.type})")
        else:
            # Enhance error message with more context
            error_msgs = result.errors or []
            enhanced_errors = [
                f"Failed to execute block: '{block_identifier}'",
                f"Reason: {result.message}"
            ]
            enhanced_errors.extend(error_msgs)
            enhanced_errors.append("Use 'listblocks' to see available blocks")
            result.errors = enhanced_errors
        
        return result

    def _cmd_validate(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        project_id = args[0] if args else None
        return self.facade.validate_project(project_id)
    
    def _cmd_batch(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """
        Run batch processing.
        
        Usage:
            batch <project.ez> <input_pattern> <input_block> <output_dir> [--stop-on-error]
            
        Examples:
            batch ./pipeline.ez "./songs/*.mp3" LoadAudio1 ./exports
            batch ./sep.ez /music/tracks Separator1 ./stems --stop-on-error
        """
        if len(args) < 4:
            return CommandResult.error_result(
                message="Usage: batch <project.ez> <input_pattern> <input_block> <output_dir>",
                errors=[
                    "project.ez   - Path to project file (template)",
                    "input_pattern - File path, glob pattern, or directory",
                    "input_block  - Block name to inject input (e.g., LoadAudio1)",
                    "output_dir   - Base directory for outputs",
                    "",
                    "Options:",
                    "  --stop-on-error  Stop on first error (default: continue)",
                    "  --command=<cmd>  Input command (default: set_path)",
                    "",
                    "Examples:",
                    '  batch ./pipeline.ez "./songs/*.mp3" LoadAudio1 ./exports',
                    "  batch ./separator.ez /music/tracks Separator1 ./stems"
                ]
            )
        
        project_path = args[0]
        input_pattern = args[1]
        input_block = args[2]
        output_dir = args[3]
        
        stop_on_error = kwargs.get('stop-on-error', 'false').lower() == 'true'
        input_command = kwargs.get('command', 'set_path')
        
        Log.info(f"Starting batch processing...")
        Log.info(f"  Project:  {project_path}")
        Log.info(f"  Input:    {input_pattern}")
        Log.info(f"  Block:    {input_block}")
        Log.info(f"  Output:   {output_dir}")
        
        return self.facade.run_batch(
            project_path=project_path,
            input_pattern=input_pattern,
            input_block=input_block,
            output_dir=output_dir,
            input_command=input_command,
            stop_on_error=stop_on_error
        )
    
    def _cmd_validate_data(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """Validate data items and check file paths"""
        project_id = args[0] if args else None
        result = self.facade.validate_data_items(project_id)
        
        # Enhanced output for data validation
        if result.success and result.data:
            data = result.data
            invalid_items = data.get('invalid_items', [])
            
            if invalid_items:
                Log.warning(f"\nInvalid Data Items ({len(invalid_items)}):")
                for item in invalid_items:
                    Log.warning(f"  • {item['block_name']}: {item['data_item_name']}")
                    Log.warning(f"    File: {item['file_path']}")
                    Log.warning(f"    Reason: {item['reason']}")
                    Log.warning("")
                
                Log.info("Action: Re-execute affected blocks or update file paths")
        
        return result
    
    def _cmd_get_block_data(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """Get all data items for a specific block"""
        if not args:
            return CommandResult.error_result("Usage: get_block_data <block_id|name>")
        
        result = self.facade.get_block_data(args[0])
        
        if result.success and result.data:
            data = result.data
            data_items = data['data_items']
            
            Log.info("=" * 80)
            Log.info(f"Data Items for Block: {data['block_name']} ({data['block_type']})")
            Log.info("=" * 80)
            
            if not data_items:
                Log.info("  No data items found (block has not been executed)")
            else:
                for item in data_items:
                    Log.info(f"\n  • {item['name']} ({item['type']})")
                    Log.info(f"    ID: {item['id']}")
                    Log.info(f"    Port: {item['output_port']}")
                    
                    if item.get('file_path'):
                        status = "" if item.get('file_valid') else ""
                        Log.info(f"    File: {status} {item['file_path']}")
                    
                    if item['type'] == 'Audio':
                        if item.get('sample_rate'):
                            Log.info(f"    Sample Rate: {item['sample_rate']} Hz")
                        if item.get('length_ms'):
                            duration_sec = round(item['length_ms'] / 1000, 2)
                            Log.info(f"    Duration: {duration_sec}s")
                    
                    elif item['type'] == 'Event':
                        event_count = item.get('event_count', 0)
                        has_events = item.get('has_events', False)
                        status = f"{event_count} events" if has_events else f"{event_count} events (not in memory)"
                        Log.info(f"    Events: {status}")
            
            Log.info("=" * 80)
        
        return result
    
    def _cmd_get_data_item_details(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """Get detailed information about a specific data item"""
        if not args:
            return CommandResult.error_result("Usage: get_data_item <data_item_id>")
        
        result = self.facade.get_data_item_details(args[0])
        
        if result.success and result.data:
            data = result.data
            
            Log.info("=" * 80)
            Log.info(f"Data Item Details: {data['name']}")
            Log.info("=" * 80)
            Log.info(f"  ID: {data['id']}")
            Log.info(f"  Type: {data['type']}")
            Log.info(f"  Block ID: {data['block_id']}")
            Log.info(f"  Port: {data['output_port']}")
            Log.info(f"  Created: {data['created_at']}")
            
            if data.get('file_path'):
                Log.info(f"\n  File Information:")
                status = " Exists" if data.get('file_exists') else " Missing"
                Log.info(f"    Status: {status}")
                Log.info(f"    Path: {data['file_path']}")
                
                if data.get('file_exists'):
                    Log.info(f"    Name: {data['file_name']}")
                    Log.info(f"    Size: {data['file_size_mb']} MB")
                    Log.info(f"    Directory: {data['file_directory']}")
            
            if data['type'] == 'Audio' and 'audio' in data:
                audio = data['audio']
                Log.info(f"\n  Audio Properties:")
                Log.info(f"    Sample Rate: {audio['sample_rate']} Hz")
                if audio.get('length_seconds'):
                    Log.info(f"    Duration: {audio['length_seconds']}s")
                if audio.get('channels'):
                    Log.info(f"    Channels: {audio['channels']}")
                if audio.get('original_path'):
                    Log.info(f"    Original: {audio['original_path']}")
            
            elif data['type'] == 'Event' and 'events' in data:
                events = data['events']
                Log.info(f"\n  Event Properties:")
                Log.info(f"    Count: {events['event_count']}")
                Log.info(f"    In Memory: {events['has_events_in_memory']}")
                
                if events.get('source_audio'):
                    Log.info(f"    Source: {events['source_audio']}")
                if events.get('extractor'):
                    Log.info(f"    Extractor: {events['extractor']}")
                
                if events.get('sample_events'):
                    Log.info(f"\n    Sample Events (first 5):")
                    for i, event in enumerate(events['sample_events'], 1):
                        Log.info(f"      {i}. Time: {event['time']}s, Class: {event['classification']}, Duration: {event['duration']}s")
            
            Log.info("=" * 80)
        
        return result
    
    def _cmd_get_port_data_summary(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        """Get summary of data on a specific port"""
        if len(args) < 2:
            return CommandResult.error_result("Usage: get_port_data <block_id|name> <port_name>")
        
        block_identifier = args[0]
        port_name = args[1]
        
        result = self.facade.get_port_data_summary(block_identifier, port_name)
        
        if result.success and result.data:
            data = result.data
            
            Log.info("=" * 80)
            Log.info(f"Port Data Summary: {data['block_name']}.{data['port_name']}")
            Log.info("=" * 80)
            Log.info(f"  Block: {data['block_name']} ({data['block_type']})")
            Log.info(f"  Port: {data['port_name']} ({data['port_type']})")
            Log.info(f"  Item Count: {data['item_count']}")
            
            if data.get('total_size_mb', 0) > 0:
                Log.info(f"  Total Size: {data['total_size_mb']} MB")
            
            if data['items']:
                Log.info(f"\n  Items:")
                for item in data['items']:
                    status = "" if item.get('file_valid') else ""
                    Log.info(f"    • {item['name']} ({item['type']}) {status}")
                    
                    if item.get('file_size_mb'):
                        Log.info(f"      Size: {item['file_size_mb']} MB")
                    
                    if item.get('duration_ms'):
                        duration_sec = round(item['duration_ms'] / 1000, 2)
                        Log.info(f"      Duration: {duration_sec}s")
                    
                    if item.get('event_count'):
                        Log.info(f"      Events: {item['event_count']}")
            
            Log.info("=" * 80)
        
        return result

    def _cmd_help(self, args: List[str], kwargs: Dict[str, str]) -> CommandResult:
        return self._show_help()

    def _log_result(self, result: CommandResult):
        """
        Log command result appropriately for CLI.
        
        Args:
            result: CommandResult to log
        """
        # Log message
        if result.success:
            Log.info(result.message)
        elif result.status == ResultStatus.WARNING:
            Log.warning(result.message)
        else:
            Log.error(result.message)
        
        # Log warnings
        for warning in result.warnings:
            Log.warning(f"  {warning}")
        
        # Log errors
        for error in result.errors:
            Log.error(f"  {error}")
    
    def set_current_project(self, project_id: str):
        """Set the current active project (compatibility helper)."""
        self.facade.current_project_id = project_id

    def get_current_project_id(self) -> Optional[str]:
        """Return the current active project ID."""
        return self.facade.current_project_id

    def get_return_data(self):
        """Return the payload from the last command result."""
        return self._last_result.data if self._last_result else None

    def set_execution_engine(self, execution_engine):
        """No-op shim for legacy tests/commands that expect this method."""
        pass
    
    def get_last_result(self) -> Optional[CommandResult]:
        """Return the most recent CommandResult produced by parse_and_execute."""
        return self._last_result
    def _format_block_list(self, blocks):
        """Format block list for CLI output"""
        if not blocks:
            return
        
        Log.info(f"Blocks in project ({len(blocks)}):")
        for block in blocks:
            # Check if block has cached outputs
            has_data = self.facade.has_cached_outputs(block.id)
            data_count = self.facade.get_cached_output_count(block.id)
            
            if has_data:
                data_status = f"[Has Data: {data_count} item(s)]"
            else:
                data_status = "[No Data]"
            
            Log.info(f"  - {block.name} (type: {block.type}) {data_status}")
    
    def _format_block_types(self, categories: Dict):
        """Format block types for CLI output"""
        Log.info(f"Available Block Types:")
        Log.info("=" * 60)
        
        for category in sorted(categories.keys()):
            Log.info(f"\n{category}:")
            for bt in sorted(categories[category], key=lambda x: x['name']):
                Log.info(f"  {bt['type_id']:20} - {bt['name']}")
                if bt['description']:
                    Log.info(f"    {bt['description']}")
                
                # Show ports
                if bt['inputs']:
                    inputs_str = ", ".join([f"{name}({pt})" for name, pt in bt['inputs'].items()])
                    Log.info(f"    Inputs: {inputs_str}")
                if bt['outputs']:
                    outputs_str = ", ".join([f"{name}({pt})" for name, pt in bt['outputs'].items()])
                    Log.info(f"    Outputs: {outputs_str}")
        
        Log.info("=" * 60)
        Log.info("\nUsage: add_block <type_id> [block_name]")
    
    def _format_block_help(self, data):
        """Format block-specific command help for CLI output"""
        if not data:
            return

        block = data.get("block")
        commands = data.get("commands", [])
        inputs = data.get("inputs", {})
        outputs = data.get("outputs", {})
        metadata = data.get("block_type_metadata")

        if not block:
            return

        Log.info(f"Block help for '{block.name}' (type: {block.type}, id: {block.id})")
        
        # Show cached data status
        has_data = self.facade.has_cached_outputs(block.id)
        data_count = self.facade.get_cached_output_count(block.id)
        
        if has_data:
            Log.info(f"  Cached Data: {data_count} output item(s) available")
        else:
            Log.info(f"  Cached Data: No cached outputs")
        
        Log.info("")  # Blank line for readability

        if commands:
            Log.info("  Commands:")
            for cmd in commands:
                usage = cmd.get("usage") or cmd.get("name")
                description = cmd.get("description", "")
                Log.info(f"    {cmd.get('name'):15} - {description}")
                if usage:
                    Log.info(f"        Usage: {usage}")
        else:
            Log.info("  No block-specific commands available.")

        self._log_block_ports("Input Types", metadata.inputs if metadata else {}, inputs)
        self._log_block_ports("Output Types", metadata.outputs if metadata else {}, outputs)

    def _log_block_ports(self, title, metadata_ports, current_ports):
        """Log port definitions combining metadata and actual port info."""
        if not metadata_ports and not current_ports:
            return

        Log.info(f"  {title}:")
        if metadata_ports:
            for name, port in metadata_ports.items():
                type_name = getattr(port, "name", str(port))
                Log.info(f"    {name} ({type_name}) [defined]")
        elif current_ports:
            for name, type_name in current_ports.items():
                Log.info(f"    {name} ({type_name})")
    
    def _format_connection_list(self, connections):
        """Format connection list for CLI output"""
        if not connections:
            return
        
        Log.info(f"Connections in project ({len(connections)}):")
        for conn in connections:
            Log.info(f"  {conn.source_block_id}.{conn.source_output_name} -> "
                    f"{conn.target_block_id}.{conn.target_input_name}")
    
    def _format_execution_result(self, exec_result):
        """Format execution result for CLI output"""
        if not exec_result:
            return
        
        if exec_result.output_data:
            Log.info("Output data:")
            for block_id, outputs in exec_result.output_data.items():
                for port_name, data_value in outputs.items():
                    if isinstance(data_value, list):
                        # Handle list of data items
                        if len(data_value) == 1:
                            Log.info(f"  {block_id}.{port_name}: {data_value[0].name} ({data_value[0].type})")
                        else:
                            Log.info(f"  {block_id}.{port_name}: [{len(data_value)} items]")
                            for item in data_value:
                                Log.info(f"    - {item.name} ({item.type})")
                    else:
                        # Handle single data item
                        Log.info(f"  {block_id}.{port_name}: {data_value.name} ({data_value.type})")
    
    def _show_help(self) -> CommandResult:
        """Show help information"""
        Log.info("=" * 60)
        Log.info("EchoZero Command Reference")
        Log.info("=" * 60)
        Log.info("")
        Log.info("Project Commands:")
        Log.info("  new                              - Create new untitled project")
        Log.info("  load <project_id|name>            - Load existing project")
        Log.info("  save                              - Save current project")
        Log.info("  save_as <directory> [name=<name>] - Save project with location and name")
        Log.info("                                    (name defaults to directory name if not provided)")
        Log.info("  delete_project [project_id]       - Delete project")
        Log.info("")
        Log.info("Block Commands:")
        Log.info("  add_block <type> [name]           - Add block to current project")
        Log.info("  delete_block <block_id|name>      - Delete block")
        Log.info("  list_blocks                       - List all blocks in project (shows cached data)")
        Log.info("  list_block_types                  - List all available block types")
        Log.info("  list_separator_models             - List available Demucs models")
        Log.info("  rename_block <id> <new_name>     - Rename block")
        Log.info("  block_help <block_id|name>        - Show commands & ports for a block")
        Log.info("  <block_id|name> <command> [...]    - Execute block-specific command")
        Log.info("")
        Log.info("Connection Commands:")
        Log.info("  connect <src_id> <out> <tgt_id> <in> - Connect blocks")
        Log.info("  disconnect <connection_id>        - Disconnect by connection ID")
        Log.info("  disconnect <block> <port>         - Disconnect specific input port")
        Log.info("  disconnect <block> all            - Disconnect all connections from block")
        Log.info("  list_connections                  - List all connections in project")
        Log.info("")
        Log.info("Execution Commands:")
        Log.info("  execute_block <block_id|name>     - Execute single block (uses cached inputs)")
        Log.info("  run_block <block_id|name>         - Alias for execute_block")
        Log.info("  validate [project_id]             - Validate project graph")
        Log.info("")
        Log.info("Data Management:")
        Log.info("  listblocks shows cached data status - [Has Data: X item(s)] or [No Data]")
        Log.info("  execute_block - preserves cached data, tests single block")
        Log.info("")
        Log.info("Help:")
        Log.info("  help                              - Show this help")
        Log.info("  ?                                 - Alias for help")
        Log.info("")
        Log.info("=" * 60)
        
        return CommandResult.success_result("Help displayed")
    
    def _parse_command_parts(self, input_string: str) -> List[str]:
        """Parse command string into parts while respecting quotes"""
        parts = []
        current_part = ""
        in_quotes = False
        i = 0
        
        while i < len(input_string):
            char = input_string[i]
            
            if char == '"' and (i == 0 or input_string[i-1] != '\\'):
                in_quotes = not in_quotes
                # Don't include quotes in final parts
            elif char.isspace() and not in_quotes:
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            else:
                current_part += char
            i += 1
        
        if current_part:
            parts.append(current_part)
        
        return parts
    
    def _parse_args(self, parts: List[str]) -> tuple[List[str], Dict[str, str]]:
        """
        Parse argument parts into args and kwargs.
        
        Args:
            parts: List of argument strings
            
        Returns:
            Tuple of (args list, kwargs dict)
        """
        args = []
        kwargs = {}
        
        for part in parts:
            if '=' in part:
                # This is a kwarg
                key, value = part.split('=', 1)
                kwargs[key.strip()] = value.strip()
            else:
                # This is a positional arg
                args.append(part)
        
        return args, kwargs
