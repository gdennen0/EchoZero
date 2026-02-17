"""
Command Line Interface for EchoZero

This module provides a simple CLI that uses the application facade.
"""

from src.utils.message import Log
from src.utils.tools import prompt, prompt_selection, prompt_yes_no
from src.cli.commands.command_parser import CommandParser
import os
import sys


class CLIInterface:
    """
    Command Line Interface that uses the application facade
    """
    
    def __init__(self, facade):
        """
        Initialize CLI interface.
        
        Args:
            facade: ApplicationFacade instance
        """
        self.facade = facade
        self.command_parser = CommandParser(facade)
        self.running = True
    
    def run(self):
        """Main CLI loop"""
        Log.info("EchoZero CLI Interface")
        Log.info("Type 'help' for available commands, 'quit' to exit")
        
        # Initialize project
        if not self._initialize_project():
            Log.error("Failed to initialize project. Exiting.")
            return
        
        # Main command loop
        while self.running:
            try:
                user_input = prompt("EZ> ").strip()
                if not user_input:
                    continue
                    
                self._process_command(user_input)
                
            except KeyboardInterrupt:
                Log.info("\\nExiting...")
                break
            except Exception as e:
                Log.error(f"Unexpected error: {e}")
    
    def _initialize_project(self) -> bool:
        """Initialize project through CLI prompts"""
        while True:
            response = prompt("Do you want to load an existing project or create a new one? (l/load, n/new): ")
            if response:
                response = response.lower()
                
            if response in ['l', 'load']:
                self._display_recent_projects()
                return self._load_project_interactive()
            elif response in ['n', 'new']:
                return self._new_project_interactive()
            else:
                Log.error("Invalid input. Please enter 'l' for load or 'n' for new.")
        
        return False
    
    def _new_project_interactive(self) -> bool:
        """Create new project interactively"""
        # Just create untitled project - name is set when saving
        command_success = self.command_parser.parse_and_execute("new")
        result = self.command_parser.get_last_result()
        if result and result.data and result.data.is_untitled():
            Log.info("Project name will be set when you save with 'save_as <directory> [name=<name>]'")
        return bool(result and result.success) or command_success
    
    def _load_project_interactive(self) -> bool:
        """Load project interactively"""
        while True:
            identifier = prompt("Enter project ID or name (or 'e' to exit): ").strip()
            if identifier.lower() == 'e':
                return False
            
            if not identifier:
                Log.error("Please enter a project ID or name.")
                continue
                
            # Use command parser
            command_success = self.command_parser.parse_and_execute(f"load {identifier}")
            result = self.command_parser.get_last_result()
            success = bool(result and result.success) or command_success
            if success:
                return True
            else:
                retry = prompt_yes_no("Try another project?")
                if not retry:
                    return False

    def _display_recent_projects(self):
        """Show the list of recent projects before loading."""
        recent = self.facade.list_recent_projects()
        if not recent:
            return

        Log.info("Recent Projects:")
        for entry in recent:
            if isinstance(entry, dict):
                name = entry.get("name", "Untitled")
                project_id = entry.get("project_id") or entry.get("id")
                directory = entry.get("save_directory", "unsaved")
            else:
                name = getattr(entry, "name", "Untitled")
                project_id = getattr(entry, "id", None)
                directory = getattr(entry, "save_directory", "unsaved")
            Log.info(f"  {name} ({project_id}) - {directory}")
    
    def _process_command(self, command: str):
        """Process a command from the user"""
        parts = command.split()
        if not parts:
            return
            
        cmd = parts[0].lower()
        
        # Built-in CLI commands
        if cmd in ['help', 'h']:
            self._show_help()
        elif cmd in ['quit', 'exit', 'q']:
            self.running = False
        else:
            # Send all other commands to the command parser
            # Parser returns CommandResult objects
            result = self.command_parser.parse_and_execute(command)
            # Result output is handled by command parser
    
    def _show_help(self):
        """Show help information"""
        # Use the command parser's help command for consistency
        self.command_parser.parse_and_execute("help")