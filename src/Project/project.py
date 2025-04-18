from src.Command.command_controller import CommandController
from src.Command.command_queue import CommandQueue
from src.Database.project_database_controller import ProjectDatabaseController
from src.Utils.message import Log
import os
import time
import json

class Project:
    """
    Project class with dedicated database and command queue.
    Manages blocks with their individual databases and provides a command interface.
    """
    
    def __init__(self, project_id=None, name="Unnamed Project"):
        """
        Initialize a project.
        
        Args:
            project_id (int): Optional project ID for loading existing project
            name (str): Project name
        """
        self.id = project_id or int(time.time() * 1000)  # Generate ID if not provided
        self.name = name
        self.description = ""
        
        # Initialize project database
        self.db = ProjectDatabaseController(self.id, self.name)
        
        # Initialize command queue
        self.command_queue = CommandQueue()
        # Start the command queue worker thread
        self.command_queue.start()
        
        # Initialize command controller
        self.command = CommandController(self.db, self)
        self.command.set_name("ProjectCommands")
        
        # Register commands
        self._register_commands()
        
        # Cache of loaded blocks, keyed by block ID
        self.blocks = {}
    
    def _register_commands(self):
        """Register project commands."""
        self.command.add("info", self.cmd_info, "Display project information")
        self.command.add("create-block", self.cmd_create_block, "Create a new block")
        self.command.add("list-blocks", self.cmd_list_blocks, "List all blocks")
        self.command.add("delete-block", self.cmd_delete_block, "Delete a block")
        self.command.add("rename", self.cmd_rename, "Rename the project")
        self.command.add("help", self.cmd_help, "Display help information")
        self.command.add("queue-status", self.cmd_queue_status, "Show command queue status")
    
    def save(self):
        """
        Save project metadata.
        
        Returns:
            bool: True if successful
        """
        return False
    
    def load(self):
        """
        Load project metadata.
        
        Returns:
            bool: True if successful
        """
        return False
    
    def close(self):
        """
        Close the project and all associated resources.
        
        Returns:
            bool: True if successful
        """
        # Stop command queue
        self.command_queue.stop()
        
        # Close database connections
        if self.db:
            self.db.close_connection()
            
        # Close block databases
        for block in self.blocks.values():
            if hasattr(block, 'db') and block.db:
                block.db.close_connection()
                
        return True
    
    def delete(self):
        """
        Delete the project, its database, and all associated blocks.
        
        Returns:
            bool: True if successful
        """
        # First close the project
        self.close()
        
        # Delete block databases
        for block in list(self.blocks.values()):
            block.delete()
            
        # Delete project database file
        if os.path.exists(self.db.db_path):
            try:
                os.remove(self.db.db_path)
                Log.info(f"Deleted project database file for project ID {self.id}")
            except Exception as e:
                Log.error(f"Failed to delete project database file: {str(e)}")
                return False
                
        return True
    
    def create_block(self, block_type, name=None):
        """
        Create a new block of the specified type.
        
        Args:
            block_type (str): Type of block to create
            name (str): Optional name for the block
            
        Returns:
            Block: Created block or None if failed
        """
        # Use command queue to safely create the block
        cmd_name = f"Create block: {block_type}"
        cmd_id = self.command_queue.create_and_enqueue(
            cmd_name,
            self._create_block_internal,
            [block_type, name]
        )
        
        # Wait for command to complete
        while True:
            cmd = self.command_queue.get_command_by_id(cmd_id)
            if cmd and cmd.status.value in ["completed", "failed"]:
                return cmd.result
            time.sleep(0.1)
    
    def _create_block_internal(self, block_type, name=None):
        """
        Internal method to create a block (executed by command queue).
        
        Args:
            block_type (str): Type of block to create
            name (str): Optional name for the block
            
        Returns:
            Block: Created block or None if failed
        """
        try:
            # Create the block using the block factory
            block = self.block_factory.create_block(block_type, self)
            
            if not block:
                Log.error(f"Failed to create block of type: {block_type}")
                return None
                
            # Set name if provided
            if name:
                block.name = name
                
            # Save the block to register it with the project
            if not block.save():
                Log.error(f"Failed to save block: {name or block_type}")
                return None
                
            # Add to cache
            self.blocks[block.id] = block
            
            Log.info(f"Created block: {block.name} (ID: {block.id})")
            return block
            
        except Exception as e:
            Log.error(f"Error creating block: {str(e)}")
            return None
    
    def get_block(self, block_id):
        """
        Get a block by ID.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            Block: Block instance or None if not found
        """
        # Check cache first
        if block_id in self.blocks:
            return self.blocks[block_id]
            
        # Not in cache, look up in database
        block_info = self.db.get_block_info(block_id=block_id)
        if not block_info:
            return None
            
        # Load the block
        block = self.block_factory.create_block(block_info['type'], self, block_id)
        if block:
            # Add to cache
            self.blocks[block_id] = block
            
        return block
    
    def get_block_by_name(self, name):
        """
        Get a block by name.
        
        Args:
            name (str): Block name
            
        Returns:
            Block: Block instance or None if not found
        """
        # Check cache first
        for block in self.blocks.values():
            if block.name == name:
                return block
                
        # Not in cache, look up in database
        block_info = self.db.get_block_info(block_name=name)
        if not block_info:
            return None
            
        # Load the block
        block = self.block_factory.create_block(block_info['type'], self, block_info['block_id'])
        if block:
            # Add to cache
            self.blocks[block.id] = block
            
        return block
    
    def get_all_blocks(self):
        """
        Get all blocks in the project.
        
        Returns:
            list: List of Block instances
        """
        # Get all block info from the database
        block_infos = self.db.get_all_blocks()
        
        # Load any blocks not already in cache
        result = []
        for info in block_infos:
            block_id = info['block_id']
            
            # Check cache first
            if block_id in self.blocks:
                result.append(self.blocks[block_id])
                continue
                
            # Load the block
            block = self.block_factory.create_block(info['type'], self, block_id)
            if block:
                # Add to cache
                self.blocks[block_id] = block
                result.append(block)
                
        return result
    
    def register_block(self, block):
        """
        Register a block with the project.
        
        Args:
            block: Block instance
            
        Returns:
            bool: True if successful
        """
        if not hasattr(block, 'id') or not hasattr(block, 'name') or not hasattr(block, 'type'):
            Log.error("Invalid block instance")
            return False
            
        # Register block in the project database
        if not hasattr(block, 'db') or not block.db:
            Log.error(f"Block '{block.name}' has no database")
            return False
            
        success = self.db.register_block(
            block.id,
            block.name,
            block.type,
            block.db.db_path
        )
        
        if success:
            # Add to cache if not already there
            if block.id not in self.blocks:
                self.blocks[block.id] = block
                
        return success
    
    def update_block(self, block_id, **kwargs):
        """
        Update block metadata in the project registry.
        
        Args:
            block_id (int): Block ID
            **kwargs: Key-value pairs of properties to update
            
        Returns:
            bool: True if successful
        """
        # Get block info
        block_info = self.db.get_block_info(block_id=block_id)
        if not block_info:
            return False
            
        # Update properties
        for key, value in kwargs.items():
            block_info[key] = value
            
        # Register updated info
        return self.db.register_block(
            block_info['block_id'],
            block_info['name'],
            block_info['type'],
            block_info['db_path']
        )
    
    def remove_block(self, block_id):
        """
        Remove a block from the project.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            bool: True if successful
        """
        # Remove from cache
        if block_id in self.blocks:
            del self.blocks[block_id]
            
        # Remove from database
        return self.db.remove_block(block_id)
    
    def connect_blocks(self, source_block_id, source_port, target_block_id, target_port):
        """
        Create a connection between two blocks.
        
        Args:
            source_block_id (int): Source block ID
            source_port (str): Source port name
            target_block_id (int): Target block ID
            target_port (str): Target port name
            
        Returns:
            int: Connection ID or None if failed
        """
        return self.db.connect_blocks(source_block_id, source_port, target_block_id, target_port)
    
    def get_block_connections(self, block_id):
        """
        Get all connections for a block.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            list: List of connection dictionaries
        """
        return self.db.get_block_connections(block_id)
    
    # Command implementations
    def cmd_info(self, args):
        """
        Display information about this project.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        Log.info(f"Project: {self.name}")
        Log.info(f"ID: {self.id}")
        if self.description:
            Log.info(f"Description: {self.description}")
            
        # Get block count
        blocks = self.get_all_blocks()
        Log.info(f"Blocks: {len(blocks)}")
        
        return True
    
    def cmd_create_block(self, args):
        """
        Command to create a new block.
        
        Args:
            args (list): Command arguments [block_type name]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: create-block <block_type> [name]")
            return False
            
        block_type = args[0]
        name = args[1] if len(args) > 1 else None
        
        block = self.create_block(block_type, name)
        return block is not None
    
    def cmd_list_blocks(self, args):
        """
        Command to list all blocks in the project.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        blocks = self.get_all_blocks()
        
        if not blocks:
            Log.info("No blocks in project")
            return True
            
        Log.info(f"Blocks in project '{self.name}':")
        for block in blocks:
            Log.info(f"  {block.name} (Type: {block.type}, ID: {block.id})")
            
        return True
    
    def cmd_delete_block(self, args):
        """
        Command to delete a block.
        
        Args:
            args (list): Command arguments [block_id or block_name]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: delete-block <block_id or block_name>")
            return False
            
        # Try to interpret as block ID
        try:
            block_id = int(args[0])
            block = self.get_block(block_id)
        except ValueError:
            # Not an ID, try name
            block = self.get_block_by_name(args[0])
            
        if not block:
            Log.error(f"Block not found: {args[0]}")
            return False
            
        # Delete the block
        if block.delete():
            Log.info(f"Deleted block: {block.name}")
            return True
            
        Log.error(f"Failed to delete block: {block.name}")
        return False
    
    def cmd_rename(self, args):
        """
        Command to rename the project.
        
        Args:
            args (list): Command arguments [new_name]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: rename <new_name>")
            return False
            
        new_name = args[0]
        old_name = self.name
        self.name = new_name
        
        # Update in database
        self.db._set_metadata("project_name", new_name)
        
        Log.info(f"Renamed project from '{old_name}' to '{new_name}'")
        return True
    
    def cmd_help(self, args):
        """
        Command to display help information.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        Log.info(f"Commands for project '{self.name}':")
        for cmd in self.command.get_commands():
            description = f" - {cmd.description}" if cmd.description else ""
            Log.info(f"  project {cmd.name}{description}")
            
        return True
    
    def cmd_queue_status(self, args):
        """
        Command to show command queue status.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        queue_size = self.command_queue.get_queue_size()
        history = self.command_queue.get_history()
        
        Log.info(f"Command queue status:")
        Log.info(f"  Pending commands: {queue_size}")
        Log.info(f"  Executed commands: {len(history)}")
        
        if args and args[0] == "history" and history:
            Log.info(f"Command history:")
            for cmd in history[-10:]:  # Show last 10 commands
                status = cmd.status.value
                execution_time = cmd.get_execution_time()
                time_str = f" ({execution_time:.2f}s)" if execution_time else ""
                Log.info(f"  {cmd.name} - {status}{time_str}")
                
        return True
    
    def __str__(self):
        """String representation of the project."""
        return f"{self.name} (ID: {self.id})" 