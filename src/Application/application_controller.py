from Project.project import ProjectV2
from src.Command.command_controller import CommandController
from src.Utils.message import Log
from src.Database.database_controller import DatabaseController
import os
import json
import time
import importlib.util
from src.Block.block_module_controller import BlockModuleController

class ApplicationController:
    """
    Main application controller that manages projects and command execution.
    Uses a command queue for safe, sequential operations.
    """
    
    def __init__(self, data_dir="data"):
        """
        Initialize the application controller.
        
        Args:
            data_dir (str): Directory for storing application data
        """
        # Create directories if they don't exist
        self.data_dir = data_dir
        self.projects_dir = os.path.join(data_dir, "projects")
        self.blocks_dir = os.path.join(data_dir, "blocks")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.projects_dir, exist_ok=True)
        os.makedirs(self.blocks_dir, exist_ok=True)
        
        self.recent_projects_file = os.path.join(self.data_dir, "recent_projects.json")
        
        # Dictionary of open projects, keyed by project ID
        self.projects = {}
        
        # Currently active project
        self.active_project = None
        
        # Application database
        self.db = None
        
        # Command controller for application-level commands
        self.command = CommandController(None, self)
        self.command.set_name("ApplicationCommands")
        self.command.queue.start()

        self._register_commands()
        
        # Try to load an existing installation from data_dir parent
        if os.path.dirname(data_dir):
            parent_dir = os.path.dirname(os.path.abspath(data_dir))
            self.check_and_load_installation(parent_dir)
        
        self.block_module_controller = BlockModuleController(self.blocks_dir) # Load block modules
        self.block_module_controller.load_modules()

    def _register_commands(self):
        """Register application-level commands."""
        self.command.add("new-project", self.cmd_new_project, "Create a new project")
        self.command.add("open-project", self.cmd_open_project, "Open an existing project")
        self.command.add("close-project", self.cmd_close_project, "Close the current project")
        self.command.add("list-projects", self.cmd_list_projects, "List all available projects")
        self.command.add("switch-project", self.cmd_switch_project, "Switch to a different project")
        self.command.add("save-all", self.cmd_save_all, "Save all open projects")
        self.command.add("exit", self.cmd_exit, "Exit the application")
        self.command.add("help", self.cmd_help, "Display help information")
        self.command.add("queue-status", self.cmd_queue_status, "Show command queue status")
        self.command.add("install", self.cmd_install, "Install application in a specified directory")
        self.command.add("load-installation", self.cmd_load_installation, "Load an existing installation")

    def install(self, install_dir):
        """
        Install the application in the specified directory.
        Creates necessary subdirectories and sets up the application database.
        
        Args:
            install_dir (str): Directory to install the application
            
        Returns:
            bool: True if installation was successful
        """
        try:
            # Create install directory if it doesn't exist
            os.makedirs(install_dir, exist_ok=True)
            
            # Create application data directory
            app_data_dir = os.path.join(install_dir, "data")
            os.makedirs(app_data_dir, exist_ok=True)
            
            # Set up database parameters
            db_path = os.path.join(app_data_dir, "application.db")
            app_id = int(time.time() * 1000)  # Generate a unique ID
            db_type = "application"
            
            # Create database controller
            # Note: There's a mismatch between __init__ (3 params) and create method (4 params)
            # The create method expects db_id, db_name, db_type, db_path but the __init__
            # only passes db_id, db_type, db_path to create.
            self.db = DatabaseController(app_id, db_type, db_path)
            
            # Manually create or set attributes
            self.db.create_attribute("install_dir", "str", install_dir)
            self.db.create_attribute("app_data_dir", "str", app_data_dir)
            self.db.create_attribute("installation_date", "str", time.strftime("%Y-%m-%d %H:%M:%S"))
            self.db.create_attribute("app_name", "str", "EchoZero")
            
            # Update instance variables to use the new directories
            self.data_dir = app_data_dir
            self.projects_dir = os.path.join(app_data_dir, "projects")
            self.blocks_dir = os.path.join(app_data_dir, "blocks")
            
            # Create subdirectories
            os.makedirs(self.projects_dir, exist_ok=True)
            os.makedirs(self.blocks_dir, exist_ok=True)
            
            # Update recent projects file path
            self.recent_projects_file = os.path.join(self.data_dir, "recent_projects.json")
            
            Log.info(f"Application installed successfully in: {install_dir}")
            return True
            
        except Exception as e:
            Log.error(f"Error installing application: {str(e)}")
            return False
            
    def create_project(self, name):
        """
        Create a new project.
        
        Args:
            name (str): Project name
            
        Returns:
            ProjectV2: Created project
        """
        # Use command queue to safely create the project
        cmd_name = f"Create project: {name}"
        cmd_id = self.command_queue.create_and_enqueue(
            cmd_name,
            self._create_project_internal,
            [name]
        )
        
        # Wait for command to complete
        while True:
            cmd = self.command_queue.get_command_by_id(cmd_id)
            if cmd and cmd.status.value in ["completed", "failed"]:
                return cmd.result
            time.sleep(0.1)
    
    def _create_project_internal(self, name):
        """
        Internal method to create a project (executed by command queue).
        
        Args:
            name (str): Project name
            
        Returns:
            ProjectV2: Created project or None if failed
        """
        try:
            # Create the project
            project = ProjectV2(name=name)
            
            # Add to open projects
            self.projects[project.id] = project
            
            # Set as active project
            self.active_project = project
            
            # Update recent projects list
            self._update_recent_projects(project.id, name)
            
            Log.info(f"Created new project: {name} (ID: {project.id})")
            return project
            
        except Exception as e:
            Log.error(f"Error creating project: {str(e)}")
            return None
    
    def open_project(self, project_id):
        """
        Open an existing project.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            ProjectV2: Opened project or None if failed
        """
        # Check if already open
        if project_id in self.projects:
            project = self.projects[project_id]
            self.active_project = project
            Log.info(f"Switched to already open project: {project.name}")
            return project
            
        # Use command queue to safely open the project
        cmd_name = f"Open project: {project_id}"
        cmd_id = self.command.queue.create_and_enqueue(
            cmd_name,
            self._open_project_internal,
            [project_id]
        )
        
        # Wait for command to complete
        while True:
            cmd = self.command.queue.get_command_by_id(cmd_id)
            if cmd and cmd.status.value in ["completed", "failed"]:
                return cmd.result
            time.sleep(0.1)
    
    def _open_project_internal(self, project_id):
        """
        Internal method to open a project (executed by command queue).
        
        Args:
            project_id (int): Project ID
            
        Returns:
            ProjectV2: Opened project or None if failed
        """
        try:
            # We need to find the project's database file
            project_db_path = os.path.join(self.projects_dir, f"project_{project_id}.db")
            
            if not os.path.exists(project_db_path):
                Log.error(f"Project database not found: {project_db_path}")
                return None
                
            # Load project metadata to get name
            project_name = self._get_project_name_from_id(project_id)
            if not project_name:
                project_name = f"Project {project_id}"
                
            # Create the project instance
            project = ProjectV2(project_id, project_name)
            
            # Add to open projects
            self.projects[project_id] = project
            
            # Set as active project
            self.active_project = project
            
            # Update recent projects list
            self._update_recent_projects(project_id, project.name)
            
            Log.info(f"Opened project: {project.name} (ID: {project_id})")
            return project
            
        except Exception as e:
            Log.error(f"Error opening project: {str(e)}")
            return None
    
    def _get_project_name_from_id(self, project_id):
        """
        Get project name from ID by looking up recent projects.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            str: Project name or None if not found
        """
        try:
            # Check recent projects
            recent_projects = self.get_recent_projects()
            for project in recent_projects:
                if project.get('id') == project_id:
                    return project.get('name')
                    
            return None
            
        except Exception:
            return None
    
    def close_project(self, project_id=None):
        """
        Close a project.
        
        Args:
            project_id (int): Project ID to close, or None for active project
            
        Returns:
            bool: True if successful
        """
        if project_id is None and self.active_project:
            project_id = self.active_project.id
            
        if project_id is None or project_id not in self.projects:
            return False
            
        project = self.projects[project_id]
        
        # Save the project
        project.save()
        
        # Close the project
        project.close()
        
        # Remove from open projects
        del self.projects[project_id]
        
        # Clear active project if it was the active one
        if self.active_project and self.active_project.id == project_id:
            self.active_project = None
            
            # Set a new active project if there are other open projects
            if self.projects:
                self.active_project = next(iter(self.projects.values()))
                
        Log.info(f"Closed project: {project.name}")
        return True
    
    def save_all_projects(self):
        """
        Save all open projects.
        
        Returns:
            bool: True if all projects saved successfully
        """
        success = True
        for project in list(self.projects.values()):
            if not project.save():
                Log.error(f"Failed to save project: {project.name}")
                success = False
                
        return success
    
    def get_active_project(self):
        """
        Get the active project.
        
        Returns:
            ProjectV2: Active project or None if no project is active
        """
        return self.active_project
    
    def set_active_project(self, project_id):
        """
        Set the active project.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            bool: True if successful
        """
        if project_id not in self.projects:
            Log.error(f"Project not open: {project_id}")
            return False
            
        self.active_project = self.projects[project_id]
        Log.info(f"Active project: {self.active_project.name}")
        return True
    
    def get_recent_projects(self):
        """
        Get list of recent projects.
        
        Returns:
            list: List of recent project dictionaries (id, name)
        """
        try:
            if os.path.exists(self.recent_projects_file):
                with open(self.recent_projects_file, 'r') as file:
                    return json.load(file)
            return []
        except Exception as e:
            Log.error(f"Error loading recent projects: {str(e)}")
            return []
    
    def _update_recent_projects(self, project_id, project_name):
        """
        Update the list of recent projects.
        
        Args:
            project_id (int): Project ID
            project_name (str): Project name
            
        Returns:
            bool: True if successful
        """
        try:
            # Load existing list
            recent_projects = self.get_recent_projects()
            if not isinstance(recent_projects, list):
                recent_projects = []
                
            # Create project entry
            project_entry = {"id": project_id, "name": project_name}
            
            # Remove if already exists
            recent_projects = [p for p in recent_projects if p.get("id") != project_id]
            
            # Add to beginning of list
            recent_projects.insert(0, project_entry)
            
            # Limit list to 10 entries
            recent_projects = recent_projects[:10]
            
            # Save list
            with open(self.recent_projects_file, 'w') as file:
                json.dump(recent_projects, file, indent=4)
                
            return True
            
        except Exception as e:
            Log.error(f"Error updating recent projects: {str(e)}")
            return False
    
    def shutdown(self):
        """
        Shutdown the application.
        
        Returns:
            bool: True if successful
        """
        # Save all projects
        self.save_all_projects()
        
        # Close all projects
        for project_id in list(self.projects.keys()):
            self.close_project(project_id)
            
        # Close application database if it exists
        if self.db:
            self.db.close_connection()
            
        # Stop command queues
        self.command.queue.stop()
        
        Log.info("Application shutdown complete")
        return True
    
    # Command implementations
    def cmd_new_project(self, args):
        """
        Command to create a new project.
        
        Args:
            args (list): Command arguments [name]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: new-project <name>")
            return False
            
        name = args[0]
        project = self.create_project(name)
        return project is not None
    
    def cmd_open_project(self, args):
        """
        Command to open a project.
        
        Args:
            args (list): Command arguments [project_id]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: open-project <project_id>")
            return False
            
        try:
            project_id = int(args[0])
            project = self.open_project(project_id)
            return project is not None
        except ValueError:
            Log.error(f"Invalid project ID: {args[0]}")
            return False
    
    def cmd_close_project(self, args):
        """
        Command to close a project.
        
        Args:
            args (list): Command arguments [project_id] (optional)
            
        Returns:
            bool: True if successful
        """
        project_id = None
        if args:
            try:
                project_id = int(args[0])
            except ValueError:
                Log.error(f"Invalid project ID: {args[0]}")
                return False
                
        return self.close_project(project_id)
    
    def cmd_list_projects(self, args):
        """
        Command to list open projects.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        if not self.projects:
            Log.info("No open projects")
            return True
            
        Log.info("Open projects:")
        for project in self.projects.values():
            active = " (active)" if project == self.active_project else ""
            Log.info(f"  ID: {project.id}, Name: {project.name}{active}")
            
        return True
    
    def cmd_switch_project(self, args):
        """
        Command to switch the active project.
        
        Args:
            args (list): Command arguments [project_id]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: switch-project <project_id>")
            return False
            
        try:
            project_id = int(args[0])
            return self.set_active_project(project_id)
        except ValueError:
            Log.error(f"Invalid project ID: {args[0]}")
            return False
    
    def cmd_save_all(self, args):
        """
        Command to save all open projects.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True if successful
        """
        return self.save_all_projects()
    
    def cmd_exit(self, args):
        """
        Command to exit the application.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: Always True
        """
        self.shutdown()
        return True
    
    def cmd_help(self, args):
        """
        Command to display help information.
        
        Args:
            args (list): Command arguments (unused)
            
        Returns:
            bool: True
        """
        Log.info("Available application commands:")
        for cmd in self.command.get_commands():
            description = f" - {cmd.description}" if cmd.description else ""
            Log.info(f"  app {cmd.name}{description}")
        
        if self.active_project:
            Log.info(f"\nActive project: {self.active_project.name}")
            Log.info("Available project commands:")
            for cmd in self.active_project.command.get_commands():
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
        
        Log.info(f"Application command queue status:")
        Log.info(f"  Pending commands: {queue_size}")
        Log.info(f"  Executed commands: {len(history)}")
        
        if args and args[0] == "history" and history:
            Log.info(f"Command history:")
            for cmd in history[-10:]:  # Show last 10 commands
                status = cmd.status.value
                execution_time = cmd.get_execution_time()
                time_str = f" ({execution_time:.2f}s)" if execution_time else ""
                Log.info(f"  {cmd.name} - {status}{time_str}")
        
        # Also show active project queue status if applicable
        if self.active_project and hasattr(self.active_project, "command_queue"):
            self.active_project.cmd_queue_status(args)
            
        return True
    
    def cmd_install(self, args):
        """
        Command to install the application in a specified directory.
        
        Args:
            args (list): Command arguments [install_dir]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: install <directory>")
            return False
            
        install_dir = args[0]
        return self.install(install_dir)

    def get_install_dir(self):
        """
        Get the installation directory from the database.
        
        Returns:
            str: Installation directory or None if not found
        """
        if self.db is None:
            return None
            
        return self.db.get_attribute("install_dir")
        
    def is_installed(self):
        """
        Check if the application is installed.
        
        Returns:
            bool: True if installed
        """
        return self.db is not None and self.get_install_dir() is not None

    def load_installation(self, install_dir):
        """
        Load an existing installation from the specified directory.
        
        Args:
            install_dir (str): Directory where the application is installed
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Check if the directory exists
            if not os.path.exists(install_dir):
                Log.error(f"Installation directory does not exist: {install_dir}")
                return False
                
            # Look for application database
            app_data_dir = os.path.join(install_dir, "data")
            db_path = os.path.join(app_data_dir, "application.db")
            
            if not os.path.exists(db_path):
                Log.error(f"Application database not found at: {db_path}")
                return False
                
            # Load the database
            # We don't know the app_id or db_type, but they'll be loaded from the DB
            # using placeholder values for initial connection
            self.db = DatabaseController(0, "application", db_path)
            
            # Update instance variables from database
            install_dir = self.get_install_dir()
            if not install_dir:
                Log.error("Failed to retrieve installation directory from database")
                return False
                
            app_data_dir = self.db.get_attribute("app_data_dir")
            
            # Update instance variables
            self.data_dir = app_data_dir
            self.projects_dir = os.path.join(app_data_dir, "projects")
            self.blocks_dir = os.path.join(app_data_dir, "blocks")
            
            # Update recent projects file path
            self.recent_projects_file = os.path.join(self.data_dir, "recent_projects.json")
            
            Log.info(f"Loaded existing installation from: {install_dir}")
            return True
            
        except Exception as e:
            Log.error(f"Error loading installation: {str(e)}")
            return False
            
    def cmd_load_installation(self, args):
        """
        Command to load an existing installation.
        
        Args:
            args (list): Command arguments [install_dir]
            
        Returns:
            bool: True if successful
        """
        if not args:
            Log.error("Usage: load-installation <directory>")
            return False
            
        install_dir = args[0]
        return self.load_installation(install_dir)

    def check_and_load_installation(self, directory=None):
        """
        Check if the specified directory (or current directory) contains an installation
        and load it if found.
        
        Args:
            directory (str, optional): Directory to check, defaults to current directory
            
        Returns:
            bool: True if installation was found and loaded
        """
        if directory is None:
            directory = os.getcwd()
            
        # Check for data directory and application.db
        app_data_dir = os.path.join(directory, "data")
        db_path = os.path.join(app_data_dir, "application.db")
        
        if os.path.exists(db_path):
            return self.load_installation(directory)
            
        # Check one level up (in case we're in a subdirectory)
        parent_dir = os.path.dirname(directory)
        if parent_dir and parent_dir != directory:
            app_data_dir = os.path.join(parent_dir, "data")
            db_path = os.path.join(app_data_dir, "application.db")
            
            if os.path.exists(db_path):
                return self.load_installation(parent_dir)
                
        return False 