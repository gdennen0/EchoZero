import sys
from PyQt5 import QtWidgets
from Application.application_controller import ApplicationControllerV2
from src.Command.command_parser import CommandParser
from src.Utils.message import Log
from src.Utils.tools import prompt

def main():
    """
    Main entry point for the application.
    Uses the new architecture with command queue and dedicated databases.
    """
    # Initialize application
    app = QtWidgets.QApplication([])
    
    # Initialize application controller
    app_controller = ApplicationControllerV2()
    
    # Prompt user to create or load a project
    Log.info("Welcome to the application!")
    Log.info("What would you like to do?")
    Log.info("  1. Create a new project")
    Log.info("  2. Open a recent project")
    
    choice = prompt("Enter your choice (1/2): ")
    
    if choice == "1":
        # Create a new project
        project_name = prompt("Enter project name: ")
        project = app_controller.create_project(project_name)
        
        if not project:
            Log.error("Failed to create project. Exiting.")
            return
            
    elif choice == "2":
        # Open a recent project
        recent_projects = app_controller.get_recent_projects()
        
        if not recent_projects:
            Log.info("No recent projects found. Creating a new project instead.")
            project_name = prompt("Enter project name: ")
            project = app_controller.create_project(project_name)
            
            if not project:
                Log.error("Failed to create project. Exiting.")
                return
        else:
            Log.info("Recent projects:")
            for i, p in enumerate(recent_projects):
                Log.info(f"  {i+1}. {p['name']} (ID: {p['id']})")
            
            project_choice = prompt("Enter project number to open (or 'n' for new): ")
            
            if project_choice.lower() == 'n':
                project_name = prompt("Enter project name: ")
                project = app_controller.create_project(project_name)
                
                if not project:
                    Log.error("Failed to create project. Exiting.")
                    return
            else:
                try:
                    idx = int(project_choice) - 1
                    if 0 <= idx < len(recent_projects):
                        project_id = recent_projects[idx]['id']
                        project = app_controller.open_project(project_id)
                        
                        if not project:
                            Log.error("Failed to open project. Exiting.")
                            return
                    else:
                        Log.error("Invalid selection. Exiting.")
                        return
                except ValueError:
                    Log.error("Invalid input. Exiting.")
                    return
    else:
        Log.error("Invalid choice. Exiting.")
        return
    
    # Initialize command parser with application controller
    parser = CommandParser(app_controller)

    Log.info(f"Project '{app_controller.get_active_project().name}' is active. Listening for commands...")
    Log.info("Type 'help' for a list of commands or 'exit' to quit.")
    
    # Main command loop
    while True:
        try:
            user_input = prompt("Enter command: ")
            
            if user_input.lower() == 'exit':
                app_controller.cmd_exit([])
                break
                
            parser.parse_and_execute(user_input)
        except KeyboardInterrupt:
            Log.info("\nCommand interrupted.")
        except Exception as e:
            Log.error(f"Error: {str(e)}")
    
    # Exit application
    sys.exit(0)

if __name__ == "__main__":
    main() 