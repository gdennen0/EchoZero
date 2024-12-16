from Utils.message import Log
from Project.project import Project
from Project.Command.command_parser import CommandParser
from Project.Interface.Types.CLI.cli_interface import CLIInterface

def main():
    project = Project() # main data structure 
    parser = CommandParser(project) # command parser module
    cli_interface = CLIInterface(parser)
    try:    
        Log.info("Listening for commands")
        while True:
            cli_interface.start()

    except KeyboardInterrupt:
        Log.info("Shutting down project...")
        project.shutdown_all_blocks()
        interface.stop()
        Log.info("Project shutdown complete.")


if __name__ == "__main__":
    main()


