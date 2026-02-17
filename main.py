"""
EchoZero Application Entry Point

Backend-only application focused on audio processing architecture.
CLI interface for interaction.
"""
import sys
import argparse

from src.utils.message import Log
from src.application.bootstrap import initialize_services
from src.cli.commands.command_parser import CommandParser
from src.cli.cli_interface import CLIInterface


def main():
    """
    EchoZero Application Entry Point (Backend Only)
    
    Initializes the architecture services and launches CLI.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EchoZero Audio Processing Application (Backend)')
    parser.add_argument('--db-path', type=str, help='Path to SQLite database file')
    parser.add_argument('--interactive', action='store_true', default=True, help='Launch interactive CLI (default)')
    parser.add_argument('--non-interactive', action='store_true', help='Run without interactive CLI')
    args = parser.parse_args()
    
    # Initialize services
    try:
        Log.info("Initializing EchoZero application...")
        services = initialize_services(db_path=args.db_path)
        Log.info("Services initialized successfully")
    except Exception as e:
        Log.error(f"Failed to initialize services: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Launch CLI interface
    if not args.non_interactive:
        try:
            Log.info("Starting CLI interface")
            # CLI uses the application facade
            cli = CLIInterface(services.facade)
            cli.run()
        except KeyboardInterrupt:
            Log.info("\nExiting...")
            return 0
        except Exception as e:
            Log.error(f"CLI error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
