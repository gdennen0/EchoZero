from Project.Command.command_item import CommandItem
from Utils.message import Log
import threading
import queue

class CommandController:
    """
    Manages command registration, queuing, and execution for a block.
    """
    def __init__(self):
        self.name = "CommandController"
        self.commands = []
        self.command_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.worker_thread.start()
        self.progress = 0  # Progress percentage (0-100)
        self.progress_lock = threading.Lock()


    def add(self, name, command):
        """
        Registers a new command.
        """
        cmd_item = CommandItem()
        cmd_item.set_name(name)
        cmd_item.set_command(command)
        cmd_item.set_controller(self)
        self.commands.append(cmd_item)

    def remove(self, name=None):
        """
        Removes a command by name.
        """
        if name is None:
            self.commands = []
            Log.info(f"All commands removed from CommandController of block '{self.name}'.")
            return

        for command in self.commands:
            if command.name == name:
                self.commands.remove(command)
                Log.info(f"Command '{name}' removed from CommandController of block '{self.name}'.")
                break
        else:
            Log.error(f"Command '{name}' not found in CommandController of block '{self.name}'.")

    def list_commands(self):
        """
        Lists all registered commands.
        """
        Log.info(f"Listing commands for block '{self.name}':")
        for command in self.commands:
            Log.info(f" - {command.name}")

    def get_commands(self):
        return self.commands

    def enqueue_command(self, command_item, *args, **kwargs):
        """
        Adds a command to the queue for execution.
        """
        self.command_queue.put((command_item, args, kwargs))
        Log.info(f"Enqueued command '{command_item.name}' for block '{self.name}'.")

    def _get_command_item(self, name):
        """
        Retrieves a CommandItem by name.
        """
        for command in self.commands:
            if command.name == name:
                return command
        return None

    def process_commands(self):
        """
        Worker thread that processes commands from the queue.
        """
        while True:
            item = self.command_queue.get()
            if item is None:
                Log.info(f"Shutting down CommandController of block '{self.name}'.")
                break  # Graceful shutdown
            command_item, args, kwargs = item
            try:
                Log.info(f"Executing command '{command_item.name}' for block '{self.name}'.")
                command_item.execute(*args, **kwargs)
                Log.info(f"Completed command '{command_item.name}' for block '{self.name}'.")
            except Exception as e:
                Log.error(f"Error executing command '{command_item.name}' in block '{self.name}': {e}")
            finally:
                self.command_queue.task_done()

    def shutdown(self):
        """
        Gracefully shuts down the worker thread.
        """
        self.command_queue.put(None)
        self.worker_thread.join()
        Log.info(f"CommandController of block '{self.name}' shutdown.") 
