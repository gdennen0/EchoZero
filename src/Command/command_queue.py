import threading
import queue
import time
from src.Utils.message import Log
from enum import Enum

class CommandStatus(Enum):
    """Possible command statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Command:
    """
    Base command class that represents an operation to be executed.
    Commands are executed in sequence by the CommandQueue.
    """
    
    def __init__(self, name, target_function, args=None, kwargs=None):
        """
        Initialize a command.
        
        Args:
            name (str): Descriptive name for the command
            target_function (callable): Function to execute
            args (list): Positional arguments for the function
            kwargs (dict): Keyword arguments for the function
        """
        self.name = name
        self.target_function = target_function
        self.args = args or []
        self.kwargs = kwargs or {}
        self.id = id(self)  # Unique ID for the command
        self.status = CommandStatus.PENDING
        self.result = None
        self.error = None
        self.timestamp_created = time.time()
        self.timestamp_started = None
        self.timestamp_completed = None
    
    def execute(self):
        """
        Execute the command.
        
        Returns:
            bool: True if command executed successfully
        """
        self.status = CommandStatus.RUNNING
        self.timestamp_started = time.time()
        
        try:
            self.result = self.target_function(*self.args, **self.kwargs)
            self.status = CommandStatus.COMPLETED
            self.timestamp_completed = time.time()
            return True
        except Exception as e:
            self.error = str(e)
            self.status = CommandStatus.FAILED
            self.timestamp_completed = time.time()
            Log.error(f"Command '{self.name}' failed: {str(e)}")
            return False
    
    def cancel(self):
        """
        Cancel the command if it's still pending.
        
        Returns:
            bool: True if command was cancelled
        """
        if self.status == CommandStatus.PENDING:
            self.status = CommandStatus.CANCELLED
            return True
        return False
    
    def get_execution_time(self):
        """
        Get the command execution time in seconds.
        
        Returns:
            float: Execution time in seconds or None if not completed
        """
        if self.timestamp_started and self.timestamp_completed:
            return self.timestamp_completed - self.timestamp_started
        return None
    
    def get_waiting_time(self):
        """
        Get the time the command spent waiting in queue.
        
        Returns:
            float: Waiting time in seconds or None if not started
        """
        if self.timestamp_started:
            return self.timestamp_started - self.timestamp_created
        return None
    
    def __str__(self):
        """String representation of the command."""
        return f"Command(id={self.id}, name='{self.name}', status={self.status.value})"

class CommandQueue:
    """
    Queue for executing commands in sequence.
    Provides thread-safe access to command execution.
    """
    
    def __init__(self, max_retries=3):
        """
        Initialize the command queue.
        
        Args:
            max_retries (int): Maximum number of retries for failed commands
        """
        self.queue = queue.Queue()
        self.history = []
        self.running = False
        self.worker_thread = None
        self.max_retries = max_retries
        self.retry_counts = {}  # Command ID to retry count
        self.lock = threading.Lock()
    
    def start(self):
        """
        Start the command queue worker thread.
        
        Returns:
            bool: True if worker thread was started
        """
        with self.lock:
            if self.running:
                return False
                
            self.running = True
            self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
            self.worker_thread.start()
            Log.info("Command queue worker started")
            return True
    
    def stop(self):
        """
        Stop the command queue worker thread.
        
        Returns:
            bool: True if worker thread was stopped
        """
        with self.lock:
            if not self.running:
                return False
                
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=1.0)
                self.worker_thread = None
            Log.info("Command queue worker stopped")
            return True
    
    def enqueue(self, command):
        """
        Add a command to the queue.
        
        Args:
            command (Command): The command to enqueue
            
        Returns:
            int: Command ID
        """
        self.queue.put(command)
        Log.info(f"Command '{command.name}' enqueued")
        return command.id
    
    def create_and_enqueue(self, name, target_function, args=None, kwargs=None):
        """
        Create and enqueue a command in one step.
        
        Args:
            name (str): Descriptive name for the command
            target_function (callable): Function to execute
            args (list): Positional arguments for the function
            kwargs (dict): Keyword arguments for the function
            
        Returns:
            int: Command ID
        """
        command = Command(name, target_function, args, kwargs)
        return self.enqueue(command)
    
    def get_command_by_id(self, command_id):
        """
        Get a command by its ID.
        
        Args:
            command_id (int): The command ID
            
        Returns:
            Command: The command or None if not found
        """
        # Check current queue
        with self.lock:
            # Check history
            for cmd in self.history:
                if cmd.id == command_id:
                    return cmd
                    
            # Not found
            return None
    
    def get_queue_size(self):
        """
        Get the current queue size.
        
        Returns:
            int: Number of pending commands
        """
        return self.queue.qsize()
    
    def get_history(self):
        """
        Get the command execution history.
        
        Returns:
            list: List of executed commands
        """
        with self.lock:
            return list(self.history)
    
    def clear_history(self):
        """Clear the command execution history."""
        with self.lock:
            self.history = []
    
    def _process_queue(self):
        """Process commands from the queue until stopped."""
        while self.running:
            try:
                # Get command from queue with timeout
                # This allows the thread to check self.running periodically
                try:
                    command = self.queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Execute command
                success = command.execute()
                
                # Handle retries for failed commands
                if not success and command.status == CommandStatus.FAILED:
                    retry_count = self.retry_counts.get(command.id, 0)
                    if retry_count < self.max_retries:
                        # Increment retry count and requeue
                        self.retry_counts[command.id] = retry_count + 1
                        Log.info(f"Retrying command '{command.name}' ({retry_count + 1}/{self.max_retries})")
                        
                        # Create a new command with the same parameters
                        retry_command = Command(
                            f"{command.name} (retry {retry_count + 1})",
                            command.target_function,
                            command.args,
                            command.kwargs
                        )
                        self.queue.put(retry_command)
                
                # Add to history
                with self.lock:
                    self.history.append(command)
                
                # Mark as done
                self.queue.task_done()
                
            except Exception as e:
                Log.error(f"Error in command queue worker: {str(e)}")
                # Continue processing the queue even if an error occurs
                
        Log.info("Command queue worker stopped")
    
    def __del__(self):
        """Ensure the worker thread is stopped when the object is deleted."""
        self.stop() 