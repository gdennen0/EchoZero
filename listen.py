import sys
from tools import prompt
from message import Log

def main_listen_loop(command):
    Log.info("Listening for commands. Type 'exit' to quit.")
    while True:
        user_input = prompt("Enter command: ")
        if user_input.lower() == 'exit':
            Log.special("Exiting application.")
            break
        process_command(user_input, command)

def process_command(input_string, command_obj):
    parts = input_string.split()
    Log.debug(f"Received parts: {parts}")
    if len(parts) < 2: # if there isnt an action with the commandw
        cmd, action = parts[0], None
    else:
        cmd, action = parts[0], parts[1]

    if cmd in command_obj.commands:
        func = command_obj.commands[cmd]
        if callable(func):
            if action:
                try:
                    Log.command(f"{cmd} {action}.")
                    method_to_call = getattr(func(), action)
                    method_to_call()  # Call the method without arguments
                    
                except Exception as e:
                    Log.error(f"Error executing {cmd}: {str(e)}")
            else:
                try:
                    Log.command(f"{cmd}")
                    func()
                except Exception as e:
                    Log.error(f"Error executing {cmd}: {str(e)}")
        else:
            Log.error(f"{cmd} is not callable.")
    else:
        Log.unknown(f"Command '{cmd}' not recognized.")
