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

# def process_command(user_input, command_obj):
#     Log.info(f"Processing command: {user_input}")
#     try:
#         parts = user_input.split('.')
#         Log.info(f"Parts {parts}")
#         # Start with the base command object, typically 'digest'
#         current_obj = command

#         # Navigate through the attributes except for the last part which should be the method
#         for part in parts[:-1]:
#             current_obj = getattr(current_obj, part, None)
#             if current_obj is None:
#                 Log.error(f"No such attribute: {part}")
#                 return

#         # The last part should be the method name
#         method_name = parts[-1]
#         if hasattr(current_obj, method_name):
#             method = getattr(current_obj, method_name)
#             if callable(method):
#                 result = method()
#                 Log.info(f"Command output: {result}")
#             else:
#                 Log.error("Command is not callable")
#         else:
#             Log.error(f"No such method: {method_name}")
#     except Exception as e:
#         Log.error(f"Error processing command: {str(e)}")

# def process_command(input_string, command_obj):
#     parts = input_string.split()
#     Log.debug(f"Received parts: {parts}")

#     if len(parts) < 2: # if there isnt an action with the command
#         cmd, action = parts[0], None
#     else:
#         cmd, action = parts[0], parts[1]

#     if cmd in command_obj.commands:
#         func = command_obj.commands[cmd]
#         if callable(func):
#             if action:
#                 try:
#                     Log.command(f"{cmd} {action}.")
#                     method_to_call = getattr(func(), action)
#                     method_to_call()  # Call the method without arguments
                    
#                 except Exception as e:
#                     Log.error(f"Error executing {cmd}: {str(e)}")
#             else:
#                 try:
#                     Log.command(f"{cmd}")
#                     func()
#                 except Exception as e:
#                     Log.error(f"Error executing {cmd}: {str(e)}")
#         else:
#             Log.error(f"{cmd} is not callable.")
#     else:
#         Log.unknown(f"Command '{cmd}' not recognized.")


# def process_command(input_string, command_obj):
#     parts = input_string.split()
#     Log.debug(f"Received parts: {parts}")
    
#     if len(parts) < 2:  # if there isn't an action with the command
#         cmd, action = parts[0], None
#     else:
#         cmd, action = parts[0], parts[1]

#     if cmd in command_obj.commands:
#         func = command_obj.commands[cmd]
#         if callable(func):
#             if action:
#                 try:
#                     Log.command(f"{cmd} {action}.")
#                     action_parts = action.split('.')
#                     current_obj = func()

#                     for part in action_parts[:-1]:
#                         current_obj = getattr(current_obj, part, None)
#                         if current_obj is None:
#                             Log.error(f"No such attribute: {part}")
#                             return

#                     method_name = action_parts[-1]
#                     if hasattr(current_obj, method_name):
#                         method_to_call = getattr(current_obj, method_name)
#                         if callable(method_to_call):
#                             method_to_call()  # Call the method without arguments
#                         else:
#                             Log.error(f"Action '{method_name}' is not callable.")
#                     else:
#                         Log.error(f"No such method: {method_name}")
#                 except Exception as e:
#                     Log.error(f"Error executing {cmd}: {str(e)}")
#             else:
#                 try:
#                     Log.command(f"{cmd}")
#                     func()
#                 except Exception as e:
#                     Log.error(f"Error executing {cmd}: {str(e)}")
#         else:
#             Log.error(f"{cmd} is not callable.")
#     else:
#         Log.unknown(f"Command '{cmd}' not recognized.")

def process_command(input_string, command_obj):
    parts = input_string.split()
    Log.debug(f"Received parts: {parts}")
    
    if len(parts) < 2:  # if there isn't an action with the command
        cmd, action = parts[0], None
    else:
        cmd, action = parts[0], parts[1]

    if cmd in command_obj.commands:
        instance = command_obj.commands[cmd]
        if action:
            try:
                Log.command(f"{cmd} {action}.")
                action_parts = action.split('.')
                current_obj = instance

                for part in action_parts[:-1]:
                    current_obj = getattr(current_obj, part, None)
                    if current_obj is None:
                        Log.error(f"No such attribute: {part}")
                        return

                method_name = action_parts[-1]
                if hasattr(current_obj, method_name):
                    method_to_call = getattr(current_obj, method_name)
                    if callable(method_to_call):
                        method_to_call()  # Call the method without arguments
                    else:
                        Log.error(f"Action '{method_name}' is not callable.")
                else:
                    Log.error(f"No such method: {method_name}")
            except Exception as e:
                Log.error(f"Error executing {cmd}: {str(e)}")
        else:
            try:
                Log.command(f"{cmd}")
                if callable(instance):
                    instance()
                else:
                    Log.error(f"{cmd} is not callable.")
            except Exception as e:
                Log.error(f"Error executing {cmd}: {str(e)}")
    else:
        Log.unknown(f"Command '{cmd}' not recognized.")