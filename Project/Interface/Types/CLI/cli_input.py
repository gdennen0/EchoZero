
def prompt(prompt_message):
    # Prompt user in terminal and return the response
    response = input(prompt_message)
    if response.lower() in ['e', 'exit']:
        Log.info("Selection exited by user.")
        return
    return response

def prompt_selection(prompt_text, options):
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        if hasattr(obj, 'name'):
            Log.info(f"{i}: {obj.name}")
        else:
            Log.info(f"{i}: {obj}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection:
            Log.info("Selection exited by user.")
            return None, None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index]
        elif selection in options_list:
            return options[selection]
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def prompt_selection_with_type(prompt_text, options):
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        Log.info(f"{i}: {obj.type}>{obj.name}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection: 
            Log.info("Selection exited by user.")
            return None, None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index], index
        elif selection in options_list:
            return options[selection], selection
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def prompt_selection_with_type_and_parent_block(prompt_text, options): #imsorryfortheshitename
    Log.info(prompt_text)
    if isinstance(options, dict):
        options_list = list(options.keys())
    else:
        options_list = options
    for i, obj in enumerate(options_list):
        Log.info(f"{i}: {obj.parent_block.name}:{obj.type}>{obj.name}")
    while True:
        selection = prompt(f"Please enter the key or index for your selection (or 'e' to exit): ")
        if not selection:
            Log.info("Selection exited by user.")
            return None
        if selection.isdigit():
            index = int(selection)
            if 0 <= index < len(options_list):
                return options_list[index]
        elif selection in options_list:
            return options[selection]
        Log.error("Invalid selection. Please enter a valid key or index, or 'e' to exit.")

def yes_no_prompt(prompt_message):
    # Prompt user with a yes/no question and return True for yes and False for no
    valid_yes = {'yes', 'y', 'ye', 'YES', 'Y', 'YE'}
    valid_no = {'no', 'n', 'NO', 'N'}
    
    while True:
        response = input(prompt_message).strip().lower()
        if response in valid_yes:
            return True
        elif response in valid_no:
            return False
        else:
            Log.prompt("Please respond with 'yes' or 'no' (or 'y' or 'n').")
