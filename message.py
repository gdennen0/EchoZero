# ---------------------
# Logging Functionality
# ---------------------
class Log:
    def __init__(self, log_type):
        pass 
    def message(text):
        log_type = "MESSAGE"
        color = "\033[97m" # White
        output_to_console(log_type,text,color=color)

    def command(text):
        log_type = "COMMAND"
        color = "\033[38;5;208m" # Orange
        output_to_console(log_type,text,color=color)

    def debug(text):
        log_type = "DEBUG"
        color = "\033[97m" # White
        output_to_console(log_type,text,color=color)

    def info(text):
        log_type = "INFO"
        color = "\033[92m" # Green
        output_to_console(log_type,text,color=color)

    def warning(text):
        log_type = "WARNING"
        color = "\033[93m" # Yellow
        output_to_console(log_type,text,color=color)

    def error(text):
        log_type = "ERROR"
        color = "\033[91m"    # Red
        output_to_console(log_type,text,color=color)

    def unknown(text):
        log_type = "UNKNOWN"
        color = "\033[90m" # Grey
        output_to_console(log_type,text,color=color)
    
    def special(text):
        log_type = "SPECIAL"
        color = "\x1b[47m"
        special_output_to_console(log_type,text,color=color)

    def prompt(text):
        log_type = "UserInput"
        color = "\033[97m"
        output_to_console(log_type,text,color=color)

    def list(title, list_items, atrib=None):
        log_type = "list"
        header_length = 60
        header = title.center(header_length, '*')
        output_to_console(log_type, header)
        for index, item in enumerate(list_items):
            if not atrib:
                Log.info("List Atribute is None")
                wrapped_text = f"Index: {index}: Item: {item[:60]}"
                output_to_console(log_type, wrapped_text)
            else:
                wrapped_text = f"Index: {index}: Item.{atrib}: {getattr(item, atrib)[:60]}"
                output_to_console(log_type, wrapped_text)
                
        output_to_console(log_type, '*' * header_length)
        
def output_to_console(log_type, text, color="\033[97m"): #color set to white by default
    from tools import get_current_time
    reset_code_color = "\033[0m"
    print(f"{get_current_time()} |{color}{log_type}{reset_code_color}| {text}")   

def special_output_to_console(log_type, text, color="\033[97m"): #color set to white by default
    from tools import get_current_time
    reset_code_color = "\033[0m"
    print(f"{get_current_time()} {color}|{log_type}| {text}{reset_code_color}")   



