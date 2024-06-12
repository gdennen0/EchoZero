from tools import get_current_time
# ---------------------
# Logging Functionality
# ---------------------
class Log:
    def __init__(self, type):
        pass 
    def message(text):
        type = "MESSAGE"
        color = "\033[97m" # White
        output_to_console(type,text,color=color)

    def command(text):
        type = "COMMAND"
        color = "\033[38;5;208m" # Orange
        output_to_console(type,text,color=color)

    def debug(text):
        type = "DEBUG"
        color = "\033[97m" # White
        output_to_console(type,text,color=color)

    def info(text):
        type = "INFO"
        color = "\033[92m" # Green
        output_to_console(type,text,color=color)

    def warning(text):
        type = "WARNING"
        color = "\033[93m" # Yellow
        output_to_console(type,text,color=color)

    def error(text):
        type = "ERROR"
        color = "\033[91m"    # Red
        output_to_console(type,text,color=color)

    def unknown(text):
        type = "UNKNOWN"
        color = "\033[90m" # Grey
        output_to_console(type,text,color=color)
    
    def special(text):
        type = "SPECIAL"
        color = "\x1b[47m"
        special_output_to_console(type,text,color=color)  

def output_to_console(type, text, color="\033[97m"): #color set to white by default
    reset_code_color = "\033[0m"
    print(f"{get_current_time()} |{color}{type}{reset_code_color}| {text}")   

def special_output_to_console(type, text, color="\033[97m"): #color set to white by default
    reset_code_color = "\033[0m"
    print(f"{get_current_time()} {color}|{type}| {text}{reset_code_color}")   