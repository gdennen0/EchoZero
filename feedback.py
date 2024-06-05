import datetime

class Feedback:
    @staticmethod
    def log(message, function_name="", class_name=""):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] {class_name}.{function_name}: {message}")

    @staticmethod
    def error(message, function_name="", class_name=""):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ERROR] [{current_time}] {class_name}.{function_name}: {message}")
