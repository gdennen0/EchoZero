
from src.API.request import Request
class cmd:
    def __init__(self, project):
        self.project = project
    def run_input_loop(self):
        while True:
            user_input = input(f"Enter a command:")
            request = Request()
            self.parse(user_input)

    def parse(self, input_string):
        # Initialize variables
        block = None
        port = None
        command_item = None
        args = []

        all_input_parts = input_string.lower().split()        # Split the input string into parts
        remaining_parts = all_input_parts.copy()

        block = self._get_matching_block(remaining_parts)
        selected_module = self.project
        if block:
            remaining_parts.pop(0)  # Remove block from the list
            selected_module = block
            input_match = self._get_matching_input(block, remaining_parts)
            output_match = self._get_matching_output(block, remaining_parts)
            if input_match or output_match:
                selected_module = input_match if input_match else output_match
                remaining_parts.pop(0)  # Remove input/output type from the list

        if remaining_parts: 
            command_item, args = self._get_command(selected_module, remaining_parts)

        return block.name, command_item.name, args

    def _get_matching_block(self, parts):
        if not parts:
            return None
        for block in self.project.get_blocks():
            if block is not None:   
                if block.name.lower() == parts[0]:
                    return block
        return None

    def _check_if_command(self, command_item):
        if command_item:
            return True
        return False

    def _get_command(self, selected_module, remaining_parts):
        """
        Attempt to find a command in the selected module's command controller
        that matches the next token in remaining_parts.
        """
        if not remaining_parts:
            return None, None
        if selected_module and hasattr(selected_module, "command"):
            for cmd_item in selected_module.command.get_commands():
                if cmd_item.name.lower() == remaining_parts[0]:
                    # Everything after the command name is considered an argument
                    return cmd_item, remaining_parts[1:]
        return None, None
