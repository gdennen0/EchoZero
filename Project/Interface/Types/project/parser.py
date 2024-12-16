class Parser:
    def __init__(self, project):
        self.project = project

    def parse(self, input_string):
        block = None
        output = None
        input = None
        command = None
        args = []
                
        input_keywords = input_string.lower().split()
        remaining_parts = input_keywords.copy()

        block = self._get_matching_block(remaining_parts)

        if block:
            remaining_parts.pop(0)  # Remove block from the list

            input = self._get_matching_input(block, remaining_parts)
            output  = self._get_matching_output(block, remaining_parts)

            if input or output:
                target_module = input if input else output
                remaining_parts.pop(0)  # Remove input/output type from the list

        if remaining_parts: 
            command, args = self._get_command(target_module, remaining_parts)
            if command:
                command = command
                args = args

        return block, input, output, command, args



    def _get_matching_block(self, parts):
        if not parts:
            return None
        for block in self.project.get_blocks():
            if block.name.lower() == parts[0]:
                return block
        return None
            
    def _get_matching_input(self, block, parts):
        if not parts:
            return None
        
        if parts[0] == "input":
            if len(parts) > 1:  # If we have a specific input port name
                for input_port in block.input.items():
                    if input_port.name.lower() == parts[1]:
                        return input_port
            return block.input
        else:
            return None

    def _get_matching_output(self, block, parts):
        if not parts:
            return None
        
        if parts[0] == "output":
            if len(parts) > 1:  # If we have a specific output port name
                for output_port in block.output.items():
                    if output_port.name.lower() == parts[1]:
                        return output_port
            return block.output
        else:
            return None
    
    def _check_if_command(self, command_item):
        if command_item:
            return True
        return False

    def _get_command(self, selected_module, remaining_parts):
        if not remaining_parts:
            return None, None
        if selected_module:
            for command in selected_module.command.get_commands():
                if command.name.lower() == remaining_parts[0]:
                    return command, remaining_parts[1:]
        return None, None