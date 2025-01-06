from src.Project.Block.Input.input import Input
from src.Utils.message import Log

class UIUpdateInput(Input):
    """
    A dedicated input type for receiving update requests (commands, etc.)
    from the UI layer.
    """
    name = "UIUpdateInput"
    type = "UIUpdateInput"

    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "UIUpdateInput"
        self.type = "UIUpdateInput"
        self.data_type = "UIUpdateInput"
        self.ui_commands = []

    def push_command(self, command_name, *args, **kwargs):
        """
        Store a user command (e.g., 'save_classification', etc.) along with arguments.
        The ManualClassifyBlock will poll these commands.
        """
        self.ui_commands.append((command_name, args, kwargs))
        Log.info(f"UIUpdateInput: {command_name} pushed to {self.name}")

    def pop_commands(self):
        """
        Retrieve and clear all stored commands.
        """
        commands_copy = list(self.ui_commands)
        self.ui_commands.clear()
        return commands_copy 