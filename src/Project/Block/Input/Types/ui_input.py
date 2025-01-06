from src.Project.Block.Input.input import Input
from src.Utils.message import Log
class UIInput(Input):
    """
    A specialized input type for user interface commands.
    The UI can push commands or data here, which the block can consume.
    """
    name = "UIInput"
    type = "UIInput"

    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "UIInput"
        self.type = "UIInput"
        self.data_type = "UIInput"
        self.data_store = {}
        self.ui_logs = []
    
    def get_data(self, key):
        """
        Return any data stored under 'key' in this UIInput.
        The ManualClassifyBlock populates this Input with data for the UI to read.
        """
        return self.data_store.get(key)

    def get_logs(self):
        """
        If the block sets logs or relevant messages here,
        the UI can read them.
        """
        return self.ui_logs[:] 