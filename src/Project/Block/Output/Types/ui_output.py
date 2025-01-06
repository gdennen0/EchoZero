from src.Project.Block.Output.output import Output

class UIOutput(Output):
    """
    A specialized output type for user interface status or data.
    The block writes updates/log info here, which the UI can poll or read.
    """

    name = "UIOutput"
    type = "UIOutput"

    def __init__(self, parent_block):
        super().__init__(parent_block)
        self.name = "UIOutput"
        self.type = "UIOutput"
        self.data_type = "UIOutput"
        self.ui_logs = []
        self.data_store = {}
        self.logs = []
    
    def write_log(self, message):
        self.ui_logs.append(message)
    
    def read_logs(self):
        """
        Returns all logs that haven't been cleared.
        For a sophisticated approach, you might keep unread vs read logs, 
        but for simplicity, we'll just return them.
        """
        return self.ui_logs

    def clear_logs(self):
        self.ui_logs.clear() 

    def write_data(self, key, value):
        """
        Store arbitrary data (e.g., figure/audio/classification info)
        so that a connected UIInput (or UI layer) can eventually read it.
        """
        self.data_store[key] = value

    def get_logs(self):
        """
        Return all stored logs as a single string, or use as needed.
        """
        return "\n".join(self.logs) 