from Block.block import Block
from message import Log

class ExportBlock(Block):
    # The export block is used to export the analysis results to a file
    def __init__(self):
        super().__init__()
        self.name = "Export"
        self.parts = []

    def add_part(self, part):
        self.parts.append(part)

    def remove_part(self, part):
        self.parts.remove(part)

    def start(self, AnalysisResults):
        Log.info(f"Exporting {len(AnalysisResults)} results")