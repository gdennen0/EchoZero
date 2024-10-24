from Container.container import Container

class GenericContainer(Container):
    def __init__(self):
        super().__init__()
        self.name = "generic"
        self.add_block(self.block_types["loadAudio"])
        self.add_block(self.block_types["transformAudio"])
        self.add_block(self.block_types["analyzeAudio"])
        self.add_block(self.block_types["export"])

