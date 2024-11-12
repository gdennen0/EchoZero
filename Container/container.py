
from command_module import CommandModule
from message import Log

# Generic Container

class Container(CommandModule):
    def __init__(self):
        super().__init__()
        self.blocks = {}
        self.block_types = {}
        self.add_command("add_block", self.add_block)
        self.add_command("remove_block", self.remove_block)
        self.add_command("list_blocks", self.list_blocks)
        self.add_command("start", self.start)

    def add_block_type(self, block_name, block_type):
        self.block_types[block_name] = block_type


    def add_block(self, block_name):
        if block_name in self.block_types:
            block = self.block_types[block_name]()
            block.set_container(self)  # Set container reference
            self.blocks[block_name] = block
            Log.info(f"Added block: {block_name}")
        else:
            raise ValueError(f"Block type '{block_name}' not found in container")
        
    def remove_block(self, block_name):
        if block_name in self.blocks:
            del self.blocks[block_name]
        else:
            raise ValueError(f"Block with name '{block_name}' not found in container")
        
    def list_blocks(self):
        for block_name, block in self.blocks.items():
            print(f"{block_name}: {block.__class__.__name__}")

            



            


  