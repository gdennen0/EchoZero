
from Command.command_controller import CommandController
from Utils.message import Log

# Generic Container

class Container():
    def __init__(self):
        Log.info(f"Creating Instance of the Container Object")
        self.command = CommandController()
        self.blocks = {}
        self.block_types = {}
        self.command.add("add_block", self.add_block)
        self.command.add("remove_block", self.remove_block)
        self.command.add("list_blocks", self.list_blocks)

    def add_block_type(self, block_name, block_type):
        self.block_types[block_name] = block_type

    def add_block(self, block_name):
        if block_name in self.block_types:
            block = self.block_types[block_name]()
            block.set_parent_container(self)  # Set container reference
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
            
    def set_name(self, name):
        self.name = name
            


  