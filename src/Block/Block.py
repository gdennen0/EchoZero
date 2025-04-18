import os
import json
from Database.database_controller import DatabaseController
from src.Utils.message import Log

class Block:
    """
    Base Block class that provides core functionality for all block types.
    This class handles database interactions and provides a framework for data transformation.
    Each specific block type should subclass this and implement the required methods.
    """
    
    def __init__(self, block_id, block_type, db_path=None):
        """
        Initialize a block with its identifier and type.
        
        Args:
            block_id (int): Unique identifier for this block
            block_type (str): Type of the block (corresponds to a module)
            db_path (str, optional): Custom database path if not using default
        """
        self.database = DatabaseController(block_id, block_type, db_path)
        self.database.add_attribute("inputs", None, "dict")
        self.database.add_attribute("outputs", None, "dict")
    def process(self):
        """
        Process input data and produce output data.
        This method should be overridden by subclasses.
        
        Returns:
            dict: Processed output data
        """
        # Default implementation just passes through data
        self.output_data = self.input_data.copy()
        Log.warning(f"Using default process implementation for block {self.block_id}. Subclasses should override this method.")
        return self.output_data
    
    def store_data_item(self, name, item_type, value, metadata=None):
        """
        Store a data item in the block's database.
        
        Args:
            name (str): Data item name
            item_type (str): Data item type
            value (any): Data item value
            metadata (dict, optional): Additional metadata for the data item
            
        Returns:
            int: ID of the created or updated data item
        """
        try:
            return self.db_controller.create_data_item(name, item_type, value, metadata)
        except Exception as e:
            Log.error(f"Error storing data item '{name}' for block {self.block_id}: {str(e)}")
            return None
    
    def retrieve_data_item(self, name=None, item_id=None):
        """
        Retrieve a data item from the block's database.
        
        Args:
            name (str, optional): Data item name
            item_id (int, optional): Data item ID
            
        Returns:
            dict: Data item or None if not found
        """
        try:
            return self.db_controller.get_data_item(item_id=item_id, name=name)
        except Exception as e:
            Log.error(f"Error retrieving data item for block {self.block_id}: {str(e)}")
            return None
    
    def get_all_data_items(self):
        """
        Get all data items for this block.
        
        Returns:
            list: List of data items
        """
        try:
            return self.db_controller.get_all_data_items()
        except Exception as e:
            Log.error(f"Error retrieving all data items for block {self.block_id}: {str(e)}")
            return []
    
    def delete_data_item(self, name=None, item_id=None):
        """
        Delete a data item from the block's database.
        
        Args:
            name (str, optional): Data item name
            item_id (int, optional): Data item ID
            
        Returns:
            bool: True if successful
        """
        try:
            return self.db_controller.delete_data_item(item_id=item_id, name=name)
        except Exception as e:
            Log.error(f"Error deleting data item for block {self.block_id}: {str(e)}")
            return False
    
    def execute(self, input_data=None):
        """
        Execute the block with provided input data.
        This is the main entry point for running a block.
        
        Args:
            input_data (dict, optional): Input data to process
            
        Returns:
            dict: Processed output data
        """
        if input_data is not None:
            self.set_input_data(input_data)
        
        # Pre-process hook (can be overridden by subclasses)
        self.pre_process()
        
        # Main processing
        result = self.process()
        
        # Post-process hook (can be overridden by subclasses)
        self.post_process()
        
        return result
    
    def pre_process(self):
        """
        Hook method called before processing.
        Can be overridden by subclasses for custom pre-processing.
        """
        pass
    
    def post_process(self):
        """
        Hook method called after processing.
        Can be overridden by subclasses for custom post-processing.
        """
        pass
    
    def validate_input(self, input_data=None):
        """
        Validate input data before processing.
        Should be overridden by subclasses to implement specific validation logic.
        
        Args:
            input_data (dict, optional): Input data to validate (uses self.input_data if None)
            
        Returns:
            bool: True if validation passes
        """
        # Default implementation always passes
        return True
    
    def to_dict(self):
        """
        Convert block to a dictionary representation.
        
        Returns:
            dict: Dictionary representation of the block
        """
        return {
            "block_id": self.block_id,
            "block_type": self.block_type,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data, db_path=None):
        """
        Create a block from a dictionary representation.
        
        Args:
            data (dict): Dictionary representation of a block
            db_path (str, optional): Custom database path
            
        Returns:
            Block: A new Block instance
        """
        block = cls(
            block_id=data.get("block_id"),
            block_type=data.get("block_type"),
            db_path=db_path
        )
        
        # Set metadata
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            block.save_metadata(key, value)
        
        return block 