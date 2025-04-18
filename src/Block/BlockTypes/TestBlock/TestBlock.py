from src.Block.Block import Block
from src.Utils.message import Log

class TestBlock(Block):
    """
    A test implementation of a Block to demonstrate subclassing.
    This block provides a simple data transformation example.
    """
    
    def __init__(self, block_id, db_path=None):
        """
        Initialize a TestBlock.
        
        Args:
            block_id (int): Unique identifier for this block
            db_path (str, optional): Custom database path if not using default
        """
        # Call parent constructor with correct block_type
        super().__init__(block_id, block_type="TestBlock", db_path=db_path)
        
        # TestBlock-specific initialization
        self.transformation_type = self.get_metadata("transformation_type", "uppercase")
    
    def pre_process(self):
        """Custom pre-processing for TestBlock."""
        Log.info(f"TestBlock {self.block_id} pre-processing...")
        
        # Make sure transformation_type is set
        if not self.get_metadata("transformation_type"):
            self.save_metadata("transformation_type", "uppercase")
            self.transformation_type = "uppercase"
    
    def process(self):
        """
        Process the input data based on the transformation type.
        
        Returns:
            dict: Transformed data
        """
        Log.info(f"TestBlock {self.block_id} processing with transformation: {self.transformation_type}")
        
        # Initialize output data
        self.output_data = {}
        
        # Apply transformation to each text input
        for key, value in self.input_data.items():
            if isinstance(value, str):
                if self.transformation_type == "uppercase":
                    self.output_data[key] = value.upper()
                elif self.transformation_type == "lowercase":
                    self.output_data[key] = value.lower()
                elif self.transformation_type == "capitalize":
                    self.output_data[key] = value.capitalize()
                elif self.transformation_type == "reverse":
                    self.output_data[key] = value[::-1]
                else:
                    # Default to original value if transformation type is not recognized
                    self.output_data[key] = value
            else:
                # Non-string values are passed through unchanged
                self.output_data[key] = value
        
        # Store a copy of the input and output data for reference
        self.store_data_item("last_input", "json", self.input_data)
        self.store_data_item("last_output", "json", self.output_data)
        
        return self.output_data
    
    def post_process(self):
        """Custom post-processing for TestBlock."""
        Log.info(f"TestBlock {self.block_id} post-processing...")
        
        # Count the number of items processed and save as metadata
        self.save_metadata("items_processed", len(self.output_data))
    
    def validate_input(self, input_data=None):
        """
        Validate that the input data has at least one string value.
        
        Args:
            input_data (dict, optional): Input data to validate
            
        Returns:
            bool: True if validation passes
        """
        data = input_data if input_data is not None else self.input_data
        
        # Validation logic: at least one key should have a string value
        has_string = False
        for value in data.values():
            if isinstance(value, str):
                has_string = True
                break
        
        if not has_string:
            Log.warning(f"TestBlock {self.block_id} validation failed: Input must contain at least one string value")
        
        return has_string
    
    def set_transformation_type(self, transformation_type):
        """
        Set the transformation type for this TestBlock.
        
        Args:
            transformation_type (str): Type of transformation to apply
                (uppercase, lowercase, capitalize, reverse)
                
        Returns:
            bool: True if successful
        """
        if transformation_type in ["uppercase", "lowercase", "capitalize", "reverse"]:
            self.transformation_type = transformation_type
            return self.save_metadata("transformation_type", transformation_type)
        else:
            Log.error(f"Invalid transformation type: {transformation_type}")
            return False
    
    def get_transformation_type(self):
        """
        Get the current transformation type.
        
        Returns:
            str: Current transformation type
        """
        return self.transformation_type
