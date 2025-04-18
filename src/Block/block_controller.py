import os
import importlib.util
from src.Utils.message import Log

class BlockController:
    """
    Controller for loading and managing block modules.
    Provides a central place to access modules for different block types.
    """
    
    def __init__(self, blocks_dir="data/blocks"):
        """
        Initialize the block module controller.
        
        Args:
            blocks_dir (str): Directory for storing block module data
        """
        self.blocks_dir = blocks_dir
        self.module_cache = {}  # Cache of loaded modules
        self.block_types_dir = "src/Project/Block/BlockTypes"  # Default directory for block types
    
    def load_modules(self):
        """
        Load all block modules found in the block types directory.
        Uses caching to avoid reloading already loaded modules.
        
        Returns:
            dict: Dictionary of loaded block modules, keyed by block type name
        """
        if self.module_cache:
            Log.info(f"Using {len(self.module_cache)} cached block modules")
            return self.module_cache
        
        module_files = self.locate_module_files()
        
        if not module_files:
            Log.warning("No block modules found to load")
            return {}
        
        for module_file in module_files:
            try:
                # Extract module name from file path
                folder_name = os.path.basename(os.path.dirname(module_file))
                module_name = folder_name
                
                spec = importlib.util.spec_from_file_location(module_name, module_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Store the module in our cache
                self.module_cache[module_name] = module
                Log.info(f"Successfully loaded block module: {module_name}")
                
            except Exception as e:
                Log.error(f"Failed to load block module from {module_file}: {str(e)}")
        
        Log.info(f"Loaded {len(self.module_cache)} block modules")
        return self.module_cache
    
    def locate_module_files(self):
        """
        Locate all block module files in the block types directory.
        
        Returns:
            list: List of paths to block module files
        """
        module_files = []
        
        # Check if the directory exists
        if not os.path.exists(self.block_types_dir):
            Log.warning(f"Block types directory not found: {self.block_types_dir}")
            return module_files
        
        # Iterate through each folder in the blocktypes directory
        for folder_name in os.listdir(self.block_types_dir):
            folder_path = os.path.join(self.block_types_dir, folder_name)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Look for a .py file with the same name as the folder
                module_file = os.path.join(folder_path, f"{folder_name}.py")
                
                if os.path.exists(module_file) and os.path.isfile(module_file):
                    module_files.append(module_file)
                    Log.info(f"Found block module: {module_file}")
                else:
                    # Also check for a BlockType.py naming pattern
                    module_file = os.path.join(folder_path, f"{folder_name}Block.py")
                    if os.path.exists(module_file) and os.path.isfile(module_file):
                        module_files.append(module_file)
                        Log.info(f"Found block module: {module_file}")
                    else:
                        Log.warning(f"No matching module file found for block type: {folder_name}")
        
        return module_files
    
    def get_module(self, module_name):
        """
        Get a loaded module by name.
        
        Args:
            module_name (str): Module name
            
        Returns:
            module: Loaded module or None if not found
        """
        # Ensure modules are loaded
        if not self.module_cache:
            self.load_modules()
        
        return self.module_cache.get(module_name)
    
    def get_all_modules(self):
        """
        Get all loaded modules.
        
        Returns:
            dict: Dictionary of loaded modules
        """
        # Ensure modules are loaded
        if not self.module_cache:
            self.load_modules()
        
        return self.module_cache
    
    def reload_module(self, module_name):
        """
        Reload a specific module.
        
        Args:
            module_name (str): Module name
            
        Returns:
            module: Reloaded module or None if not found
        """
        if module_name not in self.module_cache:
            Log.warning(f"Module not found for reloading: {module_name}")
            return None
        
        # Find the module file
        module_files = self.locate_module_files()
        module_file = None
        
        for file_path in module_files:
            folder_name = os.path.basename(os.path.dirname(file_path))
            if folder_name == module_name:
                module_file = file_path
                break
        
        if not module_file:
            Log.error(f"Module file not found for reloading: {module_name}")
            return None
        
        try:
            # Reload the module
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Update cache
            self.module_cache[module_name] = module
            Log.info(f"Successfully reloaded block module: {module_name}")
            
            return module
            
        except Exception as e:
            Log.error(f"Failed to reload block module {module_name}: {str(e)}")
            return None 