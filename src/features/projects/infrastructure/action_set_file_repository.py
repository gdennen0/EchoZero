"""
File-based Action Set Repository

Stores action sets as JSON files in the user data directory.
Action sets are global (not project-specific) so they can be reused.

Standard action sets are bundled with the app and copied to user directory on first run.
"""
import json
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.features.projects.domain.action_set import ActionSet
from src.utils.paths import get_user_data_dir, get_app_install_dir
from src.utils.message import Log


class ActionSetFileRepository:
    """
    File-based repository for action sets.
    
    Stores each action set as a separate JSON file:
    - User sets: ~/.echozero/action_sets/{name}.json
    - Standard sets: Bundled with app, copied on first run
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize repository.
        
        Args:
            base_dir: Override base directory (for testing)
        """
        if base_dir:
            self.action_sets_dir = base_dir
        else:
            self.action_sets_dir = get_user_data_dir() / "action_sets"
        
        # Ensure directory exists
        self.action_sets_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy standard sets if they don't exist
        self._init_standard_sets()
        
        Log.debug(f"ActionSetFileRepository: Initialized at {self.action_sets_dir}")
    
    def _init_standard_sets(self):
        """Copy standard action sets to user directory if they don't exist."""
        # Standard sets are stored in the app's install directory
        app_install_dir = get_app_install_dir()
        if not app_install_dir:
            Log.debug("ActionSetFileRepository: Could not determine app install directory")
            return
        
        app_data_dir = app_install_dir / "data" / "action_sets"
        
        if not app_data_dir.exists():
            Log.debug(f"ActionSetFileRepository: No standard sets found at {app_data_dir}")
            return
        
        for standard_file in app_data_dir.glob("*.json"):
            user_file = self.action_sets_dir / standard_file.name
            if not user_file.exists():
                try:
                    shutil.copy2(standard_file, user_file)
                    Log.info(f"ActionSetFileRepository: Copied standard set '{standard_file.stem}'")
                except Exception as e:
                    Log.warning(f"ActionSetFileRepository: Failed to copy standard set: {e}")
    
    def _get_file_path(self, name: str) -> Path:
        """Get file path for an action set by name."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        return self.action_sets_dir / f"{safe_name}.json"
    
    def save(self, action_set: ActionSet) -> ActionSet:
        """
        Save an action set to file.
        
        Args:
            action_set: ActionSet to save
            
        Returns:
            Saved ActionSet
        """
        file_path = self._get_file_path(action_set.name)
        
        try:
            # Update modified timestamp
            action_set.update_modified()
            
            # Serialize to JSON
            data = action_set.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            Log.info(f"ActionSetFileRepository: Saved action set '{action_set.name}' to {file_path}")
            return action_set
            
        except Exception as e:
            Log.error(f"ActionSetFileRepository: Failed to save action set: {e}")
            raise
    
    def load(self, name: str) -> Optional[ActionSet]:
        """
        Load an action set by name.
        
        Args:
            name: Action set name
            
        Returns:
            ActionSet or None if not found
        """
        file_path = self._get_file_path(name)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            action_set = ActionSet.from_dict(data)
            Log.debug(f"ActionSetFileRepository: Loaded action set '{name}'")
            return action_set
            
        except Exception as e:
            Log.error(f"ActionSetFileRepository: Failed to load action set '{name}': {e}")
            return None
    
    def load_by_id(self, action_set_id: str) -> Optional[ActionSet]:
        """
        Load an action set by ID.
        
        Args:
            action_set_id: Action set ID
            
        Returns:
            ActionSet or None if not found
        """
        # Search all files for matching ID
        for action_set in self.list_all():
            if action_set.id == action_set_id:
                return action_set
        return None
    
    def list_all(self) -> List[ActionSet]:
        """
        List all available action sets.
        
        Returns:
            List of ActionSet entities
        """
        action_sets = []
        
        for file_path in self.action_sets_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                action_set = ActionSet.from_dict(data)
                action_sets.append(action_set)
            except Exception as e:
                Log.warning(f"ActionSetFileRepository: Failed to load {file_path}: {e}")
        
        # Sort by name
        action_sets.sort(key=lambda x: x.name.lower())
        return action_sets
    
    def delete(self, name: str) -> bool:
        """
        Delete an action set.
        
        Args:
            name: Action set name
            
        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_file_path(name)
        
        if not file_path.exists():
            return False
        
        try:
            file_path.unlink()
            Log.info(f"ActionSetFileRepository: Deleted action set '{name}'")
            return True
        except Exception as e:
            Log.error(f"ActionSetFileRepository: Failed to delete action set: {e}")
            return False
    
    def exists(self, name: str) -> bool:
        """Check if an action set exists."""
        return self._get_file_path(name).exists()
    
    def get_action_sets_dir(self) -> Path:
        """Get the directory where action sets are stored."""
        return self.action_sets_dir


# Singleton instance
_file_repo: Optional[ActionSetFileRepository] = None


def get_action_set_file_repo() -> ActionSetFileRepository:
    """Get the singleton file repository instance."""
    global _file_repo
    if _file_repo is None:
        _file_repo = ActionSetFileRepository()
    return _file_repo


