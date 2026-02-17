"""
SQLite implementation of PreferencesRepository

Handles persistence of user preferences that persist across all sessions and projects.
"""
from typing import Optional, Dict, Any
from datetime import datetime

from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


class PreferencesRepository:
    """
    Repository for user preferences persistence.
    
    Preferences are application-wide settings like:
    - Default zoom level
    - Theme settings
    - Default block configurations
    - Recent files
    
    This data persists across sessions and projects.
    """
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a preference value.
        
        Args:
            key: Preference key
            value: Value to store (will be JSON serialized)
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            
            # Check if preference exists
            cursor.execute("""
                SELECT key FROM preferences WHERE key = ?
            """, (key,))
            
            exists = cursor.fetchone() is not None
            
            # Serialize value
            value_json = Database.json_encode(value) if isinstance(value, (dict, list)) else str(value)
            
            if exists:
                # Update existing
                cursor.execute("""
                    UPDATE preferences
                    SET value = ?, updated_at = ?
                    WHERE key = ?
                """, (
                    value_json,
                    datetime.now().isoformat(),
                    key
                ))
                Log.debug(f"Updated preference: {key}")
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO preferences (key, value, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    key,
                    value_json,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
                Log.debug(f"Created preference: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a preference value.
        
        Args:
            key: Preference key
            default: Default value if preference not found
            
        Returns:
            Preference value or default
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT value FROM preferences WHERE key = ?
            """, (key,))
            
            row = cursor.fetchone()
            if row is None:
                return default
            
            # Try to deserialize as JSON, otherwise return as string
            value_str = row[0]
            try:
                import json
                return json.loads(value_str)
            except (json.JSONDecodeError, TypeError):
                return value_str
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all preferences.
        
        Returns:
            Dictionary of all preference key-value pairs
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key, value FROM preferences
            """)
            
            preferences = {}
            for row in cursor.fetchall():
                key = row[0]
                value_str = row[1]
                
                # Try to deserialize as JSON
                try:
                    import json
                    preferences[key] = json.loads(value_str)
                except (json.JSONDecodeError, TypeError):
                    preferences[key] = value_str
            
            return preferences
    
    def delete(self, key: str) -> None:
        """
        Delete a preference.
        
        Args:
            key: Preference key to delete
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM preferences WHERE key = ?
            """, (key,))
            
            Log.debug(f"Deleted preference: {key}")
    
    def clear(self) -> None:
        """
        Clear all preferences.
        
        Use with caution - this removes all user preferences.
        """
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM preferences")
            Log.warning("Cleared all preferences")


