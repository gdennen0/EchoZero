import sqlite3
import os
import json
from src.Utils.message import Log

class DatabaseController:
    """
    Generic Controller for managing attributes within a database.
    """
    
    def __init__(self,db_id, db_type, db_path):
        """
        Initialize the database controller.
        
        Args:
            db_path (str, optional): Path to the database file. If provided, connects to the database.
        """
        self.connection = None
        self.db_path = None

        # Connect/create database
        if os.path.exists(db_path):
            self.connect(db_path)
        else:
            self.create(db_id, db_type, db_path)
        
    
    def create(self, db_id, db_name, db_type, db_path):
        """
        Create a new database at the specified path.
        
        Args:
            db_path (str): Path to create the database
            attributes (dict, optional): Initial attributes to set
            
        Returns:
            bool: True if database was created successfully
        """
        # Check if file already exists
        if os.path.exists(db_path):
            Log.warning(f"Database already exists at {db_path}")
            return False
            
        self.db_path = db_path
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(db_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            self.add_attribute("db_id", db_id, "int")
            self.add_attribute("db_name", db_name, "str")
            self.add_attribute("db_type", db_type, "str")
            self.add_attribute("db_path", db_path, "str")
                    
            return True
        except Exception as e:
            Log.error(f"Error creating database: {str(e)}")
            return False
    
    def connect(self, db_path):
        """
        Connect to an existing database.
        
        Args:
            db_path (str): Path to the database file
            
        Returns:
            bool: True if connected successfully
        """
        if not os.path.exists(db_path):
            Log.error(f"Database does not exist at {db_path}")
            return False
            
        try:
            # Close existing connection if any
            self.close_connection()
            
            self.db_path = db_path
            conn = self.get_connection()
            
            # Verify the database has the necessary structure
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attributes'")
            if not cursor.fetchone():
                Log.error(f"Database at {db_path} is not a valid attribute database")
                self.close_connection()
                self.db_path = None
                return False
                
            return True
        except Exception as e:
            Log.error(f"Error connecting to database: {str(e)}")
            self.db_path = None
            return False
    
    def get_connection(self):
        """
        Get or create a database connection.
        
        Returns:
            sqlite3.Connection: The database connection
        
        Raises:
            ValueError: If database path is not set
        """
        if self.connection is None:
            if not self.db_path:
                raise ValueError("Database path not set. Create or connect to a database first.")
            self.connection = sqlite3.connect(self.db_path)
            # Configure connection to return dictionaries instead of tuples
            self.connection.row_factory = sqlite3.Row
        return self.connection
        
    def close_connection(self):
        """Close the database connection if it exists."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def is_connected(self):
        """
        Check if controller is connected to a database.
        
        Returns:
            bool: True if connected to a database
        """
        return self.db_path is not None

    # Attribute operations
    def create_attribute(self, attribute_name, attribute_type, value=None ):
        """
        Add a new attribute to the database.
        
        Args:
            attribute_name (str): The attribute name
            attribute_value (any): Value for the attribute
            attribute_type (str: Type of the attribute (str, int, float, bool, json)
            
        Returns:
            bool: True if attribute was added successfully
        """
        if not self.is_connected():
            raise ValueError("Not connected to a database")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Create a table for this attribute if it doesn't exist
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {attribute_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    value TEXT
                )
            """)

            if value is None:
                value = "NULL"
            
            # Insert or update the value in the attribute-specific table
            cursor.execute(f"""
                INSERT OR REPLACE INTO {attribute_name} (name, type, value)
                VALUES (?, ?, ?)
            """, (attribute_name, attribute_type, value))
            
            conn.commit()
            return True
        
        except sqlite3.IntegrityError:
            # Attribute already exists
            Log.warning(f"Attribute '{attribute_name}' already exists")
            return False
    
    def set_attribute(self, attribute_name, value):
        """
        Set the value of an attribute. Replaces any existing value even if its multiple values. 
        
        Args:
            attribute_name (str): The attribute name
            value (any): The value to set
            
        Returns:
            bool: True if successful
        """
        if not self.is_connected():
            raise ValueError("Not connected to a database")
            
        if self.table_exists(attribute_name):
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Delete all existing data in the table
            cursor.execute(f"DELETE FROM {attribute_name}")
            
            # Insert the new value
            cursor.execute(f"""
                INSERT INTO {attribute_name} (name, type, value) 
                VALUES (?, ?, ?)
            """, (attribute_name, type(value).__name__, str(value)))
            
            conn.commit()
            return True
        else:
            Log.error(f"Attribute '{attribute_name}' does not exist")
            return False
    
    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        
        Args:
            table_name (str): The name of the table to check
            
        Returns:
            bool: True if the table exists, False otherwise
        """
        if not self.is_connected():
            return False
            
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        result = cursor.fetchone()
        
        return result is not None
    
    def get_attribute(self, attribute_name, id=None):
        """
        Get the value of an attribute.
        
        Args:
            attribute_name (str): The attribute name
            id (int, optional): The specific ID to get. If None, returns all values from the table.
            
        Returns:
            any: The attribute value (converted to appropriate type) or None if not found
                If id=None, returns a list of all values in the table.
        """
        if not self.is_connected():
            return None
            
        if not self.table_exists(attribute_name):
            return None
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if id is not None:
            # Get the specific row by ID
            cursor.execute(
                f"SELECT value, type FROM {attribute_name} WHERE id = ?", 
                (id,)
            )
            result = cursor.fetchone()
            
            if not result:
                # If no result by ID, try getting by name (backwards compatibility)
                cursor.execute(
                    f"SELECT value, type FROM {attribute_name} WHERE name = ?", 
                    (attribute_name,)
                )
                result = cursor.fetchone()
                
            if not result:
                return None
                
            value = result['value']
            return value
        else:
            # Get ALL values from the table (default)
            cursor.execute(f"SELECT id, value, type FROM {attribute_name}")
            results = cursor.fetchall()
            
            if not results:
                return None
                
            values = []
            for row in results:
                values.append({
                    'id': row['id'],
                    'value': row['value'],
                    'type': row['type']
                })
                
            return values
    
    def remove_attribute(self, attribute_name, id=None):
        """
        Remove an attribute by dropping its table.
        
        Args:
            attribute_name (str): The attribute name to remove
            id (int, optional): The specific ID to remove. If None, removes all values from the table.
        Returns:
            bool: True if successful
        """
        if not self.is_connected():
            return False
            
        if not self.table_exists(attribute_name):
            Log.warning(f"Attribute '{attribute_name}' does not exist")
            return False
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"DROP TABLE {attribute_name}")
            conn.commit()
            return True
        except sqlite3.Error as e:
            Log.error(f"Error removing attribute '{attribute_name}': {str(e)}")
            return False
    
    def get_all_attributes(self):
        """
        Get all attributes in this db and their values.
        
        Returns:
            dict: Dictionary of attribute key-value pairs
        """
        if not self.is_connected():
            return {}
            
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all table names that are attributes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        attributes = {}
        for table in tables:
            attribute_name = table['name']
            
            # Skip system tables
            if attribute_name.startswith('sqlite_'):
                continue
                
            # Get attribute value and type
            try:
                cursor.execute(f"SELECT value, type FROM {attribute_name} WHERE name = ?", (attribute_name,))
                result = cursor.fetchone()
                
                if result:
                    value_str = result['value']
                    attribute_type = result['type']
                    
                    # Convert string value to appropriate type
                    if value_str is not None:
                        if attribute_type == "int":
                            attributes[attribute_name] = int(value_str)
                        elif attribute_type == "float":
                            attributes[attribute_name] = float(value_str)
                        elif attribute_type == "bool":
                            attributes[attribute_name] = value_str == "1"
                        elif attribute_type == "json":
                            try:
                                attributes[attribute_name] = json.loads(value_str)
                            except:
                                attributes[attribute_name] = value_str
                        else:  # str or unknown type
                            attributes[attribute_name] = value_str
                    else:
                        attributes[attribute_name] = None
            except sqlite3.Error:
                # Skip tables with invalid structure
                continue
                
        return attributes