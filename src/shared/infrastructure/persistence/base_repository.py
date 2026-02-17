"""
Base Repository Pattern

Provides common CRUD patterns and utilities for repository implementations.
This eliminates ~80% of boilerplate code in repository implementations.

Usage:
    class SQLiteProjectRepository(BaseRepository[Project]):
        entity_name = "Project"
        table_name = "projects"
        id_column = "id"
        
        def _row_to_entity(self, row) -> Project:
            return Project(id=row["id"], name=row["name"], ...)
        
        def _entity_to_row(self, entity: Project) -> dict:
            return {"id": entity.id, "name": entity.name, ...}

Features:
- Type-safe generic base class
- Standard error handling with custom exceptions
- Common existence checking patterns
- Transaction management helpers
- Logging integration
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Any, Dict, Type
from dataclasses import dataclass

from src.infrastructure.persistence.sqlite.database import Database
from src.utils.message import Log


# =============================================================================
# Custom Exceptions
# =============================================================================

class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found."""
    
    def __init__(self, entity_name: str, entity_id: str, context: str = ""):
        self.entity_name = entity_name
        self.entity_id = entity_id
        self.context = context
        message = f"{entity_name} with id '{entity_id}' not found"
        if context:
            message += f" ({context})"
        super().__init__(message)


class DuplicateEntityError(RepositoryError):
    """Raised when a duplicate entity is detected."""
    
    def __init__(self, entity_name: str, field: str, value: str, context: str = ""):
        self.entity_name = entity_name
        self.field = field
        self.value = value
        self.context = context
        message = f"{entity_name} with {field} '{value}' already exists"
        if context:
            message += f" ({context})"
        super().__init__(message)


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar('T')  # Entity type


# =============================================================================
# Base Repository
# =============================================================================

class BaseRepository(ABC, Generic[T]):
    """
    Abstract base class for repository implementations.
    
    Provides common patterns for:
    - Entity existence checking
    - Unique field validation
    - Transaction management
    - Logging
    
    Subclasses must implement:
    - _row_to_entity(): Convert database row to entity
    - _entity_to_row(): Convert entity to database row (for create/update)
    
    Optional overrides:
    - _get_id(): Extract ID from entity (default: entity.id)
    - _validate_create(): Custom validation before create
    - _validate_update(): Custom validation before update
    
    Class attributes:
    - entity_name: Human-readable name for error messages (e.g., "Project")
    - table_name: Database table name
    - id_column: Primary key column name (default: "id")
    
    Example:
        class SQLiteProjectRepository(BaseRepository[Project]):
            entity_name = "Project"
            table_name = "projects"
            id_column = "id"
            
            def __init__(self, database: Database):
                super().__init__(database)
            
            def _row_to_entity(self, row) -> Project:
                return Project(
                    id=row["id"],
                    name=row["name"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
            
            def _entity_to_row(self, entity: Project) -> dict:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "created_at": entity.created_at.isoformat()
                }
    """
    
    # Class attributes - override in subclasses
    entity_name: str = "Entity"
    table_name: str = ""
    id_column: str = "id"
    
    def __init__(self, database: Database):
        """
        Initialize repository with database.
        
        Args:
            database: Database instance to use
        """
        self.db = database
    
    # =========================================================================
    # Abstract Methods (must implement)
    # =========================================================================
    
    @abstractmethod
    def _row_to_entity(self, row) -> T:
        """
        Convert a database row to an entity.
        
        Args:
            row: Database row (sqlite3.Row)
            
        Returns:
            Entity instance
        """
        pass
    
    @abstractmethod
    def _entity_to_row(self, entity: T) -> Dict[str, Any]:
        """
        Convert an entity to a dictionary for database operations.
        
        Args:
            entity: Entity instance
            
        Returns:
            Dictionary with column names as keys
        """
        pass
    
    # =========================================================================
    # Optional Overrides
    # =========================================================================
    
    def _get_id(self, entity: T) -> str:
        """
        Extract the ID from an entity.
        
        Override if the entity uses a different ID attribute.
        
        Args:
            entity: Entity instance
            
        Returns:
            Entity ID
        """
        return getattr(entity, 'id', None)
    
    def _validate_create(self, entity: T) -> None:
        """
        Validate entity before create.
        
        Override to add custom validation. Raise RepositoryError on failure.
        
        Args:
            entity: Entity to validate
        """
        pass
    
    def _validate_update(self, entity: T, existing: T) -> None:
        """
        Validate entity before update.
        
        Override to add custom validation. Raise RepositoryError on failure.
        
        Args:
            entity: New entity state
            existing: Existing entity from database
        """
        pass
    
    # =========================================================================
    # Common Utility Methods
    # =========================================================================
    
    def exists(self, entity_id: str) -> bool:
        """
        Check if an entity exists by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            True if entity exists, False otherwise
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT 1 FROM {self.table_name} WHERE {self.id_column} = ?",
                (entity_id,)
            )
            return cursor.fetchone() is not None
    
    def exists_by_field(self, field: str, value: Any, exclude_id: Optional[str] = None) -> bool:
        """
        Check if an entity exists with a specific field value.
        
        Useful for checking uniqueness constraints.
        
        Args:
            field: Field/column name to check
            value: Value to match
            exclude_id: Optional ID to exclude from check (for updates)
            
        Returns:
            True if an entity exists with that field value
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if exclude_id:
                cursor.execute(
                    f"SELECT 1 FROM {self.table_name} WHERE {field} = ? AND {self.id_column} != ?",
                    (value, exclude_id)
                )
            else:
                cursor.execute(
                    f"SELECT 1 FROM {self.table_name} WHERE {field} = ?",
                    (value,)
                )
            return cursor.fetchone() is not None
    
    def get_by_id(self, entity_id: str) -> Optional[T]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity or None if not found
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {self.table_name} WHERE {self.id_column} = ?",
                (entity_id,)
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_entity(row)
    
    def require_exists(self, entity_id: str, context: str = "") -> T:
        """
        Get an entity by ID, raising EntityNotFoundError if not found.
        
        Args:
            entity_id: Entity identifier
            context: Optional context for error message
            
        Returns:
            Entity
            
        Raises:
            EntityNotFoundError: If entity not found
        """
        entity = self.get_by_id(entity_id)
        if entity is None:
            raise EntityNotFoundError(self.entity_name, entity_id, context)
        return entity
    
    def require_unique_field(
        self,
        field: str,
        value: Any,
        exclude_id: Optional[str] = None,
        context: str = ""
    ) -> None:
        """
        Ensure a field value is unique, raising DuplicateEntityError if not.
        
        Args:
            field: Field/column name
            value: Value to check
            exclude_id: ID to exclude (for updates)
            context: Optional context for error message
            
        Raises:
            DuplicateEntityError: If duplicate found
        """
        if self.exists_by_field(field, value, exclude_id):
            raise DuplicateEntityError(self.entity_name, field, str(value), context)
    
    def list_all(self, order_by: Optional[str] = None) -> List[T]:
        """
        List all entities in the table.
        
        Args:
            order_by: Optional column to order by (e.g., "name ASC")
            
        Returns:
            List of all entities
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT * FROM {self.table_name}"
            if order_by:
                query += f" ORDER BY {order_by}"
            cursor.execute(query)
            rows = cursor.fetchall()
            return [self._row_to_entity(row) for row in rows]
    
    def count(self, where: Optional[str] = None, params: Optional[tuple] = None) -> int:
        """
        Count entities, optionally with a filter.
        
        Args:
            where: Optional WHERE clause (without "WHERE" keyword)
            params: Parameters for the WHERE clause
            
        Returns:
            Count of entities
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            if where:
                query += f" WHERE {where}"
            cursor.execute(query, params or ())
            return cursor.fetchone()[0]
    
    def delete_by_id(self, entity_id: str, require_exists: bool = True) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            entity_id: Entity identifier
            require_exists: If True, raise EntityNotFoundError if not found
            
        Returns:
            True if deleted, False if not found (when require_exists=False)
            
        Raises:
            EntityNotFoundError: If entity not found and require_exists=True
        """
        if require_exists:
            existing = self.require_exists(entity_id)
            entity_name = getattr(existing, 'name', entity_id)
        else:
            if not self.exists(entity_id):
                return False
            entity_name = entity_id
        
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE {self.id_column} = ?",
                (entity_id,)
            )
            Log.info(f"Deleted {self.entity_name}: {entity_name}")
            return True
    
    # =========================================================================
    # SQL Generation Helpers
    # =========================================================================
    
    def _build_insert_sql(self, columns: List[str]) -> str:
        """
        Build INSERT SQL statement.
        
        Args:
            columns: List of column names
            
        Returns:
            SQL INSERT statement
        """
        placeholders = ", ".join(["?"] * len(columns))
        columns_str = ", ".join(columns)
        return f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"
    
    def _build_update_sql(self, columns: List[str], where_column: str = None) -> str:
        """
        Build UPDATE SQL statement.
        
        Args:
            columns: List of column names to update
            where_column: Column for WHERE clause (default: id_column)
            
        Returns:
            SQL UPDATE statement
        """
        where_column = where_column or self.id_column
        set_clause = ", ".join([f"{col} = ?" for col in columns])
        return f"UPDATE {self.table_name} SET {set_clause} WHERE {where_column} = ?"
    
    # =========================================================================
    # Logging Helpers
    # =========================================================================
    
    def _log_create(self, entity: T, extra: str = "") -> None:
        """Log entity creation."""
        name = getattr(entity, 'name', self._get_id(entity))
        msg = f"Created {self.entity_name}: {name}"
        if extra:
            msg += f" ({extra})"
        Log.info(msg)
    
    def _log_update(self, entity: T, changes: List[str] = None) -> None:
        """Log entity update."""
        name = getattr(entity, 'name', self._get_id(entity))
        if changes:
            Log.info(f"Updated {self.entity_name}: {name} - Changes: {', '.join(changes)}")
        else:
            Log.debug(f"Updated {self.entity_name}: {name} - No changes detected")
    
    def _log_delete(self, entity: T) -> None:
        """Log entity deletion."""
        name = getattr(entity, 'name', self._get_id(entity))
        Log.info(f"Deleted {self.entity_name}: {name}")
