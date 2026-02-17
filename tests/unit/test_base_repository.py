"""
Tests for the BaseRepository pattern.

Tests common CRUD patterns, error handling, and utility methods.
"""
import pytest
import sqlite3
import tempfile
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.shared.infrastructure.persistence.base_repository import (
    BaseRepository,
    RepositoryError,
    EntityNotFoundError,
    DuplicateEntityError,
)
from src.infrastructure.persistence.sqlite.database import Database


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class TestEntity:
    """Simple test entity."""
    id: str
    name: str
    value: int = 0


class TestRepository(BaseRepository[TestEntity]):
    """Concrete repository for testing."""
    
    entity_name = "TestEntity"
    table_name = "test_entities"
    id_column = "id"
    
    def _row_to_entity(self, row) -> TestEntity:
        return TestEntity(
            id=row["id"],
            name=row["name"],
            value=row["value"]
        )
    
    def _entity_to_row(self, entity: TestEntity) -> Dict[str, Any]:
        return {
            "id": entity.id,
            "name": entity.name,
            "value": entity.value
        }
    
    def create(self, entity: TestEntity) -> TestEntity:
        """Create a test entity."""
        self._validate_create(entity)
        self.require_unique_field("name", entity.name)
        
        row = self._entity_to_row(entity)
        columns = list(row.keys())
        values = list(row.values())
        
        sql = self._build_insert_sql(columns)
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
        
        self._log_create(entity)
        return entity
    
    def update(self, entity: TestEntity) -> None:
        """Update a test entity."""
        existing = self.require_exists(entity.id)
        self._validate_update(entity, existing)
        self.require_unique_field("name", entity.name, exclude_id=entity.id)
        
        row = self._entity_to_row(entity)
        columns = [c for c in row.keys() if c != "id"]
        values = [row[c] for c in columns] + [entity.id]
        
        sql = self._build_update_sql(columns)
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
        
        changes = []
        if existing.name != entity.name:
            changes.append(f"name: '{existing.name}' -> '{entity.name}'")
        if existing.value != entity.value:
            changes.append(f"value: {existing.value} -> {entity.value}")
        self._log_update(entity, changes)


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    
    db = Database(path)
    
    # Create test table
    with db.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                value INTEGER DEFAULT 0
            )
        """)
    
    yield db
    
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def repository(test_db):
    """Create a test repository."""
    return TestRepository(test_db)


@pytest.fixture
def sample_entity():
    """Create a sample entity."""
    return TestEntity(id="test-1", name="Test Entity", value=42)


# =============================================================================
# Exception Tests
# =============================================================================

class TestExceptions:
    """Tests for custom exceptions."""
    
    def test_entity_not_found_error(self):
        """Test EntityNotFoundError message format."""
        error = EntityNotFoundError("Project", "proj-123")
        assert "Project" in str(error)
        assert "proj-123" in str(error)
        assert error.entity_name == "Project"
        assert error.entity_id == "proj-123"
    
    def test_entity_not_found_error_with_context(self):
        """Test EntityNotFoundError with context."""
        error = EntityNotFoundError("Block", "blk-456", "in project proj-123")
        assert "in project proj-123" in str(error)
        assert error.context == "in project proj-123"
    
    def test_duplicate_entity_error(self):
        """Test DuplicateEntityError message format."""
        error = DuplicateEntityError("Project", "name", "MyProject")
        assert "Project" in str(error)
        assert "name" in str(error)
        assert "MyProject" in str(error)
        assert error.entity_name == "Project"
        assert error.field == "name"
        assert error.value == "MyProject"
    
    def test_duplicate_entity_error_with_context(self):
        """Test DuplicateEntityError with context."""
        error = DuplicateEntityError("Block", "name", "MyBlock", "in project")
        assert "in project" in str(error)


# =============================================================================
# Base Repository Tests
# =============================================================================

class TestBaseRepository:
    """Tests for BaseRepository base class."""
    
    def test_exists_when_empty(self, repository):
        """Test exists returns False for empty table."""
        assert repository.exists("nonexistent") is False
    
    def test_exists_after_create(self, repository, sample_entity):
        """Test exists returns True after create."""
        repository.create(sample_entity)
        assert repository.exists(sample_entity.id) is True
    
    def test_exists_by_field(self, repository, sample_entity):
        """Test exists_by_field checks field values."""
        repository.create(sample_entity)
        assert repository.exists_by_field("name", sample_entity.name) is True
        assert repository.exists_by_field("name", "nonexistent") is False
    
    def test_exists_by_field_with_exclude(self, repository, sample_entity):
        """Test exists_by_field excludes specified ID."""
        repository.create(sample_entity)
        # Should return False when we exclude the only matching entity
        assert repository.exists_by_field("name", sample_entity.name, exclude_id=sample_entity.id) is False
    
    def test_get_by_id(self, repository, sample_entity):
        """Test get_by_id retrieves entity."""
        repository.create(sample_entity)
        result = repository.get_by_id(sample_entity.id)
        assert result is not None
        assert result.id == sample_entity.id
        assert result.name == sample_entity.name
        assert result.value == sample_entity.value
    
    def test_get_by_id_not_found(self, repository):
        """Test get_by_id returns None when not found."""
        result = repository.get_by_id("nonexistent")
        assert result is None
    
    def test_require_exists(self, repository, sample_entity):
        """Test require_exists returns entity when found."""
        repository.create(sample_entity)
        result = repository.require_exists(sample_entity.id)
        assert result.id == sample_entity.id
    
    def test_require_exists_raises(self, repository):
        """Test require_exists raises when not found."""
        with pytest.raises(EntityNotFoundError) as exc:
            repository.require_exists("nonexistent")
        assert "nonexistent" in str(exc.value)
    
    def test_require_exists_with_context(self, repository):
        """Test require_exists includes context in error."""
        with pytest.raises(EntityNotFoundError) as exc:
            repository.require_exists("nonexistent", context="during update")
        assert "during update" in str(exc.value)
    
    def test_require_unique_field(self, repository, sample_entity):
        """Test require_unique_field passes when unique."""
        # Should not raise for non-existent value
        repository.require_unique_field("name", "unique_name")
    
    def test_require_unique_field_raises(self, repository, sample_entity):
        """Test require_unique_field raises on duplicate."""
        repository.create(sample_entity)
        with pytest.raises(DuplicateEntityError) as exc:
            repository.require_unique_field("name", sample_entity.name)
        assert sample_entity.name in str(exc.value)
    
    def test_require_unique_field_with_exclude(self, repository, sample_entity):
        """Test require_unique_field allows same ID."""
        repository.create(sample_entity)
        # Should not raise when excluding own ID
        repository.require_unique_field("name", sample_entity.name, exclude_id=sample_entity.id)
    
    def test_list_all(self, repository):
        """Test list_all returns all entities."""
        entities = [
            TestEntity(id="test-1", name="Entity 1", value=1),
            TestEntity(id="test-2", name="Entity 2", value=2),
            TestEntity(id="test-3", name="Entity 3", value=3),
        ]
        for e in entities:
            repository.create(e)
        
        results = repository.list_all()
        assert len(results) == 3
    
    def test_list_all_with_order(self, repository):
        """Test list_all respects ordering."""
        entities = [
            TestEntity(id="test-2", name="Bravo", value=2),
            TestEntity(id="test-1", name="Alpha", value=1),
            TestEntity(id="test-3", name="Charlie", value=3),
        ]
        for e in entities:
            repository.create(e)
        
        results = repository.list_all(order_by="name ASC")
        assert [r.name for r in results] == ["Alpha", "Bravo", "Charlie"]
    
    def test_count(self, repository):
        """Test count returns correct count."""
        assert repository.count() == 0
        
        entities = [
            TestEntity(id="test-1", name="Entity 1", value=10),
            TestEntity(id="test-2", name="Entity 2", value=20),
            TestEntity(id="test-3", name="Entity 3", value=10),
        ]
        for e in entities:
            repository.create(e)
        
        assert repository.count() == 3
    
    def test_count_with_filter(self, repository):
        """Test count with WHERE filter."""
        entities = [
            TestEntity(id="test-1", name="Entity 1", value=10),
            TestEntity(id="test-2", name="Entity 2", value=20),
            TestEntity(id="test-3", name="Entity 3", value=10),
        ]
        for e in entities:
            repository.create(e)
        
        # Count entities with value = 10
        assert repository.count("value = ?", (10,)) == 2
    
    def test_delete_by_id(self, repository, sample_entity):
        """Test delete_by_id removes entity."""
        repository.create(sample_entity)
        assert repository.exists(sample_entity.id) is True
        
        result = repository.delete_by_id(sample_entity.id)
        
        assert result is True
        assert repository.exists(sample_entity.id) is False
    
    def test_delete_by_id_not_found_raises(self, repository):
        """Test delete_by_id raises when not found."""
        with pytest.raises(EntityNotFoundError):
            repository.delete_by_id("nonexistent")
    
    def test_delete_by_id_not_found_silent(self, repository):
        """Test delete_by_id with require_exists=False."""
        result = repository.delete_by_id("nonexistent", require_exists=False)
        assert result is False


class TestCRUDOperations:
    """Tests for full CRUD operations using the concrete repository."""
    
    def test_create(self, repository, sample_entity):
        """Test create adds entity to database."""
        result = repository.create(sample_entity)
        
        assert result.id == sample_entity.id
        assert repository.exists(sample_entity.id)
    
    def test_create_duplicate_name_raises(self, repository, sample_entity):
        """Test create raises on duplicate name."""
        repository.create(sample_entity)
        
        duplicate = TestEntity(id="test-2", name=sample_entity.name, value=99)
        with pytest.raises(DuplicateEntityError):
            repository.create(duplicate)
    
    def test_update(self, repository, sample_entity):
        """Test update modifies entity."""
        repository.create(sample_entity)
        
        sample_entity.name = "Updated Name"
        sample_entity.value = 100
        repository.update(sample_entity)
        
        result = repository.get_by_id(sample_entity.id)
        assert result.name == "Updated Name"
        assert result.value == 100
    
    def test_update_not_found_raises(self, repository, sample_entity):
        """Test update raises when entity not found."""
        with pytest.raises(EntityNotFoundError):
            repository.update(sample_entity)
    
    def test_update_duplicate_name_raises(self, repository):
        """Test update raises on duplicate name conflict."""
        entity1 = TestEntity(id="test-1", name="Entity 1", value=1)
        entity2 = TestEntity(id="test-2", name="Entity 2", value=2)
        repository.create(entity1)
        repository.create(entity2)
        
        # Try to rename entity2 to entity1's name
        entity2.name = "Entity 1"
        with pytest.raises(DuplicateEntityError):
            repository.update(entity2)


class TestSQLGeneration:
    """Tests for SQL generation helpers."""
    
    def test_build_insert_sql(self, repository):
        """Test INSERT SQL generation."""
        sql = repository._build_insert_sql(["id", "name", "value"])
        assert "INSERT INTO test_entities" in sql
        assert "(id, name, value)" in sql
        assert "(?, ?, ?)" in sql
    
    def test_build_update_sql(self, repository):
        """Test UPDATE SQL generation."""
        sql = repository._build_update_sql(["name", "value"])
        assert "UPDATE test_entities" in sql
        assert "SET name = ?, value = ?" in sql
        assert "WHERE id = ?" in sql
    
    def test_build_update_sql_custom_where(self, repository):
        """Test UPDATE SQL with custom WHERE column."""
        sql = repository._build_update_sql(["name"], where_column="custom_id")
        assert "WHERE custom_id = ?" in sql
