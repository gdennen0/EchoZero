"""
Runtime SQLite session database manager

File-based SQLite database for viewing live session state.
Database is cleared on application initialization and project load
to maintain session-only semantics - only the current project data is stored.
"""
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.message import Log


class Database:
    """
    Session-only file-based SQLite database.
    
    Provides a viewable runtime cache for the current project session.
    Database tables are cleared on app init and project load to ensure
    only the active project data is present. Source of truth remains
    the JSON .ez project files.
    """

    CURRENT_SCHEMA_VERSION = 6

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            raise ValueError("Database path is required for file-based operation")
        
        self.db_path = Path(db_path)
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to file-based database for live viewing
        self._connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        conn = self._connection
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                save_directory TEXT,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                ports TEXT,
                metadata TEXT,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                UNIQUE(project_id, name)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                id TEXT PRIMARY KEY,
                source_block_id TEXT NOT NULL,
                source_block_name TEXT,
                source_output_name TEXT NOT NULL,
                target_block_id TEXT NOT NULL,
                target_block_name TEXT,
                target_input_name TEXT NOT NULL,
                FOREIGN KEY (source_block_id) REFERENCES blocks(id) ON DELETE CASCADE,
                FOREIGN KEY (target_block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_items (
                id TEXT PRIMARY KEY,
                block_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                file_path TEXT,
                metadata TEXT,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)

        # Block local input state (references only): input_port -> data_item_id(s)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS block_local_state (
                block_id TEXT PRIMARY KEY,
                inputs_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)

        # UI state tables (Phase A Foundation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ui_state (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                entity_id TEXT,
                data TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Layer order (per-block ordering)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_orders (
                block_id TEXT PRIMARY KEY,
                order_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (block_id) REFERENCES blocks(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Setlist tables
        # One setlist per project - enforced by UNIQUE constraint on project_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS setlists (
                id TEXT PRIMARY KEY,
                audio_folder_path TEXT NOT NULL,
                project_id TEXT NOT NULL UNIQUE,
                default_actions TEXT,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS setlist_songs (
                id TEXT PRIMARY KEY,
                setlist_id TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                order_index INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                processed_at TEXT,
                action_overrides TEXT,
                error_message TEXT,
                metadata TEXT,
                FOREIGN KEY (setlist_id) REFERENCES setlists(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("SELECT COUNT(*) FROM schema_version")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (self.CURRENT_SCHEMA_VERSION,))

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocks_project_id ON blocks(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_connections_source ON connections(source_block_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_connections_target ON connections(target_block_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_items_block_id ON data_items(block_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_block_local_state_block_id ON block_local_state(block_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ui_state_type_entity ON ui_state(type, entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ui_state_entity ON ui_state(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_layer_orders_block_id ON layer_orders(block_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_setlist_songs_setlist_id ON setlist_songs(setlist_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_setlist_songs_order ON setlist_songs(setlist_id, order_index)")

        # Action sets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_sets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                actions TEXT NOT NULL,
                project_id TEXT,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_sets_project_id ON action_sets(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_sets_name ON action_sets(name)")

        # Action items table (user-configured action events within action sets)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY,
                action_set_id TEXT,
                project_id TEXT NOT NULL,
                action_type TEXT NOT NULL DEFAULT 'block',
                block_id TEXT,
                block_name TEXT,
                action_name TEXT NOT NULL,
                action_description TEXT,
                action_args TEXT,
                order_index INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_items_action_set_id ON action_items(action_set_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_items_project_id ON action_items(project_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_action_items_order ON action_items(action_set_id, order_index)")
        
        # Migration: Add action_type column if it doesn't exist
        cursor.execute("PRAGMA table_info(action_items)")
        columns = {row[1]: row for row in cursor.fetchall()}
        
        if 'action_type' not in columns:
            cursor.execute("ALTER TABLE action_items ADD COLUMN action_type TEXT NOT NULL DEFAULT 'block'")
            Log.info("Database: Added action_type column to action_items table")

        # Migration: Add block name columns to connections table if they don't exist
        cursor.execute("PRAGMA table_info(connections)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'source_block_name' not in columns:
            Log.info("Migrating connections table: adding block name columns")
            cursor.execute("ALTER TABLE connections ADD COLUMN source_block_name TEXT")
            cursor.execute("ALTER TABLE connections ADD COLUMN target_block_name TEXT")
        
        # Migration: Remove UNIQUE constraint on connections (target_block_id, target_input_name)
        # This allows multiple outputs to connect to the same input (e.g., Event ports)
        cursor.execute("PRAGMA index_list(connections)")
        indexes = cursor.fetchall()
        has_unique_target_constraint = any(
            'target_block_id' in str(idx) or 'sqlite_autoindex_connections' in str(idx)
            for idx in indexes if idx[2] == 1  # idx[2] == 1 means unique
        )
        
        # Check for autoindex which indicates inline UNIQUE constraint
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='connections'")
        table_sql = cursor.fetchone()
        if table_sql and 'UNIQUE(target_block_id, target_input_name)' in str(table_sql[0]):
            has_unique_target_constraint = True
        
        if has_unique_target_constraint:
            Log.info("Migrating connections table: removing UNIQUE constraint on (target_block_id, target_input_name)")
            # Save existing data
            cursor.execute("SELECT id, source_block_id, source_block_name, source_output_name, target_block_id, target_block_name, target_input_name FROM connections")
            existing_connections = cursor.fetchall()
            
            # Drop and recreate without UNIQUE constraint
            cursor.execute("DROP TABLE connections")
            cursor.execute("""
                CREATE TABLE connections (
                    id TEXT PRIMARY KEY,
                    source_block_id TEXT NOT NULL,
                    source_block_name TEXT,
                    source_output_name TEXT NOT NULL,
                    target_block_id TEXT NOT NULL,
                    target_block_name TEXT,
                    target_input_name TEXT NOT NULL,
                    FOREIGN KEY (source_block_id) REFERENCES blocks(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_block_id) REFERENCES blocks(id) ON DELETE CASCADE
                )
            """)
            
            # Restore data
            for connection_row in existing_connections:
                cursor.execute("""
                    INSERT INTO connections (id, source_block_id, source_block_name, source_output_name, 
                                            target_block_id, target_block_name, target_input_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, connection_row)
            
            Log.info(f"Migrated connections table: restored {len(existing_connections)} connection(s)")
        
        
        # Migration: Drop and recreate setlists table with correct schema (one per project)
        cursor.execute("PRAGMA table_info(setlists)")
        setlist_columns = [row[1] for row in cursor.fetchall()]
        
        # Check if we need to migrate to one-setlist-per-project schema
        needs_migration = False
        if setlist_columns:
            has_old_fields = 'template_project_path' in setlist_columns or 'name' in setlist_columns
            missing_new_fields = 'audio_folder_path' not in setlist_columns or 'project_id' not in setlist_columns
            
            # Check if UNIQUE constraint is on project_id (new) or audio_folder_path (old)
            cursor.execute("PRAGMA index_list(setlists)")
            indexes = cursor.fetchall()
            has_project_unique = any('project_id' in str(idx) for idx in indexes)
            has_folder_unique = any('audio_folder_path' in str(idx) for idx in indexes)
            
            if has_old_fields or missing_new_fields or (has_folder_unique and not has_project_unique):
                needs_migration = True
        
        if needs_migration:
            Log.info("Migrating setlists table: one setlist per project (UNIQUE on project_id)")
            # Save existing data
            cursor.execute("SELECT * FROM setlists")
            existing_setlists = cursor.fetchall()
            
            # Drop old table (cascade will handle setlist_songs)
            cursor.execute("DROP TABLE IF EXISTS setlists")
            
            # Recreate with correct schema (UNIQUE on project_id, not audio_folder_path)
            cursor.execute("""
                CREATE TABLE setlists (
                    id TEXT PRIMARY KEY,
                    audio_folder_path TEXT NOT NULL,
                    project_id TEXT NOT NULL UNIQUE,
                    default_actions TEXT,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Restore data (keep only one setlist per project - the most recent one)
            if existing_setlists:
                # Group by project_id and keep most recent
                projects_seen = {}
                for row in existing_setlists:
                    project_id = row[2] if len(row) > 2 else None
                    if project_id:
                        if project_id not in projects_seen:
                            projects_seen[project_id] = row
                        else:
                            # Keep the one with later modified_at
                            existing_modified = projects_seen[project_id][6] if len(projects_seen[project_id]) > 6 else ""
                            new_modified = row[6] if len(row) > 6 else ""
                            if new_modified > existing_modified:
                                projects_seen[project_id] = row
                
                # Insert one setlist per project (skip execution_strategy column)
                for row in projects_seen.values():
                    # Extract columns: id, audio_folder_path, project_id, default_actions, created_at, modified_at, metadata
                    # Skip execution_strategy (index 3 if it exists)
                    row_data = list(row)
                    if len(row_data) > 3:
                        # Remove execution_strategy if present
                        row_data.pop(3) if len(row_data) > 3 else None
                    cursor.execute("""
                        INSERT INTO setlists (
                            id, audio_folder_path, project_id, default_actions,
                            created_at, modified_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (row_data[0], row_data[1], row_data[2], row_data[4] if len(row_data) > 4 else "{}", 
                          row_data[5] if len(row_data) > 5 else "", row_data[6] if len(row_data) > 6 else "", 
                          row_data[7] if len(row_data) > 7 else "{}"))
                
                Log.info(f"Migrated {len(projects_seen)} setlist(s) (one per project)")
            
            Log.info("Recreated setlists table with one-setlist-per-project schema")
        
        # Migration: Update setlist_songs table schema if needed
        cursor.execute("PRAGMA table_info(setlist_songs)")
        song_columns = [row[1] for row in cursor.fetchall()]
        if 'action_overrides' not in song_columns:
            Log.info("Migrating setlist_songs table: adding action_overrides column")
            try:
                cursor.execute("ALTER TABLE setlist_songs ADD COLUMN action_overrides TEXT")
            except sqlite3.OperationalError:
                pass  # Column might already exist
        if 'error_message' not in song_columns:
            Log.info("Migrating setlist_songs table: adding error_message column")
            try:
                cursor.execute("ALTER TABLE setlist_songs ADD COLUMN error_message TEXT")
            except sqlite3.OperationalError:
                pass  # Column might already exist
        
        # Migration: Drop deprecated actions table (actions are now built dynamically from quick_actions)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actions'")
        if cursor.fetchone():
            Log.info("Dropping deprecated 'actions' table (actions are now built on init from quick_actions)")
            cursor.execute("DROP TABLE actions")

        conn.commit()
        Log.info(f"Runtime database initialized{' (' + str(self.db_path) + ')' if self.db_path else ''}")

    def get_connection(self) -> sqlite3.Connection:
        conn = self._connection
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def transaction(self):
        return TransactionContext(self.get_connection())

    def get_schema_version(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else 1

    def clear_runtime_tables(self) -> None:
        """
        Clear all runtime session data from the database.
        
        Called on application initialization and project load to ensure
        only the current session's project data is present.
        
        Note: preferences and session_state persist across projects,
        only ui_state (which is project-specific) is cleared.
        """
        conn = self._connection
        cursor = conn.cursor()
        for table in ("data_items", "block_local_state", "connections", "blocks", "projects", "ui_state", "layer_orders", "setlist_songs", "setlists", "action_sets", "action_items"):
            cursor.execute(f"DELETE FROM {table}")
        conn.commit()
        Log.info(f"Session data cleared: {self.db_path}")

    def reset(self) -> None:
        """
        Reset session by clearing all runtime tables.
        
        Maintains file-based connection for live viewing while
        ensuring clean session state.
        """
        self.clear_runtime_tables()

    def clear_all_data(self) -> None:
        """Clear all session data (alias for reset)."""
        self.reset()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            Log.info(f"Database connection closed: {self.db_path}")

    @staticmethod
    def json_encode(value: Optional[Dict[str, Any]]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value)

    @staticmethod
    def json_decode(value: Optional[str]) -> Dict[str, Any]:
        if value is None:
            return {}
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}


class TransactionContext:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
            Log.error(f"Transaction rolled back: {exc_val}")
        return False
