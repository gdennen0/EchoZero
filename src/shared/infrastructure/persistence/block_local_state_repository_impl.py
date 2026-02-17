"""SQLite implementation of BlockLocalStateRepository.

Persists per-block local input references in `block_local_state`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from src.utils.message import Log
from src.infrastructure.persistence.sqlite.database import Database
from src.shared.domain.repositories.block_local_state_repository import BlockLocalStateRepository, LocalInputs


class SQLiteBlockLocalStateRepository(BlockLocalStateRepository):
    def __init__(self, database: Database):
        self.db = database

    def get_inputs(self, block_id: str) -> Optional[LocalInputs]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT inputs_json
                FROM block_local_state
                WHERE block_id = ?
                """,
                (block_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            try:
                return json.loads(row[0]) if row[0] else {}
            except Exception as e:
                Log.warning(f"SQLiteBlockLocalStateRepository: Failed to decode inputs_json for {block_id}: {e}")
                return {}

    def set_inputs(self, block_id: str, inputs: LocalInputs) -> None:
        inputs_json = json.dumps(inputs or {})
        now = datetime.utcnow().isoformat()
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO block_local_state (block_id, inputs_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    inputs_json = excluded.inputs_json,
                    updated_at = excluded.updated_at
                """,
                (block_id, inputs_json, now),
            )

    def clear_inputs(self, block_id: str) -> None:
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM block_local_state
                WHERE block_id = ?
                """,
                (block_id,),
            )
