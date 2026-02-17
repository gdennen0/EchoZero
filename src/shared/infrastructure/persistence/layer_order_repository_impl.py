"""SQLite implementation of LayerOrderRepository.

Persists per-block layer order in `layer_orders`.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from src.infrastructure.persistence.sqlite.database import Database
from src.shared.domain.entities.layer_order import LayerOrder, LayerKey
from src.shared.domain.repositories.layer_order_repository import LayerOrderRepository
from src.utils.message import Log


class SQLiteLayerOrderRepository(LayerOrderRepository):
    """SQLite repository for layer order per block."""

    def __init__(self, database: Database):
        self.db = database

    def get_order(self, block_id: str) -> Optional[LayerOrder]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT order_json
                FROM layer_orders
                WHERE block_id = ?
                """,
                (block_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            try:
                raw = json.loads(row[0]) if row[0] else []
                order = [LayerKey.from_dict(entry) for entry in raw if isinstance(entry, dict)]
            except Exception as e:
                Log.warning(f"SQLiteLayerOrderRepository: Failed to decode order for {block_id}: {e}")
                order = []
            return LayerOrder(block_id=block_id, order=order)

    def set_order(self, layer_order: LayerOrder) -> None:
        order_json = json.dumps([key.to_dict() for key in (layer_order.order or [])])
        now = datetime.utcnow().isoformat()
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO layer_orders (block_id, order_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(block_id) DO UPDATE SET
                    order_json = excluded.order_json,
                    updated_at = excluded.updated_at
                """,
                (layer_order.block_id, order_json, now),
            )

    def clear_order(self, block_id: str) -> None:
        with self.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM layer_orders
                WHERE block_id = ?
                """,
                (block_id,),
            )
