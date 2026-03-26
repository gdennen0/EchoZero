"""
BaseRepository: Generic abstract base for all SQLite repositories.
Exists because every repository shares the same connection-handling and row-mapping pattern.
Never commits — session owns transactions via Unit of Work.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from echozero.errors import PersistenceError

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Base class for all SQLite repositories. Never commits — session owns transactions."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @abstractmethod
    def _from_row(self, row: sqlite3.Row) -> T:
        """Convert a database row to a domain/entity object."""

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            return self._conn.execute(sql, params)
        except sqlite3.Error as e:
            raise PersistenceError(str(e)) from e

    def _fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self._execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self._execute(sql, params).fetchall()
