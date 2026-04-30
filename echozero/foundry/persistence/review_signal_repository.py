"""ReviewSignalRepository persists canonical explicit review records.
Exists because review queues track work state, while review signals must stay durable.
Connects Foundry review services to one explicit-review state file per project root.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from echozero.foundry.domain.review import ReviewOutcome, ReviewPolarity, ReviewSignal

from .repositories import _read_state, _write_state
from .review_repository import deserialize_review_decision_state, serialize_review_decision_state


class ReviewSignalRepository:
    """Stores and loads durable review signals from the Foundry state directory."""

    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "review_signals.json"
        self._journal_path = root / "foundry" / "state" / "review_signals.journal.jsonl"
        self._schema = "foundry.state.review_signals.v1"
        self._cached_rows: dict[str, dict[str, Any]] | None = None
        self._cached_signals: dict[str, ReviewSignal] | None = None
        self._cached_session_index: dict[str, list[str]] = {}
        self._cached_snapshot_token: str | None = None
        self._cached_journal_token: str | None = None

    def save(self, signal: ReviewSignal) -> ReviewSignal:
        """Persist one durable review signal and return the saved record."""
        self._ensure_cache_loaded()
        row = self._serialize(signal)
        assert self._cached_rows is not None
        self._cached_rows[signal.id] = row
        if self._cached_signals is not None:
            self._cached_signals[signal.id] = signal
        session_key = str(signal.session_id).strip()
        if session_key:
            session_ids = self._cached_session_index.setdefault(session_key, [])
            if signal.id not in session_ids:
                session_ids.append(signal.id)
                session_ids.sort(
                    key=lambda candidate_id: str(
                        self._cached_rows.get(candidate_id, {}).get("updated_at")
                        or self._cached_rows.get(candidate_id, {}).get("created_at")
                        or ""
                    )
                )
        self._append_journal_row(row)
        self._maybe_compact_state()
        return signal

    def get(self, signal_id: str) -> ReviewSignal | None:
        """Load one durable review signal by id when it exists."""
        self._ensure_cache_loaded()
        assert self._cached_signals is not None
        return self._cached_signals.get(signal_id)

    def list(self) -> list[ReviewSignal]:
        """Return all persisted review signals."""
        self._ensure_cache_loaded()
        assert self._cached_signals is not None
        return list(self._cached_signals.values())

    def list_for_session(self, session_id: str) -> list[ReviewSignal]:
        """Return all durable review signals for one session id."""
        self._ensure_cache_loaded()
        assert self._cached_signals is not None
        normalized_session_id = str(session_id).strip()
        signal_ids = self._cached_session_index.get(normalized_session_id, [])
        return [
            self._cached_signals[signal_id]
            for signal_id in signal_ids
            if signal_id in self._cached_signals
        ]

    def _ensure_cache_loaded(self) -> None:
        snapshot_token = _state_token(self._path)
        journal_token = _state_token(self._journal_path)
        if (
            self._cached_rows is not None
            and snapshot_token == self._cached_snapshot_token
            and journal_token == self._cached_journal_token
        ):
            return
        rows: dict[str, dict[str, Any]] = {
            str(key): dict(value)
            for key, value in _read_state(self._path, self._schema).items()
            if isinstance(value, dict)
        }
        if self._journal_path.exists():
            for raw_line in self._journal_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip():
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                signal_id = str(entry.get("id", "")).strip()
                row = entry.get("row")
                if not signal_id or not isinstance(row, dict):
                    continue
                rows[signal_id] = dict(row)
        session_index: dict[str, list[str]] = {}
        for signal_id, row in rows.items():
            session_key = str(row.get("session_id", "")).strip()
            if not session_key:
                continue
            session_ids = session_index.setdefault(session_key, [])
            session_ids.append(signal_id)
        for session_key, signal_ids in session_index.items():
            signal_ids.sort(
                key=lambda signal_id: str(
                    rows.get(signal_id, {}).get("updated_at")
                    or rows.get(signal_id, {}).get("created_at")
                    or ""
                )
            )
            session_index[session_key] = signal_ids
        signals = {
            signal_id: self._deserialize(row)
            for signal_id, row in rows.items()
        }
        self._cached_rows = rows
        self._cached_signals = signals
        self._cached_session_index = session_index
        self._cached_snapshot_token = snapshot_token
        self._cached_journal_token = journal_token

    def _append_journal_row(self, row: dict[str, Any]) -> None:
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"op": "upsert", "id": row["id"], "row": row}
        with self._journal_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")
        self._cached_journal_token = _state_token(self._journal_path)

    def _maybe_compact_state(self) -> None:
        if not self._journal_path.exists():
            return
        try:
            journal_size = self._journal_path.stat().st_size
        except OSError:
            return
        if journal_size < 256 * 1024:
            return
        assert self._cached_rows is not None
        _write_state(self._path, self._schema, self._cached_rows)
        self._journal_path.write_text("", encoding="utf-8")
        self._cached_snapshot_token = _state_token(self._path)
        self._cached_journal_token = _state_token(self._journal_path)

    @staticmethod
    def _serialize(signal: ReviewSignal) -> dict[str, Any]:
        return {
            "id": signal.id,
            "session_id": signal.session_id,
            "item_id": signal.item_id,
            "audio_path": signal.audio_path,
            "predicted_label": signal.predicted_label,
            "target_class": signal.target_class,
            "polarity": signal.polarity.value,
            "score": signal.score,
            "source_provenance": signal.source_provenance,
            "review_outcome": signal.review_outcome.value,
            "review_decision": serialize_review_decision_state(signal.review_decision),
            "corrected_label": signal.corrected_label,
            "review_note": signal.review_note,
            "reviewed_at": signal.reviewed_at.isoformat() if signal.reviewed_at else None,
            "created_at": signal.created_at.isoformat(),
            "updated_at": signal.updated_at.isoformat(),
        }

    @staticmethod
    def _deserialize(row: dict) -> ReviewSignal:
        return ReviewSignal(
            id=row["id"],
            session_id=row["session_id"],
            item_id=row["item_id"],
            audio_path=row["audio_path"],
            predicted_label=row["predicted_label"],
            target_class=row["target_class"],
            polarity=ReviewPolarity(row["polarity"]),
            score=row.get("score"),
            source_provenance=row.get("source_provenance", {}),
            review_outcome=ReviewOutcome(row["review_outcome"]),
            review_decision=deserialize_review_decision_state(
                row.get("review_decision"),
                outcome=ReviewOutcome(row["review_outcome"]),
                corrected_label=row.get("corrected_label"),
                review_note=row.get("review_note"),
                source_provenance=row.get("source_provenance", {}),
                queue_session_ref=row["session_id"],
            ),
            corrected_label=row.get("corrected_label"),
            review_note=row.get("review_note"),
            reviewed_at=(
                datetime.fromisoformat(row["reviewed_at"])
                if row.get("reviewed_at")
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


def _state_token(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
