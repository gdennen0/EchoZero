"""ReviewSignalRepository persists canonical explicit review records.
Exists because review queues track work state, while review signals must stay durable.
Connects Foundry review services to one explicit-review state file per project root.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from echozero.foundry.domain.review import ReviewOutcome, ReviewPolarity, ReviewSignal

from .repositories import _read_state, _write_state
from .review_repository import deserialize_review_decision_state, serialize_review_decision_state


class ReviewSignalRepository:
    """Stores and loads durable review signals from the Foundry state directory."""

    def __init__(self, root: Path):
        self._path = root / "foundry" / "state" / "review_signals.json"
        self._schema = "foundry.state.review_signals.v1"

    def save(self, signal: ReviewSignal) -> ReviewSignal:
        """Persist one durable review signal and return the saved record."""
        rows = _read_state(self._path, self._schema)
        rows[signal.id] = {
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
        _write_state(self._path, self._schema, rows)
        return signal

    def get(self, signal_id: str) -> ReviewSignal | None:
        """Load one durable review signal by id when it exists."""
        row = _read_state(self._path, self._schema).get(signal_id)
        if row is None:
            return None
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

    def list(self) -> list[ReviewSignal]:
        """Return all persisted review signals."""
        signals: list[ReviewSignal] = []
        for signal_id in _read_state(self._path, self._schema).keys():
            signal = self.get(signal_id)
            if signal is not None:
                signals.append(signal)
        return signals
