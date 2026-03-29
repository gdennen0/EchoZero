"""
Staleness tracking: Human-readable reasons for why blocks are stale (D280).
Exists because users need to understand the blast radius of a settings change —
not just "stale" but WHY stale and WHAT changed.
Used by the Coordinator to accumulate reasons and by the UI to display them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StaleReason:
    """A single reason why a block became stale.

    Human-readable and specific: "Block 'Separator' setting 'model' changed
    from 'htdemucs' to 'htdemucs_ft'" — not just "settings changed."
    """

    source_block_id: str
    source_block_name: str
    description: str

    def __str__(self) -> str:
        return self.description


def setting_changed_reason(
    block_id: str,
    block_name: str,
    setting_key: str,
    old_value: Any = None,
    new_value: Any = None,
) -> StaleReason:
    """Create a StaleReason for a settings change."""
    if old_value is not None and new_value is not None:
        desc = (
            f"Block '{block_name}' setting '{setting_key}' "
            f"changed from {old_value!r} to {new_value!r}"
        )
    else:
        desc = f"Block '{block_name}' setting '{setting_key}' changed"
    return StaleReason(
        source_block_id=block_id,
        source_block_name=block_name,
        description=desc,
    )


def connection_changed_reason(
    block_id: str,
    block_name: str,
    action: str,
) -> StaleReason:
    """Create a StaleReason for a connection change (added or removed)."""
    return StaleReason(
        source_block_id=block_id,
        source_block_name=block_name,
        description=f"Connection {action} on block '{block_name}'",
    )


class StaleTracker:
    """Accumulates stale reasons per block. Reasons compound until cleared.

    Multiple changes before re-run = multiple reasons. Re-run clears all reasons.
    Thread-safe for read — mutations happen on the coordinator's thread.
    """

    def __init__(self) -> None:
        self._reasons: dict[str, list[StaleReason]] = {}

    def add_reason(self, block_id: str, reason: StaleReason) -> None:
        """Add a stale reason for a block. Reasons accumulate."""
        if block_id not in self._reasons:
            self._reasons[block_id] = []
        # Avoid duplicate identical reasons
        if not any(r.description == reason.description for r in self._reasons[block_id]):
            self._reasons[block_id].append(reason)

    def add_reason_to_downstream(
        self, block_ids: set[str], reason: StaleReason
    ) -> None:
        """Add the same reason to multiple blocks (downstream cascade)."""
        for block_id in block_ids:
            self.add_reason(block_id, reason)

    def get_reasons(self, block_id: str) -> tuple[StaleReason, ...]:
        """Get all stale reasons for a block. Empty tuple if fresh."""
        return tuple(self._reasons.get(block_id, []))

    def get_all_stale(self) -> dict[str, tuple[StaleReason, ...]]:
        """Get all blocks with stale reasons. For the global "N stale" badge."""
        return {bid: tuple(reasons) for bid, reasons in self._reasons.items() if reasons}

    def stale_count(self) -> int:
        """Number of blocks that are stale."""
        return len(self._reasons)

    def clear(self, block_id: str) -> None:
        """Clear all reasons for a block (called after successful re-run)."""
        self._reasons.pop(block_id, None)

    def clear_all(self) -> None:
        """Clear all stale reasons (e.g., after full re-run)."""
        self._reasons.clear()

    def is_stale(self, block_id: str) -> bool:
        """Check if a block has any stale reasons."""
        return bool(self._reasons.get(block_id))

    def summary(self, block_id: str) -> str | None:
        """One-line summary for UI hover tooltip. None if not stale."""
        reasons = self._reasons.get(block_id)
        if not reasons:
            return None
        if len(reasons) == 1:
            return str(reasons[0])
        return f"{len(reasons)} changes: " + "; ".join(str(r) for r in reasons)
