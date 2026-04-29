"""Runtime review bridge registry for live app-backed review writeback.
Exists because phone review may need to mutate canonical truth through an open app runtime.
Connects project-root keyed review writeback fallbacks to the live host that owns the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from echozero.foundry.domain.review import ReviewSession, ReviewSignal


class ReviewRuntimeBridge(Protocol):
    """Minimal callback surface for applying one review signal through a live runtime."""

    def apply_review_signal(
        self,
        session: ReviewSession,
        signal: ReviewSignal,
    ) -> dict[str, object] | None: ...


_BRIDGES: dict[Path, ReviewRuntimeBridge] = {}


def register_review_runtime_bridge(root: str | Path, bridge: ReviewRuntimeBridge) -> None:
    _BRIDGES[_normalize_root(root)] = bridge


def clear_review_runtime_bridge(root: str | Path) -> None:
    _BRIDGES.pop(_normalize_root(root), None)


def get_review_runtime_bridge(root: str | Path) -> ReviewRuntimeBridge | None:
    return _BRIDGES.get(_normalize_root(root))


def _normalize_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()
