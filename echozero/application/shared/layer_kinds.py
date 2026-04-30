"""Layer-kind capability helpers for timeline and sync workflows.
Exists to centralize event-lane capability checks across app, sync, and UI paths.
Connects LayerKind enums to first-principles behavior gates (edit, transfer, playback, persistence).
"""

from __future__ import annotations

from echozero.application.shared.enums import LayerKind

EVENT_LIKE_LAYER_KINDS = frozenset({LayerKind.EVENT, LayerKind.SECTION})


def is_event_like_layer_kind(kind: LayerKind | str | None) -> bool:
    """Return true when a layer kind participates in event-lane semantics."""

    if isinstance(kind, LayerKind):
        return kind in EVENT_LIKE_LAYER_KINDS
    try:
        return LayerKind(str(kind or "").strip().lower()) in EVENT_LIKE_LAYER_KINDS
    except ValueError:
        return False
