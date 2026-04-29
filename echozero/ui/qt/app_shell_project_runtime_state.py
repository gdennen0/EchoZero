"""Project runtime state persistence for the Qt app shell.
Exists to preserve active song context and timeline viewport/playhead across save/open cycles.
Connects app-shell presentation state to project-backed metadata in ProjectStorage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.ids import SongId, SongVersionId
from echozero.persistence.session import ProjectStorage

_RUNTIME_STATE_META_KEY = "app_shell_runtime_state.v1"


@dataclass(frozen=True, slots=True)
class ProjectRuntimeState:
    """Persisted app-shell runtime context loaded from project metadata."""

    active_song_id: SongId | None = None
    active_song_version_id: SongVersionId | None = None
    playhead: float = 0.0
    pixels_per_second: float = 100.0
    scroll_x: float = 0.0
    scroll_y: float = 0.0


def load_project_runtime_state(project_storage: ProjectStorage) -> ProjectRuntimeState:
    """Load one previously saved runtime-state snapshot from project metadata."""

    with project_storage.locked():
        row = project_storage.db.execute(
            "SELECT value FROM _meta WHERE key = ?",
            (_RUNTIME_STATE_META_KEY,),
        ).fetchone()
    if row is None:
        return ProjectRuntimeState()
    try:
        payload = json.loads(row["value"])
    except (TypeError, ValueError, KeyError):
        return ProjectRuntimeState()
    if not isinstance(payload, dict):
        return ProjectRuntimeState()
    return _state_from_payload(payload)


def persist_project_runtime_state(
    project_storage: ProjectStorage,
    *,
    presentation: TimelinePresentation,
    playhead: float | None = None,
) -> None:
    """Persist app-shell runtime context as project metadata."""

    payload = _payload_from_presentation(presentation, playhead=playhead)
    encoded_payload = json.dumps(payload, separators=(",", ":"))
    with project_storage.locked():
        project_storage.db.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            (_RUNTIME_STATE_META_KEY, encoded_payload),
        )


def _payload_from_presentation(
    presentation: TimelinePresentation,
    *,
    playhead: float | None,
) -> dict[str, Any]:
    active_song_id = (presentation.active_song_id or "").strip()
    active_song_version_id = (presentation.active_song_version_id or "").strip()
    resolved_playhead = (
        _non_negative_float(playhead)
        if playhead is not None
        else _non_negative_float(presentation.playhead)
    )
    return {
        "active_song_id": active_song_id or None,
        "active_song_version_id": active_song_version_id or None,
        "playhead": resolved_playhead,
        "pixels_per_second": _positive_float(presentation.pixels_per_second, fallback=100.0),
        "scroll_x": _non_negative_float(presentation.scroll_x),
        "scroll_y": _non_negative_float(presentation.scroll_y),
    }


def _state_from_payload(payload: dict[str, Any]) -> ProjectRuntimeState:
    active_song_id = _optional_text(payload.get("active_song_id"))
    active_song_version_id = _optional_text(payload.get("active_song_version_id"))
    return ProjectRuntimeState(
        active_song_id=SongId(active_song_id) if active_song_id is not None else None,
        active_song_version_id=(
            SongVersionId(active_song_version_id)
            if active_song_version_id is not None
            else None
        ),
        playhead=_non_negative_float(payload.get("playhead")),
        pixels_per_second=_positive_float(payload.get("pixels_per_second"), fallback=100.0),
        scroll_x=_non_negative_float(payload.get("scroll_x")),
        scroll_y=_non_negative_float(payload.get("scroll_y")),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _non_negative_float(value: object) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    if resolved < 0.0:
        return 0.0
    return resolved


def _positive_float(value: object, *, fallback: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return fallback
    if resolved <= 0.0:
        return fallback
    return resolved


__all__ = [
    "ProjectRuntimeState",
    "load_project_runtime_state",
    "persist_project_runtime_state",
]
