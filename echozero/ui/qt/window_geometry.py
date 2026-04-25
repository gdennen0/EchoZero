"""Qt window geometry helpers for the canonical shell surface.
Exists to keep startup sizing screen-aware across host displays.
Connects launcher construction to deterministic, bounded initial dimensions.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtGui import QCursor, QGuiApplication

DEFAULT_INITIAL_WINDOW_WIDTH = 1440
DEFAULT_INITIAL_WINDOW_HEIGHT = 720
INITIAL_WINDOW_WIDTH_RATIO = 0.75
INITIAL_WINDOW_HEIGHT_RATIO = 0.75
INITIAL_WINDOW_EDGE_MARGIN_PX = 16


def compute_initial_window_size_for_screen(
    screen_width: int,
    screen_height: int,
) -> tuple[int, int]:
    """Compute startup window dimensions from available screen space."""

    if screen_width <= 0 or screen_height <= 0:
        return (
            DEFAULT_INITIAL_WINDOW_WIDTH,
            DEFAULT_INITIAL_WINDOW_HEIGHT,
        )

    target_width = int(screen_width * INITIAL_WINDOW_WIDTH_RATIO)
    target_height = int(screen_height * INITIAL_WINDOW_HEIGHT_RATIO)
    max_initial_width = max(1, int(screen_width) - (INITIAL_WINDOW_EDGE_MARGIN_PX * 2))
    max_initial_height = max(1, int(screen_height) - (INITIAL_WINDOW_EDGE_MARGIN_PX * 2))
    return (
        max(1, min(target_width, screen_width, max_initial_width)),
        max(1, min(target_height, screen_height, max_initial_height)),
    )


def resolve_initial_window_size(widget: Any) -> tuple[int, int]:
    """Resolve startup window dimensions for a widget's host display."""

    screen = resolve_target_screen(widget)
    if screen is None:
        return (
            DEFAULT_INITIAL_WINDOW_WIDTH,
            DEFAULT_INITIAL_WINDOW_HEIGHT,
        )
    available_geometry = _screen_available_geometry(screen)
    if available_geometry is None:
        return (
            DEFAULT_INITIAL_WINDOW_WIDTH,
            DEFAULT_INITIAL_WINDOW_HEIGHT,
        )
    return compute_initial_window_size_for_screen(
        available_geometry.width(),
        available_geometry.height(),
    )


def fit_window_to_available_screen(widget: Any) -> None:
    """Clamp a shown top-level window so its frame fits available screen bounds."""

    screen = resolve_target_screen(widget)
    if screen is None:
        return
    available_geometry = _screen_available_geometry(screen)
    if available_geometry is None:
        return
    width_getter = getattr(widget, "width", None)
    height_getter = getattr(widget, "height", None)
    if not callable(width_getter) or not callable(height_getter):
        return
    frame_geometry = _widget_frame_geometry(widget)
    widget_width = int(width_getter())
    widget_height = int(height_getter())
    frame_width = int(_geometry_dimension(frame_geometry, "width", widget_width))
    frame_height = int(_geometry_dimension(frame_geometry, "height", widget_height))
    frame_overhead_width = max(0, frame_width - widget_width)
    frame_overhead_height = max(0, frame_height - widget_height)
    available_width = max(
        1,
        int(available_geometry.width()) - (INITIAL_WINDOW_EDGE_MARGIN_PX * 2),
    )
    available_height = max(
        1,
        int(available_geometry.height()) - (INITIAL_WINDOW_EDGE_MARGIN_PX * 2),
    )
    max_client_width = max(1, available_width - frame_overhead_width)
    max_client_height = max(1, available_height - frame_overhead_height)
    target_width = min(widget_width, max_client_width)
    target_height = min(widget_height, max_client_height)
    if target_width != widget_width or target_height != widget_height:
        resize = getattr(widget, "resize", None)
        if callable(resize):
            resize(target_width, target_height)

    frame_geometry = _widget_frame_geometry(widget)
    if frame_geometry is None:
        return
    left = int(_geometry_dimension(available_geometry, "left", 0)) + INITIAL_WINDOW_EDGE_MARGIN_PX
    top = int(_geometry_dimension(available_geometry, "top", 0)) + INITIAL_WINDOW_EDGE_MARGIN_PX
    right = int(_geometry_dimension(available_geometry, "right", left)) - INITIAL_WINDOW_EDGE_MARGIN_PX
    bottom = int(_geometry_dimension(available_geometry, "bottom", top)) - INITIAL_WINDOW_EDGE_MARGIN_PX
    frame_width = int(_geometry_dimension(frame_geometry, "width", target_width))
    frame_height = int(_geometry_dimension(frame_geometry, "height", target_height))
    x = int(_geometry_dimension(frame_geometry, "x", left))
    y = int(_geometry_dimension(frame_geometry, "y", top))
    max_x = max(left, right - frame_width + 1)
    max_y = max(top, bottom - frame_height + 1)
    clamped_x = min(max(x, left), max_x)
    clamped_y = min(max(y, top), max_y)
    if clamped_x != x or clamped_y != y:
        move = getattr(widget, "move", None)
        if callable(move):
            move(clamped_x, clamped_y)


def resolve_target_screen(widget: Any) -> Any | None:
    """Resolve the most likely host screen for startup geometry decisions."""

    return _widget_screen(widget) or _screen_at_cursor() or _primary_screen()


def _widget_screen(widget: Any) -> Any | None:
    screen_getter = getattr(widget, "screen", None)
    if callable(screen_getter):
        return screen_getter()
    return None


def _screen_at_cursor() -> Any | None:
    return QGuiApplication.screenAt(QCursor.pos())


def _primary_screen() -> Any | None:
    return QGuiApplication.primaryScreen()


def _screen_available_geometry(screen: Any) -> Any | None:
    geometry_getter = getattr(screen, "availableGeometry", None)
    if callable(geometry_getter):
        return geometry_getter()
    return None


def _widget_frame_geometry(widget: Any) -> Any | None:
    frame_geometry_getter = getattr(widget, "frameGeometry", None)
    if callable(frame_geometry_getter):
        return frame_geometry_getter()
    return None


def _geometry_dimension(geometry: Any | None, name: str, fallback: int) -> int:
    if geometry is None:
        return fallback
    accessor = getattr(geometry, name, None)
    if callable(accessor):
        return int(accessor())
    return fallback
