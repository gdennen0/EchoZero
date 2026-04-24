"""Timeline viewport sizing and follow-scroll helpers.
Exists to keep timeline span and viewport math out of the Qt widget shell.
Connects presentation timing state to horizontal scroll and follow-mode behavior.
"""

from __future__ import annotations

from echozero.application.presentation.models import TimelinePresentation
from echozero.application.shared.enums import FollowMode
from echozero.ui.FEEL import LAYER_HEADER_WIDTH_PX, TIMELINE_RIGHT_PADDING_PX

_SpanLayerSignature = tuple[int, tuple[int, ...]]
_SpanSignature = tuple[int, tuple[_SpanLayerSignature, ...], str]

_SPAN_CACHE: dict[_SpanSignature, float] = {}
_MAX_SPAN_CACHE_ENTRIES = 24


def _span_signature(presentation: TimelinePresentation) -> _SpanSignature:
    layer_sig: list[_SpanLayerSignature] = []
    for layer in presentation.layers:
        take_sig = tuple(id(take.events) for take in layer.takes)
        layer_sig.append((id(layer.events), take_sig))
    return (id(presentation.layers), tuple(layer_sig), presentation.end_time_label)


def _parse_time_label_seconds(label: str | None) -> float:
    if not label:
        return 0.0
    text = label.strip()
    if not text:
        return 0.0
    try:
        if ":" in text:
            mins_txt, secs_txt = text.split(":", 1)
            return max(0.0, int(mins_txt) * 60 + float(secs_txt))
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return 0.0


def estimate_timeline_span_seconds(presentation: TimelinePresentation) -> float:
    """Best-effort duration estimate for viewport and scroll math."""

    key = _span_signature(presentation)
    cached = _SPAN_CACHE.get(key)
    if cached is not None:
        return cached

    span = max(0.0, presentation.playhead)
    for layer in presentation.layers:
        for event in layer.events:
            span = max(span, event.end)
        for take in layer.takes:
            for event in take.events:
                span = max(span, event.end)

    span = max(span, _parse_time_label_seconds(presentation.end_time_label))
    resolved = max(0.0, span)

    _SPAN_CACHE[key] = resolved
    if len(_SPAN_CACHE) > _MAX_SPAN_CACHE_ENTRIES:
        oldest = next(iter(_SPAN_CACHE.keys()))
        _SPAN_CACHE.pop(oldest, None)
    return resolved


def compute_scroll_bounds(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    right_padding_px: int = TIMELINE_RIGHT_PADDING_PX,
) -> tuple[int, int]:
    """Return `(content_width, max_scroll_x)` for horizontal timeline navigation."""

    viewport = max(1, int(viewport_width))
    span = estimate_timeline_span_seconds(presentation)
    content_width = max(
        viewport,
        int(header_width + (span * presentation.pixels_per_second) + right_padding_px),
    )
    max_scroll = max(0, content_width - viewport)
    return content_width, max_scroll


def compute_follow_scroll_x(
    presentation: TimelinePresentation,
    viewport_width: int,
    *,
    header_width: int = LAYER_HEADER_WIDTH_PX,
    content_padding_px: int = 24,
) -> float:
    """Compute the follow-mode adjusted scroll target for the current playhead."""

    if presentation.follow_mode == FollowMode.OFF or not presentation.is_playing:
        return presentation.scroll_x

    viewport = max(1, int(viewport_width))
    content_width = max(1.0, viewport - header_width)
    pps = max(1.0, presentation.pixels_per_second)
    timeline_x = presentation.playhead * pps
    current = presentation.scroll_x
    left_bound = current + content_padding_px
    right_bound = current + max(content_padding_px + 1.0, content_width - content_padding_px)

    target = current
    if presentation.follow_mode == FollowMode.PAGE:
        if timeline_x < left_bound:
            target = max(0.0, timeline_x - content_padding_px)
        elif timeline_x > right_bound:
            target = max(0.0, timeline_x - content_padding_px)
    elif presentation.follow_mode == FollowMode.CENTER:
        target = max(0.0, timeline_x - (content_width * 0.5))
    elif presentation.follow_mode == FollowMode.SMOOTH:
        target = max(0.0, timeline_x - (content_width * 0.75))

    _, max_scroll = compute_scroll_bounds(presentation, viewport, header_width=header_width)
    return float(max(0.0, min(target, max_scroll)))


__all__ = [
    "compute_follow_scroll_x",
    "compute_scroll_bounds",
    "estimate_timeline_span_seconds",
]
