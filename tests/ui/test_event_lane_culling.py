from PyQt6.QtGui import QImage, QPainter

from echozero.application.presentation.models import EventPresentation
from echozero.application.shared.ids import EventId
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation


def _event(idx: int, start: float, end: float) -> EventPresentation:
    return EventPresentation(
        event_id=EventId(f"e{idx}"),
        start=start,
        end=end,
        label=f"E{idx}",
        color="#66a3ff",
    )


def test_event_lane_culls_offscreen_events_by_time_window():
    events = [_event(i, float(i), float(i) + 0.1) for i in range(0, 200)]
    presentation = EventLanePresentation(
        layer_id="layer_1",
        events=events,
        pixels_per_second=100.0,
        scroll_x=5000.0,  # around event ~50s
        header_width=320,
        event_height=22,
        viewport_width=1320,
    )

    image = QImage(1320, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        rects = EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()

    assert rects
    assert len(rects) < 30
    starts = [float(r[0].left()) for r in rects]
    assert min(starts) >= 300


def test_event_lane_hides_labels_when_event_is_too_narrow():
    tiny = _event(1, 1.0, 1.01)
    presentation = EventLanePresentation(
        layer_id="layer_1",
        events=[tiny],
        pixels_per_second=100.0,
        scroll_x=0.0,
        header_width=320,
        event_height=22,
        viewport_width=800,
    )

    image = QImage(800, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        rects = EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()

    assert len(rects) == 1
    assert rects[0][0].width() >= 2.0
