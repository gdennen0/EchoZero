from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

from echozero.application.presentation.models import EventPresentation
from echozero.application.shared.enums import LayerKind
from echozero.application.shared.ids import EventId
from echozero.ui.FEEL import EVENT_MIN_HIT_WIDTH_PX, EVENT_MIN_VISIBLE_WIDTH_PX
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock, EventLanePresentation
from echozero.ui.qt.timeline.style import TIMELINE_STYLE
from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    clear_waveform_cache,
    _put_cached_waveform,
)

import numpy as np


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
        take_id=None,
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


def test_event_lane_zooms_out_tiny_events_with_clickable_hit_width():
    tiny = _event(1, 1.0, 1.01)
    presentation = EventLanePresentation(
        layer_id="layer_1",
        take_id=None,
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
    assert rects[0][0].width() >= float(EVENT_MIN_HIT_WIDTH_PX)
    assert image.pixelColor(421, 31).alpha() > 0


def test_event_lane_uses_event_audio_waveform_duration_for_visual_width():
    app = QApplication.instance() or QApplication([])
    clear_waveform_cache()
    _put_cached_waveform(
        "event-hit",
        CachedWaveform(
            sample_rate=10,
            window_size=1,
            peaks=np.array([[-0.4, 0.6]] * 5, dtype=np.float32),
            duration_seconds=0.5,
        ),
    )
    tiny = _event(1, 1.0, 1.01)
    presentation = EventLanePresentation(
        layer_id="layer_1",
        take_id=None,
        events=[tiny],
        pixels_per_second=100.0,
        scroll_x=0.0,
        header_width=320,
        event_height=22,
        viewport_width=800,
        waveform_key="event-hit",
        render_audio_shape=True,
    )

    image = QImage(800, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        rects = EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()
        clear_waveform_cache()
        app.processEvents()

    assert len(rects) == 1
    assert rects[0][0].width() >= 50.0
    assert image.pixelColor(465, 31).alpha() > 0


def test_event_lane_hides_labels_when_event_is_too_narrow():
    tiny = _event(1, 1.0, 1.01)
    presentation = EventLanePresentation(
        layer_id="layer_1",
        take_id=None,
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
    assert rects[0][0].width() >= float(EVENT_MIN_VISIBLE_WIDTH_PX)


def test_event_lane_uses_presented_default_fill_when_event_has_no_color():
    event = EventPresentation(
        event_id=EventId("unstyled"),
        start=1.0,
        end=1.3,
        label="Unstyled",
        color=None,
    )
    presentation = EventLanePresentation(
        layer_id="layer_1",
        take_id=None,
        events=[event],
        pixels_per_second=100.0,
        scroll_x=0.0,
        header_width=320,
        event_height=22,
        viewport_width=800,
        default_fill_hex="#123456",
    )

    image = QImage(800, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()

    sample = image.pixelColor(430, 31)
    assert sample.red() == 0x12
    assert sample.green() == 0x34
    assert sample.blue() == 0x56


def test_event_lane_renders_demoted_badge_with_gray_fill():
    event = EventPresentation(
        event_id=EventId("demoted"),
        start=1.0,
        end=1.3,
        label="Demoted",
        color="#22cc88",
        badges=["demoted"],
    )
    presentation = EventLanePresentation(
        layer_id="layer_1",
        take_id=None,
        events=[event],
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
        EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()

    sample = image.pixelColor(430, 31)
    expected = TIMELINE_STYLE.event_lane.demoted_fill_hex
    assert sample.name().lower() == expected.lower()


def test_selected_tiny_demoted_event_has_stronger_outer_outline():
    unselected = EventPresentation(
        event_id=EventId("demoted_unselected"),
        start=1.0,
        end=1.01,
        label="Demoted",
        color="#22cc88",
        badges=["demoted"],
        is_selected=False,
    )
    selected = EventPresentation(
        event_id=EventId("demoted_selected"),
        start=1.0,
        end=1.01,
        label="Demoted",
        color="#22cc88",
        badges=["demoted"],
        is_selected=True,
    )

    def _paint_single(event: EventPresentation) -> QImage:
        presentation = EventLanePresentation(
            layer_id="layer_1",
            take_id=None,
            events=[event],
            pixels_per_second=20.0,
            scroll_x=0.0,
            header_width=320,
            event_height=22,
            viewport_width=800,
        )
        image = QImage(800, 120, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        try:
            EventLaneBlock().paint(painter, 20, presentation)
        finally:
            painter.end()
        return image

    unselected_image = _paint_single(unselected)
    selected_image = _paint_single(selected)

    # Sample just outside the tiny event body where the zoomed-out selection outline should appear.
    unselected_sample = unselected_image.pixelColor(339, 31)
    selected_sample = selected_image.pixelColor(339, 31)
    assert selected_sample.alpha() > unselected_sample.alpha() + 20
    assert selected_sample.blue() > unselected_sample.blue() + 20


def test_section_lane_renders_zero_width_section_cues_as_minimum_event_width():
    section = EventPresentation(
        event_id=EventId("section"),
        start=1.0,
        end=1.01,
        label="Verse",
        cue_ref="Q7A",
        color="#f0b74f",
    )
    presentation = EventLanePresentation(
        layer_id="layer_section",
        take_id=None,
        events=[section],
        pixels_per_second=100.0,
        scroll_x=0.0,
        header_width=320,
        layer_kind=LayerKind.SECTION,
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
    assert rects[0][0].width() >= float(EVENT_MIN_VISIBLE_WIDTH_PX)
    sample = image.pixelColor(421, 28)
    assert sample.red() > 0


def test_section_lane_can_expand_hit_width_without_changing_rendered_minimum_width():
    section = EventPresentation(
        event_id=EventId("section_hit"),
        start=1.0,
        end=1.01,
        label="Verse",
        cue_ref="Q7B",
        color="#f0b74f",
    )
    presentation = EventLanePresentation(
        layer_id="layer_section",
        take_id=None,
        events=[section],
        pixels_per_second=100.0,
        scroll_x=0.0,
        header_width=320,
        layer_kind=LayerKind.SECTION,
        event_height=22,
        viewport_width=800,
        event_hit_min_width_px=12.0,
    )

    image = QImage(800, 120, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        rects = EventLaneBlock().paint(painter, 20, presentation)
    finally:
        painter.end()

    assert len(rects) == 1
    assert rects[0][0].width() >= 12.0
    assert rects[0][0].left() >= 320.0
