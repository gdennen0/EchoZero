from types import SimpleNamespace

from echozero.ui.qt.timeline.note_contour_overlay import (
    build_note_contour_path,
    contour_samples_from_events,
    note_contour_y_for_midi,
)


def test_contour_samples_from_events_extracts_midi_metadata():
    events = [
        SimpleNamespace(
            start=0.0,
            end=0.4,
            label="C2",
            detection_metadata={"midi_note": 36},
        ),
        SimpleNamespace(
            start=0.4,
            end=0.9,
            label="G2",
            detection_metadata={"midi_note": 43},
        ),
    ]

    samples = contour_samples_from_events(events)

    assert [sample.midi_note for sample in samples] == [36, 43]
    assert samples[0].center_time == 0.2
    assert samples[1].center_time == 0.65


def test_note_contour_y_for_midi_maps_higher_notes_upward():
    low = note_contour_y_for_midi(36, min_midi=36, max_midi=48, top=10.0, row_height=60.0)
    high = note_contour_y_for_midi(48, min_midi=36, max_midi=48, top=10.0, row_height=60.0)

    assert high < low


def test_build_note_contour_path_returns_smooth_path_for_multiple_samples():
    samples = contour_samples_from_events(
        [
            SimpleNamespace(
                start=0.0,
                end=0.2,
                label="C2",
                detection_metadata={"midi_note": 36},
            ),
            SimpleNamespace(
                start=0.2,
                end=0.5,
                label="E2",
                detection_metadata={"midi_note": 40},
            ),
            SimpleNamespace(
                start=0.5,
                end=0.8,
                label="G2",
                detection_metadata={"midi_note": 43},
            ),
        ]
    )

    path = build_note_contour_path(
        samples,
        scroll_x=0.0,
        pixels_per_second=100.0,
        content_start_x=320.0,
        top=0.0,
        row_height=60.0,
    )

    assert path is not None
    assert path.elementCount() >= 4
