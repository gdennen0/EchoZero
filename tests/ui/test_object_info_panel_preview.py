"""Object-info preview rendering tests.
Exists to keep compact waveform preview coverage separate from the wider timeline shell cases.
Connects inspector preview resampling to a stable short-clip regression proof.
"""

import numpy as np
import pytest

from echozero.ui.qt.timeline.object_info_panel_preview import clip_waveform_columns
from echozero.ui.qt.timeline.waveform_cache import CachedWaveform


def test_clip_waveform_columns_resamples_short_clip_across_preview_width():
    cached = CachedWaveform(
        sample_rate=4,
        window_size=1,
        peaks=np.array(
            [
                [-0.1, 0.1],
                [-0.8, 0.6],
                [-0.3, 0.2],
                [-0.5, 0.9],
            ],
            dtype=np.float32,
        ),
    )

    columns = clip_waveform_columns(
        cached,
        start_seconds=0.25,
        end_seconds=0.50,
        column_count=8,
    )

    assert len(columns) == 8
    for vmin, vmax in columns:
        assert vmin == pytest.approx(-0.8)
        assert vmax == pytest.approx(0.6)


def test_clip_waveform_columns_blends_multiple_peak_windows_per_column():
    cached = CachedWaveform(
        sample_rate=4,
        window_size=1,
        peaks=np.array(
            [
                [-0.2, 0.1],
                [-0.8, 0.4],
                [-0.3, 0.9],
                [-0.1, 0.2],
            ],
            dtype=np.float32,
        ),
    )

    columns = clip_waveform_columns(
        cached,
        start_seconds=0.0,
        end_seconds=1.0,
        column_count=2,
    )

    assert columns[0][0] == pytest.approx(-0.8)
    assert columns[0][1] == pytest.approx(0.4)
    assert columns[1][0] == pytest.approx(-0.3)
    assert columns[1][1] == pytest.approx(0.9)
