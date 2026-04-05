import numpy as np

from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    clear_waveform_cache,
    get_cached_waveform,
    set_waveform_cache_limit_bytes,
    waveform_cache_stats,
)
import echozero.ui.qt.timeline.waveform_cache as waveform_cache


def _cached(peaks_rows: int) -> CachedWaveform:
    peaks = np.zeros((peaks_rows, 2), dtype=np.float32)
    return CachedWaveform(sample_rate=44100, window_size=256, peaks=peaks)


def test_waveform_cache_evicts_oldest_when_over_budget():
    clear_waveform_cache()
    set_waveform_cache_limit_bytes(1500)

    waveform_cache._put_cached_waveform("a", _cached(120))
    waveform_cache._put_cached_waveform("b", _cached(120))
    waveform_cache._put_cached_waveform("c", _cached(120))

    stats = waveform_cache_stats()
    assert stats["entries"] <= 2
    assert stats["bytes"] <= stats["max_bytes"]
    assert get_cached_waveform("a") is None


def test_waveform_cache_access_refreshes_lru_position():
    clear_waveform_cache()
    set_waveform_cache_limit_bytes(2600)

    waveform_cache._put_cached_waveform("a", _cached(120))
    waveform_cache._put_cached_waveform("b", _cached(120))
    assert get_cached_waveform("a") is not None  # refresh a
    waveform_cache._put_cached_waveform("c", _cached(120))

    assert get_cached_waveform("a") is not None
    assert get_cached_waveform("b") is None
