import numpy as np

from echozero.ui.qt.timeline.waveform_cache import (
    CachedWaveform,
    clear_waveform_cache,
    get_cached_waveform,
    request_waveform_from_audio_file,
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


def test_request_waveform_from_audio_file_dedupes_background_requests(monkeypatch):
    clear_waveform_cache()
    waveform_cache._PENDING_LOADS.clear()
    submitted: list[object] = []
    queued_updates: list[object] = []

    class _Executor:
        def submit(self, fn):
            submitted.append(fn)
            return object()

    def _register(key: str, audio_file: str, *, window_size: int = 256) -> CachedWaveform:
        cached = _cached(8)
        waveform_cache._put_cached_waveform(key, cached)
        return cached

    monkeypatch.setattr(waveform_cache, "_LOAD_EXECUTOR", _Executor())
    monkeypatch.setattr(waveform_cache, "register_waveform_from_audio_file", _register)
    monkeypatch.setattr(
        waveform_cache,
        "_queue_receiver_update",
        lambda receiver: queued_updates.append(receiver),
    )

    receiver = object()
    assert request_waveform_from_audio_file("song", "/tmp/song.wav", receiver=receiver) is None
    assert request_waveform_from_audio_file("song", "/tmp/song.wav", receiver=receiver) is None
    assert len(submitted) == 1
    assert get_cached_waveform("song") is None

    submitted[0]()

    assert get_cached_waveform("song") is not None
    assert queued_updates == [receiver]
    assert waveform_cache._PENDING_LOADS == set()


def test_failed_waveform_load_is_backed_off(monkeypatch):
    clear_waveform_cache()
    waveform_cache._PENDING_LOADS.clear()
    waveform_cache._FAILED_LOADS.clear()
    submitted: list[object] = []
    queued_updates: list[object] = []

    class _Executor:
        def submit(self, fn):
            submitted.append(fn)
            return object()

    def _register(key: str, audio_file: str, *, window_size: int = 256) -> CachedWaveform:
        raise OSError("bad audio")

    monkeypatch.setattr(waveform_cache, "_LOAD_EXECUTOR", _Executor())
    monkeypatch.setattr(waveform_cache, "register_waveform_from_audio_file", _register)
    monkeypatch.setattr(
        waveform_cache,
        "_queue_receiver_update",
        lambda receiver: queued_updates.append(receiver),
    )

    receiver = object()
    assert request_waveform_from_audio_file("song", "/tmp/missing.wav", receiver=receiver) is None
    assert len(submitted) == 1

    submitted[0]()

    assert waveform_cache._PENDING_LOADS == set()
    assert len(waveform_cache._FAILED_LOADS) == 1
    assert queued_updates == [receiver]

    assert request_waveform_from_audio_file("song", "/tmp/missing.wav", receiver=receiver) is None
    assert len(submitted) == 1
