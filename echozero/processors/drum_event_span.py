"""
Drum event span estimation for classified onset candidates.
Exists because review and training exports need real sample regions, not onset marker fallbacks.
Used by drum classifiers before timeline storage projects domain events into app events.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

MIN_DRUM_EVENT_DURATION_SECONDS = 0.03
MAX_DRUM_EVENT_DURATION_SECONDS = 0.50
DRUM_TAIL_FRAME_SECONDS = 0.005
DRUM_TAIL_HOP_SECONDS = 0.0025
DRUM_TAIL_HOLD_SECONDS = 0.02
DRUM_TAIL_RELATIVE_THRESHOLD = 0.08
DRUM_TAIL_PEAK_RELATIVE_THRESHOLD = 0.05
DRUM_TAIL_CUMULATIVE_ENERGY_RATIO = 0.995
DRUM_TAIL_ABSOLUTE_FLOOR = 1e-4
DRUM_TAIL_METHOD_AGREEMENT_SECONDS = 0.06


@dataclass(frozen=True, slots=True)
class DrumEventSpanEstimate:
    """Synthesized drum span estimate and the method votes that produced it."""

    duration_seconds: float
    method_durations: dict[str, float]
    agreement_seconds: float
    consensus_method: str


def estimate_drum_event_duration(
    *,
    audio: np.ndarray,
    onset_seconds: float,
    sample_rate: int,
) -> float:
    """Estimate a bounded drum sample span from audio energy after an onset."""

    return estimate_drum_event_span(
        audio=audio,
        onset_seconds=onset_seconds,
        sample_rate=sample_rate,
    ).duration_seconds


def estimate_drum_event_span(
    *,
    audio: np.ndarray,
    onset_seconds: float,
    sample_rate: int,
) -> DrumEventSpanEstimate:
    """Estimate a drum sample span by synthesizing multiple tail-end methods."""

    if sample_rate <= 0 or audio.size == 0:
        return _empty_span_estimate()

    mono_audio = _mono_float_audio(audio)
    start_index = max(0, min(mono_audio.size, int(round(float(onset_seconds) * sample_rate))))
    max_samples = max(1, int(round(MAX_DRUM_EVENT_DURATION_SECONDS * sample_rate)))
    segment = mono_audio[start_index : start_index + max_samples]
    if segment.size == 0:
        return _empty_span_estimate()

    envelope = _frame_rms_envelope(segment, sample_rate=sample_rate)
    peak_envelope = _frame_peak_envelope(segment, sample_rate=sample_rate)
    hop_seconds = _positive_sample_seconds(DRUM_TAIL_HOP_SECONDS, sample_rate=sample_rate)
    available_duration = segment.size / float(sample_rate)
    method_durations = {
        "relative_rms_decay": _relative_envelope_tail_duration(
            envelope,
            relative_threshold=DRUM_TAIL_RELATIVE_THRESHOLD,
            hop_seconds=hop_seconds,
            fallback_duration=available_duration,
        ),
        "relative_peak_decay": _relative_envelope_tail_duration(
            peak_envelope,
            relative_threshold=DRUM_TAIL_PEAK_RELATIVE_THRESHOLD,
            hop_seconds=hop_seconds,
            fallback_duration=available_duration,
        ),
        "cumulative_energy": _cumulative_energy_tail_duration(
            segment,
            sample_rate=sample_rate,
            fallback_duration=available_duration,
        ),
    }
    return _synthesize_span_estimate(method_durations)


def _empty_span_estimate() -> DrumEventSpanEstimate:
    return DrumEventSpanEstimate(
        duration_seconds=MIN_DRUM_EVENT_DURATION_SECONDS,
        method_durations={},
        agreement_seconds=0.0,
        consensus_method="minimum_fallback",
    )


def _mono_float_audio(audio: np.ndarray) -> np.ndarray:
    resolved = np.asarray(audio, dtype=np.float32)
    if resolved.ndim > 1:
        resolved = resolved.mean(axis=1)
    return resolved


def _frame_rms_envelope(segment: np.ndarray, *, sample_rate: int) -> np.ndarray:
    frame_size = max(1, int(round(DRUM_TAIL_FRAME_SECONDS * sample_rate)))
    hop_size = max(1, int(round(DRUM_TAIL_HOP_SECONDS * sample_rate)))
    envelope: list[float] = []
    for start_index in range(0, segment.size, hop_size):
        frame = segment[start_index : start_index + frame_size]
        if frame.size == 0:
            continue
        envelope.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))))
    return np.asarray(envelope, dtype=np.float32)


def _frame_peak_envelope(segment: np.ndarray, *, sample_rate: int) -> np.ndarray:
    frame_size = max(1, int(round(DRUM_TAIL_FRAME_SECONDS * sample_rate)))
    hop_size = max(1, int(round(DRUM_TAIL_HOP_SECONDS * sample_rate)))
    envelope: list[float] = []
    for start_index in range(0, segment.size, hop_size):
        frame = segment[start_index : start_index + frame_size]
        if frame.size == 0:
            continue
        envelope.append(float(np.max(np.abs(frame))))
    return np.asarray(envelope, dtype=np.float32)


def _relative_envelope_tail_duration(
    envelope: np.ndarray,
    *,
    relative_threshold: float,
    hop_seconds: float,
    fallback_duration: float,
) -> float:
    if envelope.size == 0:
        return _clamp_duration(fallback_duration)

    peak_energy = float(np.max(envelope))
    if peak_energy <= 0.0:
        return MIN_DRUM_EVENT_DURATION_SECONDS

    threshold = max(peak_energy * relative_threshold, DRUM_TAIL_ABSOLUTE_FLOOR)
    min_index = max(0, int(math.ceil(MIN_DRUM_EVENT_DURATION_SECONDS / hop_seconds)))
    hold_frames = max(1, int(math.ceil(DRUM_TAIL_HOLD_SECONDS / hop_seconds)))

    below_count = 0
    for index, energy in enumerate(envelope):
        if index < min_index:
            below_count = 0
            continue
        if float(energy) <= threshold:
            below_count += 1
            if below_count >= hold_frames:
                tail_start_index = index - hold_frames + 1
                return _clamp_duration(tail_start_index * hop_seconds)
            continue
        below_count = 0
    return _clamp_duration(fallback_duration)


def _cumulative_energy_tail_duration(
    segment: np.ndarray,
    *,
    sample_rate: int,
    fallback_duration: float,
) -> float:
    energy = np.asarray(segment, dtype=np.float64) ** 2
    total_energy = float(np.sum(energy))
    if total_energy <= 0.0:
        return MIN_DRUM_EVENT_DURATION_SECONDS
    target_energy = total_energy * DRUM_TAIL_CUMULATIVE_ENERGY_RATIO
    frame_index = int(np.searchsorted(np.cumsum(energy), target_energy, side="left"))
    duration = (frame_index + 1) / float(sample_rate)
    return _clamp_duration(min(duration, fallback_duration))


def _synthesize_span_estimate(
    method_durations: dict[str, float],
) -> DrumEventSpanEstimate:
    clamped = {
        method: _clamp_duration(duration)
        for method, duration in method_durations.items()
    }
    if not clamped:
        return _empty_span_estimate()

    cluster = _largest_agreement_cluster(clamped)
    if len(cluster) < 2:
        return DrumEventSpanEstimate(
            duration_seconds=MIN_DRUM_EVENT_DURATION_SECONDS,
            method_durations=clamped,
            agreement_seconds=max(clamped.values()) - min(clamped.values()),
            consensus_method="no_agreement",
        )
    cluster_values = [duration for _method, duration in cluster]
    duration = _median(cluster_values)
    agreement = max(cluster_values) - min(cluster_values) if len(cluster_values) > 1 else 0.0
    return DrumEventSpanEstimate(
        duration_seconds=_clamp_duration(duration),
        method_durations=clamped,
        agreement_seconds=round(float(agreement), 6),
        consensus_method="agreement",
    )


def _largest_agreement_cluster(
    method_durations: dict[str, float],
) -> list[tuple[str, float]]:
    ordered = sorted(method_durations.items(), key=lambda item: item[1])
    best: list[tuple[str, float]] = []
    for start_index, (_method, start_duration) in enumerate(ordered):
        cluster = [
            item
            for item in ordered[start_index:]
            if item[1] - start_duration <= DRUM_TAIL_METHOD_AGREEMENT_SECONDS
        ]
        if len(cluster) > len(best):
            best = cluster
            continue
        if len(cluster) == len(best) and _median([item[1] for item in cluster]) > _median(
            [item[1] for item in best]
        ):
            best = cluster
    return best or ordered


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _positive_sample_seconds(seconds: float, *, sample_rate: int) -> float:
    samples = max(1, int(round(float(seconds) * sample_rate)))
    return samples / float(sample_rate)


def _clamp_duration(duration_seconds: float) -> float:
    return min(
        MAX_DRUM_EVENT_DURATION_SECONDS,
        max(MIN_DRUM_EVENT_DURATION_SECONDS, float(duration_seconds)),
    )
