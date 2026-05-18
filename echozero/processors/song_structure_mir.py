"""MIR self-similarity song-part boundary fallback.

Exists to provide coarse part-change anchors when richer MIR tooling is unavailable.
Returns ordered segment starts for the song-sections processor to relabel as numbered cues.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SongStructureSegment:
    start_seconds: float
    cue_ref: str
    label: str
    confidence: float


def segment_song_structure_with_mir(
    *,
    file_path: str,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    boundary_sensitivity: float,
    min_section_seconds: float,
    max_sections: int,
    similarity_threshold: float,
    intro_tail_seconds: float,
    end_tail_seconds: float,
) -> tuple[SongStructureSegment, ...]:
    """Detect coarse song-part boundaries with librosa when available."""

    del similarity_threshold
    try:
        import librosa
        import numpy as np

        audio, effective_sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
        if audio.size == 0:
            return (SongStructureSegment(0.0, "Cue 1", "Cue 1", 0.75),)

        duration = float(audio.shape[0]) / float(effective_sample_rate)
        if duration <= 0.0:
            return (SongStructureSegment(0.0, "Cue 1", "Cue 1", 0.75),)

        resolved_hop = max(128, int(hop_length))
        resolved_fft = max(1024, int(n_fft))
        feature_mfcc = librosa.feature.mfcc(
            y=audio,
            sr=effective_sample_rate,
            n_mfcc=max(8, min(32, int(n_mfcc))),
            n_fft=resolved_fft,
            hop_length=resolved_hop,
        )
        feature_chroma = librosa.feature.chroma_cens(
            y=audio,
            sr=effective_sample_rate,
            hop_length=resolved_hop,
        )
        feature_rms = librosa.feature.rms(
            y=audio,
            frame_length=resolved_fft,
            hop_length=resolved_hop,
        )
        features = np.concatenate((feature_mfcc, feature_chroma, feature_rms), axis=0)
        features = librosa.util.normalize(features, axis=1)

        target_sections = _target_section_count(
            duration_seconds=duration,
            min_section_seconds=min_section_seconds,
            max_sections=max_sections,
            boundary_sensitivity=boundary_sensitivity,
        )
        boundary_frames = librosa.segment.agglomerative(features, k=target_sections)
        boundary_frames = _dedupe_boundary_frames(
            boundary_frames=boundary_frames,
            duration_seconds=duration,
            sample_rate=effective_sample_rate,
            hop_length=resolved_hop,
            min_section_seconds=min_section_seconds,
        )
        if not boundary_frames:
            boundary_frames = [0]

        boundary_strength = _boundary_strengths(features)
        segments: list[SongStructureSegment] = []
        for index, frame_index in enumerate(boundary_frames, start=1):
            start_seconds = max(
                0.0,
                min(
                    duration,
                    float(
                        librosa.frames_to_time(
                            frame_index,
                            sr=effective_sample_rate,
                            hop_length=resolved_hop,
                        )
                    ),
                ),
            )
            confidence = _boundary_confidence(
                boundary_strength=boundary_strength,
                frame_index=frame_index,
                intro_tail_seconds=intro_tail_seconds,
                end_tail_seconds=end_tail_seconds,
                start_seconds=start_seconds,
                duration_seconds=duration,
            )
            segments.append(
                SongStructureSegment(
                    start_seconds=float(start_seconds),
                    cue_ref=f"Cue {index}",
                    label=f"Cue {index}",
                    confidence=confidence,
                )
            )
        return tuple(segments)
    except Exception:
        return (SongStructureSegment(0.0, "Cue 1", "Cue 1", 0.75),)


def _target_section_count(
    *,
    duration_seconds: float,
    min_section_seconds: float,
    max_sections: int,
    boundary_sensitivity: float,
) -> int:
    """Estimate a stable number of sections from duration and sensitivity."""

    min_length = max(1.0, float(min_section_seconds))
    duration_limited = max(2, int(duration_seconds // min_length) + 1)
    hard_cap = duration_limited if int(max_sections) <= 0 else min(duration_limited, int(max_sections))
    normalized_sensitivity = max(0.0, min(1.0, float(boundary_sensitivity)))
    scaled = 2 + int(round(normalized_sensitivity * max(0, hard_cap - 2)))
    return max(2, min(hard_cap, scaled))


def _dedupe_boundary_frames(
    *,
    boundary_frames,
    duration_seconds: float,
    sample_rate: int,
    hop_length: int,
    min_section_seconds: float,
) -> list[int]:
    """Keep only real section starts and remove near-duplicate boundaries."""

    unique_frames = sorted({max(0, int(frame)) for frame in boundary_frames})
    if not unique_frames:
        return [0]
    if unique_frames[0] != 0:
        unique_frames.insert(0, 0)

    min_gap_frames = max(
        1,
        int(round(max(0.5, float(min_section_seconds)) * sample_rate / hop_length)),
    )
    max_start_seconds = max(0.0, float(duration_seconds) - max(0.5, float(min_section_seconds)))

    filtered: list[int] = []
    for frame_index in unique_frames:
        start_seconds = float(frame_index * hop_length) / float(sample_rate)
        if filtered and frame_index - filtered[-1] < min_gap_frames:
            continue
        if start_seconds > max_start_seconds:
            continue
        filtered.append(frame_index)
    return filtered or [0]


def _boundary_strengths(features) -> list[float]:
    """Measure local change magnitude around each frame."""

    import numpy as np

    frame_features = np.asarray(features.T, dtype=np.float32)
    if frame_features.shape[0] <= 1:
        return [0.0]
    deltas = np.linalg.norm(np.diff(frame_features, axis=0), axis=1)
    peak = float(np.max(deltas))
    if peak > 1e-6:
        deltas = deltas / peak
    return [0.0, *[float(value) for value in deltas]]


def _boundary_confidence(
    *,
    boundary_strength: list[float],
    frame_index: int,
    intro_tail_seconds: float,
    end_tail_seconds: float,
    start_seconds: float,
    duration_seconds: float,
) -> float:
    """Score one boundary from local change plus light edge bias."""

    import numpy as np

    left = max(0, int(frame_index) - 2)
    right = min(len(boundary_strength), int(frame_index) + 3)
    local = boundary_strength[left:right]
    base = 0.55 if not local else 0.45 + (0.45 * float(np.mean(local)))
    if start_seconds <= max(0.0, float(intro_tail_seconds)):
        base += 0.05
    if duration_seconds - start_seconds <= max(0.0, float(end_tail_seconds)):
        base -= 0.08
    return round(max(0.35, min(0.98, base)), 3)


__all__ = ["SongStructureSegment", "segment_song_structure_with_mir"]
