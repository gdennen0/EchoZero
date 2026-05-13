"""
SongStructureMIR: Beat-synchronous MIR structure analysis for section detection.
Exists because song sections need repetition-aware features, not only frame-local novelty.
Builds chroma/MFCC/tempogram embeddings into a self-similarity matrix and section cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SongStructureSegment:
    """One structure section estimated from beat-synchronous MIR features."""

    start_seconds: float
    cue_ref: str
    label: str
    confidence: float


@dataclass(frozen=True)
class SongStructureAnalysis:
    """Reusable MIR analysis artifacts for visual inspection and section generation."""

    duration_seconds: float
    beat_times_seconds: tuple[float, ...]
    boundaries_seconds: tuple[float, ...]
    novelty_curve: tuple[float, ...]
    self_similarity_matrix: tuple[tuple[float, ...], ...]
    chroma_matrix: tuple[tuple[float, ...], ...]
    mel_spectrogram_db: tuple[tuple[float, ...], ...]
    embedding_points_2d: tuple[tuple[float, float], ...]
    segments: tuple[SongStructureSegment, ...]


def analyze_song_structure(
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
) -> SongStructureAnalysis:
    """Analyze one song into MIR features, a similarity map, and repeated structure segments."""

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError(
            "MIR structure analysis requires librosa and numpy. "
            "Install with: pip install librosa numpy"
        ) from exc

    resolved_sample_rate = max(8000, int(sample_rate))
    resolved_fft = max(512, int(n_fft))
    resolved_hop = max(64, int(hop_length))

    audio, effective_sample_rate = librosa.load(file_path, sr=resolved_sample_rate, mono=True)
    if audio.size == 0:
        empty_segment = SongStructureSegment(0.0, "intro_01", "Intro", 1.0)
        return SongStructureAnalysis(
            duration_seconds=0.0,
            beat_times_seconds=(0.0,),
            boundaries_seconds=(0.0,),
            novelty_curve=(0.0,),
            self_similarity_matrix=((1.0,),),
            chroma_matrix=tuple(tuple(0.0 for _ in range(1)) for _ in range(12)),
            mel_spectrogram_db=tuple(tuple(0.0 for _ in range(1)) for _ in range(32)),
            embedding_points_2d=((0.0, 0.0),),
            segments=(empty_segment,),
        )

    duration_seconds = float(audio.shape[0]) / float(effective_sample_rate)
    harmonic, percussive = librosa.effects.hpss(audio)
    onset_envelope = librosa.onset.onset_strength(
        y=percussive,
        sr=effective_sample_rate,
        hop_length=resolved_hop,
    )
    _, beat_frames = librosa.beat.beat_track(
        y=audio,
        sr=effective_sample_rate,
        hop_length=resolved_hop,
        onset_envelope=onset_envelope,
    )
    beat_frames = _coerce_beat_frames(
        beat_frames=beat_frames,
        frame_count=max(1, int(onset_envelope.shape[0])),
        duration_seconds=duration_seconds,
        sample_rate=effective_sample_rate,
        hop_length=resolved_hop,
        min_section_seconds=min_section_seconds,
    )

    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=effective_sample_rate,
        hop_length=resolved_hop,
    )
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=effective_sample_rate,
        n_mfcc=max(8, int(n_mfcc)),
        n_fft=resolved_fft,
        hop_length=resolved_hop,
    )
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=effective_sample_rate,
        hop_length=resolved_hop,
    )
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=effective_sample_rate,
        n_fft=resolved_fft,
        hop_length=resolved_hop,
        n_mels=96,
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    chroma, mfcc, tempogram = _align_feature_frame_counts(chroma, mfcc, tempogram)
    aligned_frame_count = int(chroma.shape[1])
    beat_frames = _align_beat_frames_to_feature_count(
        beat_frames=beat_frames,
        feature_frame_count=aligned_frame_count,
    )
    feature_stack = np.vstack((chroma, mfcc, tempogram)).astype(np.float32, copy=False)
    beat_sync = librosa.util.sync(feature_stack, beat_frames, aggregate=np.mean)
    normalized_beat_sync = _normalize_columns(beat_sync)
    self_similarity = np.matmul(normalized_beat_sync.T, normalized_beat_sync)
    embedding_points = _project_feature_vectors_2d(normalized_beat_sync.T)

    novelty = _compute_checkerboard_novelty(self_similarity)
    smoothed_novelty = _smooth_curve(novelty, width=5)
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=effective_sample_rate,
        hop_length=resolved_hop,
    )
    seconds_per_beat = _median_step_seconds(
        beat_times,
        fallback=max(0.5, min_section_seconds / 4.0),
    )
    min_gap_beats = max(1, int(round(max(0.25, min_section_seconds) / seconds_per_beat)))
    boundary_indices = _pick_boundary_indices(
        smoothed_novelty,
        sensitivity=boundary_sensitivity,
        min_gap_frames=min_gap_beats,
        max_sections=max_sections,
    )
    boundary_indices = sorted(set([0, *boundary_indices]))
    boundary_times = [
        max(0.0, min(duration_seconds, float(beat_times[index])))
        for index in boundary_indices
        if 0 <= index < len(beat_times)
    ]
    if not boundary_times:
        boundary_times = [0.0]

    segment_embeddings, rms_values = _segment_descriptors(
        normalized_beat_sync=normalized_beat_sync,
        beat_times=beat_times,
        audio=audio,
        sample_rate=effective_sample_rate,
        boundaries_seconds=boundary_times,
        duration_seconds=duration_seconds,
    )
    labels = _label_segments(
        embeddings=segment_embeddings,
        rms_values=rms_values,
        boundaries_seconds=boundary_times,
        duration_seconds=duration_seconds,
        similarity_threshold=similarity_threshold,
        intro_tail_seconds=intro_tail_seconds,
        end_tail_seconds=end_tail_seconds,
    )

    segments: list[SongStructureSegment] = []
    for index, (start_seconds, label) in enumerate(
        zip(boundary_times, labels, strict=True),
        start=1,
    ):
        cue_ref = f"{label.lower()}_{index:02d}"
        confidence = _boundary_confidence(
            novelty_curve=smoothed_novelty,
            boundary_index=boundary_indices[index - 1],
        )
        segments.append(
            SongStructureSegment(
                start_seconds=float(start_seconds),
                cue_ref=cue_ref,
                label=label,
                confidence=confidence,
            )
        )

    return SongStructureAnalysis(
        duration_seconds=float(duration_seconds),
        beat_times_seconds=tuple(float(value) for value in beat_times.tolist()),
        boundaries_seconds=tuple(float(value) for value in boundary_times),
        novelty_curve=tuple(float(value) for value in smoothed_novelty.tolist()),
        self_similarity_matrix=_matrix_to_tuple(self_similarity),
        chroma_matrix=_matrix_to_tuple(chroma),
        mel_spectrogram_db=_matrix_to_tuple(mel_spectrogram_db),
        embedding_points_2d=_points_to_tuple(embedding_points),
        segments=tuple(segments),
    )


def segment_song_structure_with_mir(**kwargs: Any) -> tuple[SongStructureSegment, ...]:
    """Generate section segments from the MIR analysis path."""

    return analyze_song_structure(**kwargs).segments


def _coerce_beat_frames(
    *,
    beat_frames: Any,
    frame_count: int,
    duration_seconds: float,
    sample_rate: int,
    hop_length: int,
    min_section_seconds: float,
):
    import numpy as np

    beats = np.asarray(beat_frames, dtype=np.int32)
    if beats.size >= 2:
        return beats
    fallback_step_seconds = max(0.5, float(min_section_seconds) / 4.0)
    fallback_step_frames = max(1, int(round((fallback_step_seconds * sample_rate) / hop_length)))
    fallback = np.arange(0, max(frame_count, 1), fallback_step_frames, dtype=np.int32)
    if fallback.size == 0 or fallback[0] != 0:
        fallback = np.insert(fallback, 0, 0)
    if duration_seconds > 0.0 and fallback[-1] < frame_count - 1:
        fallback = np.append(fallback, frame_count - 1)
    return fallback


def _normalize_columns(values: Any):
    import numpy as np

    centered = values - np.mean(values, axis=1, keepdims=True)
    scales = np.std(centered, axis=1, keepdims=True)
    scales = np.maximum(scales, 1e-8)
    normalized = centered / scales
    norms = np.linalg.norm(normalized, axis=0, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normalized / norms


def _align_feature_frame_counts(*features: Any) -> tuple[Any, ...]:
    import numpy as np

    if not features:
        return ()
    frame_count = min(int(np.asarray(feature).shape[1]) for feature in features)
    if frame_count <= 0:
        raise ValueError("MIR structure analysis requires feature matrices with at least one frame.")
    return tuple(np.asarray(feature)[:, :frame_count] for feature in features)


def _align_beat_frames_to_feature_count(*, beat_frames: Any, feature_frame_count: int):
    import numpy as np

    if feature_frame_count <= 0:
        raise ValueError("feature_frame_count must be positive")
    clipped = np.asarray(beat_frames, dtype=np.int32)
    clipped = clipped[(clipped >= 0) & (clipped < feature_frame_count)]
    if clipped.size == 0 or clipped[0] != 0:
        clipped = np.insert(clipped, 0, 0)
    if clipped[-1] != feature_frame_count - 1:
        clipped = np.append(clipped, feature_frame_count - 1)
    return np.unique(clipped)


def _compute_checkerboard_novelty(self_similarity: Any, *, radius: int = 4):
    import numpy as np

    beat_count = int(self_similarity.shape[0])
    novelty = np.zeros((beat_count,), dtype=np.float32)
    if beat_count <= 1:
        return novelty
    for center in range(radius, max(radius + 1, beat_count - radius)):
        left = self_similarity[center - radius : center, center - radius : center]
        right = self_similarity[center : center + radius, center : center + radius]
        cross_a = self_similarity[center - radius : center, center : center + radius]
        cross_b = self_similarity[center : center + radius, center - radius : center]
        novelty[center] = float(
            np.mean(left) + np.mean(right) - np.mean(cross_a) - np.mean(cross_b)
        )
    peak = float(np.max(np.abs(novelty)))
    if peak <= 1e-8:
        return novelty
    return novelty / peak


def _smooth_curve(values: Any, *, width: int):
    import numpy as np

    if width <= 1 or int(values.shape[0]) <= 2:
        return values
    kernel = np.ones((max(1, int(width)),), dtype=np.float32)
    kernel = kernel / float(kernel.sum())
    pad = max(1, int(width // 2))
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def _median_step_seconds(values: Any, *, fallback: float) -> float:
    import numpy as np

    if len(values) <= 1:
        return float(fallback)
    diffs = np.diff(np.asarray(values, dtype=np.float32))
    finite = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if finite.size <= 0:
        return float(fallback)
    return float(np.median(finite))


def _pick_boundary_indices(
    novelty_curve: Any,
    *,
    sensitivity: float,
    min_gap_frames: int,
    max_sections: int,
) -> list[int]:
    import numpy as np

    beat_count = int(novelty_curve.shape[0])
    if beat_count <= 1:
        return [0]
    quantile = max(0.35, min(0.95, 0.92 - (0.5 * max(0.0, min(1.0, sensitivity)))))
    threshold = float(np.quantile(novelty_curve[1:], quantile))
    threshold = max(0.08, threshold)
    candidates = [
        index
        for index in range(1, beat_count - 1)
        if novelty_curve[index] >= threshold
        and novelty_curve[index] >= novelty_curve[index - 1]
        and novelty_curve[index] >= novelty_curve[index + 1]
    ]
    ranked = sorted(candidates, key=lambda index: float(novelty_curve[index]), reverse=True)
    selected = [0]
    for index in ranked:
        if any(abs(index - existing) < min_gap_frames for existing in selected):
            continue
        selected.append(index)
    if max_sections > 0 and len(selected) > max_sections:
        nonzero = [index for index in selected if index != 0]
        keep = sorted(nonzero, key=lambda index: float(novelty_curve[index]), reverse=True)[
            : max(0, max_sections - 1)
        ]
        selected = [0, *keep]
    return sorted(set(selected))


def _segment_descriptors(
    *,
    normalized_beat_sync: Any,
    beat_times: Any,
    audio: Any,
    sample_rate: int,
    boundaries_seconds: list[float],
    duration_seconds: float,
):
    import numpy as np

    beat_seconds_with_end = [*boundaries_seconds, float(duration_seconds)]
    embeddings: list[Any] = []
    rms_values: list[float] = []
    beat_times_array = np.asarray(beat_times, dtype=np.float32)
    for start_seconds, end_seconds in zip(
        beat_seconds_with_end,
        beat_seconds_with_end[1:],
        strict=False,
    ):
        start_index = int(
            np.searchsorted(beat_times_array, float(start_seconds), side="left")
        )
        end_index = int(np.searchsorted(beat_times_array, float(end_seconds), side="left"))
        end_index = max(start_index + 1, min(normalized_beat_sync.shape[1], end_index))
        beat_slice = normalized_beat_sync[:, start_index:end_index]
        embeddings.append(np.mean(beat_slice, axis=1))
        start_sample = max(0, int(round(float(start_seconds) * sample_rate)))
        end_sample = min(
            audio.shape[0],
            max(start_sample + 1, int(round(float(end_seconds) * sample_rate))),
        )
        audio_slice = audio[start_sample:end_sample]
        rms_values.append(
            float(np.sqrt(np.mean(np.square(audio_slice)))) if audio_slice.size else 0.0
        )
    return np.asarray(embeddings, dtype=np.float32), rms_values


def _label_segments(
    *,
    embeddings: Any,
    rms_values: list[float],
    boundaries_seconds: list[float],
    duration_seconds: float,
    similarity_threshold: float,
    intro_tail_seconds: float,
    end_tail_seconds: float,
) -> list[str]:
    from echozero.processors.song_sections_determine_style import label_segments_from_embeddings

    return label_segments_from_embeddings(
        embeddings=embeddings,
        rms_values=rms_values,
        boundaries_seconds=boundaries_seconds,
        duration_seconds=duration_seconds,
        similarity_threshold=similarity_threshold,
        intro_tail_seconds=intro_tail_seconds,
        end_tail_seconds=end_tail_seconds,
    )


def _boundary_confidence(*, novelty_curve: Any, boundary_index: int) -> float:
    import numpy as np

    if int(novelty_curve.shape[0]) <= 0:
        return 0.5
    clamped_index = max(0, min(int(boundary_index), int(novelty_curve.shape[0]) - 1))
    local = novelty_curve[max(0, clamped_index - 1) : clamped_index + 2]
    peak = float(np.max(np.abs(novelty_curve)))
    if local.size == 0 or peak <= 1e-8:
        return 0.5
    return round(0.45 + (0.5 * float(np.mean(np.abs(local)) / peak)), 3)


def _matrix_to_tuple(matrix: Any) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _points_to_tuple(points: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in points.tolist())


def _project_feature_vectors_2d(feature_vectors: Any):
    import numpy as np

    points = np.asarray(feature_vectors, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] <= 0:
        return np.zeros((1, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    centered = points - np.mean(points, axis=0, keepdims=True)
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((points.shape[0], 2), dtype=np.float32)
    projected = centered @ vt[:2].T
    if projected.shape[1] == 1:
        projected = np.concatenate(
            (projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)),
            axis=1,
        )
    scales = np.max(np.abs(projected), axis=0, keepdims=True)
    scales = np.maximum(scales, 1e-6)
    normalized = projected / scales
    return normalized.astype(np.float32, copy=False)
