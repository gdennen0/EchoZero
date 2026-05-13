"""
Determine-sections style segmenter for song structure detection.
Exists to provide a second section detector mode inspired by pnlong/determine_sections.
Used by SongSectionsProcessor when detect_method selects determine_sections_style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MFCC_SEQUENCE_POOLING_METHOD = "mfcc_sequence_pooling"
DETERMINE_SECTIONS_STYLE_METHOD = "determine_sections_style"
MIR_SELF_SIMILARITY_METHOD = "mir_self_similarity"


@dataclass(frozen=True)
class DetermineSectionsStyleSegment:
    """Detected song section in determine-sections style mode."""

    start_seconds: float
    cue_ref: str
    label: str
    confidence: float


def resolve_detect_method(value: Any) -> str:
    """Normalize detect_method settings to a known detector key."""

    token = str(value or "").strip().lower().replace("-", "_")
    if token in {DETERMINE_SECTIONS_STYLE_METHOD, "determine_sections"}:
        return DETERMINE_SECTIONS_STYLE_METHOD
    if token in {MIR_SELF_SIMILARITY_METHOD, "mir", "self_similarity", "mir_structure"}:
        return MIR_SELF_SIMILARITY_METHOD
    return MFCC_SEQUENCE_POOLING_METHOD


def determine_sections_style_segments(
    file_path: str,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    history_pool_frames: int,
    boundary_sensitivity: float,
    min_section_seconds: float,
    max_sections: int,
    similarity_threshold: float,
    intro_tail_seconds: float,
    end_tail_seconds: float,
) -> tuple[DetermineSectionsStyleSegment, ...]:
    """Estimate sections via sliding-window change scores and heuristic section labels."""

    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError(
            "Section auto-generation requires librosa and numpy. Install with: pip install librosa numpy"
        ) from exc

    audio, effective_sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return (
            DetermineSectionsStyleSegment(
                start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0
            ),
        )

    duration_seconds = float(audio.shape[0]) / float(effective_sample_rate)
    if duration_seconds <= 0.0:
        return (
            DetermineSectionsStyleSegment(
                start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0
            ),
        )

    resolved_hop = max(64, int(hop_length))
    resolved_fft = max(512, int(n_fft))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=effective_sample_rate,
        n_mfcc=max(4, int(n_mfcc)),
        n_fft=resolved_fft,
        hop_length=resolved_hop,
    )
    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=effective_sample_rate,
        n_fft=resolved_fft,
        hop_length=resolved_hop,
    )
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=effective_sample_rate,
        n_fft=resolved_fft,
        hop_length=resolved_hop,
    )
    rms = librosa.feature.rms(y=audio, frame_length=resolved_fft, hop_length=resolved_hop)
    onset_envelope = librosa.onset.onset_strength(
        y=audio, sr=effective_sample_rate, hop_length=resolved_hop
    )

    min_frames = min(
        int(mfcc.shape[1]),
        int(chroma.shape[1]),
        int(spectral_contrast.shape[1]),
        int(rms.shape[1]),
        int(onset_envelope.shape[0]),
    )
    if min_frames <= 2:
        return (
            DetermineSectionsStyleSegment(
                start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0
            ),
            DetermineSectionsStyleSegment(
                start_seconds=max(0.0, duration_seconds - 0.01),
                cue_ref="outro_02",
                label="Outro",
                confidence=0.85,
            ),
        )

    features = np.concatenate(
        (
            mfcc[:, :min_frames],
            chroma[:, :min_frames],
            spectral_contrast[:, :min_frames],
            rms[:, :min_frames],
            onset_envelope[:min_frames][None, :],
        ),
        axis=0,
    ).T.astype(np.float32)

    normalized_features = _normalize_rows(features)
    change_curve = _window_change_curve(
        normalized_features,
        window_size=max(6, int(history_pool_frames)),
    )
    change_curve = _moving_average(change_curve, width=7)

    seconds_per_frame = float(resolved_hop) / float(effective_sample_rate)
    min_gap_frames = max(1, int(round(max(0.25, float(min_section_seconds)) / seconds_per_frame)))
    boundary_frames = _pick_change_peaks(
        change_curve,
        sensitivity=float(boundary_sensitivity),
        min_gap_frames=min_gap_frames,
        max_sections=max(2, int(max_sections)),
    )
    boundaries_seconds = sorted(
        max(0.0, min(duration_seconds, float(frame_index) * seconds_per_frame))
        for frame_index in boundary_frames
    )
    if not boundaries_seconds:
        boundaries_seconds = [0.0]

    labels = _label_boundaries(
        normalized_features=normalized_features,
        audio=audio,
        sample_rate=effective_sample_rate,
        hop_length=resolved_hop,
        boundaries_seconds=boundaries_seconds,
        duration_seconds=duration_seconds,
        similarity_threshold=float(similarity_threshold),
        intro_tail_seconds=float(intro_tail_seconds),
        end_tail_seconds=float(end_tail_seconds),
    )

    segments: list[DetermineSectionsStyleSegment] = []
    for start_seconds, label in zip(boundaries_seconds, labels, strict=True):
        frame_index = max(
            0,
            min(
                int(round(start_seconds / max(seconds_per_frame, 1e-5))), change_curve.shape[0] - 1
            ),
        )
        local = change_curve[max(0, frame_index - 2) : frame_index + 3]
        confidence = (
            0.55 if local.size == 0 else 0.45 + 0.55 * float(np.clip(np.mean(local), 0.0, 1.0))
        )
        label_slug = (
            "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_") or "section"
        )
        segments.append(
            DetermineSectionsStyleSegment(
                start_seconds=float(start_seconds),
                cue_ref=f"{label_slug}_{len(segments) + 1:02d}",
                label=label,
                confidence=round(float(np.clip(confidence, 0.0, 1.0)), 3),
            )
        )

    return tuple(segments)


def _normalize_rows(values: Any):
    import numpy as np

    centered = values - np.mean(values, axis=0, keepdims=True)
    scales = np.std(centered, axis=0, keepdims=True)
    scales = np.maximum(scales, 1e-8)
    normalized = centered / scales
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normalized / norms


def _window_change_curve(features: Any, *, window_size: int):
    import numpy as np

    frame_count = int(features.shape[0])
    if frame_count <= 2:
        return np.zeros((frame_count,), dtype=np.float32)

    half_window = max(1, int(window_size // 2))
    changes = np.zeros((frame_count,), dtype=np.float32)
    for center in range(half_window, max(half_window + 1, frame_count - half_window)):
        left = features[max(0, center - half_window) : center]
        right = features[center : min(frame_count, center + half_window)]
        if left.size == 0 or right.size == 0:
            continue
        delta = np.mean(right, axis=0) - np.mean(left, axis=0)
        changes[center] = float(np.linalg.norm(delta))

    peak = float(np.max(changes))
    if peak <= 1e-8:
        return changes
    return changes / peak


def _moving_average(values: Any, *, width: int):
    import numpy as np

    if width <= 1 or int(values.shape[0]) <= 2:
        return values
    kernel = np.ones((max(1, int(width)),), dtype=np.float32)
    kernel = kernel / float(kernel.sum())
    pad = max(1, int(width // 2))
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def _pick_change_peaks(
    change_curve: Any, *, sensitivity: float, min_gap_frames: int, max_sections: int
) -> list[int]:
    import numpy as np

    frame_count = int(change_curve.shape[0])
    if frame_count <= 1:
        return [0]

    normalized_sensitivity = max(0.0, min(1.0, sensitivity))
    threshold = float(np.quantile(change_curve[1:], 0.90 - (0.5 * normalized_sensitivity)))
    threshold = max(0.2, threshold)

    candidate_frames = [
        frame_index
        for frame_index in range(1, frame_count - 1)
        if change_curve[frame_index] >= threshold
        and change_curve[frame_index] >= change_curve[frame_index - 1]
        and change_curve[frame_index] >= change_curve[frame_index + 1]
    ]
    ranked = sorted(candidate_frames, key=lambda index: float(change_curve[index]), reverse=True)

    selected = [0]
    for frame_index in ranked:
        if any(abs(frame_index - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(frame_index)

    selected = sorted(set(selected))
    if max_sections > 0 and len(selected) > max_sections:
        nonzero = [frame for frame in selected if frame != 0]
        keep = sorted(nonzero, key=lambda index: float(change_curve[index]), reverse=True)[
            : max(0, max_sections - 1)
        ]
        selected = sorted({0, *keep})
    return selected


def _label_boundaries(
    *,
    normalized_features: Any,
    audio: Any,
    sample_rate: int,
    hop_length: int,
    boundaries_seconds: list[float],
    duration_seconds: float,
    similarity_threshold: float,
    intro_tail_seconds: float,
    end_tail_seconds: float,
) -> list[str]:
    import numpy as np

    boundary_count = len(boundaries_seconds)
    labels = ["Verse" for _ in range(boundary_count)]
    if not labels:
        return ["Intro"]
    labels[0] = "Intro"

    boundary_seconds_with_end = [*boundaries_seconds, duration_seconds]
    embeddings: list[Any] = []
    energies: list[float] = []
    for start_seconds, end_seconds in zip(
        boundary_seconds_with_end, boundary_seconds_with_end[1:], strict=False
    ):
        start_frame = max(
            0,
            min(
                normalized_features.shape[0] - 1,
                int(round(start_seconds * sample_rate / hop_length)),
            ),
        )
        end_frame = max(
            start_frame + 1,
            min(normalized_features.shape[0], int(round(end_seconds * sample_rate / hop_length))),
        )
        frame_slice = normalized_features[start_frame:end_frame]
        embeddings.append(np.mean(frame_slice, axis=0))

        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(
            audio.shape[0], max(start_sample + 1, int(round(end_seconds * sample_rate)))
        )
        audio_slice = audio[start_sample:end_sample]
        energy = float(np.sqrt(np.mean(np.square(audio_slice)))) if audio_slice.size else 0.0
        energies.append(energy)

    similarity = np.matmul(
        np.asarray(embeddings, dtype=np.float32), np.asarray(embeddings, dtype=np.float32).T
    )
    repeat_threshold = max(0.0, min(0.99, similarity_threshold))
    repeat_score = [0.0 for _ in range(boundary_count)]
    for index in range(boundary_count):
        for other_index in range(boundary_count):
            if index == other_index or abs(index - other_index) <= 1:
                continue
            if float(similarity[index, other_index]) >= repeat_threshold:
                repeat_score[index] += float(similarity[index, other_index])

    if boundary_count > 2:
        chorus_index = max(range(1, boundary_count - 1), key=lambda index: repeat_score[index])
        if repeat_score[chorus_index] > 0.0:
            for index in range(1, boundary_count - 1):
                if float(similarity[index, chorus_index]) >= repeat_threshold:
                    labels[index] = "Chorus"

    median_energy = float(np.median(energies)) if energies else 0.0
    for index in range(1, max(1, boundary_count - 1)):
        if labels[index] == "Chorus":
            continue
        if median_energy > 0.0 and energies[index] <= median_energy * 0.6:
            labels[index] = "Buildup"

    tail_start = boundaries_seconds[-1]
    if (
        duration_seconds - tail_start <= max(4.0, end_tail_seconds)
        or tail_start >= duration_seconds - end_tail_seconds
    ):
        labels[-1] = "Outro"

    if boundaries_seconds[0] > max(0.05, intro_tail_seconds):
        labels[0] = "Intro"

    return labels


def label_segments_from_embeddings(
    *,
    embeddings: Any,
    rms_values: list[float],
    boundaries_seconds: list[float],
    duration_seconds: float,
    similarity_threshold: float,
    intro_tail_seconds: float,
    end_tail_seconds: float,
) -> list[str]:
    """Label section boundaries using embedding similarity and simple energy heuristics."""

    import numpy as np

    segment_count = len(boundaries_seconds)
    if segment_count <= 0:
        return ["Intro"]
    if segment_count == 1:
        return ["Intro"]

    labels = ["Verse" for _ in range(segment_count)]
    labels[0] = "Intro"

    normalized_similarity_threshold = max(0.0, min(0.99, similarity_threshold))
    similarity = np.matmul(embeddings, embeddings.T)

    repeat_scores: list[float] = []
    for segment_index in range(segment_count):
        score = 0.0
        for other_index in range(segment_count):
            if segment_index == other_index:
                continue
            if abs(segment_index - other_index) <= 1:
                continue
            if float(similarity[segment_index, other_index]) >= normalized_similarity_threshold:
                score += float(similarity[segment_index, other_index])
        repeat_scores.append(score)

    if segment_count > 2:
        chorus_index = max(range(1, segment_count - 1), key=lambda index: repeat_scores[index])
        if repeat_scores[chorus_index] > 0.0:
            for segment_index in range(1, segment_count - 1):
                if (
                    float(similarity[segment_index, chorus_index])
                    >= normalized_similarity_threshold
                ):
                    labels[segment_index] = "Chorus"

    median_rms = float(np.median(rms_values)) if rms_values else 0.0
    for segment_index in range(1, max(1, segment_count - 1)):
        if labels[segment_index] == "Chorus":
            continue
        if median_rms > 0.0 and rms_values[segment_index] <= median_rms * 0.55:
            labels[segment_index] = "Instrumental"

    if segment_count > 3:
        late_candidates = range(max(1, segment_count // 2), max(1, segment_count - 1))
        bridge_index = max(
            late_candidates,
            key=lambda index: float(
                1.0
                - max(
                    similarity[index, max(0, index - 1)],
                    similarity[index, min(segment_count - 1, index + 1)],
                )
            ),
        )
        if labels[bridge_index] not in {"Chorus", "Instrumental"}:
            labels[bridge_index] = "Bridge"

    tail_start = boundaries_seconds[-1]
    tail_duration = max(0.0, duration_seconds - tail_start)
    if (
        tail_duration <= max(4.0, end_tail_seconds)
        or tail_start >= duration_seconds - end_tail_seconds
    ):
        labels[-1] = "End"

    if duration_seconds > 0.0 and labels[0] != "Intro":
        labels[0] = "Intro"
    if boundaries_seconds[0] > max(0.05, intro_tail_seconds):
        labels[0] = "Intro"

    return labels
