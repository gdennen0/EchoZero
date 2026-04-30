"""
SongSectionsProcessor: MFCC-based section boundary and label generation for song audio.
Exists to auto-build section cue layers from source audio using sequence-style MFCC pooling.
Used by ExecutionEngine when running blocks of type 'DetectSongSections'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from echozero.domain.types import AudioData, Event, EventData, Layer
from echozero.errors import ExecutionError
from echozero.execution import ExecutionContext
from echozero.progress import ProgressReport
from echozero.result import Result, err, ok


@dataclass(frozen=True)
class _SectionLabel:
    start_seconds: float
    cue_ref: str
    label: str
    confidence: float


SegmentSongSectionsFn = Callable[
    [
        str,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
        int,
        float,
        float,
        float,
    ],
    tuple[_SectionLabel, ...],
]


def _default_segment_song_sections(
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
) -> tuple[_SectionLabel, ...]:
    try:
        import librosa
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError(
            "Section auto-generation requires librosa and numpy. Install with: pip install librosa numpy"
        ) from exc

    audio, effective_sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
    if audio.size == 0:
        return (_SectionLabel(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0),)

    duration_seconds = float(audio.shape[0]) / float(effective_sample_rate)
    if duration_seconds <= 0.0:
        return (_SectionLabel(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0),)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=effective_sample_rate,
        n_mfcc=max(4, int(n_mfcc)),
        n_fft=max(512, int(n_fft)),
        hop_length=max(64, int(hop_length)),
    )
    frame_features = np.asarray(mfcc.T, dtype=np.float32)
    if frame_features.shape[0] <= 2:
        return (
            _SectionLabel(start_seconds=0.0, cue_ref="intro_01", label="Intro", confidence=1.0),
            _SectionLabel(
                start_seconds=max(0.0, duration_seconds - 0.01),
                cue_ref="end_02",
                label="End",
                confidence=0.9,
            ),
        )

    pooled_features = _pool_feature_history(
        frame_features,
        pool_size=max(1, int(history_pool_frames)),
    )
    normalized_features = _normalize_feature_vectors(pooled_features)
    novelty = _feature_novelty_curve(normalized_features)
    novelty = _smooth_signal(novelty, width=7)

    seconds_per_frame = float(max(64, int(hop_length))) / float(effective_sample_rate)
    min_gap_frames = max(1, int(round(max(0.25, float(min_section_seconds)) / seconds_per_frame)))
    boundary_frames = _select_boundary_frames(
        novelty,
        sensitivity=float(boundary_sensitivity),
        min_gap_frames=min_gap_frames,
        max_sections=max(2, int(max_sections)),
    )
    boundaries_seconds = sorted(
        max(0.0, min(duration_seconds, frame * seconds_per_frame)) for frame in boundary_frames
    )

    segment_embeddings, segment_rms = _segment_descriptors(
        normalized_features=normalized_features,
        audio=audio,
        boundaries_seconds=boundaries_seconds,
        sample_rate=effective_sample_rate,
        hop_length=max(64, int(hop_length)),
    )
    labels = _label_segments(
        embeddings=segment_embeddings,
        rms_values=segment_rms,
        boundaries_seconds=boundaries_seconds,
        duration_seconds=duration_seconds,
        similarity_threshold=float(similarity_threshold),
        intro_tail_seconds=float(intro_tail_seconds),
        end_tail_seconds=float(end_tail_seconds),
    )

    section_labels: list[_SectionLabel] = []
    for index, (start_seconds, label_text) in enumerate(
        zip(boundaries_seconds, labels, strict=True),
        start=1,
    ):
        cue_ref = f"{label_text.lower()}_{index:02d}"
        confidence = _section_confidence(novelty=novelty, seconds_per_frame=seconds_per_frame, start_seconds=start_seconds)
        section_labels.append(
            _SectionLabel(
                start_seconds=float(start_seconds),
                cue_ref=cue_ref,
                label=label_text,
                confidence=confidence,
            )
        )

    return tuple(section_labels)


def _pool_feature_history(frame_features: Any, *, pool_size: int):
    import numpy as np

    frame_count, feature_count = frame_features.shape
    pooled = np.zeros((frame_count, feature_count), dtype=np.float32)
    cumulative = np.cumsum(frame_features, axis=0, dtype=np.float64)
    for frame_index in range(frame_count):
        start_index = max(0, frame_index - pool_size + 1)
        window_count = frame_index - start_index + 1
        window_sum = cumulative[frame_index]
        if start_index > 0:
            window_sum = window_sum - cumulative[start_index - 1]
        pooled[frame_index] = (window_sum / float(window_count)).astype(np.float32)
    return pooled


def _normalize_feature_vectors(values: Any):
    import numpy as np

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return values / norms


def _feature_novelty_curve(values: Any):
    import numpy as np

    if values.shape[0] <= 1:
        return np.zeros((values.shape[0],), dtype=np.float32)
    deltas = np.diff(values, axis=0)
    novelty = np.linalg.norm(deltas, axis=1)
    return np.concatenate((np.array([0.0], dtype=np.float32), novelty.astype(np.float32)))


def _smooth_signal(values: Any, *, width: int):
    import numpy as np

    if width <= 1 or values.size <= 2:
        return values
    pad = max(1, int(width // 2))
    kernel = np.ones((max(1, int(width)),), dtype=np.float32)
    kernel = kernel / float(kernel.sum())
    padded = np.pad(values, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:-pad]


def _select_boundary_frames(
    novelty: Any,
    *,
    sensitivity: float,
    min_gap_frames: int,
    max_sections: int,
) -> list[int]:
    import numpy as np

    frame_count = int(novelty.shape[0])
    if frame_count <= 1:
        return [0]

    normalized_sensitivity = max(0.0, min(1.0, sensitivity))
    quantile = max(0.20, min(0.95, 0.90 - (0.55 * normalized_sensitivity)))
    threshold = float(np.quantile(novelty[1:], quantile)) if frame_count > 2 else float(novelty.max())

    candidate_frames = [
        frame_index
        for frame_index in range(1, frame_count - 1)
        if novelty[frame_index] >= threshold
        and novelty[frame_index] >= novelty[frame_index - 1]
        and novelty[frame_index] >= novelty[frame_index + 1]
    ]
    ranked_candidates = sorted(candidate_frames, key=lambda index: float(novelty[index]), reverse=True)

    selected = [0]
    for frame_index in ranked_candidates:
        if any(abs(frame_index - chosen) < min_gap_frames for chosen in selected):
            continue
        selected.append(frame_index)

    selected = sorted(set(selected))
    if max_sections > 0 and len(selected) > max_sections:
        frame_scores = {frame: float(novelty[frame]) for frame in selected if frame != 0}
        highest_scored = sorted(frame_scores, key=frame_scores.get, reverse=True)[: max(0, max_sections - 1)]
        selected = sorted({0, *highest_scored})

    return selected


def _segment_descriptors(
    *,
    normalized_features: Any,
    audio: Any,
    boundaries_seconds: list[float],
    sample_rate: int,
    hop_length: int,
) -> tuple[Any, list[float]]:
    import numpy as np

    segment_embeddings: list[Any] = []
    segment_rms: list[float] = []
    total_frames = int(normalized_features.shape[0])
    duration_seconds = float(audio.shape[0]) / float(sample_rate)

    boundary_seconds_with_end = [*boundaries_seconds, duration_seconds]
    for start_seconds, end_seconds in zip(boundary_seconds_with_end, boundary_seconds_with_end[1:], strict=False):
        start_frame = max(0, min(total_frames - 1, int(round(start_seconds * sample_rate / hop_length))))
        end_frame = max(start_frame + 1, min(total_frames, int(round(end_seconds * sample_rate / hop_length))))
        frame_slice = normalized_features[start_frame:end_frame]
        if frame_slice.size == 0:
            frame_slice = normalized_features[start_frame : start_frame + 1]
        segment_embeddings.append(np.mean(frame_slice, axis=0))

        start_sample = max(0, int(round(start_seconds * sample_rate)))
        end_sample = min(audio.shape[0], max(start_sample + 1, int(round(end_seconds * sample_rate))))
        audio_slice = audio[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(np.square(audio_slice)))) if audio_slice.size else 0.0
        segment_rms.append(rms)

    return np.asarray(segment_embeddings, dtype=np.float32), segment_rms


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
                if float(similarity[segment_index, chorus_index]) >= normalized_similarity_threshold:
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
            key=lambda index: float(1.0 - max(similarity[index, max(0, index - 1)], similarity[index, min(segment_count - 1, index + 1)])),
        )
        if labels[bridge_index] not in {"Chorus", "Instrumental"}:
            labels[bridge_index] = "Bridge"

    tail_start = boundaries_seconds[-1]
    tail_duration = max(0.0, duration_seconds - tail_start)
    if tail_duration <= max(4.0, end_tail_seconds) or tail_start >= duration_seconds - end_tail_seconds:
        labels[-1] = "End"

    if duration_seconds > 0.0 and labels[0] != "Intro":
        labels[0] = "Intro"
    if boundaries_seconds[0] > max(0.05, intro_tail_seconds):
        labels[0] = "Intro"

    return labels


def _section_confidence(*, novelty: Any, seconds_per_frame: float, start_seconds: float) -> float:
    import numpy as np

    frame_index = max(0, min(int(round(start_seconds / max(seconds_per_frame, 1e-5))), novelty.shape[0] - 1))
    local = novelty[max(0, frame_index - 2) : frame_index + 3]
    if local.size == 0:
        return 0.5
    score = float(np.mean(local))
    normalized = max(0.0, min(1.0, score / max(1e-6, float(np.max(novelty) + 1e-6))))
    return round(0.4 + (0.6 * normalized), 3)


class SongSectionsProcessor:
    """Generate section cue events from source audio using MFCC sequence analysis."""

    def __init__(
        self,
        segment_song_sections_fn: SegmentSongSectionsFn | None = None,
    ) -> None:
        self._segment_song_sections_fn = segment_song_sections_fn or _default_segment_song_sections

    def execute(self, block_id: str, context: ExecutionContext) -> Result[EventData]:
        """Read song audio, infer section starts/labels, and emit one section cue layer."""

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="song_sections",
                percent=0.0,
                message="Starting section auto-generation",
            )
        )

        audio = context.get_input(block_id, "audio_in", AudioData)
        if audio is None:
            return err(
                ExecutionError(
                    f"Block '{block_id}' has no audio input — connect an audio source to 'audio_in'"
                )
            )

        block = context.graph.blocks.get(block_id)
        if block is None:
            return err(ExecutionError(f"Block not found: {block_id}"))

        settings = block.settings
        sample_rate = int(settings.get("sample_rate", audio.sample_rate or 22050))
        n_mfcc = int(settings.get("n_mfcc", 20))
        n_fft = int(settings.get("n_fft", 8192))
        hop_length = int(settings.get("hop_length", 4096))
        history_pool_frames = int(settings.get("history_pool_frames", 160))
        boundary_sensitivity = float(settings.get("boundary_sensitivity", 0.60))
        min_section_seconds = float(settings.get("min_section_seconds", 8.0))
        max_sections = int(settings.get("max_sections", 14))
        similarity_threshold = float(settings.get("similarity_threshold", 0.84))
        intro_tail_seconds = float(settings.get("intro_tail_seconds", 14.0))
        end_tail_seconds = float(settings.get("end_tail_seconds", 16.0))

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="song_sections",
                percent=0.35,
                message="Computing MFCC features and section boundaries",
            )
        )

        try:
            section_labels = self._segment_song_sections_fn(
                audio.file_path,
                sample_rate,
                n_mfcc,
                n_fft,
                hop_length,
                history_pool_frames,
                boundary_sensitivity,
                min_section_seconds,
                max_sections,
                similarity_threshold,
                intro_tail_seconds,
                end_tail_seconds,
            )
        except Exception as exc:
            return err(
                ExecutionError(
                    f"Section auto-generation failed for block '{block_id}': {type(exc).__name__}: {exc}"
                )
            )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="song_sections",
                percent=0.75,
                message="Building section cue events",
            )
        )

        events: list[Event] = []
        for cue_number, section in enumerate(section_labels, start=1):
            events.append(
                Event(
                    id=f"{block_id}_section_{cue_number:03d}",
                    time=float(section.start_seconds),
                    duration=0.0,
                    classifications={
                        "label": section.label,
                        "confidence": float(section.confidence),
                    },
                    metadata={
                        "cue_number": cue_number,
                        "cue_ref": section.cue_ref,
                        "section_label": section.label,
                        "confidence": float(section.confidence),
                        "generator": "mfcc_sequence_pooling_v1",
                    },
                    origin=block_id,
                )
            )

        layer = Layer(
            id=f"{block_id}_sections",
            name="Sections",
            events=tuple(events),
        )

        context.progress_bus.publish(
            ProgressReport(
                block_id=block_id,
                phase="song_sections",
                percent=1.0,
                message="Section auto-generation complete",
            )
        )

        return ok(EventData(layers=(layer,)))
