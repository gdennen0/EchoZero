"""
Song-parts preview service: build rich structure-preview artifacts for the settings UI.
Exists so song-part dialogs can visualize merged audio-structure evidence without Qt knowing MIR details.
Connects audio-file bindings and detector settings to a shared 2D vector-space preview contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from echozero.processors.song_sections import (
    _default_segment_song_sections,
    _determine_sections_style_segment_song_sections,
)
from echozero.processors.song_sections_determine_style import (
    DETERMINE_SECTIONS_STYLE_METHOD,
    MFCC_SEQUENCE_POOLING_METHOD,
    MIR_SELF_SIMILARITY_METHOD,
    resolve_detect_method,
)
from echozero.processors.song_structure_mir import SongStructureAnalysis, analyze_song_structure


@dataclass(frozen=True)
class SongPartsPreviewPoint:
    """One beat-synchronous point in the merged song-structure vector space."""

    x: float
    y: float
    time_seconds: float
    novelty: float
    repetition: float
    segment_index: int
    is_boundary: bool


@dataclass(frozen=True)
class SongPartsPreviewSegment:
    """One preview-ready song segment with an explicit end time."""

    start_seconds: float
    end_seconds: float
    cue_ref: str
    label: str
    confidence: float


@dataclass(frozen=True)
class SongPartsPreviewData:
    """All preview artifacts needed to render one song-parts analysis card."""

    source_audio_path: str
    detect_method: str
    detect_method_label: str
    duration_seconds: float
    points: tuple[SongPartsPreviewPoint, ...]
    segments: tuple[SongPartsPreviewSegment, ...]
    summary_text: str
    detail_text: str


def build_song_parts_preview(
    *,
    source_audio_path: str,
    settings: Mapping[str, object],
) -> SongPartsPreviewData:
    """Build a 2D vector-space preview for the current song-parts settings."""

    audio_path = Path(str(source_audio_path or "")).expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(f"Source audio file not found: {audio_path}")

    resolved_settings = _resolved_preview_settings(settings)
    detect_method = resolve_detect_method(resolved_settings["detect_method"])
    baseline_analysis = analyze_song_structure(
        file_path=str(audio_path),
        sample_rate=int(resolved_settings["sample_rate"]),
        n_mfcc=int(resolved_settings["n_mfcc"]),
        n_fft=int(resolved_settings["n_fft"]),
        hop_length=int(resolved_settings["hop_length"]),
        boundary_sensitivity=float(resolved_settings["boundary_sensitivity"]),
        min_section_seconds=float(resolved_settings["min_section_seconds"]),
        max_sections=int(resolved_settings["max_sections"]),
        similarity_threshold=float(resolved_settings["similarity_threshold"]),
        intro_tail_seconds=float(resolved_settings["intro_tail_seconds"]),
        end_tail_seconds=float(resolved_settings["end_tail_seconds"]),
    )
    preview_segments = _segments_for_method(
        detect_method=detect_method,
        source_audio_path=str(audio_path),
        settings=resolved_settings,
        baseline_analysis=baseline_analysis,
    )
    preview_points = _preview_points_for_analysis(
        analysis=baseline_analysis,
        segments=preview_segments,
    )
    return SongPartsPreviewData(
        source_audio_path=str(audio_path),
        detect_method=detect_method,
        detect_method_label=_detect_method_label(detect_method),
        duration_seconds=float(baseline_analysis.duration_seconds),
        points=preview_points,
        segments=preview_segments,
        summary_text=(
            "Vector-space preview combines beat-synchronous chroma, MFCC, tempogram, "
            "novelty, and repetition evidence into one 2D structure map."
        ),
        detail_text=(
            f"{len(preview_points)} merged beat points · "
            f"{len(preview_segments)} predicted sections · "
            f"Method: {_detect_method_label(detect_method)}"
        ),
    )


def _resolved_preview_settings(settings: Mapping[str, object]) -> dict[str, object]:
    return {
        "detect_method": settings.get("detect_method", MIR_SELF_SIMILARITY_METHOD),
        "sample_rate": int(settings.get("sample_rate", 22050)),
        "n_mfcc": int(settings.get("n_mfcc", 20)),
        "n_fft": int(settings.get("n_fft", 8192)),
        "hop_length": int(settings.get("hop_length", 4096)),
        "history_pool_frames": int(settings.get("history_pool_frames", 160)),
        "boundary_sensitivity": float(settings.get("boundary_sensitivity", 0.60)),
        "min_section_seconds": float(settings.get("min_section_seconds", 8.0)),
        "max_sections": int(settings.get("max_sections", 14)),
        "similarity_threshold": float(settings.get("similarity_threshold", 0.84)),
        "intro_tail_seconds": float(settings.get("intro_tail_seconds", 14.0)),
        "end_tail_seconds": float(settings.get("end_tail_seconds", 16.0)),
    }


def _segments_for_method(
    *,
    detect_method: str,
    source_audio_path: str,
    settings: Mapping[str, object],
    baseline_analysis: SongStructureAnalysis,
) -> tuple[SongPartsPreviewSegment, ...]:
    if detect_method == MIR_SELF_SIMILARITY_METHOD:
        starts = [float(segment.start_seconds) for segment in baseline_analysis.segments]
        labels = [
            (
                float(segment.start_seconds),
                str(segment.cue_ref),
                str(segment.label),
                float(segment.confidence),
            )
            for segment in baseline_analysis.segments
        ]
    elif detect_method == DETERMINE_SECTIONS_STYLE_METHOD:
        labels = [
            (
                float(segment.start_seconds),
                str(segment.cue_ref),
                str(segment.label),
                float(segment.confidence),
            )
            for segment in _determine_sections_style_segment_song_sections(
                source_audio_path,
                int(settings["sample_rate"]),
                int(settings["n_mfcc"]),
                int(settings["n_fft"]),
                int(settings["hop_length"]),
                int(settings["history_pool_frames"]),
                float(settings["boundary_sensitivity"]),
                float(settings["min_section_seconds"]),
                int(settings["max_sections"]),
                float(settings["similarity_threshold"]),
                float(settings["intro_tail_seconds"]),
                float(settings["end_tail_seconds"]),
            )
        ]
        starts = [start for start, _cue_ref, _label, _confidence in labels]
    else:
        labels = [
            (
                float(segment.start_seconds),
                str(segment.cue_ref),
                str(segment.label),
                float(segment.confidence),
            )
            for segment in _default_segment_song_sections(
                source_audio_path,
                int(settings["sample_rate"]),
                int(settings["n_mfcc"]),
                int(settings["n_fft"]),
                int(settings["hop_length"]),
                int(settings["history_pool_frames"]),
                float(settings["boundary_sensitivity"]),
                float(settings["min_section_seconds"]),
                int(settings["max_sections"]),
                float(settings["similarity_threshold"]),
                float(settings["intro_tail_seconds"]),
                float(settings["end_tail_seconds"]),
            )
        ]
        starts = [start for start, _cue_ref, _label, _confidence in labels]
    return _preview_segments_from_labels(
        labels=labels,
        starts=starts,
        duration_seconds=float(baseline_analysis.duration_seconds),
    )


def _preview_segments_from_labels(
    *,
    labels: list[tuple[float, str, str, float]],
    starts: list[float],
    duration_seconds: float,
) -> tuple[SongPartsPreviewSegment, ...]:
    if not labels:
        labels = [(0.0, "intro_01", "Intro", 1.0)]
        starts = [0.0]
    segments: list[SongPartsPreviewSegment] = []
    for index, (start_seconds, cue_ref, label, confidence) in enumerate(labels):
        end_seconds = (
            starts[index + 1]
            if index + 1 < len(starts)
            else max(float(start_seconds), float(duration_seconds))
        )
        segments.append(
            SongPartsPreviewSegment(
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                cue_ref=cue_ref,
                label=label,
                confidence=confidence,
            )
        )
    return tuple(segments)


def _preview_points_for_analysis(
    *,
    analysis: SongStructureAnalysis,
    segments: tuple[SongPartsPreviewSegment, ...],
) -> tuple[SongPartsPreviewPoint, ...]:
    beat_times = analysis.beat_times_seconds or (0.0,)
    coordinates = analysis.embedding_points_2d or ((0.0, 0.0),)
    novelty = _normalize_series(analysis.novelty_curve, minimum=0.1)
    repetition = _normalize_series(_repetition_scores(analysis), minimum=0.15)
    boundary_times = {round(float(segment.start_seconds), 3) for segment in segments}
    points: list[SongPartsPreviewPoint] = []
    for index, (time_seconds, coordinate) in enumerate(zip(beat_times, coordinates, strict=False)):
        segment_index = _segment_index_for_time(float(time_seconds), segments=segments)
        is_boundary = round(float(time_seconds), 3) in boundary_times
        points.append(
            SongPartsPreviewPoint(
                x=float(coordinate[0]),
                y=float(coordinate[1]),
                time_seconds=float(time_seconds),
                novelty=novelty[index] if index < len(novelty) else 0.1,
                repetition=repetition[index] if index < len(repetition) else 0.15,
                segment_index=segment_index,
                is_boundary=is_boundary,
            )
        )
    if not points:
        points.append(
            SongPartsPreviewPoint(
                x=0.0,
                y=0.0,
                time_seconds=0.0,
                novelty=0.1,
                repetition=0.15,
                segment_index=0,
                is_boundary=True,
            )
        )
    return tuple(points)


def _segment_index_for_time(
    time_seconds: float,
    *,
    segments: tuple[SongPartsPreviewSegment, ...],
) -> int:
    for index, segment in enumerate(segments):
        if segment.start_seconds <= time_seconds < segment.end_seconds:
            return index
    return max(0, len(segments) - 1)


def _repetition_scores(analysis: SongStructureAnalysis) -> tuple[float, ...]:
    scores: list[float] = []
    matrix = analysis.self_similarity_matrix
    for row_index, row in enumerate(matrix):
        peers = [float(value) for index, value in enumerate(row) if abs(index - row_index) > 1]
        if not peers:
            scores.append(0.0)
            continue
        scores.append(max(0.0, min(1.0, max(peers))))
    return tuple(scores)


def _normalize_series(values: tuple[float, ...], *, minimum: float) -> list[float]:
    if not values:
        return [minimum]
    low = min(values)
    high = max(values)
    if abs(high - low) <= 1e-6:
        return [minimum for _value in values]
    return [
        minimum + ((float(value) - float(low)) / float(high - low)) * (1.0 - minimum)
        for value in values
    ]


def _detect_method_label(detect_method: str) -> str:
    if detect_method == MIR_SELF_SIMILARITY_METHOD:
        return "MIR Self-Similarity"
    if detect_method == DETERMINE_SECTIONS_STYLE_METHOD:
        return "Experimental determine_sections-style"
    if detect_method == MFCC_SEQUENCE_POOLING_METHOD:
        return "MFCC Sequence Pooling"
    return detect_method.replace("_", " ").title()
