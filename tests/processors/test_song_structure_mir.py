"""
song_structure_mir tests: repetition-aware structure helpers.
Exists because MIR structure analysis needs direct proof for novelty and boundary selection logic.
Verifies the self-similarity novelty kernel and spacing-aware boundary picking.
"""

from __future__ import annotations

import numpy as np

from echozero.processors.song_structure_mir import (
    _align_beat_frames_to_feature_count,
    _align_feature_frame_counts,
    _compute_checkerboard_novelty,
    _pick_boundary_indices,
    _project_feature_vectors_2d,
)


def test_checkerboard_novelty_peaks_at_structure_change() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.85],
            [0.1, 0.1, 0.85, 1.0],
        ],
        dtype=np.float32,
    )

    novelty = _compute_checkerboard_novelty(similarity, radius=1)

    assert float(novelty[2]) > float(novelty[1])
    assert float(np.max(novelty)) <= 1.0


def test_pick_boundary_indices_enforces_min_gap() -> None:
    novelty = np.asarray([0.0, 0.8, 0.75, 0.2, 0.9, 0.1], dtype=np.float32)

    boundaries = _pick_boundary_indices(
        novelty,
        sensitivity=0.6,
        min_gap_frames=2,
        max_sections=3,
    )

    assert boundaries[0] == 0
    assert 4 in boundaries
    assert 2 not in boundaries
    assert len(boundaries) <= 3


def test_align_feature_frame_counts_trims_to_shared_width() -> None:
    chroma = np.ones((12, 11227), dtype=np.float32)
    mfcc = np.ones((20, 1404), dtype=np.float32)
    tempogram = np.ones((32, 1404), dtype=np.float32)

    aligned_chroma, aligned_mfcc, aligned_tempogram = _align_feature_frame_counts(
        chroma,
        mfcc,
        tempogram,
    )

    assert aligned_chroma.shape == (12, 1404)
    assert aligned_mfcc.shape == (20, 1404)
    assert aligned_tempogram.shape == (32, 1404)


def test_align_beat_frames_to_feature_count_clips_and_anchors_endpoints() -> None:
    beat_frames = np.asarray([3, 15, 28, 400], dtype=np.int32)

    aligned = _align_beat_frames_to_feature_count(
        beat_frames=beat_frames,
        feature_frame_count=30,
    )

    assert aligned.tolist() == [0, 3, 15, 28, 29]


def test_project_feature_vectors_2d_returns_normalized_coordinates() -> None:
    feature_vectors = np.asarray(
        [
            [0.0, 1.0, 0.5],
            [1.0, 0.5, 0.25],
            [2.0, 0.25, 0.0],
        ],
        dtype=np.float32,
    )

    projected = _project_feature_vectors_2d(feature_vectors)

    assert projected.shape == (3, 2)
    assert np.max(np.abs(projected)) <= 1.0
