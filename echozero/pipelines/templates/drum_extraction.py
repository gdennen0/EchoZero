"""Shared builders for classified drum extraction pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from echozero.pipelines.block_specs import AudioFilter, BinaryDrumClassify, DetectOnsets
from echozero.pipelines.pipeline import BlockHandle, Pipeline, PortRef


@dataclass(frozen=True)
class ClassifiedDrumBranchSettings:
    """Resolved settings for the shared kick/snare extraction subgraph."""

    kick_model_path: str = ""
    snare_model_path: str = ""
    device: str = "auto"
    kick_positive_threshold: float = 0.50
    snare_positive_threshold: float = 0.65
    kick_filter_enabled: bool = True
    kick_filter_freq: float = 180.0
    kick_onset_threshold: float = 0.150
    snare_filter_enabled: bool = True
    snare_filter_freq: float = 180.0
    snare_onset_threshold: float = 0.150
    kick_filter_type: str = "lowpass"
    kick_onset_min_gap: float = 0.08
    kick_onset_method: str = "default"
    kick_onset_backtrack: bool = True
    kick_onset_timing_offset_ms: float = 0.0
    snare_filter_type: str = "highpass"
    snare_onset_min_gap: float = 0.05
    snare_onset_method: str = "default"
    snare_onset_backtrack: bool = True
    snare_onset_timing_offset_ms: float = 0.0
    assignment_mode: str = "independent"
    winner_margin: float = 0.05
    event_match_window_ms: float = 40.0


def add_classified_drum_branches(
    pipeline: Pipeline,
    *,
    audio_in: PortRef,
    settings: ClassifiedDrumBranchSettings,
) -> BlockHandle:
    """Add the canonical filter -> onset -> binary classify drum subgraph."""

    kick_filter = pipeline.add(
        AudioFilter(
            enabled=settings.kick_filter_enabled,
            filter_type=settings.kick_filter_type,
            freq=settings.kick_filter_freq,
        ),
        id="kick_filter",
        audio_in=audio_in,
    )
    kick_onsets = pipeline.add(
        DetectOnsets(
            threshold=settings.kick_onset_threshold,
            min_gap=settings.kick_onset_min_gap,
            method=settings.kick_onset_method,
            backtrack=settings.kick_onset_backtrack,
            timing_offset_ms=settings.kick_onset_timing_offset_ms,
        ),
        id="kick_onsets",
        audio_in=kick_filter.audio_out,
    )
    snare_filter = pipeline.add(
        AudioFilter(
            enabled=settings.snare_filter_enabled,
            filter_type=settings.snare_filter_type,
            freq=settings.snare_filter_freq,
        ),
        id="snare_filter",
        audio_in=audio_in,
    )
    snare_onsets = pipeline.add(
        DetectOnsets(
            threshold=settings.snare_onset_threshold,
            min_gap=settings.snare_onset_min_gap,
            method=settings.snare_onset_method,
            backtrack=settings.snare_onset_backtrack,
            timing_offset_ms=settings.snare_onset_timing_offset_ms,
        ),
        id="snare_onsets",
        audio_in=snare_filter.audio_out,
    )
    return pipeline.add(
        BinaryDrumClassify(
            kick_model_path=settings.kick_model_path,
            snare_model_path=settings.snare_model_path,
            device=settings.device,
            kick_positive_threshold=settings.kick_positive_threshold,
            snare_positive_threshold=settings.snare_positive_threshold,
            assignment_mode=settings.assignment_mode,
            winner_margin=settings.winner_margin,
            event_match_window_ms=settings.event_match_window_ms,
        ),
        id="classify_drums",
        audio_in=audio_in,
        events_in=kick_onsets.events_out,
        kick_events_in=kick_onsets.events_out,
        snare_events_in=snare_onsets.events_out,
    )
