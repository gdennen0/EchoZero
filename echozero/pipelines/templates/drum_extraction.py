"""Shared builders for classified drum extraction pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from echozero.pipelines.block_specs import AudioFilter, BinaryDrumClassify, DetectOnsets
from echozero.pipelines.pipeline import BlockHandle, Pipeline, PortRef


@dataclass(frozen=True)
class ClassifiedDrumBranchSettings:
    """Resolved settings for the shared per-label drum extraction subgraph."""

    target_labels: tuple[str, ...] = ("kick", "snare")
    kick_model_path: str = ""
    snare_model_path: str = ""
    clap_model_path: str = ""
    cymbal_model_path: str = ""
    device: str = "auto"
    positive_threshold: float = 0.60
    kick_positive_threshold: float = 0.50
    snare_positive_threshold: float = 0.65
    clap_positive_threshold: float = 0.60
    cymbal_positive_threshold: float = 0.60
    kick_min_event_peak: float = 0.0010
    kick_min_event_rms: float = 0.0002
    kick_min_separation_ms: float = 80.0
    snare_min_event_peak: float = 0.0010
    snare_min_event_rms: float = 0.0002
    snare_min_separation_ms: float = 50.0
    clap_min_event_peak: float = 0.0015
    clap_min_event_rms: float = 0.0003
    clap_min_separation_ms: float = 55.0
    cymbal_min_event_peak: float = 0.0008
    cymbal_min_event_rms: float = 0.00015
    cymbal_min_separation_ms: float = 90.0
    kick_filter_enabled: bool = True
    kick_filter_freq: float = 180.0
    kick_onset_threshold: float = 0.150
    snare_filter_enabled: bool = True
    snare_filter_freq: float = 180.0
    snare_onset_threshold: float = 0.150
    clap_filter_enabled: bool = True
    clap_filter_freq: float = 1_200.0
    clap_onset_threshold: float = 0.35
    cymbal_filter_enabled: bool = True
    cymbal_filter_freq: float = 3_200.0
    cymbal_onset_threshold: float = 0.40
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
    clap_filter_type: str = "bandpass"
    clap_filter_q: float = 1.2
    clap_onset_min_gap: float = 0.055
    clap_onset_method: str = "hfc"
    clap_onset_backtrack: bool = True
    clap_onset_timing_offset_ms: float = 0.0
    cymbal_filter_type: str = "highpass"
    cymbal_filter_q: float = 1.0
    cymbal_onset_min_gap: float = 0.09
    cymbal_onset_method: str = "hfc"
    cymbal_onset_backtrack: bool = True
    cymbal_onset_timing_offset_ms: float = 0.0
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
    clap_filter = pipeline.add(
        AudioFilter(
            enabled=settings.clap_filter_enabled,
            filter_type=settings.clap_filter_type,
            freq=settings.clap_filter_freq,
            Q=settings.clap_filter_q,
        ),
        id="clap_filter",
        audio_in=audio_in,
    )
    clap_onsets = pipeline.add(
        DetectOnsets(
            threshold=settings.clap_onset_threshold,
            min_gap=settings.clap_onset_min_gap,
            method=settings.clap_onset_method,
            backtrack=settings.clap_onset_backtrack,
            timing_offset_ms=settings.clap_onset_timing_offset_ms,
        ),
        id="clap_onsets",
        audio_in=clap_filter.audio_out,
    )
    cymbal_filter = pipeline.add(
        AudioFilter(
            enabled=settings.cymbal_filter_enabled,
            filter_type=settings.cymbal_filter_type,
            freq=settings.cymbal_filter_freq,
            Q=settings.cymbal_filter_q,
        ),
        id="cymbal_filter",
        audio_in=audio_in,
    )
    cymbal_onsets = pipeline.add(
        DetectOnsets(
            threshold=settings.cymbal_onset_threshold,
            min_gap=settings.cymbal_onset_min_gap,
            method=settings.cymbal_onset_method,
            backtrack=settings.cymbal_onset_backtrack,
            timing_offset_ms=settings.cymbal_onset_timing_offset_ms,
        ),
        id="cymbal_onsets",
        audio_in=cymbal_filter.audio_out,
    )
    return pipeline.add(
        BinaryDrumClassify(
            target_labels=settings.target_labels,
            kick_model_path=settings.kick_model_path,
            snare_model_path=settings.snare_model_path,
            clap_model_path=settings.clap_model_path,
            cymbal_model_path=settings.cymbal_model_path,
            device=settings.device,
            positive_threshold=settings.positive_threshold,
            kick_positive_threshold=settings.kick_positive_threshold,
            snare_positive_threshold=settings.snare_positive_threshold,
            clap_positive_threshold=settings.clap_positive_threshold,
            cymbal_positive_threshold=settings.cymbal_positive_threshold,
            kick_min_event_peak=settings.kick_min_event_peak,
            kick_min_event_rms=settings.kick_min_event_rms,
            kick_min_separation_ms=settings.kick_min_separation_ms,
            snare_min_event_peak=settings.snare_min_event_peak,
            snare_min_event_rms=settings.snare_min_event_rms,
            snare_min_separation_ms=settings.snare_min_separation_ms,
            clap_min_event_peak=settings.clap_min_event_peak,
            clap_min_event_rms=settings.clap_min_event_rms,
            clap_min_separation_ms=settings.clap_min_separation_ms,
            cymbal_min_event_peak=settings.cymbal_min_event_peak,
            cymbal_min_event_rms=settings.cymbal_min_event_rms,
            cymbal_min_separation_ms=settings.cymbal_min_separation_ms,
            assignment_mode=settings.assignment_mode,
            winner_margin=settings.winner_margin,
            event_match_window_ms=settings.event_match_window_ms,
        ),
        id="classify_drums",
        audio_in=audio_in,
        events_in=kick_onsets.events_out,
        kick_events_in=kick_onsets.events_out,
        snare_events_in=snare_onsets.events_out,
        clap_audio_in=clap_filter.audio_out,
        clap_events_in=clap_onsets.events_out,
        cymbal_audio_in=cymbal_filter.audio_out,
        cymbal_events_in=cymbal_onsets.events_out,
    )
