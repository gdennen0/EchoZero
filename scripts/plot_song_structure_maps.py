"""
plot_song_structure_maps: Render MIR structure plots for songs and stems.
Exists because section work needs a visual proof lane for repetition and contrast across the mix.
Loads audio files, runs beat-synchronous MIR analysis,
and saves spectrogram/chroma/similarity plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from echozero.processors.song_structure_mir import analyze_song_structure


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one structure-plot rendering run."""

    parser = argparse.ArgumentParser(
        description="Render MIR structure maps for songs or stems."
    )
    parser.add_argument("inputs", nargs="+", help="Audio file(s) to analyze.")
    parser.add_argument("--output-dir", default="artifacts/song_structure_maps")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--hop-length", type=int, default=1024)
    parser.add_argument("--boundary-sensitivity", type=float, default=0.6)
    parser.add_argument("--min-section-seconds", type=float, default=8.0)
    parser.add_argument("--max-sections", type=int, default=14)
    parser.add_argument("--similarity-threshold", type=float, default=0.84)
    parser.add_argument("--intro-tail-seconds", type=float, default=14.0)
    parser.add_argument("--end-tail-seconds", type=float, default=16.0)
    return parser.parse_args()


def main() -> int:
    """Render structure figures for each requested input file."""

    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in args.inputs:
        resolved_input = Path(input_path).expanduser().resolve()
        analysis = analyze_song_structure(
            file_path=str(resolved_input),
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            boundary_sensitivity=args.boundary_sensitivity,
            min_section_seconds=args.min_section_seconds,
            max_sections=args.max_sections,
            similarity_threshold=args.similarity_threshold,
            intro_tail_seconds=args.intro_tail_seconds,
            end_tail_seconds=args.end_tail_seconds,
        )
        render_structure_figure(
            analysis=analysis,
            input_path=resolved_input,
            output_path=output_dir / f"{resolved_input.stem}_structure_map.png",
        )
        print(build_summary_line(resolved_input, analysis))
    return 0


def render_structure_figure(*, analysis, input_path: Path, output_path: Path) -> None:
    """Render one four-panel MIR analysis figure to disk."""

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Structure plotting requires matplotlib and numpy. "
            "Install with: pip install matplotlib numpy"
        ) from exc

    mel = np.asarray(analysis.mel_spectrogram_db, dtype=np.float32)
    chroma = np.asarray(analysis.chroma_matrix, dtype=np.float32)
    similarity = np.asarray(analysis.self_similarity_matrix, dtype=np.float32)
    novelty = np.asarray(analysis.novelty_curve, dtype=np.float32)
    beat_times = np.asarray(analysis.beat_times_seconds, dtype=np.float32)
    boundary_times = np.asarray(analysis.boundaries_seconds, dtype=np.float32)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle(f"MIR structure map: {input_path.name}", fontsize=14, fontweight="bold")

    mel_axis = axes[0][0]
    mel_image = mel_axis.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    mel_axis.set_title("Mel Spectrogram")
    mel_axis.set_ylabel("Mel bin")
    figure.colorbar(mel_image, ax=mel_axis, fraction=0.046)
    _draw_vertical_boundaries(mel_axis, boundary_times, analysis.duration_seconds)

    chroma_axis = axes[0][1]
    chroma_image = chroma_axis.imshow(chroma, aspect="auto", origin="lower", cmap="cividis")
    chroma_axis.set_title("Chromagram")
    chroma_axis.set_ylabel("Pitch class")
    figure.colorbar(chroma_image, ax=chroma_axis, fraction=0.046)
    _draw_vertical_boundaries(chroma_axis, boundary_times, analysis.duration_seconds)

    similarity_axis = axes[1][0]
    similarity_image = similarity_axis.imshow(
        similarity,
        aspect="equal",
        origin="lower",
        cmap="viridis",
    )
    similarity_axis.set_title("Self-Similarity Matrix")
    similarity_axis.set_xlabel("Beat index")
    similarity_axis.set_ylabel("Beat index")
    figure.colorbar(similarity_image, ax=similarity_axis, fraction=0.046)
    _draw_diagonal_boundaries(similarity_axis, beat_times, boundary_times)

    novelty_axis = axes[1][1]
    novelty_axis.plot(beat_times[: novelty.shape[0]], novelty, color="#1f77b4", linewidth=1.8)
    novelty_axis.set_title("Novelty Curve + Section Boundaries")
    novelty_axis.set_xlabel("Time (s)")
    novelty_axis.set_ylabel("Novelty")
    for boundary_seconds, segment in zip(boundary_times, analysis.segments, strict=True):
        novelty_axis.axvline(boundary_seconds, color="#ff7f0e", linewidth=1.0, alpha=0.8)
        novelty_axis.text(
            boundary_seconds,
            novelty_axis.get_ylim()[1] * 0.92,
            segment.label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _draw_vertical_boundaries(
    axis,
    boundary_times: Sequence[float],
    duration_seconds: float,
) -> None:
    for boundary_seconds in boundary_times:
        x_position = (
            0.0
            if duration_seconds <= 0.0
            else float(boundary_seconds) / float(duration_seconds)
        )
        axis.axvline(x_position * axis.get_xlim()[1], color="white", linewidth=0.8, alpha=0.55)


def _draw_diagonal_boundaries(
    axis,
    beat_times_seconds: Sequence[float],
    boundary_times: Sequence[float],
) -> None:
    if not beat_times_seconds:
        return
    import numpy as np

    beat_times = np.asarray(beat_times_seconds, dtype=np.float32)
    for boundary_seconds in boundary_times:
        beat_index = int(np.searchsorted(beat_times, float(boundary_seconds), side="left"))
        axis.axvline(beat_index, color="white", linewidth=0.7, alpha=0.45)
        axis.axhline(beat_index, color="white", linewidth=0.7, alpha=0.45)


def build_summary_line(input_path: Path, analysis) -> str:
    """Build a compact CLI summary of the detected structure."""

    labels = ", ".join(
        f"{segment.label}@{segment.start_seconds:.1f}s"
        for segment in analysis.segments
    )
    return f"{input_path.name}: {labels}"


if __name__ == "__main__":
    raise SystemExit(main())
