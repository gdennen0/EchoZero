from __future__ import annotations

import argparse
from pathlib import Path

from echozero.ui.qt.timeline.real_data_fixture import (
    build_real_data_presentation,
    build_real_data_variants,
)
from echozero.ui.qt.timeline.test_harness import capture_presentation_screenshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture timeline screenshots from real analyzed audio data.")
    parser.add_argument(
        "--audio",
        default=r"C:\Users\griff\Desktop\Doechii_NissanAltima_117bpm_SPMTE_v02 [chan 1].wav",
        help="Path to audio track file (program audio, not timecode).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("artifacts") / "timeline-real-data"),
        help="Output directory for screenshots.",
    )
    parser.add_argument(
        "--working-root",
        default=r"C:\Users\griff\.openclaw\workspace\tmp\ez2-real-data",
        help="Working directory root for transient project DB/audio copy.",
    )
    args = parser.parse_args()

    presentation, summary = build_real_data_presentation(
        audio_path=Path(args.audio),
        working_root=Path(args.working_root),
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, variant in build_real_data_variants(presentation).items():
        path = capture_presentation_screenshot(variant, out / f"timeline_{name}.png")
        print(path)

    print("REAL_DATA_SUMMARY")
    print(f"audio={summary.audio_path}")
    print(f"working_dir={summary.working_dir}")
    print(f"song_version_id={summary.song_version_id}")
    print(f"layers={summary.layer_count}")
    print(f"takes={summary.take_count}")
    print(f"main_events={summary.event_count_main}")


if __name__ == "__main__":
    main()
