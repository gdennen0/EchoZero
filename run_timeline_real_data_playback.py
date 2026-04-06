from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from echozero.ui.qt.timeline.demo_app import build_real_data_demo_app
from echozero.ui.qt.timeline.widget import TimelineWidget


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the realtime Stage Zero timeline with real audio playback.")
    parser.add_argument(
        "--audio",
        default=r"C:\Users\griff\Desktop\Doechii_NissanAltima_117bpm_SPMTE_v02 [chan 1].wav",
        help="Path to the source audio file.",
    )
    parser.add_argument(
        "--working-root",
        default=str(Path("artifacts") / "timeline-real-data-runtime"),
        help="Working directory root for transient project data.",
    )
    parser.add_argument(
        "--song-title",
        default="Doechii Nissan Altima",
        help="Song title shown in the timeline.",
    )
    args = parser.parse_args()

    qt_app = QApplication(sys.argv)
    demo, summary = build_real_data_demo_app(
        audio_path=Path(args.audio),
        working_root=Path(args.working_root),
        song_title=args.song_title,
    )
    widget = TimelineWidget(
        demo.presentation(),
        on_intent=demo.dispatch,
        runtime_audio=demo.runtime_audio,
    )
    widget.resize(1440, 720)
    widget.show()

    print("REALTIME_REAL_DATA_SUMMARY")
    print(f"audio={summary.audio_path}")
    print(f"working_dir={summary.working_dir}")
    print(f"song_version_id={summary.song_version_id}")
    print(f"layers={summary.layer_count}")
    print(f"takes={summary.take_count}")
    print(f"main_events={summary.event_count_main}")

    try:
        return qt_app.exec()
    finally:
        if demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
