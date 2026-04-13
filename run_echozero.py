from __future__ import annotations

import argparse
import sys

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import TimelineWidget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the EchoZero Stage Zero shell.")
    parser.add_argument(
        "--smoke-exit-seconds",
        type=float,
        default=None,
        help="If set to a positive number, close the app after that many seconds.",
    )
    parsed, qt_args = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    demo = build_demo_app()
    widget = TimelineWidget(
        demo.presentation(),
        on_intent=demo.dispatch,
        runtime_audio=demo.runtime_audio,
    )
    widget.resize(1440, 720)
    widget.setWindowTitle("EchoZero")
    widget.show()

    smoke_exit_seconds = parsed.smoke_exit_seconds
    if smoke_exit_seconds is not None and smoke_exit_seconds > 0:
        QTimer.singleShot(int(smoke_exit_seconds * 1000), widget.close)

    try:
        return app.exec()
    finally:
        if demo.runtime_audio is not None:
            demo.runtime_audio.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
