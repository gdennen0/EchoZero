from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from echozero.application.timeline.intents import Pause, Play, Seek, SelectTake, ToggleTakeSelector
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.widget import TimelineWidget


def main() -> int:
    app = QApplication(sys.argv)
    demo = build_demo_app()
    widget = TimelineWidget(demo.presentation(), on_intent=demo.dispatch)
    widget.resize(1400, 420)
    widget.show()

    presentation = demo.presentation()
    layer = next((candidate for candidate in presentation.layers if candidate.takes), presentation.layers[0])
    alt_take = layer.takes[1] if len(layer.takes) > 1 else layer.takes[0]

    actions: list[tuple[int, callable]] = [
        (600, lambda: widget.set_presentation(demo.dispatch(ToggleTakeSelector(layer.id)))),
        (1600, lambda: widget.set_presentation(demo.dispatch(SelectTake(layer.id, alt_take.id)))),
        (2600, lambda: widget.set_presentation(demo.dispatch(Seek(3.4)))),
        (3600, lambda: widget.set_presentation(demo.dispatch(Play()))),
        (5200, lambda: widget.set_presentation(demo.dispatch(Pause()))),
        (6500, lambda: widget.set_presentation(demo.dispatch(ToggleTakeSelector(layer.id)))),
        (7600, app.quit),
    ]

    for delay_ms, fn in actions:
        QTimer.singleShot(delay_ms, fn)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
