"""
prototype.py - Main entry point for EchoZero 2 Timeline Prototype
Run with: python prototype.py
Requires PyQt6
"""
from __future__ import annotations

import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QFrame

from model import generate_fake_data
from canvas import TimelineCanvas
from ruler import TimeRuler
from layers_panel import LayersPanel

DARK_STYLE = """
QWidget {
    background-color: #16161a;
    color: #bebecf;
    font-family: "Segoe UI";
    font-size: 11px;
}
QScrollBar:vertical {
    background: #1e1e26;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #3c3c52;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar:horizontal {
    background: #1e1e26;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #3c3c52;
    border-radius: 4px;
    min-width: 20px;
}
"""


class TimelinePrototype(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EchoZero 2 \u2014 Timeline Prototype")
        self.resize(1400, 800)

        self.state = generate_fake_data(num_events=500, num_layers=10, duration=300.0)

        self.canvas = TimelineCanvas(self.state, self)
        self.ruler = TimeRuler(self.state, self)
        self.layers_panel = LayersPanel(self.state, self)

        # Spacer in top-left corner (above layers panel, left of ruler)
        ruler_spacer = QFrame(self)
        ruler_spacer.setFixedWidth(self.layers_panel.width())
        ruler_spacer.setFixedHeight(self.ruler.height())
        ruler_spacer.setStyleSheet(
            "background-color: #12121a; border-right: 1px solid #32323c;"
        )

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(ruler_spacer,      0, 0)
        layout.addWidget(self.ruler,        0, 1)
        layout.addWidget(self.layers_panel, 1, 0)
        layout.addWidget(self.canvas,       1, 1)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)
        self.setLayout(layout)

        # Ruler click -> sync playhead -> repaint canvas
        self.ruler.playhead_changed.connect(self._on_playhead_changed)

        # Patch canvas.update so ruler + layers panel stay in sync
        _orig_update = self.canvas.update

        def _synced_update(*args):
            _orig_update(*args)
            self.ruler.update()
            self.layers_panel.sync_scroll(self.state.scroll_y)

        self.canvas.update = _synced_update

        # Playhead advances in real-time
        self._last_ns: int = time.monotonic_ns()
        self._playhead_timer = QTimer(self)
        self._playhead_timer.setInterval(16)
        self._playhead_timer.timeout.connect(self._advance_playhead)
        self._playhead_timer.start()

    def _advance_playhead(self):
        now_ns = time.monotonic_ns()
        dt = (now_ns - self._last_ns) / 1e9
        self._last_ns = now_ns
        self.state.playhead_time += dt
        if self.state.events:
            end = max(e.time + e.duration for e in self.state.events)
            if self.state.playhead_time > end:
                self.state.playhead_time = 0.0
        self.canvas.update()
        self.ruler.update()

    def _on_playhead_changed(self, t: float):
        self.state.playhead_time = t
        self.canvas.update()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    win = TimelinePrototype()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
