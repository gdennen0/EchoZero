"""
Execution Panel -- compact bottom-dock status and log.

Optimized for horizontal bottom-dock placement: fixed height header bar,
collapsible monospaced log area, and minimal vertical footprint.

Layout (collapsed):
  +-----------------------------------------------------------------+
  | [dot] Status        [======== 35%]         [Log] [Clear] [Stop] |
  +-----------------------------------------------------------------+

Layout (expanded):
  +-----------------------------------------------------------------+
  | [dot] Running: X    [========== 72%]       [Log] [Clear] [Stop] |
  +-----------------------------------------------------------------+
  | [12:34:56] Starting: SeparatorBlock                             |
  | [12:34:57] 72% - Processing stems...                            |
  +-----------------------------------------------------------------+
"""

from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QPlainTextEdit, QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ui.qt_gui.design_system import Colors, border_radius, on_theme_changed


class ExecutionPanel(QWidget):
    """Compact execution status panel for bottom-dock placement."""

    cancel_requested = pyqtSignal()

    _HEADER_H = 30
    _LOG_H = 100

    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_lines: list[str] = []
        self._log_visible = False
        self._state = "idle"
        self._setup_ui()
        self._apply_styles()
        self._on_theme_cb = self._apply_styles
        on_theme_changed(self._on_theme_cb)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- header row ---
        header = QWidget()
        header.setObjectName("ExecHeader")
        header.setFixedHeight(self._HEADER_H)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(10, 0, 6, 0)
        hl.setSpacing(8)

        self._dot = QLabel()
        self._dot.setFixedSize(8, 8)
        hl.addWidget(self._dot)

        self._status = QLabel("Idle")
        self._status.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        hl.addWidget(self._status)

        hl.addSpacing(4)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        self._bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        hl.addWidget(self._bar, stretch=1)

        self._pct = QLabel("")
        self._pct.setFixedWidth(34)
        self._pct.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        hl.addWidget(self._pct)

        hl.addSpacing(8)

        self._btn_log = QPushButton("Log")
        self._btn_log.setObjectName("ExecBtn")
        self._btn_log.setCheckable(True)
        self._btn_log.setFixedHeight(20)
        self._btn_log.clicked.connect(self._on_toggle_log)
        hl.addWidget(self._btn_log)

        self._btn_clear = QPushButton("Clear")
        self._btn_clear.setObjectName("ExecBtn")
        self._btn_clear.setFixedHeight(20)
        self._btn_clear.clicked.connect(self.clear_log)
        hl.addWidget(self._btn_clear)

        self._btn_cancel = QPushButton("Stop")
        self._btn_cancel.setObjectName("ExecCancelBtn")
        self._btn_cancel.setFixedHeight(20)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self.cancel_requested.emit)
        hl.addWidget(self._btn_cancel)

        root.addWidget(header)

        # --- collapsible log ---
        self._log = QPlainTextEdit()
        self._log.setObjectName("ExecLog")
        self._log.setReadOnly(True)
        self._log.setFixedHeight(self._LOG_H)
        self._log.setPlaceholderText("Output appears here during execution.")
        self._log.setVisible(False)
        root.addWidget(self._log)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_running(self, block_name: str):
        self._status.setText(f"Running: {block_name}")
        self._set_state("running")
        self._btn_cancel.setEnabled(True)
        self._bar.setValue(0)
        self._pct.setText("0%")
        if not self._log_visible:
            self._show_log(True)
            self._btn_log.setChecked(True)

    def set_idle(self):
        self._status.setText("Idle")
        self._set_state("idle")
        self._btn_cancel.setEnabled(False)

    def set_error(self, message: str):
        truncated = (message[:80] + "...") if len(message) > 80 else message
        self._status.setText(truncated)
        self._set_state("error")
        self._btn_cancel.setEnabled(False)

    def set_progress(self, value: int):
        v = max(0, min(100, value))
        self._bar.setValue(v)
        self._pct.setText(f"{v}%")

    def append_log(self, line: str, is_error: bool = False):
        ts = datetime.now().strftime("%H:%M:%S")
        stamped = f"[{ts}] {line}"
        self._log_lines.append(stamped)
        self._log.appendPlainText(stamped)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        if is_error:
            self.set_error(line)

    def clear_log(self):
        self._log_lines.clear()
        self._log.clear()

    @property
    def log_lines(self) -> list[str]:
        return list(self._log_lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_toggle_log(self, checked: bool):
        self._show_log(checked)

    def _show_log(self, visible: bool):
        self._log_visible = visible
        self._log.setVisible(visible)

    def _set_state(self, state: str):
        self._state = state
        if state == "running":
            dot_c = Colors.ACCENT_GREEN.name()
            txt_c = Colors.ACCENT_GREEN.name()
            bar_c = Colors.ACCENT_BLUE.name()
        elif state == "error":
            dot_c = Colors.ACCENT_RED.name()
            txt_c = Colors.ACCENT_RED.name()
            bar_c = Colors.ACCENT_RED.name()
        else:
            dot_c = Colors.TEXT_DISABLED.name()
            txt_c = Colors.TEXT_PRIMARY.name()
            bar_c = Colors.ACCENT_BLUE.name()

        self._dot.setStyleSheet(
            f"background-color: {dot_c};"
            f"border-radius: 4px; border: none;"
        )
        self._status.setStyleSheet(
            f"color: {txt_c}; font-weight: bold; font-size: 12px;"
            f"border: none; background: transparent;"
        )
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_DARK.name()};
                border: none;
                border-radius: {border_radius(2)};
            }}
            QProgressBar::chunk {{
                background-color: {bar_c};
                border-radius: {border_radius(2)};
            }}
        """)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_styles(self):
        bg = Colors.BG_MEDIUM.name()
        border = Colors.BORDER.name()
        txt2 = Colors.TEXT_SECONDARY.name()
        txt_d = Colors.TEXT_DISABLED.name()
        bg_dk = Colors.BG_DARK.name()

        self.setStyleSheet(f"""
            ExecutionPanel {{
                background-color: {bg};
                border: none;
            }}
            #ExecHeader {{
                background-color: {bg};
            }}
        """)

        self._pct.setStyleSheet(
            f"color: {txt2}; font-size: 11px;"
            f"border: none; background: transparent;"
        )

        btn_ss = f"""
            QPushButton#ExecBtn {{
                background-color: transparent;
                color: {txt2};
                border: 1px solid {border};
                border-radius: {border_radius(3)};
                padding: 1px 8px;
                font-size: 11px;
            }}
            QPushButton#ExecBtn:hover {{
                background-color: {Colors.HOVER.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QPushButton#ExecBtn:checked {{
                background-color: {Colors.BG_LIGHT.name()};
                color: {Colors.TEXT_PRIMARY.name()};
            }}
            QPushButton#ExecBtn:disabled {{
                color: {txt_d};
                border-color: {Colors.BG_LIGHT.name()};
            }}
        """
        self._btn_log.setStyleSheet(btn_ss)
        self._btn_clear.setStyleSheet(btn_ss)

        cancel_ss = f"""
            QPushButton#ExecCancelBtn {{
                background-color: transparent;
                color: {txt2};
                border: 1px solid {border};
                border-radius: {border_radius(3)};
                padding: 1px 8px;
                font-size: 11px;
            }}
            QPushButton#ExecCancelBtn:hover {{
                background-color: {Colors.ACCENT_RED.name()};
                color: {Colors.TEXT_PRIMARY.name()};
                border-color: {Colors.ACCENT_RED.name()};
            }}
            QPushButton#ExecCancelBtn:disabled {{
                color: {txt_d};
                border-color: {Colors.BG_LIGHT.name()};
            }}
        """
        self._btn_cancel.setStyleSheet(cancel_ss)

        self._log.setStyleSheet(f"""
            QPlainTextEdit#ExecLog {{
                background-color: {bg_dk};
                color: {txt2};
                border: none;
                border-top: 1px solid {border};
                font-family: "SF Mono", "Menlo", "Consolas", monospace;
                font-size: 11px;
                padding: 4px 10px;
            }}
        """)

        self._set_state(self._state)
