"""Layer routing dialog for timeline audio layers.
Exists to keep output-bus selection in one operator-facing entrypoint.
Connects layer routing settings actions to validated output-bus tokens.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


class LayerRoutingSettingsDialog(QDialog):
    """Dialog that configures one layer output route."""

    def __init__(
        self,
        *,
        layer_title: str,
        playback_output_channels: int,
        current_output_bus: str | None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._channel_count = max(1, min(16, int(playback_output_channels or 2)))
        self.setWindowTitle("Layer Routing Settings")
        self.resize(440, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layer_label = str(layer_title or "").strip() or "Selected Layer"
        intro = QLabel(
            f"Choose audio output routing for '{layer_label}'.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        default_suffix = "1" if self._channel_count == 1 else "1/2"
        self._default_radio = QRadioButton(
            f"Use Default Output ({default_suffix})",
            self,
        )
        self._custom_radio = QRadioButton("Use Custom Channel Range", self)
        layout.addWidget(self._default_radio)
        layout.addWidget(self._custom_radio)

        grid = QGridLayout()
        grid.setContentsMargins(12, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self._start_spin = QSpinBox(self)
        self._start_spin.setRange(1, self._channel_count)
        self._end_spin = QSpinBox(self)
        self._end_spin.setRange(1, self._channel_count)
        grid.addWidget(QLabel("Start Channel", self), 0, 0)
        grid.addWidget(self._start_spin, 0, 1)
        grid.addWidget(QLabel("End Channel", self), 1, 0)
        grid.addWidget(self._end_spin, 1, 1)
        layout.addLayout(grid)

        self._summary = QLabel("", self)
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText("Apply Routing")
        layout.addWidget(self._buttons)

        parsed = self._parse_output_bus(current_output_bus)
        if parsed is None:
            self._default_radio.setChecked(True)
            self._start_spin.setValue(1)
            self._end_spin.setValue(min(2, self._channel_count))
        else:
            self._custom_radio.setChecked(True)
            self._start_spin.setValue(parsed[0])
            self._end_spin.setValue(parsed[1])

        self._default_radio.toggled.connect(self._sync_controls)
        self._custom_radio.toggled.connect(self._sync_controls)
        self._start_spin.valueChanged.connect(self._sync_controls)
        self._end_spin.valueChanged.connect(self._sync_controls)
        self._sync_controls()

    def selected_output_bus(self) -> str | None:
        """Return the selected output bus token, or None for default."""

        if self._default_radio.isChecked():
            return None
        start_channel = int(self._start_spin.value())
        end_channel = int(self._end_spin.value())
        if end_channel < start_channel:
            end_channel = start_channel
        return f"outputs_{start_channel}_{end_channel}"

    def _sync_controls(self) -> None:
        custom_enabled = self._custom_radio.isChecked()
        self._start_spin.setEnabled(custom_enabled)
        self._end_spin.setEnabled(custom_enabled)
        if int(self._end_spin.value()) < int(self._start_spin.value()):
            self._end_spin.setValue(int(self._start_spin.value()))
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        if self._default_radio.isChecked():
            if self._channel_count == 1:
                self._summary.setText("Routing summary: Default output (Output 1).")
            else:
                self._summary.setText("Routing summary: Default output (Outputs 1/2).")
            return
        start_channel = int(self._start_spin.value())
        end_channel = int(self._end_spin.value())
        if end_channel == start_channel:
            self._summary.setText(f"Routing summary: Output {start_channel}.")
            return
        if end_channel == start_channel + 1:
            self._summary.setText(
                f"Routing summary: Outputs {start_channel}/{end_channel}."
            )
            return
        self._summary.setText(f"Routing summary: Outputs {start_channel}-{end_channel}.")

    def _parse_output_bus(self, output_bus: str | None) -> tuple[int, int] | None:
        if not isinstance(output_bus, str):
            return None
        token = output_bus.strip().lower()
        if not token:
            return None
        parts = token.split("_")
        if len(parts) != 3 or parts[0] != "outputs":
            return None
        if not parts[1].isdigit() or not parts[2].isdigit():
            return None
        start_channel = int(parts[1])
        end_channel = int(parts[2])
        if start_channel < 1 or end_channel < start_channel:
            return None
        if start_channel > self._channel_count or end_channel > self._channel_count:
            return None
        return start_channel, end_channel
