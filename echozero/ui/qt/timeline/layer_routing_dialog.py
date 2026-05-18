"""Layer routing dialog for timeline audio layers.
Exists to keep output-bus selection in one operator-facing entrypoint.
Connects layer routing settings actions to validated output-bus tokens.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from echozero.output_routing import (
    canonical_layer_output_bus,
    output_bus_label,
    output_bus_options,
)

from echozero.ui.style.qt import ensure_qt_theme_installed


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
        ensure_qt_theme_installed()
        self._channel_count = max(1, min(16, int(playback_output_channels or 2)))
        self._route_buttons: list[tuple[str, QRadioButton]] = []
        self.setWindowTitle("Layer Routing Settings")
        self.resize(440, 320)

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

        self._default_radio = QRadioButton(
            "Use Master Output",
            self,
        )
        layout.addWidget(self._default_radio)

        route_intro = QLabel("Route this layer to one physical output:", self)
        route_intro.setWordWrap(True)
        layout.addWidget(route_intro)

        for route in output_bus_options(self._channel_count, include_stereo_pairs=True):
            button = QRadioButton(route.label, self)
            button.toggled.connect(self._sync_controls)
            self._route_buttons.append((route.token, button))
            layout.addWidget(button)

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

        selected_output_bus = canonical_layer_output_bus(
            current_output_bus,
            max_channel=self._channel_count,
            reject_invalid=False,
        )
        selected = False
        if selected_output_bus is not None:
            for token, button in self._route_buttons:
                if token == selected_output_bus:
                    button.setChecked(True)
                    selected = True
                    break
        if not selected:
            self._default_radio.setChecked(True)

        self._default_radio.toggled.connect(self._sync_controls)
        self._sync_controls()

    def selected_output_bus(self) -> str | None:
        """Return selected output bus token, or None for default."""

        if self._default_radio.isChecked():
            return None
        for token, button in self._route_buttons:
            if button.isChecked():
                return token
        return None

    def _sync_controls(self) -> None:
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(
                self._default_radio.isChecked() or self.selected_output_bus() is not None
            )
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        selected_output_bus = self.selected_output_bus()
        if selected_output_bus is None:
            self._summary.setText("Routing summary: Master/default output.")
            return
        self._summary.setText(
            f"Routing summary: {output_bus_label(selected_output_bus)}."
        )
