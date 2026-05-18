"""Dedicated OSC settings dialog for machine-local EchoZero preferences.
Exists because OSC receive/send setup and health checks need a focused operator surface.
Connects app-settings persistence to reusable OSC status + ping probes in one modal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from echozero.application.settings import (
    AppSettingsService,
    AppSettingsUpdateResult,
    AppSettingsValidationError,
    SettingsPage,
)
from echozero.application.sync.ma3_connection_check import MA3OscLiveBridge
from echozero.ui.qt.osc_settings_panel import OscSettingsPanel
from echozero.ui.qt.settings_page_form import SettingsPageForm
from echozero.ui.style.qt import ensure_qt_theme_installed


class OscSettingsDialog(QDialog):
    """Modal editor for machine-local OSC settings and connection health checks."""

    def __init__(
        self,
        settings_service: AppSettingsService,
        *,
        on_saved: Callable[[AppSettingsUpdateResult], None] | None = None,
        monitor_provider: Callable[[], list[Mapping[str, object]]] | None = None,
        clear_monitor: Callable[[], None] | None = None,
        live_bridge_provider: Callable[[], MA3OscLiveBridge | None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("oscSettingsDialog")
        ensure_qt_theme_installed()
        self._settings_service = settings_service
        self._on_saved = on_saved
        self.resize(900, 640)
        self.setMinimumSize(760, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        self._header = QFrame(self)
        self._header.setObjectName("oscSettingsDialogHeader")
        self._header.setProperty("section", True)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(12)
        header_copy = QVBoxLayout()
        header_copy.setContentsMargins(0, 0, 0, 0)
        header_copy.setSpacing(2)
        self._eyebrow = QLabel("NETWORK CONTROL", self._header)
        self._eyebrow.setObjectName("oscSettingsDialogEyebrow")
        header_copy.addWidget(self._eyebrow)
        self._title = QLabel(self._header)
        self._title.setObjectName("oscSettingsDialogTitle")
        self._title.setWordWrap(True)
        header_copy.addWidget(self._title)
        self._summary = QLabel(self._header)
        self._summary.setObjectName("oscSettingsDialogSummary")
        self._summary.setWordWrap(True)
        header_copy.addWidget(self._summary)
        header_layout.addLayout(header_copy, 1)
        self._store_path = QLabel(self._header)
        self._store_path.setObjectName("oscSettingsDialogStorePath")
        self._store_path.setWordWrap(True)
        self._store_path.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        header_layout.addWidget(self._store_path)
        self._warnings = QLabel(self._header)
        self._warnings.setObjectName("oscSettingsDialogWarnings")
        self._warnings.setWordWrap(True)
        header_layout.addWidget(self._warnings)
        layout.addWidget(self._header)

        self._form = SettingsPageForm(self)
        self._form.field_value_changed.connect(self._on_field_value_changed)
        self._form.setMinimumHeight(128)
        self._form.setMaximumHeight(160)
        self._form.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self._panel = OscSettingsPanel(
            values_provider=self._form.values,
            values_applier=self._form.set_values,
            monitor_provider=monitor_provider,
            clear_monitor=clear_monitor,
            live_bridge_provider=live_bridge_provider,
            parent=self,
        )
        self._panel.setMinimumHeight(275)

        self._splitter = QSplitter(Qt.Orientation.Vertical, self)
        self._splitter.setObjectName("oscSettingsDialogSplitter")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._form)
        self._splitter.addWidget(self._panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setHandleWidth(5)
        self._splitter.setSizes([150, 425])
        layout.addWidget(self._splitter, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Save,
            self,
        )
        self._buttons.setObjectName("oscSettingsDialogButtons")
        self._restore_defaults = QPushButton("Restore OSC Defaults", self)
        self._restore_defaults.setProperty("appearance", "subtle")
        self._restore_defaults.clicked.connect(self._on_restore_defaults)
        self._buttons.addButton(self._restore_defaults, QDialogButtonBox.ButtonRole.ResetRole)
        save_button = self._require_button(QDialogButtonBox.StandardButton.Save)
        save_button.setProperty("appearance", "primary")
        save_button.clicked.connect(self._on_save)
        close_button = self._require_button(QDialogButtonBox.StandardButton.Close)
        close_button.setProperty("appearance", "subtle")
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._render_page()

    def _render_page(self) -> None:
        page = self._osc_settings_page()
        self.setWindowTitle(page.title)
        self._title.setText(page.title)
        self._summary.setText(
            "Set the local listener and MA3 destination, then verify the round trip before saving."
        )
        self._store_path.setText("Local machine settings\napp-settings.json")
        self._store_path.setToolTip(str(self._settings_service.store_path))
        self._warnings.setVisible(bool(page.warnings))
        self._warnings.setText("\n".join(page.warnings))
        self._form.set_page(
            page,
            empty_message="No OSC settings are currently available.",
        )

    def _osc_settings_page(self) -> SettingsPage:
        base_page = self._settings_service.describe()
        osc_sections = tuple(
            replace(section, description="", preferred_columns=3)
            for section in base_page.sections
            if section.key in {"osc_receive", "osc_send"}
        )
        warnings = tuple(
            warning
            for warning in base_page.warnings
            if "OSC" in warning or "config JSON" in warning
        )
        return SettingsPage(
            key="osc_settings",
            title="OSC Settings",
            summary=(
                "Machine-local OSC receive/send endpoints with connection status and ping checks."
            ),
            sections=osc_sections,
            warnings=warnings,
        )

    def _on_restore_defaults(self) -> None:
        defaults = {
            key: value
            for key, value in self._settings_service.default_values().items()
            if key.startswith("osc_")
        }
        self._form.set_values(defaults)
        self._panel.mark_settings_dirty()

    def _on_save(self) -> None:
        try:
            result = self._settings_service.apply_updates(self._form.values())
        except AppSettingsValidationError as exc:
            QMessageBox.warning(self, "Invalid OSC Settings", str(exc))
            return
        if self._on_saved is not None:
            try:
                self._on_saved(result)
            except Exception as exc:
                QMessageBox.warning(self, "Apply OSC Settings", str(exc))
                return
        self.accept()

    def _on_field_value_changed(self, key: str, _value: object) -> None:
        if key.startswith("osc_"):
            self._panel.mark_settings_dirty()

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Missing dialog button for standard button {standard_button!r}")
        return button
