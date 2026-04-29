"""Dedicated OSC settings dialog for machine-local EchoZero preferences.
Exists because OSC receive/send setup and health checks need a focused operator surface.
Connects app-settings persistence to reusable OSC status + ping probes in one modal.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from echozero.application.settings import (
    AppSettingsService,
    AppSettingsValidationError,
    SettingsPage,
)
from echozero.ui.qt.osc_settings_panel import OscSettingsPanel
from echozero.ui.qt.settings_page_form import SettingsPageForm
from echozero.ui.style.qt import ensure_qt_theme_installed


class OscSettingsDialog(QDialog):
    """Modal editor for machine-local OSC settings and connection health checks."""

    def __init__(
        self,
        settings_service: AppSettingsService,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("oscSettingsDialog")
        ensure_qt_theme_installed()
        self._settings_service = settings_service
        self.resize(700, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._header = QFrame(self)
        self._header.setObjectName("oscSettingsDialogHeader")
        self._header.setProperty("section", True)
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(14, 14, 14, 14)
        header_layout.setSpacing(6)
        self._eyebrow = QLabel("OSC SETTINGS", self._header)
        self._eyebrow.setObjectName("oscSettingsDialogEyebrow")
        header_layout.addWidget(self._eyebrow)
        self._title = QLabel(self._header)
        self._title.setObjectName("oscSettingsDialogTitle")
        self._title.setWordWrap(True)
        header_layout.addWidget(self._title)
        self._summary = QLabel(self._header)
        self._summary.setObjectName("oscSettingsDialogSummary")
        self._summary.setWordWrap(True)
        header_layout.addWidget(self._summary)
        self._store_path = QLabel(self._header)
        self._store_path.setObjectName("oscSettingsDialogStorePath")
        self._store_path.setWordWrap(True)
        header_layout.addWidget(self._store_path)
        self._warnings = QLabel(self._header)
        self._warnings.setObjectName("oscSettingsDialogWarnings")
        self._warnings.setWordWrap(True)
        header_layout.addWidget(self._warnings)
        layout.addWidget(self._header)

        self._form = SettingsPageForm(self)
        self._form.field_value_changed.connect(self._on_field_value_changed)
        layout.addWidget(self._form, 1)

        self._panel = OscSettingsPanel(values_provider=self._form.values, parent=self)
        layout.addWidget(self._panel)

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
        self._summary.setText(page.summary)
        self._store_path.setText(f"Stored locally at {self._settings_service.store_path}")
        self._warnings.setVisible(bool(page.warnings))
        self._warnings.setText("\n".join(page.warnings))
        self._form.set_page(
            page,
            empty_message="No OSC settings are currently available.",
        )

    def _osc_settings_page(self) -> SettingsPage:
        base_page = self._settings_service.describe()
        osc_sections = tuple(
            section
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

        if result.restart_required:
            QMessageBox.information(
                self,
                "Restart Required",
                "\n".join(result.restart_reasons),
            )
        self.accept()

    def _on_field_value_changed(self, key: str, _value: object) -> None:
        if key.startswith("osc_"):
            self._panel.mark_settings_dirty()

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Missing dialog button for standard button {standard_button!r}")
        return button
