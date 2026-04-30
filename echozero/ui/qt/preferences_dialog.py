"""Preferences dialog for machine-local EchoZero application settings.
Exists because app-local audio and OSC settings need one bounded reusable editing surface.
Connects AppSettingsService to the neutral settings form and local JSON config editing.
"""

from __future__ import annotations

from collections.abc import Callable

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
    AppSettingsUpdateResult,
    AppSettingsValidationError,
    SettingsPage,
)
from echozero.ui.qt.settings_page_form import SettingsPageForm
from echozero.ui.style.qt import ensure_qt_theme_installed


class PreferencesDialog(QDialog):
    """Modal editor for machine-local EchoZero application settings."""

    def __init__(
        self,
        settings_service: AppSettingsService,
        *,
        on_saved: Callable[[AppSettingsUpdateResult], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("preferencesDialog")
        ensure_qt_theme_installed()
        self._settings_service = settings_service
        self._on_saved = on_saved
        self.resize(720, 620)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._header = QFrame(self)
        self._header.setObjectName("preferencesDialogHeader")
        self._header.setProperty("section", True)
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(14, 14, 14, 14)
        header_layout.setSpacing(6)
        self._eyebrow = QLabel("APPLICATION SETTINGS", self._header)
        self._eyebrow.setObjectName("preferencesDialogEyebrow")
        header_layout.addWidget(self._eyebrow)
        self._title = QLabel(self._header)
        self._title.setObjectName("preferencesDialogTitle")
        self._title.setWordWrap(True)
        header_layout.addWidget(self._title)
        self._summary = QLabel(self._header)
        self._summary.setObjectName("preferencesDialogSummary")
        self._summary.setWordWrap(True)
        header_layout.addWidget(self._summary)
        self._store_path = QLabel(self._header)
        self._store_path.setObjectName("preferencesDialogStorePath")
        self._store_path.setWordWrap(True)
        header_layout.addWidget(self._store_path)
        self._warnings = QLabel(self._header)
        self._warnings.setObjectName("preferencesDialogWarnings")
        self._warnings.setWordWrap(True)
        header_layout.addWidget(self._warnings)
        layout.addWidget(self._header)

        self._form = SettingsPageForm(self)
        layout.addWidget(self._form, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Save,
            self,
        )
        self._buttons.setObjectName("preferencesDialogButtons")
        self._restore_defaults = QPushButton("Restore Defaults", self)
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
        page = self._settings_service.describe()
        page = SettingsPage(
            key=page.key,
            title=page.title,
            summary=page.summary,
            sections=tuple(
                section
                for section in page.sections
                if section.key not in {"osc_receive", "osc_send"}
            ),
            warnings=page.warnings,
        )
        self.setWindowTitle(page.title)
        self._title.setText(page.title)
        self._summary.setText(page.summary)
        self._store_path.setText(f"Stored locally at {self._settings_service.store_path}")
        self._warnings.setVisible(bool(page.warnings))
        self._warnings.setText("\n".join(page.warnings))
        self._form.set_page(
            page,
            empty_message="No application settings are currently available.",
        )

    def _on_restore_defaults(self) -> None:
        self._form.set_values(self._settings_service.default_values())

    def _on_save(self) -> None:
        try:
            result = self._settings_service.apply_updates(self._form.values())
        except AppSettingsValidationError as exc:
            QMessageBox.warning(self, "Invalid Settings", str(exc))
            return
        if self._on_saved is not None:
            try:
                self._on_saved(result)
            except Exception as exc:
                QMessageBox.warning(self, "Apply Settings", str(exc))
                return
        self.accept()

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Missing dialog button for standard button {standard_button!r}")
        return button
