"""Dialog host for reusable action settings forms.
Exists because settings editing needs one bounded surface with clear scope and copy semantics.
Connects reusable settings form rendering to save, copy-preview, and rerun dialog actions.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ChangeSessionScope,
    ObjectActionSettingsSession,
    PreviewCopySource,
    SaveAndRunSession,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.ui.qt.settings_form import ActionSettingsForm
from echozero.ui.style.qt.qss import build_foundry_shell_qss


class ActionSettingsDialog(QDialog):
    """Modal wrapper around the shared action settings form."""

    def __init__(
        self,
        session: ObjectActionSettingsSession,
        *,
        dispatch_command: Callable[[str, object], ObjectActionSettingsSession],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._dispatch_command = dispatch_command
        self.setStyleSheet(build_foundry_shell_qss())
        self.resize(640, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._context = QLabel(self)
        self._context.setWordWrap(True)
        layout.addWidget(self._context)

        scope_group = QGroupBox("Editing Scope", self)
        scope_layout = QGridLayout(scope_group)
        scope_layout.addWidget(QLabel("Scope", scope_group), 0, 0)
        self._scope = QComboBox(scope_group)
        self._scope.currentIndexChanged.connect(self._on_scope_changed)
        scope_layout.addWidget(self._scope, 0, 1)
        self._scope_hint = QLabel(scope_group)
        self._scope_hint.setWordWrap(True)
        scope_layout.addWidget(self._scope_hint, 1, 0, 1, 2)
        layout.addWidget(scope_group)

        self._copy_group = QGroupBox(self)
        copy_layout = QGridLayout(self._copy_group)
        copy_layout.addWidget(QLabel("Source", self._copy_group), 0, 0)
        self._copy_source = QComboBox(self._copy_group)
        self._copy_source.currentIndexChanged.connect(self._on_copy_source_changed)
        copy_layout.addWidget(self._copy_source, 0, 1)
        self._apply_copy = QPushButton("Use Source Values", self._copy_group)
        self._apply_copy.clicked.connect(self._on_apply_copy)
        copy_layout.addWidget(self._apply_copy, 0, 2)
        self._copy_hint = QLabel(self._copy_group)
        self._copy_hint.setWordWrap(True)
        copy_layout.addWidget(self._copy_hint, 1, 0, 1, 3)
        self._copy_preview = QLabel(self._copy_group)
        self._copy_preview.setWordWrap(True)
        copy_layout.addWidget(self._copy_preview, 2, 0, 1, 3)
        layout.addWidget(self._copy_group)

        self._stage_group = QGroupBox(self)
        settings_layout = QVBoxLayout(self._stage_group)
        self._form = ActionSettingsForm(self._stage_group)
        self._form.field_value_changed.connect(self._on_field_value_changed)
        settings_layout.addWidget(self._form)
        layout.addWidget(self._stage_group, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close
            | QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._on_save)
        self._buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Save And Rerun")
        self._buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_run)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._render_session(session)

    def _render_session(self, session: ObjectActionSettingsSession) -> None:
        self._session = session
        self.setWindowTitle(self._dialog_title(session))
        self._form.set_plan(session.plan)
        self._context.setText(self._context_text(session))
        self._scope.blockSignals(True)
        self._scope.clear()
        for choice in session.scope_choices:
            self._scope.addItem(choice.label, choice.scope)
        index = self._scope.findData(session.scope)
        if index >= 0:
            self._scope.setCurrentIndex(index)
        self._scope.blockSignals(False)
        self._scope.setEnabled(len(session.scope_choices) > 1)

        self._copy_source.blockSignals(True)
        self._copy_source.clear()
        self._copy_source.addItem("Choose Source", "")
        for source in session.copy_sources:
            self._copy_source.addItem(source.label, source.source_id)
        selected = session.selected_copy_source_id or ""
        source_index = self._copy_source.findData(selected)
        if source_index >= 0:
            self._copy_source.setCurrentIndex(source_index)
        self._copy_source.blockSignals(False)
        self._sync_session_controls(session)

    def _sync_session_controls(self, session: ObjectActionSettingsSession) -> None:
        self._copy_group.setTitle(f"Copy Settings Into {session.current_scope_state.label}")
        self._stage_group.setTitle(f"Stage Settings: {session.plan.title}")
        self._scope_hint.setText(self._scope_hint_text(session))
        self._copy_group.setVisible(bool(session.copy_sources))
        self._apply_copy.setEnabled(bool(session.copy_sources) and bool(self._copy_source.currentData()))
        self._copy_hint.setText(self._copy_hint_text(session))
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(session.can_save)
        run_button = self._buttons.button(QDialogButtonBox.StandardButton.Apply)
        run_button.setEnabled(session.can_save_and_run)
        run_button.setToolTip(session.run_disabled_reason)
        self._copy_preview.setText(self._copy_preview_text(session))

    def _on_field_value_changed(self, key: str, value: object) -> None:
        self._session = self._dispatch_command(
            self._session.session_id,
            SetSessionFieldValue(key, value),
        )
        self._context.setText(self._context_text(self._session))
        self._sync_session_controls(self._session)

    def _on_scope_changed(self) -> None:
        scope = self._scope.currentData()
        if not scope:
            return
        self._render_session(self._dispatch_command(self._session.session_id, ChangeSessionScope(scope)))

    def _on_copy_source_changed(self) -> None:
        self._apply_copy.setEnabled(bool(self._copy_source.currentData()))
        source_id = self._copy_source.currentData()
        if not source_id:
            self._copy_preview.setText("")
            self._copy_hint.setText(self._copy_hint_text(self._session))
            return
        self._render_session(self._dispatch_command(self._session.session_id, PreviewCopySource(source_id)))

    def _on_apply_copy(self) -> None:
        source_id = self._copy_source.currentData()
        if not source_id:
            return
        self._render_session(self._dispatch_command(self._session.session_id, ApplyCopySource(source_id)))

    def _on_save(self) -> None:
        self._dispatch_command(self._session.session_id, SaveSession())
        self.accept()

    def _on_run(self) -> None:
        self._dispatch_command(self._session.session_id, SaveAndRunSession())
        self.accept()

    @staticmethod
    def _dialog_title(session: ObjectActionSettingsSession) -> str:
        return f"Pipeline Settings: {session.current_scope_state.label} · {session.plan.title}"

    @staticmethod
    def _context_text(session: ObjectActionSettingsSession) -> str:
        target_summary = session.plan.summary or session.plan.object_id or session.plan.object_type
        status = (
            "Unsaved changes are ready to save."
            if session.has_unsaved_changes
            else "Stored settings are up to date."
        )
        return "\n".join(
            (
                f"Stage: {session.plan.title}",
                f"Target: {target_summary}",
                f"Scope: {session.current_scope_state.label}",
                status,
            )
        )

    @staticmethod
    def _scope_hint_text(session: ObjectActionSettingsSession) -> str:
        if session.scope == "song_default":
            return (
                "Song Default edits the baseline recipe for this song. "
                "Switch back to This Version when you want to rerun this stage."
            )
        return (
            "This Version is the live copy for the current version. "
            "Save here to rerun this stage on what you are editing now."
        )

    def _copy_hint_text(self, session: ObjectActionSettingsSession) -> str:
        if not session.copy_sources:
            return ""
        source_id = self._copy_source.currentData()
        if not source_id:
            return "Choose a saved source to preview what would change in this scope."
        source = next((item for item in session.copy_sources if item.source_id == source_id), None)
        if source is None:
            return "Choose a saved source to preview what would change in this scope."
        return source.description or "Preview copy impact before applying it."

    @staticmethod
    def _copy_preview_text(session: ObjectActionSettingsSession) -> str:
        preview = session.copy_preview
        if preview is None:
            return ""
        if not preview.changes:
            return f"{preview.summary}\nNo settings would change."
        count = len(preview.changes)
        noun = "setting" if count == 1 else "settings"
        lines = [f"{preview.summary} · {count} {noun} will change"]
        lines.extend(
            f"{key.replace('_', ' ').title()}: {before} -> {after}"
            for key, before, after in preview.changes
        )
        return "\n".join(lines)
