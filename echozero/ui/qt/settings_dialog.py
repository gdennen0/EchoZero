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
    QFrame,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from echozero.application.timeline.object_actions import (
    ApplyCopySource,
    ChangeSessionScope,
    LoadSessionProfile,
    ObjectActionSettingsSession,
    PreviewCopySource,
    ResetSessionDefaults,
    SaveAndRunSession,
    SaveSessionProfile,
    SaveSessionToDefaults,
    SaveSession,
    SetSessionFieldValue,
)
from echozero.ui.qt.action_settings_summary import build_action_confirmation_summary
from echozero.ui.qt.song_parts_preview import SongPartsPreviewPanel
from echozero.ui.qt.settings_form import ActionSettingsForm
from echozero.ui.style.qt import ensure_qt_theme_installed


class ActionSettingsDialog(QDialog):
    """Modal wrapper around the shared action settings form."""

    def __init__(
        self,
        session: ObjectActionSettingsSession,
        *,
        dispatch_command: Callable[[str, object], ObjectActionSettingsSession],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("actionSettingsDialog")
        ensure_qt_theme_installed()
        self._session = session
        self._dispatch_command = dispatch_command
        self.resize(560, 460)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        self._header = QFrame(self)
        self._header.setObjectName("actionSettingsDialogHeader")
        self._header.setProperty("section", True)
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(8, 8, 8, 8)
        header_layout.setSpacing(2)
        self._title = QLabel(self._header)
        self._title.setObjectName("actionSettingsDialogTitle")
        self._title.setWordWrap(False)
        header_layout.addWidget(self._title)
        self._context = QLabel(self._header)
        self._context.setObjectName("actionSettingsDialogContext")
        self._context.setWordWrap(True)
        header_layout.addWidget(self._context)
        self._confirmation = QLabel(self._header)
        self._confirmation.setObjectName("actionSettingsDialogConfirmation")
        self._confirmation.setWordWrap(True)
        self._confirmation.setVisible(False)
        header_layout.addWidget(self._confirmation)
        layout.addWidget(self._header)

        self._scope_group = QGroupBox("Version", self)
        self._scope_group.setProperty("section", True)
        self._scope_group.setProperty("compact", True)
        scope_layout = QGridLayout(self._scope_group)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setHorizontalSpacing(8)
        scope_layout.setVerticalSpacing(2)
        scope_layout.setColumnStretch(1, 1)
        scope_layout.addWidget(QLabel("Edit", self._scope_group), 0, 0)
        self._scope = QComboBox(self._scope_group)
        self._scope.currentIndexChanged.connect(self._on_scope_changed)
        scope_layout.addWidget(self._scope, 0, 1)

        self._copy_group = QGroupBox("Copy", self)
        self._copy_group.setProperty("section", True)
        self._copy_group.setProperty("compact", True)
        copy_layout = QGridLayout(self._copy_group)
        copy_layout.setContentsMargins(0, 0, 0, 0)
        copy_layout.setHorizontalSpacing(8)
        copy_layout.setVerticalSpacing(2)
        copy_layout.setColumnStretch(1, 1)
        copy_layout.addWidget(QLabel("From", self._copy_group), 0, 0)
        self._copy_source = QComboBox(self._copy_group)
        self._copy_source.currentIndexChanged.connect(self._on_copy_source_changed)
        copy_layout.addWidget(self._copy_source, 0, 1)
        self._apply_copy = QPushButton("Apply", self._copy_group)
        self._set_button_appearance(self._apply_copy, "subtle")
        self._apply_copy.clicked.connect(self._on_apply_copy)
        copy_layout.addWidget(self._apply_copy, 0, 2)
        self._copy_preview = QLabel(self._copy_group)
        self._copy_preview.setObjectName("actionSettingsCopyPreview")
        self._copy_preview.setWordWrap(True)
        self._copy_preview.setVisible(False)
        copy_layout.addWidget(self._copy_preview, 1, 0, 1, 3)

        controls_layout = QGridLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(0)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 2)
        controls_layout.addWidget(self._scope_group, 0, 0)
        controls_layout.addWidget(self._copy_group, 0, 1)
        self._profile_group = QGroupBox("Profiles", self)
        self._profile_group.setProperty("section", True)
        self._profile_group.setProperty("compact", True)
        profile_layout = QGridLayout(self._profile_group)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        profile_layout.setHorizontalSpacing(8)
        profile_layout.setVerticalSpacing(2)
        profile_layout.setColumnStretch(1, 1)
        profile_layout.addWidget(QLabel("Saved", self._profile_group), 0, 0)
        self._profile_name = QComboBox(self._profile_group)
        self._profile_name.currentIndexChanged.connect(self._on_profile_selection_changed)
        profile_layout.addWidget(self._profile_name, 0, 1)
        self._load_profile = QPushButton("Load", self._profile_group)
        self._set_button_appearance(self._load_profile, "subtle")
        self._load_profile.clicked.connect(self._on_load_profile)
        profile_layout.addWidget(self._load_profile, 0, 2)
        self._save_profile = QPushButton("Save Profile", self._profile_group)
        self._set_button_appearance(self._save_profile, "subtle")
        self._save_profile.clicked.connect(self._on_save_profile)
        profile_layout.addWidget(self._save_profile, 0, 3)
        controls_layout.addWidget(self._profile_group, 1, 0, 1, 2)
        layout.addLayout(controls_layout)

        self._stage_group = QGroupBox(self)
        self._stage_group.setProperty("section", True)
        settings_layout = QVBoxLayout(self._stage_group)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(6)
        self._form = ActionSettingsForm(self._stage_group)
        self._form.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._form.field_value_changed.connect(self._on_field_value_changed)
        settings_layout.addWidget(self._form)
        self._song_parts_preview = SongPartsPreviewPanel(self)
        layout.addWidget(self._song_parts_preview)
        layout.addWidget(self._stage_group, 1)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close
            | QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Apply,
            self,
        )
        self._buttons.setObjectName("actionSettingsButtons")
        self._save_defaults = QPushButton("Save to Defaults", self)
        self._set_button_appearance(self._save_defaults, "subtle")
        self._save_defaults.clicked.connect(self._on_save_defaults)
        self._buttons.addButton(self._save_defaults, QDialogButtonBox.ButtonRole.ActionRole)
        self._reset_defaults = QPushButton("Reset to Defaults", self)
        self._set_button_appearance(self._reset_defaults, "subtle")
        self._reset_defaults.clicked.connect(self._on_reset_defaults)
        self._buttons.addButton(self._reset_defaults, QDialogButtonBox.ButtonRole.ResetRole)
        save_button = self._require_button(QDialogButtonBox.StandardButton.Save)
        self._set_button_appearance(save_button, "subtle")
        save_button.clicked.connect(self._on_save)
        apply_button = self._require_button(QDialogButtonBox.StandardButton.Apply)
        self._set_button_appearance(apply_button, "primary")
        apply_button.setText("Save And Rerun")
        apply_button.clicked.connect(self._on_run)
        close_button = self._require_button(QDialogButtonBox.StandardButton.Close)
        self._set_button_appearance(close_button, "subtle")
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)
        self._render_session(session)

    def _render_session(self, session: ObjectActionSettingsSession) -> None:
        self._session = session
        self._configure_dialog_for_session(session)
        self.setWindowTitle(self._dialog_title(session))
        self._title.setText(self._header_title_text(session))
        self._form.set_plan(session.plan)
        self._context.setText(self._context_text(session))
        self._sync_confirmation_summary(session)
        self._song_parts_preview.set_session(session)
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
        self._copy_source.addItem("Select...", "")
        for source in session.copy_sources:
            self._copy_source.addItem(source.label, source.source_id)
        selected = session.selected_copy_source_id or ""
        source_index = self._copy_source.findData(selected)
        if source_index >= 0:
            self._copy_source.setCurrentIndex(source_index)
        self._copy_source.blockSignals(False)
        self._profile_name.blockSignals(True)
        self._profile_name.clear()
        self._profile_name.addItem("Select...", "")
        for name in session.profile_names:
            self._profile_name.addItem(name, name)
        self._profile_name.blockSignals(False)
        self._sync_session_controls(session)

    def _sync_session_controls(self, session: ObjectActionSettingsSession) -> None:
        self._copy_group.setTitle(f"Copy to {session.current_scope_state.label}")
        self._stage_group.setTitle(session.plan.title)
        scope_hint = self._scope_hint_text(session)
        self._scope_group.setToolTip(scope_hint)
        self._scope.setToolTip(scope_hint)
        self._copy_group.setVisible(bool(session.copy_sources))
        self._apply_copy.setEnabled(
            bool(session.copy_sources) and bool(self._copy_source.currentData())
        )
        copy_hint = self._copy_hint_text(session)
        self._copy_group.setToolTip(copy_hint)
        self._copy_source.setToolTip(copy_hint)
        self._apply_copy.setToolTip(copy_hint)
        self._profile_group.setVisible(session.can_manage_profiles)
        profile_hint = self._profile_hint_text(session)
        self._profile_group.setToolTip(profile_hint)
        self._profile_name.setToolTip(profile_hint)
        self._load_profile.setToolTip(profile_hint)
        self._save_profile.setToolTip(profile_hint)
        self._load_profile.setEnabled(self._can_load_profile(session))
        self._save_profile.setEnabled(session.can_manage_profiles)
        self._require_button(QDialogButtonBox.StandardButton.Save).setEnabled(session.can_save)
        run_button = self._require_button(QDialogButtonBox.StandardButton.Apply)
        run_button.setEnabled(session.can_save_and_run)
        run_button.setToolTip(session.run_disabled_reason)
        run_button.setText(self._run_button_text(session))
        can_save_defaults = self._can_save_defaults(session)
        self._save_defaults.setEnabled(can_save_defaults)
        self._save_defaults.setToolTip(
            self._save_defaults_hint_text(session, can_save=can_save_defaults)
        )
        can_reset_defaults = self._can_reset_defaults(session)
        self._reset_defaults.setEnabled(can_reset_defaults)
        self._reset_defaults.setToolTip(
            self._reset_hint_text(session, can_reset=can_reset_defaults)
        )
        preview_text = self._copy_preview_text(session)
        self._copy_preview.setVisible(bool(preview_text))
        self._copy_preview.setText(preview_text)

    def _on_field_value_changed(self, key: str, value: object) -> None:
        self._session = self._dispatch_command(
            self._session.session_id,
            SetSessionFieldValue(key, value),
        )
        self._context.setText(self._context_text(self._session))
        self._sync_confirmation_summary(self._session)
        self._sync_session_controls(self._session)

    def _on_scope_changed(self) -> None:
        scope = self._scope.currentData()
        if not scope:
            return
        self._render_session(
            self._dispatch_command(self._session.session_id, ChangeSessionScope(scope))
        )

    def _on_load_profile(self) -> None:
        profile_name = self._selected_profile_name()
        if not profile_name:
            return
        self._render_session(
            self._dispatch_command(
                self._session.session_id,
                LoadSessionProfile(profile_name),
            )
        )

    def _on_save_profile(self) -> None:
        profile_name = self._prompt_profile_name()
        if not profile_name:
            return
        self._render_session(
            self._dispatch_command(
                self._session.session_id,
                SaveSessionProfile(profile_name),
            )
        )

    def _on_copy_source_changed(self) -> None:
        self._apply_copy.setEnabled(bool(self._copy_source.currentData()))
        source_id = self._copy_source.currentData()
        if not source_id:
            self._copy_preview.setVisible(False)
            self._copy_preview.setText("")
            hint_text = self._copy_hint_text(self._session)
            self._copy_group.setToolTip(hint_text)
            self._copy_source.setToolTip(hint_text)
            self._apply_copy.setToolTip(hint_text)
            return
        self._render_session(
            self._dispatch_command(self._session.session_id, PreviewCopySource(source_id))
        )

    def _on_apply_copy(self) -> None:
        source_id = self._copy_source.currentData()
        if not source_id:
            return
        self._render_session(
            self._dispatch_command(self._session.session_id, ApplyCopySource(source_id))
        )

    def _on_save(self) -> None:
        self._dispatch_command(self._session.session_id, SaveSession())
        self.accept()

    def _on_save_defaults(self) -> None:
        self._render_session(
            self._dispatch_command(self._session.session_id, SaveSessionToDefaults())
        )

    def _on_reset_defaults(self) -> None:
        self._render_session(
            self._dispatch_command(self._session.session_id, ResetSessionDefaults())
        )

    def _on_run(self) -> None:
        self._dispatch_command(self._session.session_id, SaveAndRunSession())
        self.accept()

    def _on_profile_selection_changed(self) -> None:
        self._load_profile.setEnabled(self._can_load_profile(self._session))

    def _can_load_profile(self, session: ObjectActionSettingsSession) -> bool:
        name = self._selected_profile_name()
        return bool(name) and name in session.profile_names

    def _selected_profile_name(self) -> str:
        return str(self._profile_name.currentData() or "").strip()

    def _prompt_profile_name(self) -> str | None:
        initial_name = self._selected_profile_name()
        profile_name, accepted = QInputDialog.getText(
            self,
            "Save Pipeline Profile",
            "Profile name:",
            text=initial_name,
        )
        name = str(profile_name).strip()
        if not accepted or not name:
            return None
        return name

    def _require_button(self, standard_button: QDialogButtonBox.StandardButton) -> QPushButton:
        button = self._buttons.button(standard_button)
        if button is None:
            raise RuntimeError(f"Missing dialog button for standard button {standard_button!r}")
        return button

    def _sync_confirmation_summary(self, session: ObjectActionSettingsSession) -> None:
        summary = build_action_confirmation_summary(session)
        self._confirmation.setVisible(bool(summary))
        self._confirmation.setText(summary)

    @staticmethod
    def _set_button_appearance(button: QPushButton, appearance: str) -> None:
        button.setProperty("appearance", appearance)
        style = button.style()
        if style is not None:
            style.unpolish(button)
            style.polish(button)
        button.update()

    @staticmethod
    def _dialog_title(session: ObjectActionSettingsSession) -> str:
        if session.action_id == "timeline.extract_song_sections":
            return "Confirm Song Parts Detection"
        return f"Pipeline Settings · {session.plan.title}"

    def _configure_dialog_for_session(self, session: ObjectActionSettingsSession) -> None:
        if session.action_id in {
            "timeline.extract_classified_drums",
            "timeline.extract_song_drum_events",
        }:
            self.resize(max(self.width(), 920), max(self.height(), 760))
            self.setMinimumWidth(820)
            return
        if session.action_id == "timeline.extract_song_sections":
            self.resize(max(self.width(), 980), max(self.height(), 860))
            self.setMinimumWidth(900)
            return
        self.setMinimumWidth(520)

    @staticmethod
    def _header_title_text(session: ObjectActionSettingsSession) -> str:
        if session.action_id in {
            "timeline.extract_classified_drums",
            "timeline.extract_song_drum_events",
        }:
            return "Extract Drum Events Setup"
        if session.action_id == "timeline.extract_song_sections":
            return "Confirm Song Parts Detection"
        return "Pipeline Settings"

    @staticmethod
    def _run_button_text(session: ObjectActionSettingsSession) -> str:
        if session.action_id in {
            "timeline.extract_classified_drums",
            "timeline.extract_song_drum_events",
        }:
            return "Confirm And Extract Drums"
        if session.action_id == "timeline.extract_song_sections":
            return "Confirm Method And Run"
        return "Save And Rerun"

    @staticmethod
    def _context_text(session: ObjectActionSettingsSession) -> str:
        target_summary = session.plan.summary or session.plan.object_id or session.plan.object_type
        status = "Unsaved changes" if session.has_unsaved_changes else "Up to date"
        parts = [
            session.plan.title,
            session.current_scope_state.label,
            f"Target: {target_summary}",
        ]
        detect_method_label = _selected_option_label(session, key="detect_method")
        if session.action_id == "timeline.extract_song_sections" and detect_method_label:
            parts.append(f"Method: {detect_method_label}")
        parts.append(status)
        return " · ".join(parts)

    @staticmethod
    def _scope_hint_text(session: ObjectActionSettingsSession) -> str:
        if session.scope == "app_default":
            return (
                "Application Default edits the machine-local baseline every new song starts from. "
                "Open a song version when you want to run this stage."
            )
        if session.scope == "song_default":
            return (
                "Song Default edits the baseline recipe for this song. "
                "Switch back to This Version when you want to rerun this stage."
            )
        return (
            "This Version is the live copy for the current version. "
            "Save here to rerun this stage on what you are editing now."
        )

    @staticmethod
    def _can_save_defaults(session: ObjectActionSettingsSession) -> bool:
        return (
            session.default_save_scope is not None
            and session.scope != session.default_save_scope
        )

    @staticmethod
    def _save_defaults_hint_text(
        session: ObjectActionSettingsSession,
        *,
        can_save: bool,
    ) -> str:
        if session.default_save_scope is None:
            return "Saving to defaults is unavailable in this runtime."
        if can_save:
            return f"Save current stage values into {session.default_save_label.lower()}."
        return f"You are already editing {session.default_save_label.lower()}."

    @staticmethod
    def _can_reset_defaults(session: ObjectActionSettingsSession) -> bool:
        fields = (*session.plan.editable_fields, *session.plan.advanced_fields)
        return any(field.value != field.default_value for field in fields)

    @staticmethod
    def _reset_hint_text(
        session: ObjectActionSettingsSession,
        *,
        can_reset: bool,
    ) -> str:
        if not (*session.plan.editable_fields, *session.plan.advanced_fields):
            return "This stage has no editable settings."
        if can_reset:
            return "Reset all stage settings in this scope to template defaults."
        return "All settings in this scope already match template defaults."

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

    def _profile_hint_text(self, session: ObjectActionSettingsSession) -> str:
        if not session.can_manage_profiles:
            return ""
        name = self._selected_profile_name()
        if not name:
            return "Save the current stage values as a machine-local pipeline profile."
        if name in session.profile_names:
            return "Load or overwrite this saved machine-local pipeline profile."
        return "Save the current stage values as a new machine-local pipeline profile."

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
        preview_limit = 4
        lines.extend(
            f"{key.replace('_', ' ').title()}: {before} -> {after}"
            for key, before, after in preview.changes[:preview_limit]
        )
        if count > preview_limit:
            lines.append(f"...and {count - preview_limit} more settings.")
        return "\n".join(lines)


def _selected_option_label(session: ObjectActionSettingsSession, *, key: str) -> str:
    value = session.values.get(key)
    for field in (*session.plan.editable_fields, *session.plan.advanced_fields):
        if field.key != key:
            continue
        for option in field.options:
            if option.value == value:
                return option.label
        break
    return str(value or "").strip()
