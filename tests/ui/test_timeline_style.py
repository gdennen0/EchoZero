from dataclasses import replace

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QImage, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from echozero.application.settings import SettingsFieldSurface
from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContextSection,
    InspectorContract,
    InspectorFactRow,
    InspectorSection,
    build_timeline_inspector_contract,
)
from echozero.application.timeline.object_actions import (
    ObjectActionSessionFieldValue,
    ObjectActionSettingField,
    ObjectActionSettingOption,
    ObjectActionSettingsPlan,
    ObjectActionSettingsScopeState,
    ObjectActionSettingsSession,
    ResetSessionDefaults,
    SaveSessionToDefaults,
    SetSessionFieldValue,
)
from echozero.ui.FEEL import (
    TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX,
    TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX,
    TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX,
    TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX,
    TIMELINE_TRANSPORT_HEIGHT_PX,
)
from echozero.ui.qt.pipeline_settings_browser_dialog import PipelineSettingsBrowserDialog
from echozero.ui.qt.settings_dialog import ActionSettingsDialog
from echozero.ui.qt.settings_form import ActionSettingsForm
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock
from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.ruler import RulerBlock, visible_ruler_marks
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
from echozero.ui.qt.timeline.blocks.transport_bar import TransportLayout
from echozero.ui.qt.timeline.blocks.transport_bar_block import TransportBarBlock
from echozero.ui.qt.timeline.blocks.waveform_lane import WaveformLaneBlock
from echozero.ui.qt.timeline.demo_app import build_demo_app
from echozero.ui.qt.timeline.style import (
    TIMELINE_STYLE,
    build_object_palette_stylesheet,
    build_timeline_scroll_area_stylesheet,
    fixture_color,
    fixture_take_action_label,
)
from echozero.ui.qt.timeline.widget import ObjectInfoPanel, TimelineEditorModeBar, TimelineWidget
from echozero.ui.style.qt import ensure_qt_theme_installed
from echozero.ui.style.qt.qss import build_echozero_app_qss
from tests.ui.timeline_shell_shared_support import _song_switching_presentation


def test_object_palette_stylesheet_uses_shared_tokens():
    stylesheet = build_object_palette_stylesheet()

    assert TIMELINE_STYLE.object_palette.background_hex in stylesheet
    assert TIMELINE_STYLE.object_palette.border_hex in stylesheet
    assert TIMELINE_STYLE.object_palette.title_object_name in stylesheet
    assert TIMELINE_STYLE.object_palette.body_object_name in stylesheet


def test_object_info_panel_layout_comes_from_style_module():
    app = QApplication.instance() or QApplication([])
    panel = ObjectInfoPanel()
    try:
        margins = panel.layout().contentsMargins()
        style = TIMELINE_STYLE.object_palette

        assert panel.objectName() == style.frame_object_name
        assert margins.left() == style.content_padding.left
        assert margins.top() == style.content_padding.top
        assert margins.right() == style.content_padding.right
        assert margins.bottom() == style.content_padding.bottom
        assert panel.layout().spacing() == style.section_spacing_px
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_collapses_to_compact_rail():
    app = QApplication.instance() or QApplication([])
    panel = ObjectInfoPanel()
    try:
        assert panel.is_collapsed is False
        assert panel.target_width() == panel.expanded_width
        assert panel._title.isHidden() is False
        assert panel._collapse_button.text() == ">"

        panel.toggle_collapsed()
        app.processEvents()

        assert panel.is_collapsed is True
        assert panel.target_width() == 28
        assert panel.minimumWidth() == 28
        assert panel.maximumWidth() == 28
        assert panel._title.isHidden() is True
        assert panel._content_splitter.isHidden() is True
        assert panel._collapse_button.text() == "<"
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_uses_sectioned_summary_and_hidden_playback_surface_by_default():
    app = QApplication.instance() or QApplication([])
    panel = ObjectInfoPanel()
    try:
        assert panel._selection_card.property("section") is True
        assert panel._event_preview_card.property("section") is True
        assert panel._layer_controls.property("section") is True
        assert panel._kind.objectName() == "timeline_object_info_kind"
        assert panel._event_preview_card.isHidden()
        assert panel._layer_controls.isHidden()
        assert panel._content_splitter.handleWidth() == TIMELINE_OBJECT_INFO_SPLITTER_HANDLE_PX
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_keeps_metadata_compact_and_scrollable():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    try:
        rows = tuple(
            InspectorFactRow(label=f"field {index}", value="value " * 8) for index in range(12)
        )
        contract = InspectorContract(
            title="Dense Layer",
            sections=(InspectorSection(section_id="details", label="Details", rows=rows),),
        )

        panel.resize(panel.maximumWidth(), 320)
        panel.show()
        panel.set_contract(demo.presentation(), contract)
        app.processEvents()

        sizes = panel._content_splitter.sizes()

        assert panel._content_splitter.orientation() == Qt.Orientation.Vertical
        assert panel._body.isReadOnly()
        assert panel._body.minimumHeight() == TIMELINE_OBJECT_INFO_METADATA_MIN_HEIGHT_PX
        assert panel._body.verticalScrollBar().maximum() > 0
        assert sizes[0] >= TIMELINE_OBJECT_INFO_METADATA_DEFAULT_HEIGHT_PX
        assert sizes[0] < sizes[1]
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_renders_contract_actions_without_hard_wired_buttons():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    try:
        contract = build_timeline_inspector_contract(demo.presentation())
        panel.set_contract(demo.presentation(), contract)

        assert "song.add" not in panel._action_buttons
        assert panel._action_buttons
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_renders_pipeline_action_settings_rows():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    try:
        presentation = demo.presentation()
        contract = InspectorContract(
            title="Layer Source Audio",
            context_sections=(
                InspectorContextSection(
                    section_id="pipelines",
                    label="Pipelines",
                    actions=(
                        InspectorAction(
                            action_id="timeline.extract_stems",
                            label="Extract Stems",
                        ),
                    ),
                ),
            ),
        )
        panel.set_contract(presentation, contract)
        panel.set_action_settings_plans(
            (
                ObjectActionSettingsPlan(
                    action_id="timeline.extract_stems",
                    title="Extract Stems",
                    object_id="source_audio",
                    object_type="layer",
                    pipeline_template_id="stem_separation",
                    editable_fields=(
                        ObjectActionSettingField(
                            key="mode",
                            label="Mode",
                            value="merge",
                            default_value="merge",
                            widget="dropdown",
                            options=(
                                ObjectActionSettingOption(value="merge", label="Merge"),
                                ObjectActionSettingOption(value="overwrite", label="Overwrite"),
                            ),
                        ),
                    ),
                    locked_bindings=(("target", "timeline"),),
                    run_label="Run",
                ),
            )
        )

        assert "timeline.extract_stems" in panel._pipeline_action_rows
        assert panel._pipeline_action_rows["timeline.extract_stems"].objectName() == (
            "timeline_object_info_action_row"
        )
        assert panel._settings_buttons["timeline.extract_stems"].property("appearance") == "subtle"
        assert panel._settings_buttons["timeline.extract_stems"].text() == "..."
        assert panel._settings_buttons["timeline.extract_stems"].toolTip() == "Open Settings"
        assert panel._action_buttons["timeline.extract_stems"].property("appearance") == "primary"
        assert panel._action_buttons["timeline.extract_stems"].text() == "Extract Stems"
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_labels_confirmation_pipeline_runs_as_review():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    try:
        presentation = demo.presentation()
        contract = InspectorContract(
            title="Layer Drums",
            context_sections=(
                InspectorContextSection(
                    section_id="pipelines",
                    label="Pipelines",
                    actions=(
                        InspectorAction(
                            action_id="timeline.extract_classified_drums",
                            label="Extract Classified Drums",
                            requires_settings_confirmation=True,
                        ),
                    ),
                ),
            ),
        )
        panel.set_contract(presentation, contract)
        panel.set_action_settings_plans(
            (
                ObjectActionSettingsPlan(
                    action_id="timeline.extract_classified_drums",
                    title="Extract Classified Drums",
                    object_id="drums",
                    object_type="layer",
                    pipeline_template_id="extract_classified_drums",
                    run_label="Review & Run",
                    requires_settings_confirmation=True,
                ),
            )
        )

        button = panel._action_buttons["timeline.extract_classified_drums"]
        assert button.text() == "Review"
        assert button.toolTip() == "Review & Run"
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_scrolls_pipeline_rows_instead_of_compressing_them():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    try:
        actions = tuple(
            InspectorAction(
                action_id=f"timeline.synthetic_pipeline_{index}",
                label=f"Synthetic Pipeline {index}",
            )
            for index in range(12)
        )
        contract = InspectorContract(
            title="Layer Source Audio",
            context_sections=(
                InspectorContextSection(
                    section_id="synthetic-pipelines",
                    label="Pipelines",
                    actions=actions,
                ),
            ),
        )
        plans = tuple(
            ObjectActionSettingsPlan(
                action_id=action.action_id,
                title=action.label,
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id=action.action_id,
                editable_fields=(
                    ObjectActionSettingField(
                        key="threshold",
                        label="Threshold",
                        value=0.5,
                        default_value=0.5,
                        widget="number",
                    ),
                ),
                locked_bindings=(("target", "source_audio"),),
                summary="Long pipeline summary that should keep each action row readable.",
                run_label="Run",
            )
            for action in actions
        )

        panel.resize(panel.maximumWidth(), 240)
        panel.show()
        panel.set_contract(demo.presentation(), contract)
        panel.set_action_settings_plans(plans)
        app.processEvents()

        assert panel._actions_scroll.widget() is panel._action_sections
        assert panel._actions_scroll.verticalScrollBar().maximum() > 0
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_emits_pipeline_action_with_saved_settings():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    emitted = []
    try:
        presentation = demo.presentation()
        contract = InspectorContract(
            title="Layer Source Audio",
            context_sections=(
                InspectorContextSection(
                    section_id="pipelines",
                    label="Pipelines",
                    actions=(
                        InspectorAction(
                            action_id="timeline.extract_stems",
                            label="Extract Stems",
                        ),
                    ),
                ),
            ),
        )
        panel.set_contract(presentation, contract)
        panel.set_action_settings_plans(
            (
                ObjectActionSettingsPlan(
                    action_id="timeline.extract_stems",
                    title="Extract Stems",
                    object_id="source_audio",
                    object_type="layer",
                    pipeline_template_id="stem_separation",
                    editable_fields=(
                        ObjectActionSettingField(
                            key="mode",
                            label="Mode",
                            value="merge",
                            default_value="merge",
                            widget="dropdown",
                            options=(
                                ObjectActionSettingOption(value="merge", label="Merge"),
                                ObjectActionSettingOption(value="overwrite", label="Overwrite"),
                            ),
                        ),
                    ),
                    run_label="Run",
                ),
            )
        )
        panel.action_requested.connect(emitted.append)

        panel._action_buttons["timeline.extract_stems"].click()

        assert len(emitted) == 1
        assert emitted[0].action_id == "timeline.extract_stems"
    finally:
        panel.close()
        app.processEvents()


def test_object_info_panel_emits_settings_request_from_pipeline_row():
    app = QApplication.instance() or QApplication([])
    demo = build_demo_app()
    panel = ObjectInfoPanel()
    emitted = []
    try:
        presentation = demo.presentation()
        contract = InspectorContract(
            title="Layer Source Audio",
            context_sections=(
                InspectorContextSection(
                    section_id="pipelines",
                    label="Pipelines",
                    actions=(
                        InspectorAction(
                            action_id="timeline.extract_stems",
                            label="Extract Stems",
                        ),
                    ),
                ),
            ),
        )
        panel.set_contract(presentation, contract)
        panel.set_action_settings_plans(
            (
                ObjectActionSettingsPlan(
                    action_id="timeline.extract_stems",
                    title="Extract Stems",
                    object_id="source_audio",
                    object_type="layer",
                    pipeline_template_id="stem_separation",
                    editable_fields=(
                        ObjectActionSettingField(
                            key="mode",
                            label="Mode",
                            value="merge",
                            default_value="merge",
                            widget="dropdown",
                            options=(
                                ObjectActionSettingOption(value="merge", label="Merge"),
                                ObjectActionSettingOption(value="overwrite", label="Overwrite"),
                            ),
                        ),
                    ),
                    run_label="Run",
                ),
            )
        )
        panel.settings_requested.connect(emitted.append)

        panel._settings_buttons["timeline.extract_stems"].click()

        assert len(emitted) == 1
        assert emitted[0].action_id == "timeline.extract_stems"
    finally:
        panel.close()
        app.processEvents()


def test_action_settings_dialog_makes_scope_explicit_in_title_and_copy_target():
    app = QApplication.instance() or QApplication([])
    session = ObjectActionSettingsSession(
        session_id="session_1",
        action_id="timeline.extract_stems",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="stem_separation",
            editable_fields=(
                ObjectActionSettingField(
                    key="model",
                    label="Model",
                    value="mdx_extra",
                    default_value="mdx_extra",
                ),
            ),
            summary="Source Audio · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="mdx_extra",
                        draft_value="mdx_extra_q",
                    ),
                ),
                can_run=True,
            ),
            ObjectActionSettingsScopeState(
                scope="song_default",
                label="Song Default",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="mdx_extra",
                        draft_value="mdx_extra",
                    ),
                ),
                can_run=False,
            ),
        ),
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, _command: session,
    )
    try:
        assert dialog.objectName() == "actionSettingsDialog"
        assert dialog.windowTitle() == "Pipeline Settings · Extract Stems"
        assert dialog._header.property("section") is True
        assert dialog._title.text() == "Pipeline Settings"
        assert dialog._copy_group.title() == "Copy to This Version"
        assert dialog._stage_group.title() == "Extract Stems"
        assert "This Version" in dialog._context.text()
        assert dialog._buttons.button(QDialogButtonBox.StandardButton.Save).property(
            "appearance"
        ) == ("subtle")
        assert dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).property(
            "appearance"
        ) == ("primary")
        assert (
            dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).text()
            == "Save And Rerun"
        )
    finally:
        dialog.close()
        app.processEvents()


def test_action_settings_dialog_uses_bounded_section_surfaces():
    app = QApplication.instance() or QApplication([])
    session = ObjectActionSettingsSession(
        session_id="session_2",
        action_id="timeline.extract_stems",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="stem_separation",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(scope="version", label="This Version"),
            ObjectActionSettingsScopeState(scope="song_default", label="Song Default"),
        ),
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, _command: session,
    )
    try:
        assert dialog._scope_group.property("section") is True
        assert dialog._scope_group.property("compact") is True
        assert dialog._copy_group.property("section") is True
        assert dialog._copy_group.property("compact") is True
        assert dialog._stage_group.property("section") is True
        assert dialog._apply_copy.property("appearance") == "subtle"
        assert dialog._save_defaults.property("appearance") == "subtle"
        assert dialog._copy_preview.objectName() == "actionSettingsCopyPreview"
        assert dialog._reset_defaults.isEnabled() is False
    finally:
        dialog.close()
        app.processEvents()


def test_action_settings_dialog_song_parts_shows_preview_panel():
    app = QApplication.instance() or QApplication([])
    session = ObjectActionSettingsSession(
        session_id="session_song_parts",
        action_id="timeline.extract_song_sections",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_song_sections",
            title="Detect Song Parts",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="extract_song_sections",
            summary="Source Audio · This Version",
            locked_bindings=(("audio_file", "/tmp/source.wav"),),
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="detect_method",
                        persisted_value="mir_self_similarity",
                        draft_value="mir_self_similarity",
                    ),
                ),
                can_run=True,
            ),
        ),
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, _command: session,
    )
    try:
        assert dialog.windowTitle() == "Confirm Song Parts Detection"
        assert dialog._song_parts_preview.isHidden() is False
        assert (
            dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).text()
            == "Confirm Method And Run"
        )
    finally:
        dialog.close()
        app.processEvents()


def test_action_settings_dialog_drum_confirmation_summary_stays_visible():
    app = QApplication.instance() or QApplication([])
    session = ObjectActionSettingsSession(
        session_id="session_drum_confirm",
        action_id="timeline.extract_song_drum_events",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_song_drum_events",
            title="Extract Drum Events",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="extract_song_drum_events",
            summary="Source Audio · This Version",
            requires_settings_confirmation=True,
            editable_fields=(
                ObjectActionSettingField(
                    key="target_drum_labels",
                    label="Drum Outputs",
                    value=("kick", "snare"),
                    default_value=("kick", "snare"),
                    widget="checkbox_group",
                    options=(
                        ObjectActionSettingOption(value="kick", label="Kick"),
                        ObjectActionSettingOption(value="snare", label="Snare"),
                    ),
                ),
                ObjectActionSettingField(
                    key="sensitivity_preset",
                    label="Sensitivity",
                    value="balanced",
                    default_value="balanced",
                    widget="dropdown",
                    options=(
                        ObjectActionSettingOption(value="balanced", label="Balanced"),
                    ),
                ),
                ObjectActionSettingField(
                    key="kick_model_path",
                    label="Kick Model",
                    value="/models/kick.manifest.json",
                    default_value="",
                    widget="dropdown",
                    options=(
                        ObjectActionSettingOption(
                            value="/models/kick.manifest.json",
                            label="Kick Bundle",
                            metadata={"status": "ready"},
                        ),
                    ),
                ),
            ),
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="target_drum_labels",
                        persisted_value=("kick", "snare"),
                        draft_value=("kick", "snare"),
                    ),
                    ObjectActionSessionFieldValue(
                        key="sensitivity_preset",
                        persisted_value="balanced",
                        draft_value="balanced",
                    ),
                    ObjectActionSessionFieldValue(
                        key="kick_model_path",
                        persisted_value="/models/kick.manifest.json",
                        draft_value="/models/kick.manifest.json",
                    ),
                ),
                can_run=True,
            ),
        ),
        can_save_and_run=True,
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, _command: session,
    )
    try:
        confirmation = dialog._confirmation.text()
        assert dialog._confirmation.isHidden() is False
        assert "Confirm before run" in confirmation
        assert "Event types: Kick, Snare" in confirmation
        assert "Sensitivity: Balanced" in confirmation
        assert "Models: 1 ready" in confirmation
        assert (
            dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).text()
            == "Confirm And Extract Drums"
        )
    finally:
        dialog.close()
        app.processEvents()


def test_action_settings_dialog_reset_defaults_dispatches_session_command():
    app = QApplication.instance() or QApplication([])
    dispatched: list[object] = []
    session = ObjectActionSettingsSession(
        session_id="session_reset",
        action_id="timeline.extract_stems",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="stem_separation",
            editable_fields=(
                ObjectActionSettingField(
                    key="model",
                    label="Model",
                    value="mdx_extra_q",
                    default_value="latest_model",
                    persisted_value="mdx_extra_q",
                    is_dirty=False,
                ),
            ),
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="mdx_extra_q",
                        draft_value="mdx_extra_q",
                    ),
                ),
                can_run=True,
            ),
        ),
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, command: (dispatched.append(command) or session),
    )
    try:
        assert dialog._reset_defaults.property("appearance") == "subtle"
        assert dialog._reset_defaults.text() == "Reset to Defaults"
        assert dialog._reset_defaults.isEnabled() is True

        dialog._reset_defaults.click()

        assert len(dispatched) == 1
        assert isinstance(dispatched[0], ResetSessionDefaults)
    finally:
        dialog.close()
        app.processEvents()


def test_action_settings_dialog_save_to_defaults_dispatches_session_command():
    app = QApplication.instance() or QApplication([])
    dispatched: list[object] = []
    session = ObjectActionSettingsSession(
        session_id="session_save_defaults",
        action_id="timeline.extract_stems",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="stem_separation",
            editable_fields=(
                ObjectActionSettingField(
                    key="model",
                    label="Model",
                    value="mdx_extra",
                    default_value="latest_model",
                    persisted_value="latest_model",
                    is_dirty=True,
                ),
            ),
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="latest_model",
                        draft_value="mdx_extra",
                    ),
                ),
                can_run=True,
            ),
            ObjectActionSettingsScopeState(
                scope="song_default",
                label="Song Default",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="latest_model",
                        draft_value="latest_model",
                    ),
                ),
                can_run=False,
            ),
        ),
    )
    dialog = ActionSettingsDialog(
        session,
        dispatch_command=lambda _session_id, command: (dispatched.append(command) or session),
    )
    try:
        assert dialog._save_defaults.text() == "Save to Defaults"
        assert dialog._save_defaults.isEnabled() is True

        dialog._save_defaults.click()

        assert len(dispatched) == 1
        assert isinstance(dispatched[0], SaveSessionToDefaults)
    finally:
        dialog.close()
        app.processEvents()


def test_pipeline_settings_browser_dialog_renders_stages_and_disables_run_without_target():
    app = QApplication.instance() or QApplication([])
    model_field = ObjectActionSettingField(
        key="model",
        label="Model",
        value="latest_model",
        default_value="latest_model",
        persisted_value="latest_model",
    )
    field_values = (
        ObjectActionSessionFieldValue(
            key="model",
            persisted_value="latest_model",
            draft_value="latest_model",
        ),
    )
    stems_session = ObjectActionSettingsSession(
        session_id="pipeline_stems",
        action_id="timeline.extract_stems",
        object_id="",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="",
            object_type="layer",
            pipeline_template_id="stem_separation",
            editable_fields=(model_field,),
            summary="No target layer selected · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=field_values,
                can_run=False,
            ),
        ),
        can_save=True,
        can_save_and_run=False,
        run_disabled_reason="Select a target layer before running this stage.",
    )
    drums_session = ObjectActionSettingsSession(
        session_id="pipeline_drums",
        action_id="timeline.extract_drum_events",
        object_id="",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_drum_events",
            title="Extract Onsets",
            object_id="",
            object_type="layer",
            pipeline_template_id="onset_detection",
            editable_fields=(model_field,),
            summary="No target layer selected · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=field_values,
                can_run=False,
            ),
        ),
        can_save=True,
        can_save_and_run=False,
        run_disabled_reason="Select a target layer before running this stage.",
    )

    dialog = PipelineSettingsBrowserDialog(
        (stems_session, drums_session),
        dispatch_command=lambda session_id, _command: (
            stems_session if session_id == stems_session.session_id else drums_session
        ),
    )
    try:
        assert dialog._action_list.count() == 2
        assert dialog._stage_group.title() == "Extract Stems"
        assert dialog._require_button(QDialogButtonBox.StandardButton.Apply).isEnabled() is False

        dialog._action_list.setCurrentRow(1)
        app.processEvents()

        assert dialog._stage_group.title() == "Extract Onsets"
    finally:
        dialog.close()
        app.processEvents()


def test_pipeline_settings_browser_dialog_honors_initial_action_selection():
    app = QApplication.instance() or QApplication([])
    model_field = ObjectActionSettingField(
        key="model",
        label="Model",
        value="latest_model",
        default_value="latest_model",
        persisted_value="latest_model",
    )
    field_values = (
        ObjectActionSessionFieldValue(
            key="model",
            persisted_value="latest_model",
            draft_value="latest_model",
        ),
    )
    stems_session = ObjectActionSettingsSession(
        session_id="pipeline_stems",
        action_id="timeline.extract_stems",
        object_id="",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_stems",
            title="Extract Stems",
            object_id="",
            object_type="layer",
            pipeline_template_id="stem_separation",
            editable_fields=(model_field,),
            summary="No target layer selected · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=field_values,
                can_run=False,
            ),
        ),
        can_save=True,
        can_save_and_run=False,
        run_disabled_reason="Select a target layer before running this stage.",
    )
    drums_session = ObjectActionSettingsSession(
        session_id="pipeline_drums",
        action_id="timeline.extract_drum_events",
        object_id="",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_drum_events",
            title="Extract Onsets",
            object_id="",
            object_type="layer",
            pipeline_template_id="onset_detection",
            editable_fields=(model_field,),
            summary="No target layer selected · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=field_values,
                can_run=False,
            ),
        ),
        can_save=True,
        can_save_and_run=False,
        run_disabled_reason="Select a target layer before running this stage.",
    )

    dialog = PipelineSettingsBrowserDialog(
        (stems_session, drums_session),
        dispatch_command=lambda session_id, _command: (
            stems_session if session_id == stems_session.session_id else drums_session
        ),
        initial_action_id="timeline.extract_drum_events",
    )
    try:
        assert dialog._stage_group.title() == "Extract Onsets"
    finally:
        dialog.close()
        app.processEvents()


def test_pipeline_settings_browser_dialog_song_parts_shows_preview_panel():
    app = QApplication.instance() or QApplication([])
    field_values = (
        ObjectActionSessionFieldValue(
            key="detect_method",
            persisted_value="mir_self_similarity",
            draft_value="mir_self_similarity",
        ),
    )
    session = ObjectActionSettingsSession(
        session_id="pipeline_song_parts",
        action_id="timeline.extract_song_sections",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_song_sections",
            title="Detect Song Parts",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="extract_song_sections",
            summary="Source Audio · This Version",
            locked_bindings=(("audio_file", "/tmp/source.wav"),),
            requires_settings_confirmation=True,
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=field_values,
                can_run=True,
            ),
        ),
        can_save=True,
        can_save_and_run=True,
    )
    dialog = PipelineSettingsBrowserDialog(
        (session,),
        dispatch_command=lambda _session_id, _command: session,
    )
    try:
        assert dialog._song_parts_preview.isHidden() is False
        assert dialog._title.text() == "Confirm Song Parts Detection"
        assert dialog._require_button(QDialogButtonBox.StandardButton.Apply).text() == (
            "Confirm Method And Run"
        )
    finally:
        dialog.close()
        app.processEvents()


def test_pipeline_settings_browser_dialog_field_edit_does_not_rebuild_form():
    app = QApplication.instance() or QApplication([])
    dispatched: list[object] = []
    threshold_field = ObjectActionSettingField(
        key="threshold",
        label="Threshold",
        value=0.3,
        default_value=0.3,
        persisted_value=0.3,
        widget="number",
    )
    session = ObjectActionSettingsSession(
        session_id="pipeline_threshold",
        action_id="timeline.extract_drum_events",
        object_id="source_audio",
        object_type="layer",
        scope="version",
        plan=ObjectActionSettingsPlan(
            action_id="timeline.extract_drum_events",
            title="Extract Onsets",
            object_id="source_audio",
            object_type="layer",
            pipeline_template_id="onset_detection",
            editable_fields=(threshold_field,),
            summary="Source Audio · This Version",
        ),
        scope_states=(
            ObjectActionSettingsScopeState(
                scope="version",
                label="This Version",
                field_values=(
                    ObjectActionSessionFieldValue(
                        key="threshold",
                        persisted_value=0.3,
                        draft_value=0.3,
                    ),
                ),
                can_run=True,
            ),
        ),
        can_save=True,
        can_save_and_run=True,
    )

    dialog = PipelineSettingsBrowserDialog(
        (session,),
        dispatch_command=lambda _session_id, command: (dispatched.append(command) or session),
    )
    try:
        widget_before = dialog._form._inputs["threshold"]
        assert isinstance(widget_before, QDoubleSpinBox)

        widget_before.setValue(0.45)
        app.processEvents()

        assert len(dispatched) == 1
        assert isinstance(dispatched[0], SetSessionFieldValue)
        assert dialog._form._inputs["threshold"] is widget_before
    finally:
        dialog.close()
        app.processEvents()


def test_timeline_editor_mode_bar_groups_tools_and_syncs_state():
    app = QApplication.instance() or QApplication([])
    bar = TimelineEditorModeBar()
    try:
        bar.set_state(
            edit_mode="draw",
            snap_enabled=True,
            grid_mode="beat",
            beat_available=True,
        )

        assert bar.objectName() == "timelineEditorModeBar"
        assert bar.findChild(QWidget, "timelineEditorModeGroup") is not None
        assert bar.findChild(QWidget, "timelineEditorAssistGroup") is not None
        assert bar.findChild(QWidget, "timelineEditorShellGroup") is not None
        assert bar._settings_button.objectName() == "timelineEditorSettingsButton"
        assert bar._osc_settings_button.objectName() == "timelineEditorOscSettingsButton"
        assert bar._pipeline_settings_button.objectName() == "timelineEditorPipelineSettingsButton"
        assert list(bar._mode_buttons.keys()) == [
            "select",
            "move",
            "draw",
            "erase",
            "fix",
        ]
        assert bar._mode_buttons["select"].isHidden() is False
        assert bar._mode_buttons["move"].isHidden() is True
        assert bar._mode_buttons["draw"].isHidden() is False
        assert bar._mode_buttons["erase"].isHidden() is True
        assert bar._mode_buttons["fix"].isHidden() is False
        assert bar._mode_buttons["move"].text() == "↔ Move"
        assert bar._mode_buttons["draw"].text() == "+ Add"
        assert bar._mode_buttons["erase"].text() == "- Erase"
        assert bar._mode_buttons["fix"].text() == "🩹 Fix"
        assert bar._fix_action_buttons["select"].objectName() == "timelineEditorFixSelectButton"
        assert bar._mode_buttons["draw"].isChecked()
        assert bar._snap_button.isChecked()
        assert bar._grid_button.text() == "▦ Grid: Beats"

        bar.set_state(
            edit_mode="erase",
            snap_enabled=True,
            grid_mode="beat",
            beat_available=True,
        )

        assert bar._mode_buttons["erase"].isHidden() is False
        assert bar._mode_buttons["erase"].isChecked() is True
        assert bar._mode_buttons["move"].isHidden() is True
    finally:
        bar.close()
        app.processEvents()


def test_timeline_editor_mode_bar_emits_pipeline_settings_requested_signal():
    app = QApplication.instance() or QApplication([])
    bar = TimelineEditorModeBar()
    emitted: list[bool] = []
    try:
        bar.pipeline_settings_requested.connect(lambda: emitted.append(True))

        bar._pipeline_settings_button.click()

        assert emitted == [True]
    finally:
        bar.close()
        app.processEvents()


def test_timeline_editor_mode_bar_emits_osc_settings_requested_signal():
    app = QApplication.instance() or QApplication([])
    bar = TimelineEditorModeBar()
    emitted: list[bool] = []
    try:
        bar.osc_settings_requested.connect(lambda: emitted.append(True))

        bar._osc_settings_button.click()

        assert emitted == [True]
    finally:
        bar.close()
        app.processEvents()


def test_timeline_editor_mode_bar_switches_to_compact_density_when_narrow():
    app = QApplication.instance() or QApplication([])
    bar = TimelineEditorModeBar()
    try:
        bar.set_state(
            edit_mode="select",
            snap_enabled=True,
            grid_mode="auto",
            beat_available=True,
        )
        bar.resize(900, 56)
        bar.show()
        app.processEvents()

        assert bar.property("compact") is True
        assert bar._mode_buttons["select"].text() == "↖"
        assert bar._mode_buttons["move"].text() == "↔"
        assert bar._mode_buttons["move"].isHidden() is True
        assert bar._mode_buttons["erase"].isHidden() is True
        assert bar._settings_button.text() == ""
        assert bar._osc_settings_button.text() == "OSC"
        assert bar._pipeline_settings_button.text() == "Pipe"
        assert bar._grid_button.text() == "▦T"
        assert bar._fix_action_buttons["select"].isVisible() is False
    finally:
        bar.close()
        app.processEvents()


def test_timeline_editor_pipeline_button_routes_to_pipeline_settings_browser(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    opened: list[bool] = []
    try:
        monkeypatch.setattr(
            widget._action_router,
            "open_pipeline_settings_browser",
            lambda: opened.append(True),
        )

        widget._editor_bar._pipeline_settings_button.click()

        assert opened == [True]
    finally:
        widget.close()
        app.processEvents()


def test_timeline_editor_pipeline_button_opens_application_defaults_without_active_song(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    opened_scopes: list[tuple[str, ...]] = []
    try:
        empty_presentation = replace(
            widget.presentation,
            active_song_id="",
            active_song_version_id="",
            active_song_title="",
            active_song_version_label="",
            layers=[],
            selected_layer_id=None,
            selected_layer_ids=[],
        )
        widget.set_presentation(empty_presentation)

        class _Runtime:
            def presentation(self):
                return empty_presentation

            def open_object_action_session(self, action_id, _params, **_kwargs):
                model_field = ObjectActionSettingField(
                    key="model",
                    label="Model",
                    value="latest_model",
                    default_value="latest_model",
                    persisted_value="latest_model",
                )
                field_values = (
                    ObjectActionSessionFieldValue(
                        key="model",
                        persisted_value="latest_model",
                        draft_value="latest_model",
                    ),
                )
                return ObjectActionSettingsSession(
                    session_id=f"session_{action_id}",
                    action_id=action_id,
                    object_id="",
                    object_type="layer",
                    scope="app_default",
                    plan=ObjectActionSettingsPlan(
                        action_id=action_id,
                        title="Stage",
                        object_id="",
                        object_type="layer",
                        pipeline_template_id="stem_separation",
                        editable_fields=(model_field,),
                        summary="No target layer selected · Application Default",
                    ),
                    scope_states=(
                        ObjectActionSettingsScopeState(
                            scope="app_default",
                            label="Application Default",
                            field_values=field_values,
                            can_run=False,
                        ),
                    ),
                    default_save_scope="app_default",
                    default_save_label="Application Default",
                    can_save=True,
                    can_save_and_run=False,
                    run_disabled_reason="Application defaults are the reusable baseline.",
                )

            def dispatch_object_action_command(self, *_args, **_kwargs):
                raise AssertionError("dialog should not dispatch in this route test")

        class _Dialog:
            def __init__(self, sessions, **_kwargs):
                opened_scopes.append(tuple(session.scope for session in sessions))

            def exec(self):
                return 0

        monkeypatch.setattr(widget._action_router, "_resolve_runtime_shell", lambda: _Runtime())
        monkeypatch.setattr(
            widget._action_router,
            "_pipeline_settings_browser_dialog_class",
            _Dialog,
        )

        widget._editor_bar._pipeline_settings_button.click()

        assert opened_scopes
        assert all(scope == "app_default" for scope in opened_scopes[0])
    finally:
        widget.close()
        app.processEvents()


def test_open_object_action_settings_routes_to_focused_settings_dialog(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    captured: list[InspectorAction | None] = []
    try:
        monkeypatch.setattr(
            widget._action_router,
            "_open_single_action_settings_dialog",
            lambda *, action: captured.append(action),
        )

        action = InspectorAction(
            action_id="timeline.extract_stems",
            label="Extract Stems",
            kind="settings",
            params={"layer_id": "source_audio"},
        )
        widget._action_router.open_object_action_settings(action)

        assert captured == [action]
    finally:
        widget.close()
        app.processEvents()


def test_transport_layout_centers_compact_bar_content():
    layout = TransportLayout.create(width=1280)
    center_lines = (
        layout.controls_rect.center().y(),
        layout.time_rect.center().y(),
        layout.meta_rect.center().y(),
    )

    assert layout.rect.height() == TIMELINE_TRANSPORT_HEIGHT_PX
    assert max(center_lines) - min(center_lines) <= 0.5


def test_transport_layout_keeps_time_and_meta_within_bounds_when_width_is_narrow():
    horizontal_padding = 24.0
    for width in (1280, 900, 720, 640, 560, 500):
        layout = TransportLayout.create(width=width)
        inner_right = width - horizontal_padding

        assert layout.controls_rect.left() >= horizontal_padding
        assert layout.controls_rect.right() <= inner_right + 0.5

        if layout.time_rect.width() > 0.0:
            assert layout.time_rect.left() >= layout.controls_rect.right()
            assert layout.time_rect.right() <= inner_right + 0.5

        if layout.meta_rect.width() > 0.0:
            assert layout.meta_rect.left() >= layout.time_rect.right()
            assert layout.meta_rect.right() <= inner_right + 0.5


def test_transport_bar_meta_text_keeps_stopped_state_visible_when_width_is_small():
    app = QApplication.instance() or QApplication([])
    presentation = build_demo_app().presentation()
    block = TransportBarBlock()
    expected_status = "PLAYING" if presentation.is_playing else "STOPPED"

    text = block._status_meta_text(
        presentation=presentation,
        available_width=120.0,
        font_metrics=QFontMetrics(app.font()),
    )

    assert expected_status in text


def test_transport_bar_meta_text_promotes_bpm_when_available():
    app = QApplication.instance() or QApplication([])
    presentation = replace(
        build_demo_app().presentation(),
        bpm=128.0,
        bpm_confidence=0.9,
    )
    block = TransportBarBlock()

    text = block._status_meta_text(
        presentation=presentation,
        available_width=240.0,
        font_metrics=QFontMetrics(app.font()),
    )

    assert "128 BPM" in text


def test_transport_bar_block_paint_returns_control_rects_without_error():
    presentation = build_demo_app().presentation()
    block = TransportBarBlock()
    layout = TransportLayout.create(width=1280, height=TIMELINE_TRANSPORT_HEIGHT_PX)
    image = QImage(1280, TIMELINE_TRANSPORT_HEIGHT_PX, QImage.Format.Format_ARGB32_Premultiplied)
    painter = QPainter(image)
    try:
        control_rects = block.paint(painter, layout, presentation)
    finally:
        painter.end()

    assert set(control_rects) == {"play", "stop", "follow"}
    assert control_rects["play"].width() > 0.0
    assert control_rects["stop"].width() > 0.0
    assert control_rects["follow"].width() > 0.0


def test_visible_ruler_marks_use_bars_when_bpm_and_anchor_are_available():
    presentation = replace(
        _song_switching_presentation(),
        bpm=120.0,
        beat_anchor_seconds=0.5,
        pixels_per_second=120.0,
        scroll_x=0.0,
    )

    marks = visible_ruler_marks(
        presentation=presentation,
        content_width=720.0,
        content_start_x=160.0,
    )

    assert marks
    assert marks[0][0] == "1|1"


def test_timeline_widget_top_chrome_is_compact_and_centered():
    app = QApplication.instance() or QApplication([])
    ensure_qt_theme_installed(app)
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        widget.resize(1400, 900)
        widget.show()
        app.processEvents()

        mode_group = widget._editor_bar.findChild(QWidget, "timelineEditorModeGroup")
        assert mode_group is not None
        edit_label = next(
            label
            for label in mode_group.findChildren(QLabel)
            if label.property("timelineToolbarLabel") is True
        )
        select_button = widget._editor_bar._mode_buttons["select"]
        bar_center_y = widget._editor_bar.rect().center().y()
        group_center_y = mode_group.rect().center().y()

        assert widget._transport.height() == TIMELINE_TRANSPORT_HEIGHT_PX
        assert widget._editor_bar.height() <= 60
        assert abs(mode_group.geometry().center().y() - bar_center_y) <= 2
        assert abs(edit_label.geometry().center().y() - group_center_y) <= 2
        assert abs(select_button.geometry().center().y() - group_center_y) <= 2
        assert edit_label.height() >= TIMELINE_EDITOR_BUTTON_MIN_HEIGHT_PX
        assert abs(edit_label.height() - select_button.height()) <= 4
    finally:
        widget.close()
        app.processEvents()


def test_action_settings_form_returns_current_values():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_stems",
                title="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="stem_separation",
                editable_fields=(
                    ObjectActionSettingField(
                        key="mode",
                        label="Mode",
                        value="merge",
                        default_value="merge",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(value="merge", label="Merge"),
                            ObjectActionSettingOption(value="overwrite", label="Overwrite"),
                        ),
                    ),
                ),
                advanced_fields=(
                    ObjectActionSettingField(
                        key="shifts",
                        label="Shifts",
                        value=1,
                        default_value=1,
                        widget="number",
                    ),
                ),
            )
        )

        form._inputs["mode"].setCurrentIndex(1)
        form._inputs["shifts"].setValue(3)

        assert form.values() == {"mode": "overwrite", "shifts": 3}
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_groups_extract_classified_drums_intentionally():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_classified_drums",
                title="Extract Classified Drums",
                object_id="drums",
                object_type="layer",
                pipeline_template_id="extract_classified_drums",
                editable_fields=(
                    ObjectActionSettingField(
                        key="target_drum_labels",
                        label="Drum Outputs",
                        value=("kick", "snare", "clap"),
                        default_value=("kick", "snare"),
                        widget="checkbox_group",
                        options=(
                            ObjectActionSettingOption(value="kick", label="Kick"),
                            ObjectActionSettingOption(value="snare", label="Snare"),
                            ObjectActionSettingOption(value="clap", label="Clap"),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="sensitivity_preset",
                        label="Sensitivity",
                        value="balanced",
                        default_value="balanced",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(value="more_events", label="More Events"),
                            ObjectActionSettingOption(value="balanced", label="Balanced"),
                            ObjectActionSettingOption(value="fewer_events", label="Fewer Events"),
                            ObjectActionSettingOption(value="custom", label="Custom / Advanced"),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="kick_model_path",
                        label="Kick Model",
                        value="/models/kick.manifest.json",
                        default_value="",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(
                                value="/models/kick.manifest.json",
                                label="Kick Bundle",
                                metadata={"status": "ready"},
                            ),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="snare_model_path",
                        label="Snare Model",
                        value="/models/snare.manifest.json",
                        default_value="",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(
                                value="/models/snare.manifest.json",
                                label="Snare Bundle",
                            ),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="clap_model_path",
                        label="Clap Model",
                        value="/models/clap.manifest.json",
                        default_value="",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(
                                value="/models/clap.manifest.json",
                                label="Clap Bundle",
                            ),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="cymbal_model_path",
                        label="Cymbal Model",
                        value="/models/cymbal.manifest.json",
                        default_value="",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(
                                value="/models/cymbal.manifest.json",
                                label="Cymbal Bundle",
                            ),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="kick_positive_threshold",
                        label="Kick Classification Threshold",
                        value=0.5,
                        default_value=0.5,
                        widget="number",
                    ),
                    ObjectActionSettingField(
                        key="snare_positive_threshold",
                        label="Snare Classification Threshold",
                        value=0.5,
                        default_value=0.5,
                        widget="number",
                    ),
                    ObjectActionSettingField(
                        key="clap_positive_threshold",
                        label="Clap Classification Threshold",
                        value=0.5,
                        default_value=0.5,
                        widget="number",
                    ),
                    ObjectActionSettingField(
                        key="cymbal_positive_threshold",
                        label="Cymbal Classification Threshold",
                        value=0.5,
                        default_value=0.5,
                        widget="number",
                    ),
                    ObjectActionSettingField(
                        key="kick_min_separation_ms",
                        label="Kick Dedup Window (ms)",
                        value=80.0,
                        default_value=80.0,
                        widget="number",
                    ),
                    ObjectActionSettingField(
                        key="kick_filter_enabled",
                        label="Kick Filter Enabled",
                        value=True,
                        default_value=True,
                        widget="toggle",
                    ),
                    ObjectActionSettingField(
                        key="snare_filter_enabled",
                        label="Snare Filter Enabled",
                        value=True,
                        default_value=True,
                        widget="toggle",
                    ),
                    ObjectActionSettingField(
                        key="clap_filter_enabled",
                        label="Clap Filter Enabled",
                        value=True,
                        default_value=True,
                        widget="toggle",
                    ),
                    ObjectActionSettingField(
                        key="cymbal_filter_enabled",
                        label="Cymbal Filter Enabled",
                        value=True,
                        default_value=True,
                        widget="toggle",
                    ),
                ),
                advanced_fields=(
                    ObjectActionSettingField(
                        key="assignment_mode",
                        label="Assignment Mode",
                        value="independent",
                        default_value="independent",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(
                                value="independent",
                                label="Independent (Allow Overlaps)",
                            ),
                            ObjectActionSettingOption(
                                value="exclusive_max",
                                label="Winner Takes Similar Hits",
                            ),
                        ),
                    ),
                ),
            )
        )

        labels = {label.text() for label in form.findChildren(QLabel)}
        assert "Events to find" in labels
        assert "Quality" in labels
        assert "Model status" in labels
        assert "Advanced / Custom recovery" in labels
        assert form._page.title == "Extract Drum Events"
        assert set(form._inputs) >= {
            "target_drum_labels",
            "sensitivity_preset",
            "kick_model_path",
            "snare_model_path",
            "clap_model_path",
            "cymbal_model_path",
            "kick_positive_threshold",
            "snare_positive_threshold",
            "clap_positive_threshold",
            "cymbal_positive_threshold",
            "kick_min_separation_ms",
            "kick_filter_enabled",
            "snare_filter_enabled",
            "clap_filter_enabled",
            "cymbal_filter_enabled",
            "assignment_mode",
        }

        sections_by_title = {
            section.title: section for section in form._page.sections if section.fields
        }
        assert [field.key for field in sections_by_title["Events to find"].fields] == [
            "target_drum_labels"
        ]
        assert [field.key for field in sections_by_title["Quality"].fields] == [
            "sensitivity_preset"
        ]
        assert [field.key for field in sections_by_title["Model status"].fields] == [
            "kick_model_path",
            "snare_model_path",
            "clap_model_path",
            "cymbal_model_path",
        ]
        assert {
            field.surface for field in sections_by_title["Model status"].fields
        } == {SettingsFieldSurface.ADVANCED}
        assert {
            field.surface for field in sections_by_title["Advanced / Custom recovery"].fields
        } == {SettingsFieldSurface.ADVANCED}
        advanced_keys = {
            field.key for field in sections_by_title["Advanced / Custom recovery"].fields
        }
        assert {
            "kick_positive_threshold",
            "snare_positive_threshold",
            "clap_positive_threshold",
            "cymbal_positive_threshold",
            "kick_filter_enabled",
            "assignment_mode",
        } <= advanced_keys
        worksheet_grids = [
            layout
            for layout in form.findChildren(QGridLayout)
            if layout.property("worksheet") == "classifier_models"
        ]
        assert worksheet_grids == []
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_compacts_song_drum_events_primary_controls():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_song_drum_events",
                title="Extract Drum Events",
                object_id="song",
                object_type="layer",
                pipeline_template_id="extract_song_drum_events",
                editable_fields=(
                    ObjectActionSettingField(
                        key="target_drum_labels",
                        label="Drum Outputs",
                        value=("kick", "snare"),
                        default_value=("kick", "snare"),
                        widget="checkbox_group",
                        options=(
                            ObjectActionSettingOption(value="kick", label="Kick"),
                            ObjectActionSettingOption(value="snare", label="Snare"),
                        ),
                    ),
                    ObjectActionSettingField(
                        key="sensitivity_preset",
                        label="Sensitivity",
                        value="balanced",
                        default_value="balanced",
                        widget="dropdown",
                        options=(ObjectActionSettingOption(value="balanced", label="Balanced"),),
                    ),
                    ObjectActionSettingField(
                        key="kick_onset_threshold",
                        label="Kick Detection Threshold",
                        value=0.25,
                        default_value=0.25,
                        widget="number",
                    ),
                ),
            )
        )

        sections_by_title = {
            section.title: section for section in form._page.sections if section.fields
        }
        assert [field.key for field in sections_by_title["Events to find"].fields] == [
            "target_drum_labels"
        ]
        assert [field.key for field in sections_by_title["Quality"].fields] == [
            "sensitivity_preset"
        ]
        assert [
            field.key for field in sections_by_title["Advanced / Custom recovery"].fields
        ] == ["kick_onset_threshold"]
        assert {
            field.surface for field in sections_by_title["Advanced / Custom recovery"].fields
        } == {SettingsFieldSurface.ADVANCED}
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_emits_field_value_changed_for_operator_edits():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    emitted = []
    try:
        form.field_value_changed.connect(lambda key, value: emitted.append((key, value)))
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_stems",
                title="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="stem_separation",
                editable_fields=(
                    ObjectActionSettingField(
                        key="mode",
                        label="Mode",
                        value="merge",
                        default_value="merge",
                        widget="dropdown",
                        options=(
                            ObjectActionSettingOption(value="merge", label="Merge"),
                            ObjectActionSettingOption(value="overwrite", label="Overwrite"),
                        ),
                    ),
                ),
                advanced_fields=(
                    ObjectActionSettingField(
                        key="shifts",
                        label="Shifts",
                        value=1,
                        default_value=1,
                        widget="number",
                    ),
                ),
            )
        )

        form._inputs["mode"].setCurrentIndex(1)
        form._inputs["shifts"].setValue(3)

        assert emitted == [("mode", "overwrite"), ("shifts", 3)]
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_number_inputs_disable_keyboard_tracking():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_stems",
                title="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="stem_separation",
                editable_fields=(
                    ObjectActionSettingField(
                        key="threshold",
                        label="Threshold",
                        value=0.3,
                        default_value=0.3,
                        widget="number",
                    ),
                ),
                advanced_fields=(
                    ObjectActionSettingField(
                        key="shifts",
                        label="Shifts",
                        value=1,
                        default_value=1,
                        widget="number",
                    ),
                ),
            )
        )

        threshold_widget = form._inputs["threshold"]
        shifts_widget = form._inputs["shifts"]
        assert isinstance(threshold_widget, QDoubleSpinBox)
        assert isinstance(shifts_widget, QSpinBox)
        assert threshold_widget.keyboardTracking() is False
        assert shifts_widget.keyboardTracking() is False
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_shows_threshold_tooltips_on_label_and_input():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_stems",
                title="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="stem_separation",
                editable_fields=(
                    ObjectActionSettingField(
                        key="threshold",
                        label="Threshold",
                        value=0.3,
                        default_value=0.3,
                        widget="number",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.05,
                    ),
                ),
            )
        )

        widget_tooltip = form._inputs["threshold"].toolTip()
        label = next(
            widget for widget in form.findChildren(QLabel) if widget.text() == "Threshold"
        )

        assert "Threshold strictness" in widget_tooltip
        assert "Range: 0 to 1" in widget_tooltip
        assert "Default: 0.3" in widget_tooltip
        assert label.toolTip() == widget_tooltip
    finally:
        form.close()
        app.processEvents()


def test_action_settings_form_switching_plans_keeps_single_section_title():
    app = QApplication.instance() or QApplication([])
    form = ActionSettingsForm()
    try:
        form.show()
        app.processEvents()
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_stems",
                title="Extract Stems",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="stem_separation",
                editable_fields=(
                    ObjectActionSettingField(
                        key="alpha",
                        label="Alpha",
                        value="a",
                        default_value="a",
                    ),
                ),
            )
        )
        form.set_plan(
            ObjectActionSettingsPlan(
                action_id="timeline.extract_drum_events",
                title="Extract Onsets",
                object_id="source_audio",
                object_type="layer",
                pipeline_template_id="onset_detection",
                editable_fields=(
                    ObjectActionSettingField(
                        key="beta",
                        label="Beta",
                        value="b",
                        default_value="b",
                    ),
                ),
            )
        )
        app.processEvents()

        visible_stage_titles = [
            label
            for label in form.findChildren(QLabel)
            if label.text() == "Stage Settings" and label.isVisible()
        ]
        assert len(visible_stage_titles) == 1
        assert set(form._inputs.keys()) == {"beta"}
    finally:
        form.close()
        app.processEvents()


def test_timeline_scroll_area_stylesheet_uses_shared_tokens():
    stylesheet = build_timeline_scroll_area_stylesheet()

    assert TIMELINE_STYLE.scroll_area_background_hex in stylesheet
    assert "border: none;" in stylesheet


def test_timeline_blocks_default_to_shared_style_tokens():
    assert TransportBarBlock().style is TIMELINE_STYLE.transport_bar
    assert LayerHeaderBlock().style is TIMELINE_STYLE.layer_header
    assert TakeRowBlock().style is TIMELINE_STYLE.take_row
    assert EventLaneBlock().style is TIMELINE_STYLE.event_lane
    assert WaveformLaneBlock().style is TIMELINE_STYLE.waveform_lane

    ruler = RulerBlock()
    assert ruler.style is TIMELINE_STYLE.ruler
    assert ruler.playhead_color_hex == TIMELINE_STYLE.playhead.color_hex


def test_timeline_widget_shell_uses_shared_style_tokens():
    app = QApplication.instance() or QApplication([])
    ensure_qt_theme_installed(app)
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        assert widget.windowTitle() == TIMELINE_STYLE.window_title
        assert widget._canvas._style is TIMELINE_STYLE
        assert app.styleSheet() == build_echozero_app_qss()
        assert widget._editor_bar.objectName() == "timelineEditorModeBar"
        assert widget._shell_splitter.objectName() == "timelineShellSplitter"
        assert widget._main_splitter.objectName() == "timelineMainSplitter"
        assert widget._editor_bar.styleSheet() == ""
        assert widget._scroll.styleSheet() == ""
        assert widget._shell_splitter.styleSheet() == ""
        assert widget._main_splitter.styleSheet() == ""
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_song_browser_splitter_restores_resized_width():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    try:
        widget.resize(1600, 900)
        widget.show()
        app.processEvents()

        widget._shell_splitter.setSizes([340, 1260])
        app.processEvents()
        expected_width = widget._shell_splitter.sizes()[0]
        widget._sync_song_browser_splitter_width()
        assert widget._song_browser_panel.expanded_width == expected_width

        widget._song_browser_panel.toggle_collapsed()
        app.processEvents()
        assert widget._shell_splitter.sizes()[0] <= 80

        widget._song_browser_panel.toggle_collapsed()
        app.processEvents()
        assert widget._shell_splitter.sizes()[0] >= 220
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_object_info_splitter_restores_resized_width():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    try:
        widget.resize(1600, 900)
        widget.show()
        app.processEvents()

        widget._main_splitter.setSizes([1220, 380])
        app.processEvents()
        expected_width = widget._main_splitter.sizes()[1]
        widget._sync_object_info_splitter_width()
        assert widget._object_info.expanded_width == expected_width

        widget._object_info.toggle_collapsed()
        app.processEvents()
        assert widget._main_splitter.sizes()[1] <= 80

        widget._object_info.toggle_collapsed()
        app.processEvents()
        assert widget._main_splitter.sizes()[1] >= 220
    finally:
        widget.close()
        app.processEvents()


def test_timeline_widget_can_condense_width_below_legacy_canvas_floor():
    app = QApplication.instance() or QApplication([])
    widget = TimelineWidget(_song_switching_presentation())
    try:
        widget.resize(1600, 900)
        widget.show()
        app.processEvents()

        widget.resize(1280, 900)
        app.processEvents()

        assert widget.width() <= 1280
        assert widget._canvas.minimumWidth() < 600
    finally:
        widget.close()
        app.processEvents()


def test_timeline_fixture_tokens_are_discoverable_from_style_module():
    assert fixture_color("song") == TIMELINE_STYLE.fixture.layer_color_tokens["song"]
    assert fixture_color("sync") == TIMELINE_STYLE.fixture.layer_color_tokens["sync"]
    assert fixture_take_action_label("overwrite_main") == "Overwrite Main"
    assert fixture_take_action_label("merge_main") == "Merge Main"
    assert fixture_take_action_label("delete_take") == "Delete Take"
