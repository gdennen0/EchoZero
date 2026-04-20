from PyQt6.QtWidgets import QApplication, QDialogButtonBox

from echozero.application.presentation.inspector_contract import (
    InspectorAction,
    InspectorContextSection,
    InspectorContract,
    build_timeline_inspector_contract,
)
from echozero.application.timeline.object_actions import (
    ObjectActionSessionFieldValue,
    ObjectActionSettingField,
    ObjectActionSettingOption,
    ObjectActionSettingsPlan,
    ObjectActionSettingsScopeState,
    ObjectActionSettingsSession,
)
from echozero.ui.qt.settings_dialog import ActionSettingsDialog
from echozero.ui.qt.settings_form import ActionSettingsForm
from echozero.ui.qt.timeline.blocks.event_lane import EventLaneBlock
from echozero.ui.qt.timeline.blocks.layer_header import LayerHeaderBlock
from echozero.ui.qt.timeline.blocks.ruler import RulerBlock
from echozero.ui.qt.timeline.blocks.take_row import TakeRowBlock
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
from echozero.ui.qt.timeline.widget import ObjectInfoPanel, TimelineWidget


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
        assert panel._settings_buttons["timeline.extract_stems"].text() == "Open Settings"
        assert panel._action_buttons["timeline.extract_stems"].text() == "Run"
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
        assert dialog.windowTitle() == "Pipeline Settings: This Version · Extract Stems"
        assert dialog._copy_group.title() == "Copy Settings Into This Version"
        assert dialog._stage_group.title() == "Stage Settings: Extract Stems"
        assert "Scope: This Version" in dialog._context.text()
        assert (
            dialog._buttons.button(QDialogButtonBox.StandardButton.Apply).text()
            == "Save And Rerun"
        )
    finally:
        dialog.close()
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
    widget = TimelineWidget(build_demo_app().presentation())
    try:
        assert widget.windowTitle() == TIMELINE_STYLE.window_title
        assert widget._canvas._style is TIMELINE_STYLE
        assert build_timeline_scroll_area_stylesheet(widget._style) == widget._scroll.styleSheet()
    finally:
        widget.close()
        app.processEvents()


def test_timeline_fixture_tokens_are_discoverable_from_style_module():
    assert fixture_color("song") == TIMELINE_STYLE.fixture.layer_color_tokens["song"]
    assert fixture_color("sync") == TIMELINE_STYLE.fixture.layer_color_tokens["sync"]
    assert fixture_take_action_label("overwrite_main") == "Overwrite Main"
    assert fixture_take_action_label("merge_main") == "Merge Main"
